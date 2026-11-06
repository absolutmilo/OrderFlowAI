"""
FastAPI Server and WebSocket Broadcaster.
Streams aggregated order flow and real-time microstructure probabilities to UI clients.
Supports multi-exchange aggregation (Binance + Bybit) replicating SignificantTrades / aggr.trade.
"""

import asyncio
import json
import logging
import os
import sqlite3
import time
from dataclasses import asdict
from pathlib import Path
from typing import Set, List
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from src.core.trade_model import AggregatedTrade, RawTrade
from src.core.aggregator import TradeAggregator
from src.core.classifier import MicrostructureClassifier
from src.core.market_data_store import MarketDataStore
from src.core.scalping import STRATEGIES, ScalpingEngine
from src.connectors.binance import BinanceConnector
from src.connectors.bybit import BybitLinearConnector
from src.api.chart_routes import router as chart_router
from config import CONFIG

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("OrderFlowAI.Server")

app = FastAPI(title="OrderFlowAI - Real-time Market Microstructure & Sonification")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CONFIG.server.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared components
classifier = MicrostructureClassifier()
scalp_classifier = MicrostructureClassifier(inference_mode="heuristic")
database_path = Path(__file__).resolve().parents[2] / CONFIG.data_capture.database_path
market_data_store = MarketDataStore(
    database_path,
    CONFIG.data_capture.label_horizons_seconds,
    CONFIG.data_capture.label_min_move_bps,
)
scalping_engine = ScalpingEngine(database_path, CONFIG.scalping)
active_websockets: Set[WebSocket] = set()
connectors: List = []
trade_counter = 0
exchange_counts = {"Binance": 0, "Bybit": 0}
last_capture_at_ms = 0
data_capture_status = "starting"


def get_inference_mode() -> str:
    return classifier.inference_mode


def snapshot_scores(heuristic_probabilities, hmm_probabilities=None) -> dict:
    scores = asdict(heuristic_probabilities)
    scores["hmm_4"] = (
        {"available": True, **asdict(hmm_probabilities)}
        if hmm_probabilities is not None
        else {"available": False}
    )
    return scores


async def broadcast_message(message: dict):
    """Broadcasts a JSON payload to all connected browser clients."""
    if not active_websockets:
        return
    payload = json.dumps(message)
    dead_sockets = set()
    for ws in active_websockets:
        try:
            await ws.send_text(payload)
        except Exception:
            dead_sockets.add(ws)
    for ws in dead_sockets:
        active_websockets.discard(ws)


async def handle_aggregated_trade(agg_trade: AggregatedTrade):
    """Classify each aggregated event and periodically persist labeled training data."""
    global last_capture_at_ms, data_capture_status
    global trade_counter
    trade_counter += 1

    # Update classifier
    classifier.add_trade(agg_trade)
    probabilities = classifier.classify()
    metrics = classifier.get_current_metrics()

    now_ms = int(time.time() * 1000)
    if agg_trade.exchange == CONFIG.scalping.exchange:
        scalp_classifier.add_trade(agg_trade)
        scalp_probabilities = scalp_classifier.classify()
        scalp_metrics = scalp_classifier.get_current_metrics()

        if (
            scalp_metrics is not None
            and scalp_classifier.is_warmed_up()
            and now_ms - last_capture_at_ms >= CONFIG.data_capture.sample_interval_seconds * 1000
        ):
            previous_capture_at_ms = last_capture_at_ms
            last_capture_at_ms = now_ms
            try:
                hmm_probabilities = None
                if scalp_classifier.hmm_model is not None:
                    if (
                        previous_capture_at_ms
                        and now_ms - previous_capture_at_ms
                        > CONFIG.scalping.max_snapshot_gap_seconds * 1000
                    ):
                        scalp_classifier.hmm_model.reset()
                    hmm_probabilities = scalp_classifier.classify(mode="model")

                scalp_features = asdict(scalp_metrics)
                market_data_store.record_observation(
                    observed_at_ms=now_ms,
                    symbol=agg_trade.symbol,
                    price=scalp_metrics.current_price,
                    cvd_usd=scalp_metrics.cvd_usd,
                    features=scalp_features,
                    heuristic_scores=snapshot_scores(
                        scalp_probabilities,
                        hmm_probabilities,
                    ),
                    configuration=asdict(CONFIG),
                    market_source=CONFIG.scalping.exchange,
                )
                scalping_engine.observe(
                    now_ms,
                    scalp_metrics.current_price,
                    scalp_features,
                )
                data_capture_status = "recording"
            except sqlite3.Error:
                logger.exception("No se pudo guardar el snapshot Bybit Linear en SQLite.")
                data_capture_status = "error"

    # Pattern tags
    patterns = []
    if probabilities.algo_institutional > 70.0:
        patterns.append("ALGO")
    if probabilities.iceberg > 60.0:
        patterns.append(f"ICEBERG-{probabilities.iceberg_side or 'LIMIT'}")
    if probabilities.absorption > 60.0:
        patterns.append("ABSORPTION")
    if probabilities.hft_competition > 75.0:
        patterns.append("HFT")
    if probabilities.breakout > 70.0:
        patterns.append("BREAKOUT")
    agg_trade.detected_patterns = patterns

    # Broadcast to web dashboard
    broadcast_data = {
        "type": "order_flow_tick",
        "trade": agg_trade.to_dict(),
        "probabilities": probabilities.to_dict(),
        "metrics": metrics.to_dict() if metrics else None,
        "exchange_stats": exchange_counts,
        "inference_mode": classifier.inference_mode,
        "data_capture_status": data_capture_status,
        "detector_config": asdict(CONFIG.classifier),
        "scalping": {
            "signal": scalping_engine.last_signal or {"status": "no_setup"},
            "alert_enabled": scalping_engine.parameters is not None,
        },
    }
    await broadcast_message(broadcast_data)


aggregator = TradeAggregator(on_aggregated_trade=handle_aggregated_trade)


async def handle_raw_trade(raw_trade: RawTrade):
    """Feeds incoming exchange raw trades into the aggregator."""
    if "Binance" in raw_trade.exchange:
        exchange_counts["Binance"] += 1
    elif "Bybit" in raw_trade.exchange:
        exchange_counts["Bybit"] += 1

    await aggregator.process_raw_trade(raw_trade)


@app.on_event("startup")
async def on_startup():
    global connectors, data_capture_status, last_capture_at_ms
    connectors.clear()
    last_capture_at_ms = 0
    market_data_store.initialize()
    data_capture_status = "waiting_for_warmup"
    logger.info("Iniciando conectores de OrderFlowAI...")

    # 1. Binance Spot (Unrestricted, high volume, global)
    if CONFIG.exchange.enable_binance_spot:
        binance_conn = BinanceConnector(
            symbol=CONFIG.exchange.symbol,
            on_trade_callback=handle_raw_trade,
            use_futures=False
        )
        connectors.append(binance_conn)
        await binance_conn.start()

    # 2. Bybit Linear (Perpetual futures order flow)
    if CONFIG.exchange.enable_bybit_linear:
        bybit_conn = BybitLinearConnector(
            symbol=CONFIG.scalping.symbol,
            on_trade_callback=handle_raw_trade
        )
        connectors.append(bybit_conn)
        await bybit_conn.start()

    logger.info(f"OrderFlowAI pipeline activo con {len(connectors)} conectores de exchanges.")


@app.on_event("shutdown")
async def on_shutdown():
    global connectors, data_capture_status
    for conn in connectors:
        await conn.stop()
    connectors.clear()
    market_data_store.close()
    data_capture_status = "stopped"
    logger.info("OrderFlowAI detenido.")


@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_websockets.add(websocket)
    logger.info(f"Cliente UI conectado. Total clientes: {len(active_websockets)}")
    try:
        # Send initial snapshot immediately upon connection
        metrics = classifier.get_current_metrics()
        probs = classifier.classify()
        await websocket.send_text(json.dumps({
            "type": "snapshot",
            "probabilities": probs.to_dict(),
            "metrics": metrics.to_dict() if metrics else None,
            "exchange_stats": exchange_counts,
            "inference_mode": classifier.inference_mode,
            "data_capture_status": data_capture_status,
            "detector_config": asdict(CONFIG.classifier),
            "scalping": {
                "signal": scalping_engine.last_signal or {"status": "no_setup"},
                "alert_enabled": scalping_engine.parameters is not None,
            },
        }))

        while True:
            raw_msg = await websocket.receive_text()
            try:
                msg_data = json.loads(raw_msg)
                action = msg_data.get("action")
                if action == "toggle_inference_mode":
                    new_mode = classifier.toggle_inference_mode()
                    logger.info(f"Toggle inference mode vía WS: {new_mode}")
                    await broadcast_message({
                        "type": "mode_change",
                        "inference_mode": new_mode,
                    })
                elif action == "set_inference_mode":
                    new_mode = classifier.set_inference_mode(msg_data.get("mode", "model"))
                    logger.info(f"Set inference mode vía WS: {new_mode}")
                    await broadcast_message({
                        "type": "mode_change",
                        "inference_mode": new_mode,
                    })
            except (json.JSONDecodeError, AttributeError, ValueError, RuntimeError) as error:
                logger.warning("Acción WebSocket inválida: %s", error)
                await websocket.send_text(json.dumps({
                    "type": "mode_error",
                    "message": str(error),
                }))
    except WebSocketDisconnect:
        active_websockets.discard(websocket)
        logger.info(f"Cliente desconectado. Total clientes: {len(active_websockets)}")
    except Exception as e:
        active_websockets.discard(websocket)
        logger.warning(f"Error WebSocket: {e}")


@app.post("/api/inference-mode/toggle")
async def toggle_inference_mode_endpoint():
    try:
        new_mode = classifier.toggle_inference_mode()
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    await broadcast_message({
        "type": "mode_change",
        "inference_mode": new_mode,
    })
    return JSONResponse({"status": "ok", "inference_mode": new_mode})


@app.post("/api/inference-mode")
async def set_inference_mode_endpoint(payload: dict):
    target = payload.get("mode", "model")
    try:
        new_mode = classifier.set_inference_mode(target)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    await broadcast_message({
        "type": "mode_change",
        "inference_mode": new_mode,
    })
    return JSONResponse({"status": "ok", "inference_mode": new_mode})


@app.get("/api/scalping/status")
async def get_scalping_status():
    if database_path.exists():
        with sqlite3.connect(database_path, timeout=5) as connection:
            row = connection.execute(
                """
                SELECT COUNT(*), MIN(observed_at_ms), MAX(observed_at_ms),
                       SUM(
                           CASE
                               WHEN CASE
                                   WHEN json_valid(heuristic_scores_json)
                                   THEN json_extract(heuristic_scores_json, '$.hmm_4.available')
                                   ELSE 0
                               END = 1
                               THEN 1 ELSE 0
                           END
                       )
                FROM market_observations
                WHERE market_source = ? AND symbol = ?
                """,
                (CONFIG.scalping.exchange, CONFIG.scalping.symbol),
            ).fetchone()
    else:
        row = (0, None, None, 0)

    return JSONResponse({
        **scalping_engine.status(),
        "observations": row[0],
        "first_observation_ms": row[1],
        "last_observation_ms": row[2],
        "hmm_snapshot_count": row[3] or 0,
        "minimum_training_observations": CONFIG.scalping.min_training_observations,
        "minimum_observations": scalping_engine.status()["minimum_observations"],
        "max_snapshot_gap_seconds": CONFIG.scalping.max_snapshot_gap_seconds,
        "minimum_snapshot_coverage": CONFIG.scalping.minimum_snapshot_coverage,
        "strategies": STRATEGIES,
        "cost_configuration": {
            "taker_fee_bps_per_side": CONFIG.scalping.taker_fee_bps_per_side,
            "slippage_bps_per_side": CONFIG.scalping.slippage_bps_per_side,
        },
    })


@app.post("/api/scalping/backtest")
async def run_scalping_backtest(
    strategy: str = "flow_momentum",
    timeframe_minutes: int = 5,
    direction_mode: str = "original",
):
    if strategy not in STRATEGIES:
        raise HTTPException(status_code=400, detail="Estrategia de scalping no reconocida.")
    if timeframe_minutes not in (5, 10, 15):
        raise HTTPException(status_code=400, detail="El timeframe debe ser 5, 10 o 15 minutos.")
    if direction_mode not in ("original", "inverse"):
        raise HTTPException(status_code=400, detail="La dirección debe ser original o inverse.")

    try:
        result = await asyncio.to_thread(
            scalping_engine.run_backtest,
            strategy,
            timeframe_minutes,
            direction_mode,
        )
    except (sqlite3.Error, ValueError, json.JSONDecodeError) as error:
        logger.exception("No se pudo ejecutar el backtest de scalping.")
        raise HTTPException(status_code=503, detail=str(error)) from error

    await broadcast_message({
        "type": "scalping_backtest",
        "backtest": result,
        "alert_enabled": result.get("alert_enabled", False),
    })
    return JSONResponse(result)


@app.get("/api/status")
async def get_status():
    return JSONResponse({
        "status": "online",
        "symbol": CONFIG.exchange.symbol,
        "connectors_active": len(connectors),
        "exchange_counts": exchange_counts,
        "is_warmed_up": classifier.is_warmed_up(),
        "window_trades": len(classifier.trades),
        "inference_mode": classifier.inference_mode,
        "detector_config": asdict(CONFIG.classifier),
        "data_capture_status": data_capture_status,
        "stored_observations": market_data_store.observation_count() if market_data_store.connection is not None else None,
        "database": "SQLite",
        "connected_clients": len(active_websockets),
        "total_aggregated_trades": trade_counter
    })


@app.get("/api/classification-history")
async def get_classification_history(mode: str = "model"):
    if mode not in ("model", "heuristic"):
        raise HTTPException(
            status_code=400,
            detail="El modo debe ser model o heuristic.",
        )
    if not database_path.exists():
        return JSONResponse({"mode": mode, "samples": []})

    cutoff_ms = int(time.time() * 1000) - 60 * 60 * 1000
    with sqlite3.connect(database_path, timeout=5) as connection:
        rows = connection.execute(
            """
            SELECT observed_at_ms, heuristic_scores_json
            FROM market_observations
            WHERE market_source = ? AND symbol = ? AND observed_at_ms >= ?
            ORDER BY observed_at_ms ASC
            """,
            (CONFIG.scalping.exchange, CONFIG.scalping.symbol, cutoff_ms),
        ).fetchall()

    keys = (
        "algo_institutional",
        "iceberg",
        "absorption",
        "retail",
        "hft_competition",
    )
    samples = []
    for observed_at_ms, scores_json in rows:
        try:
            scores = json.loads(scores_json)
        except json.JSONDecodeError:
            logger.warning(
                "Se omitió un snapshot con scores inválidos en %s.",
                observed_at_ms,
            )
            continue
        if not isinstance(scores, dict):
            logger.warning(
                "Se omitió un snapshot con formato de scores no válido en %s.",
                observed_at_ms,
            )
            continue
        if mode == "model":
            hmm_scores = scores.get("hmm_4")
            if not isinstance(hmm_scores, dict) or not hmm_scores.get("available"):
                continue
            scores = hmm_scores
        samples.append({
            "timestamp_ms": observed_at_ms,
            "values": {
                key: scores.get(key, 0.0)
                for key in keys
            },
        })

    return JSONResponse({"mode": mode, "samples": samples})


# Static files mount
static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/")
    async def serve_index():
        return FileResponse(
            os.path.join(static_dir, "index.html"),
            headers={"Cache-Control": "no-cache"},
        )
