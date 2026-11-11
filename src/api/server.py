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
import aiohttp
from dataclasses import asdict
from pathlib import Path
from typing import Set, List, Optional
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from src.core.trade_model import AggregatedTrade, RawTrade
from src.core.aggregator import TradeAggregator
from src.core.classifier import MicrostructureClassifier
from src.core.market_data_store import MarketDataStore
from src.core.scalping import STRATEGIES, ScalpingEngine
from src.core.candle_backtest import CandleBacktestEngine, aggregate_klines
from src.connectors.binance import BinanceConnector
from src.connectors.bybit_kline_cache import load_or_update_bybit_klines
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
candle_backtest_engine = CandleBacktestEngine(CONFIG.scalping)
candle_cache_path = Path(__file__).resolve().parents[2] / CONFIG.scalping.candle_cache_path
active_websockets: Set[WebSocket] = set()
connectors: List = []
trade_counter = 0
exchange_counts = {"Binance": 0, "Bybit": 0}
last_capture_at_ms = 0
last_bybit_trade_at_ms = 0
snapshot_capture_task: Optional[asyncio.Task] = None
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


def capture_market_snapshot(now_ms: int) -> bool:
    global last_capture_at_ms, data_capture_status

    if (
        not last_bybit_trade_at_ms
        or now_ms - last_bybit_trade_at_ms
        > CONFIG.scalping.max_snapshot_gap_seconds * 1000
    ):
        data_capture_status = "waiting_for_feed"
        return False
    if not scalp_classifier.is_warmed_up():
        data_capture_status = "waiting_for_warmup"
        return False

    scalp_metrics = scalp_classifier.get_current_metrics()
    if scalp_metrics is None:
        data_capture_status = "waiting_for_warmup"
        return False

    previous_capture_at_ms = last_capture_at_ms
    if (
        previous_capture_at_ms
        and now_ms - previous_capture_at_ms
        > CONFIG.scalping.max_snapshot_gap_seconds * 1000
        and scalp_classifier.hmm_model is not None
    ):
        scalp_classifier.hmm_model.reset()
    hmm_probabilities = (
        scalp_classifier.classify(mode="model")
        if scalp_classifier.hmm_model is not None
        else None
    )
    scalp_probabilities = scalp_classifier.classify()
    scalp_features = asdict(scalp_metrics)
    market_data_store.record_observation(
        observed_at_ms=now_ms,
        symbol=CONFIG.scalping.symbol,
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
    last_capture_at_ms = now_ms
    data_capture_status = "recording"
    return True


async def capture_market_data_loop() -> None:
    global data_capture_status
    interval_seconds = CONFIG.data_capture.sample_interval_seconds
    if interval_seconds <= 0:
        raise ValueError("El intervalo de captura debe ser mayor que cero.")

    while True:
        await asyncio.sleep(interval_seconds)
        now_ms = int(time.time() * 1000)
        if now_ms - last_capture_at_ms < interval_seconds * 1000:
            continue
        try:
            capture_market_snapshot(now_ms)
        except sqlite3.Error:
            logger.exception("No se pudo guardar el snapshot Bybit Linear en SQLite.")
            data_capture_status = "error"


async def handle_aggregated_trade(agg_trade: AggregatedTrade):
    """Classify each aggregated event and refresh the live market state."""
    global last_bybit_trade_at_ms, trade_counter
    trade_counter += 1
    now_ms = int(time.time() * 1000)
    if agg_trade.exchange == CONFIG.scalping.exchange:
        last_bybit_trade_at_ms = now_ms

    # Update classifier
    classifier.add_trade(agg_trade)
    probabilities = classifier.classify()
    metrics = classifier.get_current_metrics()

    if agg_trade.exchange == CONFIG.scalping.exchange:
        scalp_classifier.add_trade(agg_trade)

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
    global last_bybit_trade_at_ms, snapshot_capture_task
    connectors.clear()
    last_capture_at_ms = 0
    last_bybit_trade_at_ms = 0
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

    snapshot_capture_task = asyncio.create_task(capture_market_data_loop())
    logger.info(f"OrderFlowAI pipeline activo con {len(connectors)} conectores de exchanges.")


@app.on_event("shutdown")
async def on_shutdown():
    global connectors, data_capture_status, snapshot_capture_task
    if snapshot_capture_task is not None:
        snapshot_capture_task.cancel()
        try:
            await snapshot_capture_task
        except asyncio.CancelledError:
            pass
        snapshot_capture_task = None
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


def load_scalping_status_counts(now_ms: int):
    recent_cutoff_ms = now_ms - 24 * 60 * 60 * 1000
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
            recent_row = connection.execute(
                """
                WITH ordered AS (
                    SELECT observed_at_ms,
                           observed_at_ms - LAG(observed_at_ms) OVER (
                               ORDER BY observed_at_ms
                           ) AS gap_ms
                    FROM market_observations
                    WHERE market_source = ?
                      AND symbol = ?
                      AND observed_at_ms >= ?
                )
                SELECT COUNT(*),
                       COALESCE(SUM(
                           CASE WHEN gap_ms <= ? THEN gap_ms ELSE 0 END
                       ), 0),
                       COALESCE(MAX(gap_ms), 0)
                FROM ordered
                """,
                (
                    CONFIG.scalping.exchange,
                    CONFIG.scalping.symbol,
                    recent_cutoff_ms,
                    CONFIG.scalping.max_snapshot_gap_seconds * 1000,
                ),
            ).fetchone()
    else:
        row = (0, None, None, 0)
        recent_row = (0, 0, 0)
    return row, recent_row


@app.get("/api/scalping/status")
async def get_scalping_status():
    now_ms = int(time.time() * 1000)
    row, recent_row = await asyncio.to_thread(
        load_scalping_status_counts,
        now_ms,
    )

    return JSONResponse({
        **scalping_engine.status(),
        "observations": row[0],
        "first_observation_ms": row[1],
        "last_observation_ms": row[2],
        "hmm_snapshot_count": row[3] or 0,
        "recent_observations": recent_row[0],
        "recent_snapshot_coverage_pct": round(
            min(1.0, recent_row[0] / max(1.0, recent_row[1] / 1000.0)) * 100.0,
            2,
        ),
        "recent_maximum_snapshot_gap_seconds": round(recent_row[2] / 1000.0, 2),
        "recent_window_hours": 24,
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
    early_exit_on_hmm: bool = True,
):
    if strategy not in STRATEGIES:
        raise HTTPException(status_code=400, detail="Estrategia de scalping no reconocida.")
    if timeframe_minutes not in (1, 5, 10, 15):
        raise HTTPException(status_code=400, detail="El timeframe debe ser 1, 5, 10 o 15 minutos.")
    if direction_mode not in ("original", "inverse"):
        raise HTTPException(status_code=400, detail="La dirección debe ser original o inverse.")

    try:
        result = await asyncio.to_thread(
            scalping_engine.run_backtest,
            strategy,
            timeframe_minutes,
            direction_mode,
            early_exit_on_hmm,
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


@app.post("/api/scalping/candle-backtest")
async def run_candle_backtest(
    timeframe_minutes: int = 5,
    direction_mode: str = "original",
    lookback_days: int = CONFIG.scalping.candle_backtest_lookback_days,
):
    if timeframe_minutes not in (1, 5, 10, 15):
        raise HTTPException(status_code=400, detail="El timeframe debe ser 1, 5, 10 o 15 minutos.")
    if direction_mode not in ("original", "inverse"):
        raise HTTPException(status_code=400, detail="La dirección debe ser original o inverse.")
    if not 7 <= lookback_days <= 30:
        raise HTTPException(status_code=400, detail="El periodo debe estar entre 7 y 30 días.")

    source_interval_minutes = 5 if timeframe_minutes == 10 else timeframe_minutes
    source_interval = str(source_interval_minutes)
    now_ms = int(time.time() * 1000)
    try:
        downloaded, cache_status, downloaded_count = await load_or_update_bybit_klines(
            cache_dir=candle_cache_path,
            symbol=CONFIG.scalping.symbol,
            interval=source_interval,
            lookback_days=lookback_days,
            now_ms=now_ms,
        )
        lookback_start_ms = now_ms - lookback_days * 24 * 60 * 60 * 1000
        completed = [
            candle for candle in downloaded
            if (
                lookback_start_ms <= int(candle["time_ms"])
                and int(candle["time_ms"]) + source_interval_minutes * 60_000 <= now_ms
            )
        ]
        candles = aggregate_klines(
            completed,
            source_interval_minutes,
            timeframe_minutes,
        )
        result = await asyncio.to_thread(
            candle_backtest_engine.run_backtest,
            candles,
            timeframe_minutes,
            direction_mode,
            lookback_days,
        )
        result["cache_status"] = cache_status
        result["downloaded_candles"] = downloaded_count
    except (
        RuntimeError,
        ValueError,
        OSError,
        asyncio.TimeoutError,
        aiohttp.ClientError,
    ) as error:
        logger.exception("No se pudo ejecutar el backtest OHLCV descargado de Bybit.")
        raise HTTPException(status_code=503, detail=str(error)) from error

    await broadcast_message({
        "type": "scalping_candle_backtest",
        "backtest": result,
    })
    return JSONResponse(result)


@app.get("/api/status")
async def get_status():
    now_ms = int(time.time() * 1000)
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
        "snapshot_interval_seconds": CONFIG.data_capture.sample_interval_seconds,
        "last_bybit_trade_at_ms": last_bybit_trade_at_ms or None,
        "seconds_since_last_bybit_trade": (
            round((now_ms - last_bybit_trade_at_ms) / 1000.0, 2)
            if last_bybit_trade_at_ms else None
        ),
        "last_snapshot_at_ms": last_capture_at_ms or None,
        "seconds_since_last_snapshot": (
            round((now_ms - last_capture_at_ms) / 1000.0, 2)
            if last_capture_at_ms else None
        ),
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
        columns = {
            col[1] for col in connection.execute("PRAGMA table_info(market_observations)").fetchall()
        }
        has_features = "features_json" in columns
        select_cols = "observed_at_ms, heuristic_scores_json" + (", features_json" if has_features else "")
        rows = connection.execute(
            f"""
            SELECT {select_cols}
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
    for row in rows:
        observed_at_ms = row[0]
        scores_json = row[1]
        features_json = row[2] if has_features and len(row) > 2 else None
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

        cvd_val = 0.0
        if features_json:
            try:
                feat = json.loads(features_json)
                cvd_val = float(feat.get("cvd_usd", 0.0))
            except Exception:
                pass

        sample_values = {
            key: scores.get(key, 0.0)
            for key in keys
        }
        sample_values["cvd_usd"] = cvd_val
        samples.append({
            "timestamp_ms": observed_at_ms,
            "values": sample_values,
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
