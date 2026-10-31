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
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from src.core.trade_model import AggregatedTrade, RawTrade
from src.core.aggregator import TradeAggregator
from src.core.classifier import MicrostructureClassifier
from src.core.market_data_store import MarketDataStore
from src.connectors.binance import BinanceConnector
from src.connectors.bybit import BybitLinearConnector
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
database_path = Path(__file__).resolve().parents[2] / CONFIG.data_capture.database_path
market_data_store = MarketDataStore(
    database_path,
    CONFIG.data_capture.label_horizons_seconds,
    CONFIG.data_capture.label_min_move_bps,
)
active_websockets: Set[WebSocket] = set()
connectors: List = []
trade_counter = 0
exchange_counts = {"Binance": 0, "Bybit": 0}
last_capture_at_ms = 0
data_capture_status = "starting"
INFERENCE_MODE = "heuristic"


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
    if (
        metrics is not None
        and classifier.is_warmed_up()
        and now_ms - last_capture_at_ms >= CONFIG.data_capture.sample_interval_seconds * 1000
    ):
        last_capture_at_ms = now_ms
        try:
            market_data_store.record_observation(
                observed_at_ms=now_ms,
                symbol=agg_trade.symbol,
                price=metrics.current_price,
                cvd_usd=metrics.cvd_usd,
                features=asdict(metrics),
                heuristic_scores=asdict(probabilities),
                configuration=asdict(CONFIG),
            )
            data_capture_status = "recording"
        except sqlite3.Error:
            logger.exception("No se pudo guardar el snapshot de mercado en SQLite.")
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
        "inference_mode": INFERENCE_MODE,
        "data_capture_status": data_capture_status,
        "detector_config": asdict(CONFIG.classifier),
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
            symbol=CONFIG.exchange.symbol,
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
            "inference_mode": INFERENCE_MODE,
            "data_capture_status": data_capture_status,
            "detector_config": asdict(CONFIG.classifier),
        }))

        while True:
            # Keep socket alive and receive client commands
            await websocket.receive_text()
    except WebSocketDisconnect:
        active_websockets.discard(websocket)
        logger.info(f"Cliente desconectado. Total clientes: {len(active_websockets)}")
    except Exception as e:
        active_websockets.discard(websocket)
        logger.warning(f"Error WebSocket: {e}")


@app.get("/api/status")
async def get_status():
    return JSONResponse({
        "status": "online",
        "symbol": CONFIG.exchange.symbol,
        "connectors_active": len(connectors),
        "exchange_counts": exchange_counts,
        "is_warmed_up": classifier.is_warmed_up(),
        "window_trades": len(classifier.trades),
        "inference_mode": INFERENCE_MODE,
        "detector_config": asdict(CONFIG.classifier),
        "data_capture_status": data_capture_status,
        "stored_observations": market_data_store.observation_count() if market_data_store.connection is not None else None,
        "database": "SQLite",
        "connected_clients": len(active_websockets),
        "total_aggregated_trades": trade_counter
    })


# Static files mount
static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/")
    async def serve_index():
        return FileResponse(os.path.join(static_dir, "index.html"))
