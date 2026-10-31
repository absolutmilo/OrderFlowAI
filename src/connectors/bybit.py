"""
Bybit Linear Perpetuals WebSocket Connector.
Streams real-time public trade events (publicTrade.BTCUSDT) from Bybit v5 API.
"""

import asyncio
import json
import logging
from typing import Callable, Optional
import websockets

from src.connectors.base import BaseExchangeConnector
from src.core.trade_model import RawTrade, TradeSide
from config import CONFIG

logger = logging.getLogger("OrderFlowAI.Bybit")


class BybitLinearConnector(BaseExchangeConnector):
    def __init__(
        self,
        symbol: str = "BTCUSDT",
        on_trade_callback: Optional[Callable[[RawTrade], None]] = None
    ):
        super().__init__(symbol=symbol.upper(), on_trade_callback=on_trade_callback)
        self.ws_url = "wss://stream.bybit.com/v5/public/linear"
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

    async def start(self):
        self.is_running = True
        self._stop_event.clear()
        self._task = asyncio.create_task(self._run_loop())
        logger.info(f"[Bybit-Linear] Conector iniciado para {self.symbol} -> {self.ws_url}")

    async def stop(self):
        self.is_running = False
        self._stop_event.set()
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("[Bybit-Linear] Conector detenido.")

    async def _run_loop(self):
        retry_delay = CONFIG.exchange.reconnect_delay_seconds
        while self.is_running and not self._stop_event.is_set():
            try:
                logger.info(f"[Bybit-Linear] Conectando a {self.ws_url}...")
                async with websockets.connect(
                    self.ws_url,
                    ping_interval=CONFIG.exchange.ping_interval_seconds,
                    ping_timeout=10,
                    close_timeout=5
                ) as ws:
                    # Subscribe to publicTrade topic
                    sub_msg = {
                        "op": "subscribe",
                        "args": [f"publicTrade.{self.symbol}"]
                    }
                    await ws.send(json.dumps(sub_msg))
                    logger.info(f"[Bybit-Linear] Suscrito a publicTrade.{self.symbol} exitosamente!")
                    retry_delay = CONFIG.exchange.reconnect_delay_seconds

                    while self.is_running and not self._stop_event.is_set():
                        msg = await ws.recv()
                        data = json.loads(msg)

                        # Bybit payload format:
                        # {
                        #   "topic": "publicTrade.BTCUSDT",
                        #   "type": "snapshot",
                        #   "data": [
                        #     {
                        #       "T": 1672304484978,  // Timestamp
                        #       "s": "BTCUSDT",       // Symbol
                        #       "S": "Buy",          // Side: Buy or Sell
                        #       "v": "0.001",        // Size
                        #       "p": "16578.50",     // Price
                        #       "i": "20f4c9..."     // Trade ID
                        #     }
                        #   ]
                        # }
                        if data.get("topic") == f"publicTrade.{self.symbol}":
                            trade_list = data.get("data", [])
                            for t in trade_list:
                                price = float(t["p"])
                                size = float(t["v"])
                                usd_val = price * size
                                side_str = t.get("S", "Buy").upper()
                                side = TradeSide.BUY if side_str == "BUY" else TradeSide.SELL

                                trade = RawTrade(
                                    exchange="Bybit-Linear",
                                    symbol=t["s"],
                                    trade_id=str(t.get("i", "")),
                                    price=price,
                                    size=size,
                                    usd_value=usd_val,
                                    side=side,
                                    timestamp_ms=int(t["T"]),
                                    is_liquidation=False
                                )

                                if self.on_trade_callback:
                                    if asyncio.iscoroutinefunction(self.on_trade_callback):
                                        await self.on_trade_callback(trade)
                                    else:
                                        self.on_trade_callback(trade)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"[Bybit-Linear] Error en WebSocket: {e}. Reconectando en {retry_delay:.1f}s...")
                await asyncio.sleep(retry_delay)
                retry_delay = min(20.0, retry_delay * 1.5)
