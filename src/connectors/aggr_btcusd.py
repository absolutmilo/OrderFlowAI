"""
Native AGGR BTCUSD Connector.
Streams real-time BTCUSD inverse/perpetual trades directly from exchange WebSocket endpoints
uncovered from aggr.trade (Binance Coin-M Futures and Bybit Inverse), eliminating the need
for an external browser extension bridge.
"""

import asyncio
import json
import logging
import time
from typing import Callable, Optional, Tuple, List, Any
import websockets

from config import CONFIG

logger = logging.getLogger("OrderFlowAI.AggrBTCUSD")


class AggrBtcusdConnector:
    """
    Direct WebSocket client streaming real-time BTCUSD prices.
    Primary source: Binance Coin-Margined Futures (BTCUSD_PERP).
    Secondary / Fallback source: Bybit Inverse Perpetual (BTCUSD).
    """

    BINANCE_DSTREAM_URL = "wss://dstream.binance.com/ws/btcusd_perp@aggTrade"
    BYBIT_INVERSE_URL = "wss://stream.bybit.com/v5/public/inverse"

    def __init__(
        self,
        on_price_callback: Optional[Callable[[float, int, str], Any]] = None,
        preferred_source: str = "auto",  # 'auto', 'binance', 'bybit'
    ):
        self.on_price_callback = on_price_callback
        self.preferred_source = preferred_source
        self.is_running = False
        self.active_source: Optional[str] = None
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

    async def start(self):
        """Starts background WebSocket streaming task."""
        self.is_running = True
        self._stop_event.clear()
        self._task = asyncio.create_task(self._run_loop())
        logger.info("[AGGR-BTCUSD] Conector nativo iniciado.")

    async def stop(self):
        """Stops the connector and cancels tasks."""
        self.is_running = False
        self._stop_event.set()
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self.active_source = None
        logger.info("[AGGR-BTCUSD] Conector nativo detenido.")

    @staticmethod
    def parse_binance_dstream_message(data: dict) -> Optional[Tuple[float, int]]:
        """
        Parses Binance Coin-M aggTrade message.
        Schema:
        {
            "e": "aggTrade",
            "p": "104120.5",
            "T": 1731324567890
        }
        """
        if not isinstance(data, dict):
            return None
        if data.get("e") == "aggTrade" and "p" in data:
            try:
                price = float(data["p"])
                ts = int(data.get("T") or data.get("E") or (time.time() * 1000))
                return price, ts
            except (ValueError, TypeError):
                return None
        return None

    @staticmethod
    def parse_bybit_inverse_message(data: dict) -> List[Tuple[float, int]]:
        """
        Parses Bybit Inverse publicTrade message.
        Schema:
        {
            "topic": "publicTrade.BTCUSD",
            "data": [
                {"p": "104120.50", "T": 1731324567890}
            ]
        }
        """
        results: List[Tuple[float, int]] = []
        if not isinstance(data, dict):
            return results
        
        topic = data.get("topic", "")
        if "publicTrade.BTCUSD" in topic and isinstance(data.get("data"), list):
            for item in data["data"]:
                try:
                    price = float(item["p"])
                    ts = int(item.get("T") or (time.time() * 1000))
                    results.append((price, ts))
                except (ValueError, TypeError, KeyError):
                    continue
        return results

    async def _emit_price(self, price: float, timestamp_ms: int, source: str):
        if not self.on_price_callback:
            return
        if asyncio.iscoroutinefunction(self.on_price_callback):
            await self.on_price_callback(price, timestamp_ms, source)
        else:
            self.on_price_callback(price, timestamp_ms, source)

    async def _stream_binance(self) -> None:
        """Connects and streams from Binance Coin-M Futures."""
        logger.info(f"[AGGR-BTCUSD] Conectando a Binance Coin-M ({self.BINANCE_DSTREAM_URL})...")
        async with websockets.connect(
            self.BINANCE_DSTREAM_URL,
            ping_interval=CONFIG.exchange.ping_interval_seconds,
            ping_timeout=10,
            close_timeout=5,
        ) as ws:
            self.active_source = "Binance-CoinM"
            logger.info("[AGGR-BTCUSD] ¡Conectado exitosamente a Binance Coin-M (BTCUSD_PERP)!")

            while self.is_running and not self._stop_event.is_set():
                msg = await ws.recv()
                data = json.loads(msg)
                parsed = self.parse_binance_dstream_message(data)
                if parsed:
                    price, ts = parsed
                    await self._emit_price(price, ts, "Binance-CoinM")

    async def _stream_bybit(self) -> None:
        """Connects and streams from Bybit Inverse Perpetual."""
        logger.info(f"[AGGR-BTCUSD] Conectando a Bybit Inverse ({self.BYBIT_INVERSE_URL})...")
        async with websockets.connect(
            self.BYBIT_INVERSE_URL,
            ping_interval=CONFIG.exchange.ping_interval_seconds,
            ping_timeout=10,
            close_timeout=5,
        ) as ws:
            sub_msg = {
                "op": "subscribe",
                "args": ["publicTrade.BTCUSD"],
            }
            await ws.send(json.dumps(sub_msg))
            self.active_source = "Bybit-Inverse"
            logger.info("[AGGR-BTCUSD] ¡Conectado y suscrito exitosamente a Bybit Inverse (BTCUSD)!")

            while self.is_running and not self._stop_event.is_set():
                msg = await ws.recv()
                data = json.loads(msg)
                trades = self.parse_bybit_inverse_message(data)
                for price, ts in trades:
                    await self._emit_price(price, ts, "Bybit-Inverse")

    async def _run_loop(self):
        """
        Manages the connection lifecycle with automatic fallback.
        Tries Binance Coin-M first; if blocked or failing, falls back to Bybit Inverse.
        """
        retry_delay = CONFIG.exchange.reconnect_delay_seconds

        while self.is_running and not self._stop_event.is_set():
            # Decide order of preference
            if self.preferred_source == "bybit":
                endpoints = [self._stream_bybit]
            elif self.preferred_source == "binance":
                endpoints = [self._stream_binance]
            else:  # 'auto'
                endpoints = [self._stream_binance, self._stream_bybit]

            connected_any = False
            for endpoint in endpoints:
                if not self.is_running or self._stop_event.is_set():
                    break
                try:
                    await endpoint()
                    connected_any = True
                    retry_delay = CONFIG.exchange.reconnect_delay_seconds
                except asyncio.CancelledError:
                    return
                except Exception as e:
                    self.active_source = None
                    logger.warning(
                        f"[AGGR-BTCUSD] Error con {endpoint.__name__}: {e}. Intentando alternativa..."
                    )

            if not self.is_running or self._stop_event.is_set():
                break

            logger.info(f"[AGGR-BTCUSD] Reintentando conexión en {retry_delay:.1f}s...")
            try:
                await asyncio.sleep(retry_delay)
            except asyncio.CancelledError:
                break
            retry_delay = min(20.0, retry_delay * 1.5)
