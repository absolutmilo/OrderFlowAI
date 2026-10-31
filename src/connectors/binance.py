"""
Binance WebSocket Connector.
Connects to Binance high-throughput aggregated trades stream (aggTrade).
Supports both Binance Spot (stream.binance.com:9443) and Futures (fstream.binance.com).
"""

import asyncio
import json
import logging
from typing import Callable, Optional
import websockets

from src.connectors.base import BaseExchangeConnector
from src.core.trade_model import RawTrade, TradeSide
from config import CONFIG

logger = logging.getLogger("OrderFlowAI.Binance")


class BinanceConnector(BaseExchangeConnector):
    def __init__(
        self,
        symbol: str = "BTCUSDT",
        on_trade_callback: Optional[Callable[[RawTrade], None]] = None,
        use_futures: bool = False
    ):
        super().__init__(symbol=symbol.upper(), on_trade_callback=on_trade_callback)
        self.use_futures = use_futures

        # Binance Spot is universally accessible globally without regional IP restrictions
        if use_futures:
            self.ws_url = f"wss://fstream.binance.com/ws/{self.symbol.lower()}@aggTrade"
            self.exchange_name = "Binance-Futures"
        else:
            self.ws_url = f"wss://stream.binance.com:9443/ws/{self.symbol.lower()}@aggTrade"
            self.exchange_name = "Binance-Spot"

        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

    async def start(self):
        self.is_running = True
        self._stop_event.clear()
        self._task = asyncio.create_task(self._run_loop())
        logger.info(f"[{self.exchange_name}] Conector iniciado para {self.symbol} -> {self.ws_url}")

    async def stop(self):
        self.is_running = False
        self._stop_event.set()
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info(f"[{self.exchange_name}] Conector detenido.")

    async def _run_loop(self):
        retry_delay = CONFIG.exchange.reconnect_delay_seconds
        while self.is_running and not self._stop_event.is_set():
            try:
                logger.info(f"[{self.exchange_name}] Conectando a {self.ws_url}...")
                async with websockets.connect(
                    self.ws_url,
                    ping_interval=CONFIG.exchange.ping_interval_seconds,
                    ping_timeout=10,
                    close_timeout=5
                ) as ws:
                    logger.info(f"[{self.exchange_name}] ¡Conexión establecida exitosamente!")
                    retry_delay = CONFIG.exchange.reconnect_delay_seconds

                    while self.is_running and not self._stop_event.is_set():
                        msg = await ws.recv()
                        data = json.loads(msg)

                        # Schema Binance aggTrade
                        if data.get("e") == "aggTrade":
                            price = float(data["p"])
                            size = float(data["q"])
                            usd_value = price * size
                            is_buyer_maker = bool(data["m"])
                            # m = true -> Venta agresiva a mercado (hit the bid)
                            # m = false -> Compra agresiva a mercado (lift the ask)
                            side = TradeSide.SELL if is_buyer_maker else TradeSide.BUY

                            trade = RawTrade(
                                exchange=self.exchange_name,
                                symbol=data["s"],
                                trade_id=str(data["a"]),
                                price=price,
                                size=size,
                                usd_value=usd_value,
                                side=side,
                                timestamp_ms=int(data["T"]),
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
                logger.warning(f"[{self.exchange_name}] Error en WebSocket: {e}. Reconectando en {retry_delay:.1f}s...")
                await asyncio.sleep(retry_delay)
                retry_delay = min(20.0, retry_delay * 1.5)
