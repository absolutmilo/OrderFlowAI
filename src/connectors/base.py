"""
Base interface for exchange WebSocket connectors.
"""

from abc import ABC, abstractmethod
from typing import Callable, Optional
from src.core.trade_model import RawTrade


class BaseExchangeConnector(ABC):
    def __init__(self, symbol: str, on_trade_callback: Callable[[RawTrade], None]):
        self.symbol = symbol
        self.on_trade_callback = on_trade_callback
        self.is_running = False

    @abstractmethod
    async def start(self):
        """Starts the WebSocket connection and listening loop."""
        pass

    @abstractmethod
    async def stop(self):
        """Gracefully disconnects and stops the connector."""
        pass
