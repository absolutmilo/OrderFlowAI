"""
Real-time Trade Aggregator module.
Aggregates micro-transactions happening within small time frames into single cohesive tape entries,
reproducing the aggregation logic of SignificantTrades / aggr.trade.
"""

import asyncio
from typing import Callable, Optional
from src.core.trade_model import RawTrade, AggregatedTrade, TradeSide
from config import CONFIG


class TradeAggregator:
    def __init__(
        self,
        window_ms: int = CONFIG.aggregator.aggregation_window_ms,
        on_aggregated_trade: Optional[Callable[[AggregatedTrade], None]] = None
    ):
        self.window_ms = window_ms
        self.on_aggregated_trade = on_aggregated_trade
        self.current_group: Optional[AggregatedTrade] = None
        self._flush_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

    async def process_raw_trade(self, raw: RawTrade):
        """Processes an incoming raw trade, grouping it if eligible, or emitting the previous group."""
        async with self._lock:
            # Check if this raw trade can be merged with current group:
            # Criteria: same side, same price, and timestamp difference within aggregation window
            can_merge = (
                self.current_group is not None
                and self.current_group.exchange == raw.exchange
                and self.current_group.symbol == raw.symbol
                and self.current_group.side == raw.side
                and abs(self.current_group.price - raw.price) < 0.01
                and (raw.timestamp_ms - self.current_group.last_timestamp_ms) <= self.window_ms
            )

            if can_merge and self.current_group is not None:
                self.current_group.total_size += raw.size
                self.current_group.total_usd_value += raw.usd_value
                self.current_group.trade_count += 1
                self.current_group.last_timestamp_ms = raw.timestamp_ms
                if raw.is_liquidation:
                    self.current_group.is_liquidation = True
            else:
                # Flush previous group if exists
                if self.current_group is not None:
                    await self._emit_current_group()

                # Start new group
                self.current_group = AggregatedTrade(
                    exchange=raw.exchange,
                    symbol=raw.symbol,
                    first_timestamp_ms=raw.timestamp_ms,
                    last_timestamp_ms=raw.timestamp_ms,
                    price=raw.price,
                    total_size=raw.size,
                    total_usd_value=raw.usd_value,
                    side=raw.side,
                    trade_count=1,
                    is_liquidation=raw.is_liquidation,
                    detected_patterns=[]
                )

                # Schedule automatic flush after window_ms if no subsequent trade arrives
                if self._flush_task and not self._flush_task.done():
                    self._flush_task.cancel()
                self._flush_task = asyncio.create_task(self._delayed_flush(self.window_ms))

    async def _delayed_flush(self, delay_ms: int):
        try:
            await asyncio.sleep(delay_ms / 1000.0)
            async with self._lock:
                if self.current_group is not None:
                    await self._emit_current_group()
        except asyncio.CancelledError:
            pass

    async def _emit_current_group(self):
        if self.current_group is not None:
            group = self.current_group
            self.current_group = None
            if self.on_aggregated_trade:
                # Call callback (may be async or sync)
                if asyncio.iscoroutinefunction(self.on_aggregated_trade):
                    await self.on_aggregated_trade(group)
                else:
                    self.on_aggregated_trade(group)
