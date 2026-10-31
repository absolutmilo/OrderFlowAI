"""
Unit tests for the MicrostructureClassifier and TradeAggregator.
"""

import pytest
import numpy as np
from src.core.trade_model import AggregatedTrade, RawTrade, TradeSide
from src.core.classifier import MicrostructureClassifier
from src.core.aggregator import TradeAggregator


def test_algo_institutional_detection():
    """Verify that regular, equidistant, uniform orders produce a high algorithmic probability."""
    classifier = MicrostructureClassifier(window_size=100, warmup_trades=30)

    base_time = 1700000000000
    base_price = 70000.0

    # Inject 50 equidistant trades with identical sizes (perfect TWAP signature)
    for i in range(50):
        t = AggregatedTrade(
            exchange="Binance",
            symbol="BTCUSDT",
            first_timestamp_ms=base_time + (i * 250),  # exactly 250ms apart
            last_timestamp_ms=base_time + (i * 250),
            price=base_price + (i * 0.1),
            total_size=0.5,                            # exactly 0.5 BTC
            total_usd_value=35000.0,
            side=TradeSide.BUY,
            trade_count=1
        )
        classifier.add_trade(t)

    probs = classifier.classify()
    assert probs.algo_institutional > 70.0, f"Expected algo > 70%, got {probs.algo_institutional}%"


def test_iceberg_detection():
    """Verify that heavy accumulation at a single price level triggers Iceberg detection."""
    classifier = MicrostructureClassifier(window_size=100, warmup_trades=30)
    base_time = 1700000000000
    iceberg_price = 69500.0

    # Inject trades where 60% of volume is concentrated at 69500.0
    for i in range(40):
        is_iceberg_level = (i % 2 == 0)
        p = iceberg_price if is_iceberg_level else (iceberg_price + (i * 2.0))
        size = 2.0 if is_iceberg_level else 0.1
        usd = size * p

        t = AggregatedTrade(
            exchange="Binance",
            symbol="BTCUSDT",
            first_timestamp_ms=base_time + (i * 100),
            last_timestamp_ms=base_time + (i * 100),
            price=p,
            total_size=size,
            total_usd_value=usd,
            side=TradeSide.SELL if is_iceberg_level else TradeSide.BUY,
            trade_count=1
        )
        classifier.add_trade(t)

    probs = classifier.classify()
    assert probs.iceberg > 50.0, f"Expected iceberg > 50%, got {probs.iceberg}%"
    assert probs.iceberg_side == "COMPRADOR"  # Sell market orders hitting passive buyer


def test_absorption_detection():
    """Verify that aggressive selling with zero price drop triggers passive buyer absorption."""
    classifier = MicrostructureClassifier(window_size=100, warmup_trades=30)
    base_time = 1700000000000
    static_price = 70000.0

    # Intense selling, but price is completely pinned at 70000.0
    for i in range(45):
        t = AggregatedTrade(
            exchange="Binance",
            symbol="BTCUSDT",
            first_timestamp_ms=base_time + (i * 50),
            last_timestamp_ms=base_time + (i * 50),
            price=static_price,
            total_size=1.5,
            total_usd_value=105000.0,
            side=TradeSide.SELL,
            trade_count=1
        )
        classifier.add_trade(t)

    probs = classifier.classify()
    assert probs.absorption > 60.0, f"Expected absorption > 60%, got {probs.absorption}%"


@pytest.mark.asyncio
async def test_aggregator_grouping():
    """Verify that raw trades at the same timestamp and price are merged."""
    emitted = []

    def on_trade(agg):
        emitted.append(agg)

    aggregator = TradeAggregator(window_ms=50, on_aggregated_trade=on_trade)

    # Send 3 identical micro-trades
    t1 = RawTrade("Binance", "BTCUSDT", "1", 70000.0, 0.1, 7000.0, TradeSide.BUY, 1000)
    t2 = RawTrade("Binance", "BTCUSDT", "2", 70000.0, 0.2, 14000.0, TradeSide.BUY, 1020)
    t3 = RawTrade("Binance", "BTCUSDT", "3", 70000.0, 0.3, 21000.0, TradeSide.BUY, 1030)

    await aggregator.process_raw_trade(t1)
    await aggregator.process_raw_trade(t2)
    await aggregator.process_raw_trade(t3)

    # Different price flushes previous group
    t4 = RawTrade("Binance", "BTCUSDT", "4", 70001.0, 0.5, 35000.5, TradeSide.BUY, 1040)
    await aggregator.process_raw_trade(t4)

    assert len(emitted) == 1
    assert emitted[0].trade_count == 3
    assert abs(emitted[0].total_size - 0.6) < 1e-5
