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
    classifier = MicrostructureClassifier(warmup_trades=30, inference_mode="heuristic")

    base_time = 1700000000000
    base_price = 70000.0

    # Inject 50 equidistant trades with identical sizes (perfect TWAP signature)
    for i in range(50):
        t = AggregatedTrade(
            exchange="Binance",
            symbol="BTCUSDT",
            first_timestamp_ms=base_time + (i * 1000),
            last_timestamp_ms=base_time + (i * 1000),
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
    classifier = MicrostructureClassifier(warmup_trades=30, inference_mode="heuristic")
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
            first_timestamp_ms=base_time + (i * 1000),
            last_timestamp_ms=base_time + (i * 1000),
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
    classifier = MicrostructureClassifier(warmup_trades=30, inference_mode="heuristic")
    base_time = 1700000000000
    static_price = 70000.0

    # Intense selling, but price is completely pinned at 70000.0
    for i in range(45):
        t = AggregatedTrade(
            exchange="Binance",
            symbol="BTCUSDT",
            first_timestamp_ms=base_time + (i * 1000),
            last_timestamp_ms=base_time + (i * 1000),
            price=static_price,
            total_size=1.5,
            total_usd_value=105000.0,
            side=TradeSide.SELL,
            trade_count=1
        )
        classifier.add_trade(t)

    probs = classifier.classify()
    assert probs.absorption > 60.0, f"Expected absorption > 60%, got {probs.absorption}%"


def test_classifier_uses_fixed_event_time_windows():
    classifier = MicrostructureClassifier(inference_mode="heuristic")
    for index, timestamp in enumerate((1_000, 2_000, 4_000, 6_000)):
        classifier.add_trade(AggregatedTrade(
            exchange="Bybit-Linear",
            symbol="BTCUSDT",
            first_timestamp_ms=timestamp,
            last_timestamp_ms=timestamp,
            price=70_000.0 + index,
            total_size=0.1,
            total_usd_value=7_000.0,
            side=TradeSide.BUY,
            trade_count=1,
        ))

    assert len(classifier.get_window_trades(1)) == 1
    assert len(classifier.get_window_trades(5)) == 3
    assert len(classifier.get_window_trades(30)) == 4
    metrics = classifier.get_current_metrics(5)
    assert metrics is not None
    assert metrics.window_seconds == 5
    assert metrics.event_count == 3
    assert metrics.trades_per_second == pytest.approx(0.6)


def test_classifier_resets_warmup_after_feed_gap():
    classifier = MicrostructureClassifier(inference_mode="heuristic")
    for index in range(31):
        timestamp = 1_000_000 + index * 1000
        classifier.add_trade(AggregatedTrade(
            exchange="Bybit-Linear",
            symbol="BTCUSDT",
            first_timestamp_ms=timestamp,
            last_timestamp_ms=timestamp,
            price=70_000.0,
            total_size=0.1,
            total_usd_value=7_000.0,
            side=TradeSide.BUY,
            trade_count=1,
        ))

    assert classifier.is_warmed_up()
    timestamp = 1_036_000
    classifier.add_trade(AggregatedTrade(
        exchange="Bybit-Linear",
        symbol="BTCUSDT",
        first_timestamp_ms=timestamp,
        last_timestamp_ms=timestamp,
        price=70_000.0,
        total_size=0.1,
        total_usd_value=7_000.0,
        side=TradeSide.BUY,
        trade_count=1,
    ))
    assert not classifier.is_warmed_up()
    assert len(classifier.trades) == 1


def test_classifier_rejects_events_from_another_stream():
    classifier = MicrostructureClassifier(inference_mode="heuristic")
    first_trade = AggregatedTrade(
        exchange="Bybit-Linear",
        symbol="BTCUSDT",
        first_timestamp_ms=1_000,
        last_timestamp_ms=1_000,
        price=70_000.0,
        total_size=0.1,
        total_usd_value=7_000.0,
        side=TradeSide.BUY,
        trade_count=1,
    )
    other_stream_trade = AggregatedTrade(
        exchange="Binance-Spot",
        symbol="BTCUSDT",
        first_timestamp_ms=2_000,
        last_timestamp_ms=2_000,
        price=70_001.0,
        total_size=0.1,
        total_usd_value=7_000.0,
        side=TradeSide.BUY,
        trade_count=1,
    )
    classifier.add_trade(first_trade)

    with pytest.raises(ValueError, match="no acepta eventos"):
        classifier.add_trade(other_stream_trade)
    assert classifier.stream_identity == ("Bybit-Linear", "BTCUSDT")


def test_classifier_rejects_out_of_order_events():
    classifier = MicrostructureClassifier(inference_mode="heuristic")
    for timestamp in (2_000, 1_000):
        trade = AggregatedTrade(
            exchange="Bybit-Linear",
            symbol="BTCUSDT",
            first_timestamp_ms=timestamp,
            last_timestamp_ms=timestamp,
            price=70_000.0,
            total_size=0.1,
            total_usd_value=7_000.0,
            side=TradeSide.BUY,
            trade_count=1,
        )
        if timestamp == 2_000:
            classifier.add_trade(trade)
        else:
            with pytest.raises(ValueError, match="orden temporal"):
                classifier.add_trade(trade)


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


@pytest.mark.asyncio
async def test_aggregator_does_not_merge_trades_from_different_exchanges():
    emitted = []
    aggregator = TradeAggregator(on_aggregated_trade=emitted.append)
    binance_trade = RawTrade(
        "Binance-Spot", "BTCUSDT", "1", 70000.0, 0.1, 7000.0, TradeSide.BUY, 1000
    )
    bybit_trade = RawTrade(
        "Bybit-Linear", "BTCUSDT", "2", 70000.0, 0.2, 14000.0, TradeSide.BUY, 1010
    )

    await aggregator.process_raw_trade(binance_trade)
    await aggregator.process_raw_trade(bybit_trade)

    assert len(emitted) == 1
    assert emitted[0].exchange == "Binance-Spot"
    assert emitted[0].total_size == 0.1
    await aggregator._emit_current_group()
    assert len(emitted) == 2
    assert emitted[1].exchange == "Bybit-Linear"
