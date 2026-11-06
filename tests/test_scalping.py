import sqlite3

import numpy as np
import pytest

from config import ScalpingConfig
from src.core.scalping import ScalpParameters, ScalpingEngine


def test_optimizer_ignores_legacy_combined_observations(tmp_path):
    database_path = tmp_path / "scalping.sqlite3"
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE market_observations (
                observed_at_ms INTEGER,
                symbol TEXT,
                market_source TEXT,
                price REAL,
                features_json TEXT
            )
            """
        )
        connection.execute(
            "INSERT INTO market_observations VALUES (1, 'BTCUSDT', 'legacy_combined', 70000, '{}')"
        )

    engine = ScalpingEngine(
        database_path,
        ScalpingConfig(min_training_observations=1),
    )
    result = engine.run_backtest()

    assert result["status"] == "collecting_data"
    assert result["observations"] == 0


def test_signal_requires_persistent_order_flow_and_momentum():
    engine = ScalpingEngine(":memory:", ScalpingConfig())
    parameters = ScalpParameters(
        horizon_seconds=300,
        stop_atr_multiple=1.0,
        target_atr_multiple=1.5,
        imbalance_threshold=0.25,
        momentum_threshold_bps=0.0,
        persistence_seconds=3,
    )
    timestamps = np.arange(100, dtype=np.int64) * 1000
    prices = 70_000.0 * (1.0 + np.arange(100) * 0.00001)
    imbalance = np.full(100, 0.5)

    signals = engine._signal_indices(timestamps, prices, imbalance, parameters)

    assert signals
    assert signals[0][1] == 1
    assert signals[0][0] >= 63


def test_backtest_applies_round_trip_costs_to_target_exit():
    engine = ScalpingEngine(
        ":memory:",
        ScalpingConfig(
            taker_fee_bps_per_side=2.0,
            slippage_bps_per_side=1.0,
        ),
    )
    timestamps = np.arange(1000, 1005, dtype=np.int64) * 1000
    prices = np.array([100.0, 100.0, 100.2, 100.2, 100.2])
    atr = np.full(5, 10.0)
    parameters = ScalpParameters(300, 1.0, 1.0, 0.25, 0.0, 1)

    results = engine._simulate(
        timestamps,
        prices,
        atr,
        [(0, 1)],
        parameters,
        start_index=0,
        end_index=len(timestamps),
    )

    assert results[0]["gross_return_bps"] == pytest.approx(10.0)
    assert results[0]["net_return_bps"] == pytest.approx(4.0)
    assert results[0]["direction"] == "LONG"
    assert results[0]["exit_reason"] == "take_profit"


def test_inverse_signals_swap_long_and_short_without_changing_entry_indices():
    original_signals = [(12, 1), (35, -1), (48, 1)]

    inverse_signals = ScalpingEngine._invert_signals(original_signals)

    assert inverse_signals == [(12, -1), (35, 1), (48, -1)]


def test_inverse_trade_direction_reverses_stop_and_target_outcomes():
    engine = ScalpingEngine(
        ":memory:",
        ScalpingConfig(taker_fee_bps_per_side=0.0, slippage_bps_per_side=0.0),
    )
    timestamps = np.arange(1000, 1004, dtype=np.int64) * 1000
    prices = np.array([100.0, 100.0, 100.2, 100.2])
    atr = np.full(4, 10.0)
    parameters = ScalpParameters(3, 1.0, 1.0, 0.25, 0.0, 1)

    original = engine._simulate(
        timestamps, prices, atr, [(0, 1)], parameters, 0, len(timestamps)
    )
    inverse = engine._simulate(
        timestamps,
        prices,
        atr,
        engine._invert_signals([(0, 1)]),
        parameters,
        0,
        len(timestamps),
    )

    assert original[0]["exit_reason"] == "take_profit"
    assert original[0]["gross_return_bps"] == pytest.approx(10.0)
    assert inverse[0]["exit_reason"] == "stop_loss"
    assert inverse[0]["gross_return_bps"] == pytest.approx(-10.0)


def test_backtest_does_not_count_time_exit_with_incomplete_horizon():
    engine = ScalpingEngine(
        ":memory:",
        ScalpingConfig(taker_fee_bps_per_side=0.0, slippage_bps_per_side=0.0),
    )
    timestamps = np.arange(1000, 1005, dtype=np.int64) * 1000
    prices = np.full(5, 100.0)
    parameters = ScalpParameters(300, 1.0, 1.0, 0.25, 0.0, 1)

    results = engine._simulate(
        timestamps,
        prices,
        np.full(5, 10.0),
        [(0, 1)],
        parameters,
        start_index=0,
        end_index=len(timestamps),
    )

    assert results == []


def test_optimizer_requires_minimum_training_observations_after_split():
    count = 50_630
    timestamps = np.arange(count, dtype=np.int64) * 1000
    engine = ScalpingEngine(
        ":memory:",
        ScalpingConfig(min_training_observations=36_000),
    )
    engine._load_observations = lambda: (
        timestamps,
        np.full(count, 70_000.0),
        np.zeros(count),
        np.zeros(count),
    )

    result = engine.run_backtest()

    assert result["status"] == "collecting_data"
    assert result["training_observations"] == 35_441
    assert result["required_training_observations"] == 36_000
    assert result["required_observations"] == 51_429


def test_optimizer_rejects_snapshot_history_with_insufficient_coverage():
    timestamps = np.arange(0, 2400, 2, dtype=np.int64) * 1000
    count = len(timestamps)
    engine = ScalpingEngine(
        ":memory:",
        ScalpingConfig(
            min_training_observations=800,
            minimum_snapshot_coverage=0.80,
        ),
    )
    engine._load_observations = lambda: (
        timestamps,
        np.full(count, 70_000.0),
        np.zeros(count),
        np.zeros(count),
    )

    result = engine.run_backtest()

    assert result["status"] == "insufficient_data_quality"
    assert result["snapshot_coverage_pct"] == pytest.approx(50.04, abs=0.02)
    assert result["maximum_snapshot_gap_seconds"] == 2.0
    assert engine.parameters is None


def test_optimizer_evaluates_full_grid_on_chronological_holdout():
    count = 2400
    timestamps = np.arange(count, dtype=np.int64) * 1000
    prices = 70_000.0 * (
        1.0 + 0.001 * np.sin(2 * np.pi * np.arange(count) / 300)
    )
    momentum = np.zeros(count)
    momentum[60:] = np.log(prices[60:] / prices[:-60]) * 10000
    imbalance = np.where(momentum >= 0, 0.7, -0.7)
    engine = ScalpingEngine(
        ":memory:",
        ScalpingConfig(
            min_training_observations=int(count * 0.7),
            min_validation_trades=1,
            taker_fee_bps_per_side=0.1,
            slippage_bps_per_side=0.1,
        ),
    )
    engine._load_observations = lambda: (
        timestamps,
        prices,
        imbalance,
        np.zeros(count),
    )

    result = engine.run_backtest()

    assert result["candidate_count"] == 36
    assert result["training_observations"] == 1680
    assert result["validation_observations"] == 720
    assert result["training"]["trade_count"] > 0
    assert result["validation"]["trade_count"] > 0
    assert "gross_expectancy_bps" in result["validation"]
    assert set(result["validation"]["by_direction"]) == {"LONG", "SHORT"}
    assert set(result["validation"]["by_exit_reason"]) == {
        "stop_loss",
        "take_profit",
        "time",
    }
    assert result["validation_end_utc"]
    inverse_result = engine.run_backtest(direction_mode="inverse")
    assert inverse_result["direction_mode"] == "inverse"
    assert inverse_result["direction_mode_name"] == "Invertida (LONG ↔ SHORT)"
    assert inverse_result["alert_enabled"] is False


def test_aggregated_bars_use_complete_snapshot_ranges():
    timestamps = np.arange(600, dtype=np.int64) * 1000
    prices = np.arange(600, dtype=np.float64) + 100.0
    imbalance = np.full(600, 0.5)
    volume = np.full(600, 1000.0)

    bars = ScalpingEngine._aggregate_bars(
        timestamps, prices, imbalance, volume, timeframe_minutes=5
    )

    assert len(bars["close"]) == 2
    assert bars["open"].tolist() == [100.0, 400.0]
    assert bars["close"].tolist() == [399.0, 699.0]
    assert bars["snapshot_indices"].tolist() == [299, 599]


def test_aggregated_bars_reject_sparse_or_interrupted_buckets():
    timestamps = np.concatenate((
        np.arange(100, dtype=np.int64) * 1000,
        np.arange(120, 300, dtype=np.int64) * 1000,
        np.arange(300, 600, dtype=np.int64) * 1000,
    ))
    prices = np.arange(len(timestamps), dtype=np.float64) + 100.0

    bars = ScalpingEngine._aggregate_bars(
        timestamps,
        prices,
        np.full(len(timestamps), 0.5),
        np.full(len(timestamps), 1_000.0),
        timeframe_minutes=5,
    )

    assert len(bars["close"]) == 1
    assert bars["timestamps"][0] == timestamps[-1]


def test_simulation_skips_trades_crossing_unobserved_market_gaps():
    engine = ScalpingEngine(
        ":memory:",
        ScalpingConfig(
            taker_fee_bps_per_side=0.0,
            slippage_bps_per_side=0.0,
            max_snapshot_gap_seconds=5.0,
        ),
    )
    timestamps = np.array([0, 1_000, 10_000, 11_000], dtype=np.int64)
    prices = np.full(4, 100.0)
    parameters = ScalpParameters(10, 1.0, 1.0, 0.25, 0.0, 1)

    trades = engine._simulate(
        timestamps,
        prices,
        np.full(4, 10.0),
        [(0, 1)],
        parameters,
        start_index=0,
        end_index=len(timestamps),
    )

    assert trades == []


def test_atr_ignores_price_jumps_across_snapshot_gaps():
    timestamps = np.array([0, 1_000, 10_000, 11_000], dtype=np.int64)
    prices = np.array([100.0, 100.0, 110.0, 110.0])

    atr = ScalpingEngine._atr_bps(timestamps, prices, horizon_seconds=60)

    assert atr[2] == pytest.approx(0.25)
    assert atr[3] == pytest.approx(0.25)


def test_whale_trend_strategy_requires_volume_flow_and_trend_confirmation():
    count = 40
    closes = np.linspace(100.0, 110.0, count)
    bars = {
        "snapshot_indices": np.arange(count, dtype=np.int64) * 300,
        "open": closes - 0.1,
        "high": closes,
        "low": closes - 0.2,
        "close": closes,
        "imbalance": np.full(count, 0.5),
        "volume_rate": np.full(count, 100.0),
    }
    bars["volume_rate"][-1] = 300.0

    signals = ScalpingEngine._whale_trend_signals(bars)

    assert signals == [(int(bars["snapshot_indices"][-1]), 1)]


def test_fourier_strategy_needs_enough_completed_bars():
    bars = {
        "snapshot_indices": np.arange(71, dtype=np.int64),
        "high": np.ones(71),
        "low": np.ones(71),
        "close": np.ones(71),
    }

    assert ScalpingEngine._fourier_breakout_signals(bars) == []


def test_fourier_backtest_reports_when_selected_timeframe_has_too_few_bars():
    count = 51_429
    timestamps = np.arange(count, dtype=np.int64) * 1000
    prices = np.full(count, 70_000.0)
    engine = ScalpingEngine(
        ":memory:",
        ScalpingConfig(min_training_observations=36_000),
    )
    engine._load_observations = lambda: (
        timestamps,
        prices,
        np.zeros(count),
        np.full(count, 1_000.0),
    )

    result = engine.run_backtest(
        strategy="fourier_breakout",
        timeframe_minutes=15,
    )

    assert result["status"] == "collecting_timeframe_data"
    assert result["training_bars"] < result["required_training_bars"]
    assert result["required_training_bars"] == 72


def test_fourier_no_trainable_setup_reports_signal_diagnostics():
    count = 116_000
    timestamps = np.arange(count, dtype=np.int64) * 1000
    engine = ScalpingEngine(
        ":memory:",
        ScalpingConfig(min_training_observations=36_000),
    )
    engine._load_observations = lambda: (
        timestamps,
        np.full(count, 70_000.0),
        np.zeros(count),
        np.full(count, 1_000.0),
    )

    result = engine.run_backtest("fourier_breakout", 10)

    assert result["status"] == "no_trainable_setup"
    assert result["training_bars"] >= 72
    assert result["required_training_bars"] == 72
    assert result["training_signal_count"] == 0
    assert result["max_training_trades"] == 0
    assert result["signal_diagnostics"]["raw_anomalies"] == 0
    assert result["signal_diagnostics"]["signals"] == 0
    assert "no generó señales" in result["message"]
