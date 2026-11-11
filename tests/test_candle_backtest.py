import numpy as np

from config import ScalpingConfig
from src.core.candle_backtest import CandleBacktestEngine, aggregate_klines


def test_aggregate_klines_keeps_only_complete_aligned_bars():
    candles = [
        {"time_ms": time_ms, "open": 100.0, "high": 101.0, "low": 99.0,
         "close": 100.5, "volume": 2.0}
        for time_ms in (0, 300_000, 900_000)
    ]

    result = aggregate_klines(candles, 5, 10)

    assert len(result) == 1
    assert result[0] == {
        "time_ms": 0,
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.5,
        "volume": 4.0,
    }


def test_candle_signals_use_completed_bar_and_enter_on_the_next_open():
    closes = np.linspace(100.0, 150.0, 100)
    volumes = np.ones(100)

    signals = CandleBacktestEngine._signal_indices(closes, volumes)

    assert signals == [(21, 1)]


def test_candle_simulation_uses_stop_when_stop_and_target_hit_in_same_bar():
    engine = CandleBacktestEngine(
        ScalpingConfig(taker_fee_bps_per_side=0.0, slippage_bps_per_side=0.0)
    )
    opens = np.full(5, 100.0)
    highs = np.array([100.0, 102.0, 100.0, 100.0, 100.0])
    lows = np.array([100.0, 98.0, 100.0, 100.0, 100.0])
    closes = np.full(5, 100.0)
    atr_bps = np.full(5, 100.0)

    trades = engine._simulate(
        opens,
        highs,
        lows,
        closes,
        atr_bps,
        [(1, 1)],
        1.0,
        1.0,
        start_index=1,
        end_index=5,
        interval_seconds=300,
    )

    assert trades[0]["exit_reason"] == "stop_loss"
    assert trades[0]["gross_return_bps"] < 0
    assert trades[0]["duration_seconds"] == 300


def test_candle_backtest_uses_chronological_holdout_and_does_not_enable_alerts():
    count = 600
    indices = np.arange(count)
    closes = 100.0 * (1.0 + 0.01 * np.sin(2 * np.pi * indices / 45))
    candles = [
        {
            "time_ms": int(index * 300_000),
            "open": float(close),
            "high": float(close * 1.002),
            "low": float(close * 0.998),
            "close": float(close),
            "volume": float(100 + (index % 7)),
        }
        for index, close in enumerate(closes)
    ]
    engine = CandleBacktestEngine(
        ScalpingConfig(
            min_validation_trades=1,
            min_candle_training_bars=300,
            taker_fee_bps_per_side=0.1,
            slippage_bps_per_side=0.1,
        )
    )

    result = engine.run_backtest(candles, timeframe_minutes=5)

    assert result["data_source"] == "bybit_rest_klines"
    assert result["observations"] == count
    assert result["training_observations"] == 420
    assert result["validation_observations"] == 180
    assert result["candidate_count"] == 9
    assert result["alert_enabled"] is False
    assert result["strategy"] == "candle_ema_momentum"


def test_candle_backtest_rejects_gapped_history():
    candles = [
        {"time_ms": time_ms, "open": 100.0, "high": 101.0, "low": 99.0,
         "close": 100.0, "volume": 1.0}
        for time_ms in (0, 60_000, 180_000)
    ]
    engine = CandleBacktestEngine(ScalpingConfig(min_candle_training_bars=1))

    result = engine.run_backtest(candles, timeframe_minutes=1)

    assert result["status"] == "insufficient_data_quality"
