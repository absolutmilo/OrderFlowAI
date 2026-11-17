"""Unit tests for MicrostructurePatternEngine and HMM probabilistic setups."""

import numpy as np
import pytest

from src.core.microstructure_patterns import (
    MicrostructurePatternEngine,
    MarketRegimeState,
    MicrostructureFeatures,
    PatternSignal,
    AdverseWallAlert,
)
from config import ScalpingConfig
from src.core.scalping import ScalpingEngine, STRATEGIES


def test_microstructure_pattern_engine_iceberg_defense_buy():
    engine = MicrostructurePatternEngine()
    regime = MarketRegimeState(dominant_regime="ICEBERG_COMPRADOR", iceberg_prob=85.0)
    features = MicrostructureFeatures(
        price=75000.0,
        cvd_usd=15000.0,
        cvd_ratio=0.25,
        price_delta_bps=0.2,
        trades_per_second=2.0,
        volume_per_second_usd=10000.0,
        inter_arrival_cv=1.2,
        volatility_atr_bps=0.8,  # LOW volatility
    )

    signal = engine.evaluate_pattern(regime, features)
    assert signal is not None
    assert signal.pattern_name == "PASSIVE_BUYING_CANDIDATE"
    assert signal.direction == 1
    assert signal.recommended_horizon_seconds is None
    assert signal.probability_win is None
    assert signal.expectancy_bps is None
    assert signal.confidence is None


def test_microstructure_pattern_engine_iceberg_defense_sell():
    engine = MicrostructurePatternEngine()
    regime = MarketRegimeState(dominant_regime="ICEBERG_VENDEDOR", iceberg_prob=90.0)
    features = MicrostructureFeatures(
        price=75000.0,
        cvd_usd=-25000.0,
        cvd_ratio=-0.30,
        price_delta_bps=-0.3,
        trades_per_second=2.5,
        volume_per_second_usd=12000.0,
        inter_arrival_cv=1.3,
        volatility_atr_bps=2.0,  # NORMAL volatility
    )

    signal = engine.evaluate_pattern(regime, features)
    assert signal is not None
    assert signal.pattern_name == "PASSIVE_SELLING_CANDIDATE"
    assert signal.direction == -1
    assert signal.recommended_horizon_seconds is None


def test_microstructure_pattern_engine_absorption_exhaustion_long():
    engine = MicrostructurePatternEngine()
    regime = MarketRegimeState(dominant_regime="ABSORPTION_BUY", absorption_prob=88.0)
    features = MicrostructureFeatures(
        price=75000.0,
        cvd_usd=5000.0,
        cvd_ratio=0.08,
        price_delta_bps=0.5,
        trades_per_second=1.8,
        volume_per_second_usd=8000.0,
        inter_arrival_cv=1.5,
        volatility_atr_bps=1.5,
    )

    signal = engine.evaluate_pattern(
        regime,
        features,
        recent_regimes=["ABSORPTION_BUY", "ABSORPTION_BUY"],
        recent_momentums_bps=[0.4],
    )
    assert signal is not None
    assert signal.pattern_name == "LOW_PRICE_RESPONSE_REVERSAL_LONG"
    assert signal.direction == 1
    assert signal.probability_win is None
    assert signal.expectancy_bps is None


def test_check_adverse_wall_detection():
    engine = MicrostructurePatternEngine(adverse_persistence_required=3)

    # In LONG position, sellers build a persistent adverse wall
    long_dir = 1
    recent_regimes = ["ABSORPTION_SELL", "ABSORPTION_SELL", "ICEBERG_VENDEDOR"]
    alert = engine.check_adverse_wall(long_dir, recent_regimes)

    assert alert.is_adverse is True
    assert alert.opposing_direction == -1
    assert "Estado opuesto inferido" in alert.reason

    # In SHORT position, buyers build a persistent adverse wall
    short_dir = -1
    recent_regimes = ["ABSORPTION_BUY", "ABSORPTION_BUY", "ICEBERG_COMPRADOR"]
    alert = engine.check_adverse_wall(short_dir, recent_regimes)

    assert alert.is_adverse is True
    assert alert.opposing_direction == 1
    assert "Estado opuesto inferido" in alert.reason

    # No adverse wall if regime is mixed or aligned
    neutral_regimes = ["ABSORPTION_BUY", "RETAIL_DOMINANT", "ALGO_INSTITUTIONAL"]
    alert = engine.check_adverse_wall(long_dir, neutral_regimes)
    assert alert.is_adverse is False


def test_scalping_engine_hmm_probabilistic_strategy_registration():
    assert "hmm_probabilistic" in STRATEGIES
    assert "HMM-4" in STRATEGIES["hmm_probabilistic"]


def test_scalping_engine_adverse_hmm_exit_simulation():
    count = 100
    timestamps = np.arange(count, dtype=np.int64) * 1000
    prices = np.full(count, 70000.0)
    imbalance = np.zeros(count)
    volume_rate = np.zeros(count)

    engine = ScalpingEngine(":memory:", ScalpingConfig(min_training_observations=50))

    # Mock scores with persistent adverse seller absorption after index 20
    scores = []
    for i in range(count):
        if i >= 20:
            scores.append({"hmm_4": {"dominant_regime": "ABSORPTION_SELL"}})
        else:
            scores.append({"hmm_4": {"dominant_regime": "ABSORPTION_BUY"}})

    adverse_walls = ScalpingEngine._build_adverse_walls(scores, count)
    assert adverse_walls[25] == -1  # Adverse for LONG
