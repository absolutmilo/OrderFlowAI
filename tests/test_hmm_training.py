import json
import sqlite3

import numpy as np
import pytest

from config import CONFIG
from src.ml.train_hmm import (
    extract_features_from_db,
    sequence_lengths_from_timestamps,
    train_gaussian_hmm,
)


def test_hmm_training_segments_observations_after_long_capture_gaps():
    timestamps = np.array([0, 1_000, 2_000, 7_000, 8_000, 14_001, 20_000])

    assert sequence_lengths_from_timestamps(timestamps, 5.0) == (5, 1, 1)


def test_hmm_training_does_not_transition_across_independent_sequences():
    rng = np.random.default_rng(42)
    first_regime = rng.normal(-1.0, 0.05, size=(20, 5))
    second_regime = rng.normal(1.0, 0.05, size=(20, 5))
    observations = np.concatenate((first_regime, second_regime))

    single_sequence = train_gaussian_hmm(observations, max_iter=1)
    separate_sequences = train_gaussian_hmm(
        observations,
        max_iter=1,
        sequence_lengths=(20, 20),
    )

    assert not np.allclose(single_sequence[3], separate_sequences[3])
    assert not np.allclose(single_sequence[0], separate_sequences[0])


def test_hmm_training_uses_only_current_bybit_feature_schema(tmp_path):
    database_path = tmp_path / "observations.sqlite3"
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE market_observations (
                observed_at_ms INTEGER,
                symbol TEXT,
                market_source TEXT,
                features_json TEXT
            )
            """
        )
        current_features = {
            "feature_schema_version": CONFIG.classifier.feature_schema_version,
            "buy_volume_usd": 600.0,
            "sell_volume_usd": 400.0,
            "cvd_usd": 200.0,
            "volume_per_second_usd": 1_000.0,
            "inter_arrival_cv": 0.5,
            "price_delta_bps": 1.0,
            "trades_per_second": 10.0,
        }
        rows = [
            (
                timestamp,
                CONFIG.scalping.symbol,
                CONFIG.scalping.exchange,
                json.dumps(current_features),
            )
            for timestamp in range(100)
        ]
        rows.extend([
            (101, CONFIG.scalping.symbol, "Binance-Spot", json.dumps(current_features)),
            (
                102,
                CONFIG.scalping.symbol,
                CONFIG.scalping.exchange,
                json.dumps({**current_features, "feature_schema_version": "legacy"}),
            ),
        ])
        connection.executemany(
            "INSERT INTO market_observations VALUES (?, ?, ?, ?)",
            rows,
        )

    observations, timestamps, count = extract_features_from_db(database_path)

    assert count == 100
    assert observations.shape == (100, 5)
    assert timestamps.tolist() == list(range(100))


def test_hmm_training_rejects_insufficient_compatible_snapshots(tmp_path):
    database_path = tmp_path / "observations.sqlite3"
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE market_observations (
                observed_at_ms INTEGER,
                symbol TEXT,
                market_source TEXT,
                features_json TEXT
            )
            """
        )
        connection.execute(
            "INSERT INTO market_observations VALUES (?, ?, ?, ?)",
            (
                1,
                CONFIG.scalping.symbol,
                CONFIG.scalping.exchange,
                json.dumps({"feature_schema_version": "legacy"}),
            ),
        )

    with pytest.raises(ValueError, match="snapshots compatibles"):
        extract_features_from_db(database_path)
