import json
import sqlite3

from src.core.market_data_store import MarketDataStore


def test_store_snapshots_configuration_and_resolves_forward_labels(tmp_path):
    store = MarketDataStore(tmp_path / "market.sqlite3", (5,), 1.0)
    store.initialize()
    configuration = {"classifier": {"window": 200}}

    store.record_observation(
        observed_at_ms=1_000,
        symbol="BTCUSDT",
        price=70_000.0,
        cvd_usd=10_000.0,
        features={"cvd_usd": 10_000.0},
        heuristic_scores={"dominant_regime": "RETAIL_DOMINANT"},
        configuration=configuration,
    )
    store.record_observation(
        observed_at_ms=6_000,
        symbol="BTCUSDT",
        price=69_900.0,
        cvd_usd=5_000.0,
        features={"cvd_usd": 5_000.0},
        heuristic_scores={"dominant_regime": "RETAIL_DOMINANT"},
        configuration=configuration,
    )
    updated_configuration = {"classifier": {"window": 100}}
    store.record_observation(
        observed_at_ms=7_000,
        symbol="BTCUSDT",
        price=69_950.0,
        cvd_usd=-2_000.0,
        features={"cvd_usd": -2_000.0},
        heuristic_scores={"dominant_regime": "RETAIL_DOMINANT"},
        configuration=updated_configuration,
    )

    row = store._connection().execute(
        """
        SELECT return_5s_bps, label_5s, features_json, configuration_id
        FROM market_observations
        WHERE observed_at_ms = 1000
        """
    ).fetchone()
    config_count = store._connection().execute(
        "SELECT COUNT(*) FROM configuration_snapshots"
    ).fetchone()[0]

    assert row[0] < -1.0
    assert row[1] == "reversal"
    assert json.loads(row[2]) == {"cvd_usd": 10_000.0}
    assert row[3] is not None
    assert config_count == 2
    assert store.observation_count() == 3
    store.close()


def test_store_marks_small_moves_neutral(tmp_path):
    store = MarketDataStore(tmp_path / "market.sqlite3", (5,), 1.0)
    store.initialize()
    configuration = {}

    store.record_observation(1_000, "BTCUSDT", 70_000.0, 100.0, {}, {}, configuration)
    store.record_observation(6_000, "BTCUSDT", 70_004.0, 100.0, {}, {}, configuration)

    label = store._connection().execute(
        "SELECT label_5s FROM market_observations WHERE observed_at_ms = 1000"
    ).fetchone()[0]
    assert label == "neutral"
    store.close()


def test_store_migrates_existing_database_and_adds_scalping_horizons(tmp_path):
    database_path = tmp_path / "existing.sqlite3"
    old_store = MarketDataStore(database_path, (5,), 1.0)
    old_store.initialize()
    old_store.record_observation(
        1_000,
        "BTCUSDT",
        70_000.0,
        10_000.0,
        {"cvd_usd": 10_000.0},
        {},
        {},
    )
    old_store.close()

    store = MarketDataStore(database_path, (5, 300, 600, 900), 1.0)
    store.initialize()
    store.record_observation(
        2_000,
        "BTCUSDT",
        70_000.0,
        10_000.0,
        {"cvd_usd": 10_000.0},
        {},
        {},
        market_source="Bybit-Linear",
    )
    store.record_observation(
        302_000, "BTCUSDT", 70_070.0, 10_000.0, {}, {}, {},
        market_source="Bybit-Linear",
    )
    store.record_observation(
        602_000, "BTCUSDT", 70_100.0, 10_000.0, {}, {}, {},
        market_source="Bybit-Linear",
    )
    store.record_observation(
        902_000, "BTCUSDT", 70_140.0, 10_000.0, {}, {}, {},
        market_source="Bybit-Linear",
    )

    rows = store._connection().execute(
        """
        SELECT market_source, return_300s_bps, return_900s_bps
        FROM market_observations
        ORDER BY observed_at_ms
        """
    ).fetchall()
    assert rows[0] == ("legacy_combined", None, None)
    assert rows[1][0] == "Bybit-Linear"
    assert rows[1][1] == (70.0 / 70_000.0) * 10_000
    assert rows[1][2] == (140.0 / 70_000.0) * 10_000
    store.close()


def test_store_prunes_oldest_observations_to_keep_database_under_limit(tmp_path):
    database_path = tmp_path / "bounded.sqlite3"
    maximum_size = 180_000
    store = MarketDataStore(
        database_path,
        (5,),
        1.0,
        max_database_size_bytes=maximum_size,
        retention_target_fraction=0.75,
    )
    store.initialize()
    large_features = {"payload": "x" * 8_000}

    for index in range(40):
        store.record_observation(
            observed_at_ms=index * 1000,
            symbol="BTCUSDT",
            price=70_000.0,
            cvd_usd=1_000.0,
            features=large_features,
            heuristic_scores={},
            configuration={"version": 1},
        )

    retained = store._connection().execute(
        "SELECT observed_at_ms FROM market_observations ORDER BY observed_at_ms"
    ).fetchall()
    orphaned_configurations = store._connection().execute(
        """
        SELECT COUNT(*) FROM configuration_snapshots
        WHERE NOT EXISTS (
            SELECT 1 FROM market_observations
            WHERE market_observations.configuration_id = configuration_snapshots.id
        )
        """
    ).fetchone()[0]

    assert 0 < len(retained) < 40
    assert retained[-1][0] == 39_000
    assert store.storage_size_bytes() <= maximum_size
    assert orphaned_configurations == 0
    store.close()


def test_store_enforces_limit_when_opening_existing_database(tmp_path):
    database_path = tmp_path / "existing-bounded.sqlite3"
    store = MarketDataStore(database_path, (5,), 1.0)
    store.initialize()
    large_features = {"payload": "x" * 8_000}
    for index in range(20):
        store.record_observation(
            observed_at_ms=index * 1000,
            symbol="BTCUSDT",
            price=70_000.0,
            cvd_usd=1_000.0,
            features=large_features,
            heuristic_scores={},
            configuration={},
        )
    store.close()
    actual_size = database_path.stat().st_size
    store = MarketDataStore(
        database_path,
        (5,),
        1.0,
        max_database_size_bytes=max(150_000, actual_size // 2),
        retention_target_fraction=0.75,
    )

    store.initialize()

    assert store.storage_size_bytes() <= store.max_database_size_bytes
    assert store.observation_count() < 20
    assert store._connection().execute(
        "SELECT MAX(observed_at_ms) FROM market_observations"
    ).fetchone()[0] == 19_000
    store.close()
