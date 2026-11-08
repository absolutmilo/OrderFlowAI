import asyncio
import json
import sqlite3
import time
from unittest.mock import AsyncMock, Mock, patch

from fastapi import HTTPException

from src.api import server
from src.core.trade_model import ClassificationProbabilities, OrderFlowMetrics


def test_scalping_status_counts_only_the_configured_exchange(tmp_path):
    database_path = tmp_path / "status.sqlite3"
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE market_observations (
                observed_at_ms INTEGER,
                symbol TEXT,
                market_source TEXT,
                heuristic_scores_json TEXT
            )
            """
        )
        connection.executemany(
            "INSERT INTO market_observations VALUES (?, ?, ?, ?)",
            [
                (
                    1,
                    "BTCUSDT",
                    "Bybit-Linear",
                    '{"hmm_4":{"available":true}}',
                ),
                (2, "BTCUSDT", "legacy_combined", '{"hmm_4":{"available":true}}'),
                (3, "ETHUSDT", "Bybit-Linear", '{"hmm_4":{"available":true}}'),
            ],
        )

    with patch.object(server, "database_path", database_path):
        response = asyncio.run(server.get_scalping_status())

    assert response.status_code == 200
    assert b'"observations":1' in response.body
    assert b'"hmm_snapshot_count":1' in response.body
    assert b'"recent_window_hours":24' in response.body
    assert b'"recent_observations":0' in response.body
    assert b'"recent_snapshot_coverage_pct":0.0' in response.body
    assert b'"minimum_training_observations":36000' in response.body
    assert b'"minimum_observations":51429' in response.body


def test_scalping_status_reports_recent_snapshot_coverage(tmp_path):
    database_path = tmp_path / "recent-status.sqlite3"
    now_ms = int(time.time() * 1000)
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE market_observations (
                observed_at_ms INTEGER,
                symbol TEXT,
                market_source TEXT,
                heuristic_scores_json TEXT
            )
            """
        )
        connection.executemany(
            "INSERT INTO market_observations VALUES (?, ?, ?, ?)",
            [
                (now_ms - 3_000, "BTCUSDT", "Bybit-Linear", "{}"),
                (now_ms - 2_000, "BTCUSDT", "Bybit-Linear", "{}"),
                (now_ms - 1_000, "BTCUSDT", "Bybit-Linear", "{}"),
            ],
        )

    with patch.object(server, "database_path", database_path):
        response = asyncio.run(server.get_scalping_status())

    assert response.status_code == 200
    assert b'"recent_observations":3' in response.body
    assert b'"recent_snapshot_coverage_pct":100.0' in response.body
    assert b'"recent_maximum_snapshot_gap_seconds":1.0' in response.body


def test_backtest_endpoint_returns_engine_result_and_broadcasts():
    result = {"status": "validated", "alert_enabled": True}
    engine = Mock()
    engine.run_backtest.return_value = result

    with patch.object(server, "scalping_engine", engine):
        with patch.object(
            server, "broadcast_message", new_callable=AsyncMock
        ) as broadcast:
            response = asyncio.run(server.run_scalping_backtest())

    assert response.status_code == 200
    assert response.body == b'{"status":"validated","alert_enabled":true}'
    engine.run_backtest.assert_called_once_with("flow_momentum", 5, "original")
    broadcast.assert_awaited_once()


def test_snapshot_scores_preserve_heuristics_and_add_hmm4_posterior():
    heuristic = ClassificationProbabilities(
        algo_institutional=10.0,
        iceberg=20.0,
        absorption=30.0,
        retail=40.0,
        hft_competition=5.0,
        breakout=6.0,
        dominant_regime="RETAIL_DOMINANT",
    )
    hmm = ClassificationProbabilities(
        algo_institutional=60.0,
        iceberg=20.0,
        absorption=10.0,
        retail=10.0,
        hft_competition=4.0,
        breakout=6.0,
        dominant_regime="ALGO_INSTITUTIONAL",
    )

    scores = server.snapshot_scores(heuristic, hmm)

    assert scores["dominant_regime"] == "RETAIL_DOMINANT"
    assert scores["hmm_4"]["available"] is True
    assert scores["hmm_4"]["dominant_regime"] == "ALGO_INSTITUTIONAL"


def test_capture_market_snapshot_requires_a_fresh_bybit_trade():
    now_ms = int(time.time() * 1000)
    with patch.object(server, "last_bybit_trade_at_ms", now_ms - 6_000):
        with patch.object(server, "data_capture_status", "recording"):
            with patch.object(server.scalp_classifier, "is_warmed_up") as warmed_up:
                with patch.object(server.market_data_store, "record_observation") as save:
                    assert server.capture_market_snapshot(now_ms) is False
            assert server.data_capture_status == "waiting_for_feed"

    warmed_up.assert_not_called()
    save.assert_not_called()


def test_capture_market_snapshot_persists_latest_fresh_state():
    now_ms = int(time.time() * 1000)
    metrics = OrderFlowMetrics(
        current_price=100_000.0,
        cvd_usd=1_000.0,
        cvd_btc=0.01,
        buy_volume_usd=2_000.0,
        sell_volume_usd=1_000.0,
        trades_per_second=10.0,
        volume_per_second_usd=3_000.0,
        price_delta_bps=1.0,
        inter_arrival_cv=0.5,
    )
    probabilities = ClassificationProbabilities(
        algo_institutional=10.0,
        iceberg=20.0,
        absorption=30.0,
        retail=40.0,
        hft_competition=5.0,
        breakout=6.0,
        dominant_regime="RETAIL_DOMINANT",
    )
    classifier = Mock()
    classifier.is_warmed_up.return_value = True
    classifier.get_current_metrics.return_value = metrics
    classifier.classify.return_value = probabilities
    classifier.hmm_model = None

    with patch.object(server, "last_bybit_trade_at_ms", now_ms):
        with patch.object(server, "last_capture_at_ms", 0):
            with patch.object(server, "data_capture_status", "starting"):
                with patch.object(server, "scalp_classifier", classifier):
                    with patch.object(server, "market_data_store") as store:
                        with patch.object(server, "scalping_engine") as engine:
                            assert server.capture_market_snapshot(now_ms) is True
                            assert server.last_capture_at_ms == now_ms
                            assert server.data_capture_status == "recording"

    store.record_observation.assert_called_once()
    engine.observe.assert_called_once_with(
        now_ms,
        metrics.current_price,
        {
            "current_price": metrics.current_price,
            "cvd_usd": metrics.cvd_usd,
            "cvd_btc": metrics.cvd_btc,
            "buy_volume_usd": metrics.buy_volume_usd,
            "sell_volume_usd": metrics.sell_volume_usd,
            "trades_per_second": metrics.trades_per_second,
            "volume_per_second_usd": metrics.volume_per_second_usd,
            "price_delta_bps": metrics.price_delta_bps,
            "inter_arrival_cv": metrics.inter_arrival_cv,
        },
    )


def test_classification_history_returns_selected_mode_for_last_hour(tmp_path):
    database_path = tmp_path / "classification-history.sqlite3"
    now_ms = int(time.time() * 1000)
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE market_observations (
                observed_at_ms INTEGER,
                symbol TEXT,
                market_source TEXT,
                heuristic_scores_json TEXT
            )
            """
        )
        connection.executemany(
            "INSERT INTO market_observations VALUES (?, ?, ?, ?)",
            [
                (
                    now_ms - 1_000,
                    "BTCUSDT",
                    "Bybit-Linear",
                    json.dumps({
                        "algo_institutional": 10,
                        "iceberg": 20,
                        "absorption": 30,
                        "retail": 40,
                        "hft_competition": 5,
                        "hmm_4": {
                            "available": True,
                            "algo_institutional": 60,
                            "iceberg": 20,
                            "absorption": 10,
                            "retail": 10,
                            "hft_competition": 5,
                        },
                    }),
                ),
                (
                    now_ms - 3_700_000,
                    "BTCUSDT",
                    "Bybit-Linear",
                    '{"algo_institutional":99}',
                ),
                (
                    now_ms - 2_000,
                    "BTCUSDT",
                    "legacy_combined",
                    '{"algo_institutional":99}',
                ),
            ],
        )

    with patch.object(server, "database_path", database_path):
        heuristic_response = asyncio.run(
            server.get_classification_history(mode="heuristic")
        )
        model_response = asyncio.run(
            server.get_classification_history(mode="model")
        )

    heuristic_data = json.loads(heuristic_response.body)
    model_data = json.loads(model_response.body)
    assert len(heuristic_data["samples"]) == 1
    assert heuristic_data["samples"][0]["values"]["algo_institutional"] == 10
    assert len(model_data["samples"]) == 1
    assert model_data["samples"][0]["values"]["algo_institutional"] == 60


def test_classification_history_rejects_unknown_mode():
    try:
        asyncio.run(server.get_classification_history(mode="unknown"))
    except HTTPException as error:
        assert error.status_code == 400
    else:
        raise AssertionError("Expected HTTP 400 for unknown inference mode.")


def test_backtest_endpoint_rejects_unknown_strategy():
    try:
        asyncio.run(server.run_scalping_backtest(strategy="unknown"))
    except HTTPException as error:
        assert error.status_code == 400
    else:
        raise AssertionError("Expected HTTP 400 for unknown strategy.")


def test_backtest_endpoint_passes_selected_strategy_and_timeframe():
    engine = Mock()
    engine.run_backtest.return_value = {"status": "collecting_timeframe_data"}

    with patch.object(server, "scalping_engine", engine):
        with patch.object(server, "broadcast_message", new_callable=AsyncMock):
            response = asyncio.run(
                server.run_scalping_backtest(
                    strategy="fourier_breakout",
                    timeframe_minutes=10,
                    direction_mode="inverse",
                )
            )

    assert response.status_code == 200
    engine.run_backtest.assert_called_once_with("fourier_breakout", 10, "inverse")


def test_backtest_endpoint_rejects_unsupported_timeframe():
    try:
        asyncio.run(server.run_scalping_backtest(timeframe_minutes=3))
    except HTTPException as error:
        assert error.status_code == 400
    else:
        raise AssertionError("Expected HTTP 400 for unsupported timeframe.")


def test_backtest_endpoint_rejects_unsupported_direction():
    try:
        asyncio.run(server.run_scalping_backtest(direction_mode="sideways"))
    except HTTPException as error:
        assert error.status_code == 400
    else:
        raise AssertionError("Expected HTTP 400 for unsupported direction.")


def test_backtest_endpoint_reports_storage_errors():
    engine = Mock()
    engine.run_backtest.side_effect = sqlite3.OperationalError("database unavailable")

    with patch.object(server, "scalping_engine", engine):
        try:
            asyncio.run(server.run_scalping_backtest())
        except HTTPException as error:
            assert error.status_code == 503
            assert error.detail == "database unavailable"
        else:
            raise AssertionError("Expected HTTP 503 for database failure.")


def test_dashboard_index_is_revalidated_by_browsers():
    response = asyncio.run(server.serve_index())

    assert response.headers["cache-control"] == "no-cache"
