import asyncio
import json
import sqlite3
import time
from unittest.mock import AsyncMock, Mock, patch

from fastapi import HTTPException

from src.api import server
from src.core.trade_model import ClassificationProbabilities


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
    assert b'"minimum_training_observations":36000' in response.body
    assert b'"minimum_observations":51429' in response.body


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
