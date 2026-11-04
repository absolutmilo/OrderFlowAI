import asyncio
import sqlite3
from unittest.mock import AsyncMock, Mock, patch

from fastapi import HTTPException

from src.api import server


def test_scalping_status_counts_only_the_configured_exchange(tmp_path):
    database_path = tmp_path / "status.sqlite3"
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE market_observations (
                observed_at_ms INTEGER,
                symbol TEXT,
                market_source TEXT
            )
            """
        )
        connection.executemany(
            "INSERT INTO market_observations VALUES (?, ?, ?)",
            [
                (1, "BTCUSDT", "Bybit-Linear"),
                (2, "BTCUSDT", "legacy_combined"),
                (3, "ETHUSDT", "Bybit-Linear"),
            ],
        )

    with patch.object(server, "database_path", database_path):
        response = asyncio.run(server.get_scalping_status())

    assert response.status_code == 200
    assert b'"observations":1' in response.body
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
