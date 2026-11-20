import asyncio
import json
import sqlite3
import time
from unittest.mock import AsyncMock, Mock, patch

from fastapi import HTTPException

from config import CONFIG
from src.api import server
from src.core.market_data_store import MarketDataStore
from src.core.trade_model import (
    AggregatedTrade,
    ClassificationProbabilities,
    OrderFlowMetrics,
    TradeSide,
)


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


def test_orderflow_profile_endpoint_returns_selected_window_and_session_vwap(tmp_path):
    now_ms = 1_744_000_000_000
    store = MarketDataStore(tmp_path / "profile-api.sqlite3", (5,), 1.0)
    store.initialize()
    store.record_profile_trade(
        now_ms - 20_000,
        CONFIG.scalping.symbol,
        CONFIG.scalping.exchange,
        70_000.0,
        0.1,
        "BUY",
    )
    store.record_observation(
        now_ms - 20_000,
        CONFIG.scalping.symbol,
        70_000.0,
        7_000.0,
        {},
        {},
        {},
        market_source=CONFIG.scalping.exchange,
    )

    with patch.object(server, "market_data_store", store):
        with patch.object(server.time, "time", return_value=now_ms / 1000):
            response = asyncio.run(server.get_orderflow_profile(window_minutes=60))

    payload = json.loads(response.body)
    assert response.status_code == 200
    assert payload["window_minutes"] == 60
    assert payload["levels"][0]["delta_usd"] == 7_000.0
    assert payload["profile_bucket_count"] == 1
    assert payload["session_bucket_count"] == 1
    assert payload["session_vwap"] == 70_000.0
    assert payload["current_price"] == 70_000.0
    store.close()


def test_orderflow_profile_endpoint_rejects_unsupported_windows():
    try:
        asyncio.run(server.get_orderflow_profile(window_minutes=30))
    except HTTPException as error:
        assert error.status_code == 400
    else:
        raise AssertionError("Expected unsupported profile window to be rejected.")


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
    engine.run_backtest.assert_called_once_with("flow_momentum", 5, "original", True)
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


def test_market_comparison_reports_exchange_spreads_with_fresh_quotes():
    prices = {
        "Binance-Spot": {"price": 63_850.0, "updated_at_ms": 9_800},
        "Bybit-Linear": {"price": 63_890.4, "updated_at_ms": 9_900},
        "AGGR-BTCUSD": {
            "price": 64_108.6,
            "updated_at_ms": 9_950,
            "source_observed_at_ms": 9_940,
        },
    }

    with patch.object(server, "latest_exchange_prices", prices):
        with patch.object(server, "latest_exchange_metrics", {
            "Bybit-Linear": {
                "cvd_usd": -649_000.0,
                "price_delta_bps": 12.0,
                "updated_at_ms": 9_900,
            },
        }):
            with patch.object(server, "aggr_reference_quote", prices["AGGR-BTCUSD"]):
                comparison = server.market_comparison_snapshot(now_ms=10_000)

    assert comparison["spreads"]["bybit_vs_binance"]["difference_usd"] == 40.4
    assert comparison["spreads"]["aggr_vs_bybit"]["difference_usd"] == 218.2
    assert comparison["spreads"]["aggr_vs_bybit"]["difference_bps"] == 34.152
    assert comparison["flows"]["Bybit-Linear"]["cvd_usd"] == -649_000.0
    assert comparison["spreads"]["aggr_vs_bybit"]["fresh"] is True

    stale_prices = {
        "Binance-Spot": {"price": 63_850.0, "updated_at_ms": 4_999},
        "Bybit-Linear": {"price": 63_890.4, "updated_at_ms": 4_999},
    }
    with patch.object(server, "latest_exchange_prices", stale_prices):
        with patch.object(server, "latest_exchange_metrics", {}):
            with patch.object(server, "aggr_reference_quote", None):
                stale_comparison = server.market_comparison_snapshot(now_ms=10_000)

    assert stale_comparison["spreads"]["bybit_vs_binance"]["fresh"] is False


def test_aggr_reference_endpoint_accepts_chrome_extension_origin():
    payload = server.AggrReferencePrice(price=64_108.6, observed_at_ms=10_000)
    request = Mock()
    request.headers = {"origin": "chrome-extension://test-extension"}

    with patch.object(server, "aggr_reference_quote", None):
        with patch.object(
            server, "broadcast_message", new_callable=AsyncMock
        ) as broadcast:
            response = asyncio.run(
                server.receive_aggr_reference_price(payload, request)
            )

    assert response["status"] == "ok"
    broadcast.assert_awaited_once()
    assert broadcast.await_args.args[0]["market_comparison"]["prices"][
        "AGGR-BTCUSD"
    ]["price"] == 64_108.6


def test_aggr_reference_endpoint_rejects_non_extension_origin():
    payload = server.AggrReferencePrice(price=64_108.6, observed_at_ms=10_000)
    request = Mock()
    request.headers = {"origin": "https://btcusd.aggr.trade"}

    try:
        asyncio.run(server.receive_aggr_reference_price(payload, request))
    except HTTPException as error:
        assert error.status_code == 403
    else:
        raise AssertionError("Expected non-extension quote source to be rejected.")


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


def test_out_of_order_aggregated_trade_is_dropped_without_refreshing_feed():
    classifier = Mock()
    classifier.add_trade.side_effect = server.OutOfOrderTradeError(
        "Los eventos deben llegar en orden temporal."
    )
    trade = AggregatedTrade(
        exchange=CONFIG.scalping.exchange,
        symbol=CONFIG.scalping.symbol,
        first_timestamp_ms=1_000,
        last_timestamp_ms=1_000,
        price=70_000.0,
        total_size=0.1,
        total_usd_value=7_000.0,
        side=TradeSide.BUY,
        trade_count=1,
    )

    with patch.object(server, "classifier_for_stream", return_value=classifier):
        with patch.object(server, "last_bybit_trade_at_ms", 123):
            asyncio.run(server.handle_aggregated_trade(trade))
            assert server.last_bybit_trade_at_ms == 123

    classifier.classify.assert_not_called()


def test_snapshot_deadline_does_not_add_processing_time_to_the_interval():
    assert server._next_snapshot_deadline(10.0, 10.0, 1.0) == 11.0
    assert server._next_snapshot_deadline(10.0, 10.4, 1.0) == 11.4
    assert server._next_snapshot_deadline(10.0, 11.5, 1.0) == 12.5


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
            "window_seconds": metrics.window_seconds,
            "event_count": metrics.event_count,
            "window_metrics": {
                str(window): {
                    "current_price": metrics.current_price,
                    "cvd_usd": metrics.cvd_usd,
                    "cvd_btc": metrics.cvd_btc,
                    "buy_volume_usd": metrics.buy_volume_usd,
                    "sell_volume_usd": metrics.sell_volume_usd,
                    "trades_per_second": metrics.trades_per_second,
                    "volume_per_second_usd": metrics.volume_per_second_usd,
                    "price_delta_bps": metrics.price_delta_bps,
                    "inter_arrival_cv": metrics.inter_arrival_cv,
                    "window_seconds": metrics.window_seconds,
                    "event_count": metrics.event_count,
                }
                for window in CONFIG.classifier.feature_windows_seconds
            },
            "feature_schema_version": CONFIG.classifier.feature_schema_version,
        },
    )


def test_classifier_registry_keeps_exchange_symbol_streams_independent():
    binance_classifier = server.classifier_for_stream("Binance-Spot", "ETHUSDT")
    bybit_classifier = server.classifier_for_stream("Bybit-Linear", "ETHUSDT")

    assert binance_classifier is not bybit_classifier
    assert binance_classifier.stream_identity == ("Binance-Spot", "ETHUSDT")
    assert bybit_classifier.stream_identity == ("Bybit-Linear", "ETHUSDT")
    assert binance_classifier.inference_mode == "heuristic"


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
    engine.run_backtest.assert_called_once_with("fourier_breakout", 10, "inverse", True)


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


def test_candle_backtest_downloads_bybit_bars_and_never_uses_sqlite():
    now_ms = 1_800_000_000_000
    downloaded = [
        {
            "time_ms": now_ms - 4 * 300_000,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1.0,
        },
        {
            "time_ms": now_ms - 3 * 300_000,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1.0,
        },
        {
            "time_ms": now_ms - 2 * 300_000,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1.0,
        },
        {
            "time_ms": now_ms - 300_000,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1.0,
        },
    ]
    engine = Mock()
    engine.run_backtest.return_value = {"status": "collecting_candle_data"}

    with patch.object(server.time, "time", return_value=now_ms / 1000):
        with patch.object(server, "candle_backtest_engine", engine):
            with patch.object(
                server, "load_or_update_bybit_klines", new_callable=AsyncMock
            ) as load_klines:
                load_klines.return_value = (downloaded, "downloaded", len(downloaded))
                with patch.object(server, "broadcast_message", new_callable=AsyncMock):
                    response = asyncio.run(
                        server.run_candle_backtest(timeframe_minutes=10)
                    )

    assert response.status_code == 200
    load_klines.assert_awaited_once_with(
        cache_dir=server.candle_cache_path,
        symbol=server.CONFIG.scalping.symbol,
        interval="5",
        lookback_days=server.CONFIG.scalping.candle_backtest_lookback_days,
        now_ms=now_ms,
    )
    response_data = json.loads(response.body)
    assert response_data["cache_status"] == "downloaded"
    assert response_data["downloaded_candles"] == len(downloaded)
    candles, timeframe, direction, days = engine.run_backtest.call_args.args
    assert len(candles) == 2
    assert candles[0]["volume"] == 2.0
    assert timeframe == 10
    assert direction == "original"
    assert days == 30


def test_candle_backtest_rejects_out_of_range_lookback():
    try:
        asyncio.run(server.run_candle_backtest(lookback_days=31))
    except HTTPException as error:
        assert error.status_code == 400
    else:
        raise AssertionError("Expected HTTP 400 for an unsupported lookback.")


def test_dashboard_index_is_revalidated_by_browsers():
    response = asyncio.run(server.serve_index())

    assert response.headers["cache-control"] == "no-cache"
