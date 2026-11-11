import asyncio
from unittest.mock import AsyncMock, patch

from src.connectors import bybit_kline_cache
from src.connectors.bybit_rest import BybitAPIError


def _candles(first_time_ms, last_time_ms, interval_ms):
    return [
        {
            "time_ms": time_ms,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 10.0,
        }
        for time_ms in range(first_time_ms, last_time_ms + 1, interval_ms)
    ]


def test_complete_cache_is_reused_without_requesting_bybit(tmp_path):
    interval_ms = 5 * 60_000
    now_ms = 1_800_000_000_000 // interval_ms * interval_ms
    first_time_ms = (now_ms - 86_400_000) // interval_ms * interval_ms
    last_time_ms = now_ms - interval_ms
    downloaded = _candles(first_time_ms, last_time_ms, interval_ms)

    with patch.object(
        bybit_kline_cache,
        "fetch_bybit_klines_paginated",
        new_callable=AsyncMock,
        return_value=downloaded,
    ) as fetch_history:
        initial = asyncio.run(
            bybit_kline_cache.load_or_update_bybit_klines(
                tmp_path, "BTCUSDT", "5", 1, now_ms
            )
        )
        cached = asyncio.run(
            bybit_kline_cache.load_or_update_bybit_klines(
                tmp_path, "BTCUSDT", "5", 1, now_ms
            )
        )

    assert initial == (downloaded, "downloaded", len(downloaded))
    assert cached == (downloaded, "cached", 0)
    fetch_history.assert_awaited_once()


def test_cache_fetches_only_new_recent_candles(tmp_path):
    interval_ms = 5 * 60_000
    now_ms = 1_800_000_000_000 // interval_ms * interval_ms
    first_time_ms = (now_ms - 86_400_000) // interval_ms * interval_ms
    old_last_time_ms = now_ms - interval_ms
    initial_candles = _candles(first_time_ms, old_last_time_ms, interval_ms)
    next_now_ms = now_ms + interval_ms
    next_last_time_ms = next_now_ms - interval_ms
    updated_candle = _candles(next_last_time_ms, next_last_time_ms, interval_ms)

    with patch.object(
        bybit_kline_cache,
        "fetch_bybit_klines_paginated",
        new_callable=AsyncMock,
        return_value=initial_candles,
    ) as fetch_history:
        first_result = asyncio.run(
            bybit_kline_cache.load_or_update_bybit_klines(
                tmp_path, "BTCUSDT", "5", 1, now_ms
            )
        )
    with patch.object(
        bybit_kline_cache,
        "fetch_bybit_klines",
        new_callable=AsyncMock,
        return_value=updated_candle,
    ) as fetch_recent:
        updated_result = asyncio.run(
            bybit_kline_cache.load_or_update_bybit_klines(
                tmp_path, "BTCUSDT", "5", 1, next_now_ms
            )
        )

    fetch_history.assert_awaited_once()
    fetch_recent.assert_awaited_once_with(
        symbol="BTCUSDT",
        interval="5",
        limit=1000,
        start_time_ms=old_last_time_ms,
        end_time_ms=next_now_ms,
    )
    assert first_result[1] == "downloaded"
    assert updated_result[1:] == ("updated", 1)
    assert updated_result[0][-1]["time_ms"] == next_last_time_ms


def test_cache_is_separate_for_each_source_timeframe(tmp_path):
    interval_ms = 5 * 60_000
    now_ms = 1_800_000_000_000 // interval_ms * interval_ms
    first_time_ms = (now_ms - 86_400_000) // interval_ms * interval_ms
    last_time_ms = now_ms - interval_ms
    candles = _candles(first_time_ms, last_time_ms, interval_ms)

    with patch.object(
        bybit_kline_cache,
        "fetch_bybit_klines_paginated",
        new_callable=AsyncMock,
        return_value=candles,
    ) as fetch_history:
        asyncio.run(
            bybit_kline_cache.load_or_update_bybit_klines(
                tmp_path, "BTCUSDT", "5", 1, now_ms
            )
        )
        result = asyncio.run(
            bybit_kline_cache.load_or_update_bybit_klines(
                tmp_path, "BTCUSDT", "15", 1, now_ms
            )
        )

    assert result[1] == "downloaded"
    assert fetch_history.await_count == 2
    assert (tmp_path / "BTCUSDT_5m.json").exists()
    assert (tmp_path / "BTCUSDT_15m.json").exists()


def test_partial_history_is_saved_and_next_request_resumes_older_pages(tmp_path):
    interval_ms = 5 * 60_000
    now_ms = 1_800_000_000_000 // interval_ms * interval_ms
    first_time_ms = (now_ms - 86_400_000) // interval_ms * interval_ms
    last_time_ms = now_ms - interval_ms
    all_candles = _candles(first_time_ms, last_time_ms, interval_ms)
    partial_candles = all_candles[-100:]

    async def fail_after_page(*, on_page, **_kwargs):
        on_page(partial_candles)
        raise BybitAPIError("Bybit klines API error 10016", retryable=True)

    with patch.object(
        bybit_kline_cache,
        "fetch_bybit_klines_paginated",
        new_callable=AsyncMock,
        side_effect=fail_after_page,
    ):
        try:
            asyncio.run(
                bybit_kline_cache.load_or_update_bybit_klines(
                    tmp_path, "BTCUSDT", "5", 1, now_ms
                )
            )
        except BybitAPIError as error:
            assert "10016" in str(error)
        else:
            raise AssertionError("Expected simulated transient Bybit failure.")

    async def complete_remaining(*, on_page, **kwargs):
        missing_candles = all_candles[:len(all_candles) - len(partial_candles)]
        on_page(missing_candles)
        assert kwargs["end_time_ms"] == partial_candles[0]["time_ms"] - 1
        return missing_candles

    with patch.object(
        bybit_kline_cache,
        "fetch_bybit_klines_paginated",
        new_callable=AsyncMock,
        side_effect=complete_remaining,
    ) as resume_history:
        resumed = asyncio.run(
            bybit_kline_cache.load_or_update_bybit_klines(
                tmp_path, "BTCUSDT", "5", 1, now_ms
            )
        )

    assert resumed == (all_candles, "resumed", len(all_candles) - len(partial_candles))
    resume_history.assert_awaited_once()
