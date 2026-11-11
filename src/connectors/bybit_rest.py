"""Bybit V5 REST connector for public linear-market klines."""

import asyncio
import logging
from typing import Callable, Dict, List, Optional

import aiohttp


logger = logging.getLogger("OrderFlowAI.BybitREST")

BYBIT_REST_BASE = "https://api.bybit.com"
BYBIT_KLINE_ENDPOINT = f"{BYBIT_REST_BASE}/v5/market/kline"
RETRYABLE_API_CODES = {10000, 10006, 10016, 10018, 10019}
MAX_REQUEST_ATTEMPTS = 4
RETRY_DELAY_SECONDS = 1.0
SUPPORTED_INTERVALS = {
    "1", "3", "5", "15", "30", "60", "120", "240", "360", "720", "D", "W", "M",
}


class BybitAPIError(RuntimeError):
    def __init__(self, message: str, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


async def _fetch_klines(
    session: aiohttp.ClientSession,
    symbol: str,
    interval: str,
    limit: int,
    start_time_ms: Optional[int],
    end_time_ms: Optional[int],
) -> List[Dict[str, float]]:
    params = {
        "category": "linear",
        "symbol": symbol.upper(),
        "interval": interval,
        "limit": limit,
    }
    if start_time_ms is not None:
        params["start"] = start_time_ms
    if end_time_ms is not None:
        params["end"] = end_time_ms

    async with session.get(BYBIT_KLINE_ENDPOINT, params=params) as response:
        if response.status != 200:
            text = await response.text()
            raise BybitAPIError(
                f"Bybit klines HTTP {response.status}: {text[:200]}",
                retryable=response.status == 429 or response.status >= 500,
            )
        payload = await response.json()

    if not isinstance(payload, dict):
        raise BybitAPIError("Bybit klines response is not a JSON object.")
    if payload.get("retCode") != 0:
        code = payload.get("retCode")
        raise BybitAPIError(
            f"Bybit klines API error {code}: {payload.get('retMsg')}",
            retryable=code in RETRYABLE_API_CODES,
        )
    result = payload.get("result")
    raw_klines = result.get("list") if isinstance(result, dict) else None
    if not isinstance(raw_klines, list):
        raise RuntimeError("Bybit klines response does not contain a result list.")

    klines = []
    for row in raw_klines:
        if not isinstance(row, list) or len(row) < 6:
            raise RuntimeError("Bybit returned a kline with an invalid shape.")
        klines.append({
            "time_ms": int(row[0]),
            "open": float(row[1]),
            "high": float(row[2]),
            "low": float(row[3]),
            "close": float(row[4]),
            "volume": float(row[5]),
        })
    return sorted(klines, key=lambda candle: candle["time_ms"])


async def _fetch_klines_with_retries(
    session: aiohttp.ClientSession,
    symbol: str,
    interval: str,
    limit: int,
    start_time_ms: Optional[int],
    end_time_ms: Optional[int],
) -> List[Dict[str, float]]:
    for attempt in range(1, MAX_REQUEST_ATTEMPTS + 1):
        try:
            return await _fetch_klines(
                session, symbol, interval, limit, start_time_ms, end_time_ms
            )
        except BybitAPIError as error:
            if not error.retryable or attempt == MAX_REQUEST_ATTEMPTS:
                raise
            delay = RETRY_DELAY_SECONDS * (2 ** (attempt - 1))
            logger.warning(
                "Bybit klines devolvió un error temporal; reintento %d/%d en %.1fs: %s",
                attempt,
                MAX_REQUEST_ATTEMPTS - 1,
                delay,
                error,
            )
        except (aiohttp.ClientError, asyncio.TimeoutError):
            if attempt == MAX_REQUEST_ATTEMPTS:
                raise
            delay = RETRY_DELAY_SECONDS * (2 ** (attempt - 1))
            logger.warning(
                "Fallo temporal de red al pedir klines Bybit; reintento %d/%d en %.1fs.",
                attempt,
                MAX_REQUEST_ATTEMPTS - 1,
                delay,
            )
        await asyncio.sleep(delay)
    raise RuntimeError("Bybit kline retries ended without a response.")


async def fetch_bybit_klines(
    symbol: str = "BTCUSDT",
    interval: str = "1",
    limit: int = 1000,
    start_time_ms: Optional[int] = None,
    end_time_ms: Optional[int] = None,
) -> List[Dict[str, float]]:
    """Fetch one page of Bybit linear klines, sorted oldest to newest."""
    if interval not in SUPPORTED_INTERVALS:
        raise ValueError(f"Unsupported Bybit kline interval: {interval}")
    if not 1 <= limit <= 1000:
        raise ValueError("Bybit kline limit must be between 1 and 1000.")

    timeout = aiohttp.ClientTimeout(total=20)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        return await _fetch_klines_with_retries(
            session, symbol, interval, limit, start_time_ms, end_time_ms
        )


async def fetch_bybit_klines_paginated(
    symbol: str = "BTCUSDT",
    interval: str = "1",
    total_bars: int = 1000,
    end_time_ms: Optional[int] = None,
    on_page: Optional[Callable[[List[Dict[str, float]]], None]] = None,
) -> List[Dict[str, float]]:
    """Fetch up to ``total_bars`` klines by paging backward from the present."""
    if interval not in SUPPORTED_INTERVALS:
        raise ValueError(f"Unsupported Bybit kline interval: {interval}")
    if total_bars <= 0:
        raise ValueError("The requested kline count must be greater than zero.")

    timeout = aiohttp.ClientTimeout(total=20)
    all_klines: Dict[int, Dict[str, float]] = {}
    cursor_end_ms = end_time_ms
    async with aiohttp.ClientSession(timeout=timeout) as session:
        remaining = total_bars
        while remaining > 0:
            page_limit = min(remaining, 1000)
            page = await _fetch_klines_with_retries(
                session,
                symbol,
                interval,
                page_limit,
                start_time_ms=None,
                end_time_ms=cursor_end_ms,
            )
            if not page:
                break

            for candle in page:
                all_klines[int(candle["time_ms"])] = candle
            if on_page is not None:
                on_page(page)
            remaining = total_bars - len(all_klines)
            oldest_time_ms = int(page[0]["time_ms"])
            next_end_ms = oldest_time_ms - 1
            if cursor_end_ms is not None and next_end_ms >= cursor_end_ms:
                raise RuntimeError("Bybit kline pagination did not move backward.")
            cursor_end_ms = next_end_ms
            if remaining > 0:
                await asyncio.sleep(0.1)

    return sorted(all_klines.values(), key=lambda candle: candle["time_ms"])[-total_bars:]
