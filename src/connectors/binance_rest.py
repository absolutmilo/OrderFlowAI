"""
Binance REST API connector for historical OHLCV (klines) data.
Used by the CryptoraPRO chart endpoint to seed indicator computations.
"""

import asyncio
import logging
from typing import List, Optional
import aiohttp

logger = logging.getLogger("OrderFlowAI.BinanceREST")

BINANCE_SPOT_BASE = "https://api.binance.com"
BINANCE_FUTURES_BASE = "https://fapi.binance.com"

INTERVAL_MAP = {
    "1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m",
    "30m": "30m", "1h": "1h", "2h": "2h", "4h": "4h",
    "6h": "6h", "8h": "8h", "12h": "12h",
    "1d": "1d", "3d": "3d", "1w": "1w", "1M": "1M",
    # Aliases used by chart UI
    "60": "1h", "240": "4h", "D": "1d", "W": "1w",
}


async def fetch_klines(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    limit: int = 1500,
    start_time_ms: Optional[int] = None,
    end_time_ms: Optional[int] = None,
    futures: bool = False,
) -> List[dict]:
    """
    Fetches OHLCV klines from Binance REST API.
    
    Returns a list of dicts with keys:
        time (int, seconds), open, high, low, close, volume
    Sorted ascending by time.
    """
    base = BINANCE_FUTURES_BASE if futures else BINANCE_SPOT_BASE
    endpoint = f"{base}/fapi/v1/klines" if futures else f"{base}/api/v3/klines"
    mapped_interval = INTERVAL_MAP.get(interval, interval)
    
    params: dict = {
        "symbol": symbol.upper(),
        "interval": mapped_interval,
        "limit": min(limit, 1500),
    }
    if start_time_ms:
        params["startTime"] = start_time_ms
    if end_time_ms:
        params["endTime"] = end_time_ms

    timeout = aiohttp.ClientTimeout(total=15)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(endpoint, params=params) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"Binance klines HTTP {resp.status}: {text[:200]}")
            raw = await resp.json()

    # Binance kline format:
    # [open_time, O, H, L, C, V, close_time, qav, num_trades, tbbav, tbqav, ignore]
    result = []
    for k in raw:
        result.append({
            "time": int(k[0]) // 1000,          # seconds (lightweight-charts expects seconds)
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5]),
        })
    return result


async def fetch_klines_paginated(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    total_bars: int = 1500,
    futures: bool = False,
) -> List[dict]:
    """
    Fetches up to `total_bars` candles by paginating Binance klines.
    Binance limit per request is 1500. Paginates backwards if needed.
    """
    if total_bars <= 1500:
        return await fetch_klines(symbol=symbol, interval=interval, limit=total_bars, futures=futures)

    all_bars: List[dict] = []
    end_ms: Optional[int] = None
    remaining = total_bars

    while remaining > 0:
        batch_limit = min(remaining, 1500)
        batch = await fetch_klines(
            symbol=symbol, interval=interval, limit=batch_limit,
            end_time_ms=end_ms, futures=futures
        )
        if not batch:
            break
        all_bars = batch + all_bars
        remaining -= len(batch)
        if len(batch) < batch_limit:
            break  # Hit the beginning of exchange history
        # Next page ends just before first bar of this batch
        end_ms = batch[0]["time"] * 1000 - 1
        await asyncio.sleep(0.1)  # Polite rate-limit pause

    return all_bars
