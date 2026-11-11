"""Persistent local cache for Bybit historical kline downloads."""

import json
import logging
import math
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

from src.connectors.bybit_rest import fetch_bybit_klines, fetch_bybit_klines_paginated

logger = logging.getLogger("OrderFlowAI.BybitKlineCache")

_CACHE_VERSION = 1
_MINUTE_INTERVALS = {"1": 1, "5": 5, "15": 15}


def _cache_path(cache_dir: Path, symbol: str, interval: str) -> Path:
    safe_symbol = "".join(char for char in symbol.upper() if char.isalnum())
    return cache_dir / f"{safe_symbol}_{interval}m.json"


def _read_cache(path: Path, symbol: str, interval: str) -> Dict[int, Dict[str, float]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as error:
        logger.warning("No se pudo leer la caché %s; se descargará de nuevo: %s", path, error)
        return {}

    if (
        not isinstance(payload, dict)
        or payload.get("version") != _CACHE_VERSION
        or payload.get("symbol") != symbol.upper()
        or payload.get("interval") != interval
        or not isinstance(payload.get("klines"), list)
    ):
        logger.warning("La caché %s no coincide con su esquema o temporalidad; se descargará de nuevo.", path)
        return {}

    candles = {}
    for row in payload["klines"]:
        try:
            time_ms = int(row["time_ms"])
            candle = {
                key: float(row[key])
                for key in ("open", "high", "low", "close", "volume")
            }
        except (KeyError, TypeError, ValueError):
            logger.warning("La caché %s contiene una vela inválida; se descargará de nuevo.", path)
            return {}
        if (
            time_ms < 0
            or not all(math.isfinite(value) for value in candle.values())
            or min(candle["open"], candle["high"], candle["low"], candle["close"]) <= 0
            or candle["volume"] < 0
        ):
            logger.warning("La caché %s contiene valores de vela inválidos; se descargará de nuevo.", path)
            return {}
        candles[time_ms] = {"time_ms": time_ms, **candle}
    return candles


def _write_cache(
    path: Path,
    symbol: str,
    interval: str,
    candles: Dict[int, Dict[str, float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": _CACHE_VERSION,
        "symbol": symbol.upper(),
        "interval": interval,
        "klines": sorted(candles.values(), key=lambda candle: candle["time_ms"]),
    }
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f"{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(payload, temporary_file, separators=(",", ":"))
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _has_contiguous_range(
    candles: Dict[int, Dict[str, float]],
    first_time_ms: int,
    last_time_ms: int,
    interval_ms: int,
) -> bool:
    if first_time_ms > last_time_ms:
        return False
    return all(
        candle_time in candles
        for candle_time in range(first_time_ms, last_time_ms + 1, interval_ms)
    )


async def load_or_update_bybit_klines(
    cache_dir: Path,
    symbol: str,
    interval: str,
    lookback_days: int,
    now_ms: int,
) -> Tuple[List[Dict[str, float]], str, int]:
    """Return a contiguous lookback window and whether it was cached or downloaded."""
    if interval not in _MINUTE_INTERVALS:
        raise ValueError(f"Unsupported cached Bybit kline interval: {interval}")
    if lookback_days <= 0:
        raise ValueError("The requested lookback must be greater than zero.")

    symbol = symbol.upper()
    interval_minutes = _MINUTE_INTERVALS[interval]
    interval_ms = interval_minutes * 60_000
    first_time_ms = (now_ms - lookback_days * 86_400_000) // interval_ms * interval_ms
    last_time_ms = now_ms // interval_ms * interval_ms - interval_ms
    expected_count = (last_time_ms - first_time_ms) // interval_ms + 1
    path = _cache_path(cache_dir, symbol, interval)
    cached = _read_cache(path, symbol, interval)

    if _has_contiguous_range(cached, first_time_ms, last_time_ms, interval_ms):
        return (
            [cached[time_ms] for time_ms in range(first_time_ms, last_time_ms + 1, interval_ms)],
            "cached",
            0,
        )

    def save_page(page: List[Dict[str, float]]) -> None:
        retention_start_ms = first_time_ms - 5 * interval_ms
        for candle in page:
            candle_time = int(candle["time_ms"])
            if retention_start_ms <= candle_time <= last_time_ms:
                cached[candle_time] = candle
        _write_cache(path, symbol, interval, cached)

    resumable_tail = (
        bool(cached)
        and last_time_ms in cached
        and min(cached) > first_time_ms
        and _has_contiguous_range(cached, min(cached), last_time_ms, interval_ms)
    )
    if resumable_tail:
        missing_count = (min(cached) - first_time_ms) // interval_ms
        downloaded = await fetch_bybit_klines_paginated(
            symbol=symbol,
            interval=interval,
            total_bars=missing_count + 5,
            end_time_ms=min(cached) - 1,
            on_page=save_page,
        )
        for candle in downloaded:
            candle_time = int(candle["time_ms"])
            if first_time_ms - 5 * interval_ms <= candle_time <= last_time_ms:
                cached[candle_time] = candle
        _write_cache(path, symbol, interval, cached)
        return (
            [
                cached[candle_time]
                for candle_time in sorted(cached)
                if first_time_ms <= candle_time <= last_time_ms
            ],
            "resumed",
            len(downloaded),
        )

    existing_range_complete = (
        bool(cached)
        and first_time_ms in cached
        and _has_contiguous_range(
            cached,
            first_time_ms,
            min(max(cached), last_time_ms),
            interval_ms,
        )
    )
    missing_count = max(0, expected_count - len([
        time_ms for time_ms in range(first_time_ms, last_time_ms + 1, interval_ms)
        if time_ms in cached
    ]))

    if (
        existing_range_complete
        and max(cached) <= last_time_ms
        and missing_count <= 999
    ):
        updated = await fetch_bybit_klines(
            symbol=symbol,
            interval=interval,
            limit=1000,
            start_time_ms=max(cached),
            end_time_ms=now_ms,
        )
        downloaded_count = len(updated)
        for candle in updated:
            candle_time = int(candle["time_ms"])
            if candle_time + interval_ms <= now_ms:
                cached[candle_time] = candle
        cache_status = "updated"
    else:
        downloaded = await fetch_bybit_klines_paginated(
            symbol=symbol,
            interval=interval,
            total_bars=expected_count + 5,
            end_time_ms=now_ms,
            on_page=save_page,
        )
        downloaded_count = len(downloaded)
        for candle in downloaded:
            candle_time = int(candle["time_ms"])
            if first_time_ms - 5 * interval_ms <= candle_time <= last_time_ms:
                cached[candle_time] = candle
        cache_status = "downloaded"

    retention_start_ms = first_time_ms - 5 * interval_ms
    cached = {
        candle_time: candle
        for candle_time, candle in cached.items()
        if retention_start_ms <= candle_time <= last_time_ms
    }
    _write_cache(path, symbol, interval, cached)

    if not _has_contiguous_range(cached, first_time_ms, last_time_ms, interval_ms):
        return (
            [
                cached[candle_time]
                for candle_time in sorted(cached)
                if first_time_ms <= candle_time <= last_time_ms
            ],
            cache_status,
            downloaded_count,
        )
    return (
        [cached[time_ms] for time_ms in range(first_time_ms, last_time_ms + 1, interval_ms)],
        cache_status,
        downloaded_count,
    )
