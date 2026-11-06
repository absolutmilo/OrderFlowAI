"""
CryptoraPRO Chart API Router.
Provides endpoints to serve historical klines and computed indicator data
for the lightweight-charts chart page.
"""

import asyncio
import logging
from typing import Optional
from pathlib import Path
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from src.connectors.binance_rest import fetch_klines_paginated
from src.core.indicators import (
    OHLCVBar,
    CryptoraPROParams,
    compute,
    to_json_list,
)

logger = logging.getLogger("OrderFlowAI.ChartAPI")

router = APIRouter(prefix="/api/chart", tags=["chart"])

# In-memory indicator cache: keyed by (symbol, interval, limit)
# Simple cache to avoid re-computing on every page load
_cache: dict = {}
_cache_max_entries = 10


def _cache_key(symbol: str, interval: str, limit: int) -> str:
    return f"{symbol}_{interval}_{limit}"


@router.get("/klines")
async def get_klines(
    symbol: str = Query("BTCUSDT", description="Trading pair symbol"),
    interval: str = Query("1h", description="Candle interval: 1m 5m 15m 1h 4h 1d"),
    limit: int = Query(1500, ge=100, le=5000, description="Number of candles"),
):
    """
    Returns raw OHLCV klines from Binance REST for the given symbol/interval.
    Used by lightweight-charts candlestick series.
    """
    try:
        klines = await fetch_klines_paginated(
            symbol=symbol,
            interval=interval,
            total_bars=limit,
        )
    except Exception as exc:
        logger.error("Error fetching klines %s %s: %s", symbol, interval, exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return JSONResponse({"klines": klines, "count": len(klines)})


@router.get("/indicators")
async def get_indicators(
    symbol: str = Query("BTCUSDT"),
    interval: str = Query("1h"),
    limit: int = Query(1500, ge=100, le=5000),
    # Fourier
    c1_days: float = Query(20.0),
    c2_days: float = Query(60.0),
    dft_window: int = Query(1200, alias="N"),
    smooth_len: int = Query(3),
    # A/D
    nz: int = Query(300, alias="Nz"),
    z_thr: float = Query(2.7),
    ad_persist: int = Query(2),
    max_abs_move: float = Query(0.003),
    # MFI
    mfi_len: int = Query(14),
    # Whale
    whale_lookback: int = Query(30),
    whale_mult: float = Query(2.0),
    cooldown: int = Query(6),
    # Timeframe
    mins_per_bar: int = Query(60),
    tf_type: str = Query("Intradia"),
):
    """
    Fetches klines and computes all CryptoraPRO indicators.
    Returns per-bar indicator values for the chart overlay panel.
    """
    cache_key = _cache_key(symbol, interval, limit)

    # Try to fetch klines (always fresh — indicators are computed server-side)
    try:
        klines = await fetch_klines_paginated(
            symbol=symbol,
            interval=interval,
            total_bars=limit,
        )
    except Exception as exc:
        logger.error("Error fetching klines for indicators: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if not klines:
        raise HTTPException(status_code=404, detail="No kline data returned from Binance.")

    bars = [
        OHLCVBar(
            time=k["time"],
            open=k["open"],
            high=k["high"],
            low=k["low"],
            close=k["close"],
            volume=k["volume"],
        )
        for k in klines
    ]

    params = CryptoraPROParams(
        tf_type=tf_type,
        mins_per_bar=mins_per_bar,
        c1_days=c1_days,
        c2_days=c2_days,
        N=dft_window,
        smooth_len=smooth_len,
        Nz=nz,
        z_thr_ad=z_thr,
        ad_persist=ad_persist,
        max_abs_move=max_abs_move,
        mfi_length=mfi_len,
        whale_lookback=whale_lookback,
        whale_multiplier=whale_mult,
        cooldown_bars=cooldown,
    )

    try:
        results = await asyncio.to_thread(compute, bars, params)
    except Exception as exc:
        logger.error("Error computing CryptoraPRO indicators: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    indicator_data = to_json_list(results)

    return JSONResponse({
        "symbol": symbol,
        "interval": interval,
        "count": len(indicator_data),
        "params": {
            "c1_days": c1_days,
            "c2_days": c2_days,
            "mins_per_bar": mins_per_bar,
        },
        "indicators": indicator_data,
        "klines": klines,  # Include klines so frontend needs only one request
    })


@router.get("/params")
async def get_default_params():
    """Returns default CryptoraPRO parameter values for the frontend UI."""
    p = CryptoraPROParams()
    return JSONResponse({
        "tf_type": p.tf_type,
        "mins_per_bar": p.mins_per_bar,
        "c1_days": p.c1_days,
        "c2_days": p.c2_days,
        "N": p.N,
        "smooth_len": p.smooth_len,
        "dom_strong": p.dom_strong,
        "dom_weak": p.dom_weak,
        "st_short": p.st_short,
        "st_long": p.st_long,
        "use_ad_zones": p.use_ad_zones,
        "Nz": p.Nz,
        "z_thr_ad": p.z_thr_ad,
        "ad_persist": p.ad_persist,
        "max_abs_move": p.max_abs_move,
        "mfi_length": p.mfi_length,
        "mfi_overbought": p.mfi_overbought,
        "mfi_oversold": p.mfi_oversold,
        "whale_lookback": p.whale_lookback,
        "whale_multiplier": p.whale_multiplier,
        "cooldown_bars": p.cooldown_bars,
        "buffer_pct": p.buffer_pct,
        "max_zone_age": p.max_zone_age,
        "transition_low": p.transition_low,
        "transition_high": p.transition_high,
    })
