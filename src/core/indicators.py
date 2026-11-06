"""
CryptoraPRO Indicator Engine — Python/NumPy port of the PineScript strategy
"ONE PANEL: Fourier + A/D + MFI + Whale + Spec + Heatmap".

All computations match the PineScript logic bar-for-bar.
Designed to run on a list of OHLCV candles (time-ordered, ascending).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class OHLCVBar:
    time: int    # Unix seconds
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class CryptoraPROParams:
    """All configurable parameters matching PineScript inputs."""
    # Timeframe
    tf_type: str = "Intradia"        # "Intradia" | "Diario" | "Semanal" | "Mensual"
    mins_per_bar: int = 60

    # Fourier
    c1_days: float = 20.0
    c2_days: float = 60.0
    N: int = 1200                    # DFT window (bars)
    smooth_len: int = 3              # Return smoothing EMA

    # Fourier states
    dom_strong: float = 0.60
    dom_weak: float = 0.55
    st_short: int = 20               # Stability short EMA
    st_long: int = 80                # Stability long EMA

    # A/D zones
    use_ad_zones: bool = True
    Nz: int = 300                    # Residual z-score window
    z_thr_ad: float = 2.7
    ad_persist: int = 2
    max_abs_move: float = 0.003

    # Volume confirmation (A/D)
    use_vol_confirm: bool = False
    vol_lookback: int = 100
    vol_z_thr: float = 0.5

    # MFI
    mfi_length: int = 14
    mfi_overbought: int = 80
    mfi_oversold: int = 20

    # Whale
    whale_lookback: int = 30
    whale_multiplier: float = 2.0
    cooldown_bars: int = 6

    # Heatmap
    corr_len: int = 20

    # Entry alerts
    buffer_pct: float = 0.0005
    max_zone_age: int = 200
    transition_low: float = 45.0
    transition_high: float = 55.0

    # DOM switch (stability threshold)
    dom_strong_th: float = 0.60
    dom_weak_th: float = 0.55


@dataclass
class BarResult:
    """Computed indicator values for a single bar."""
    # Raw
    time: int
    close: float
    volume: float

    # Fourier
    share1: float = 0.5         # Cycle 1 energy share (0-1)
    share2: float = 0.5         # Cycle 2 energy share (0-1)
    dom_id: int = 1             # 1 or 2
    dom_share: float = 0.5
    state: int = 3              # 1=TREND, 2=WEAKENING, 3=TRANSITION
    stability: float = 0.0
    recon: float = 0.0
    resid: float = 0.0
    z: float = 0.0
    ret: float = 0.0

    # A/D
    is_accum: bool = False
    is_distrib: bool = False

    # MFI
    mfi: float = 50.0

    # Whale
    whale_buy: bool = False
    whale_sell: bool = False

    # Zone state
    last_accum_high: Optional[float] = None
    last_accum_low: Optional[float] = None
    last_distrib_high: Optional[float] = None
    last_distrib_low: Optional[float] = None
    run_acc_high: Optional[float] = None
    run_acc_low: Optional[float] = None
    run_dis_high: Optional[float] = None
    run_dis_low: Optional[float] = None

    # Entry signals
    long_confirmed: bool = False
    short_confirmed: bool = False
    enter_accum_zone: bool = False
    enter_distrib_zone: bool = False

    # Dominance switch
    dom_switch: bool = False


# ---------------------------------------------------------------------------
# Rolling helpers (pure NumPy, vectorised)
# ---------------------------------------------------------------------------

def _rolling_sum(x: np.ndarray, w: int) -> np.ndarray:
    """Rolling sum with NaN for the first (w-1) elements."""
    n = len(x)
    cs = np.cumsum(x)
    result = np.full(n, np.nan)
    result[w - 1:] = cs[w - 1:]
    result[w:] -= cs[:-w]
    return result


def _rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    return _rolling_sum(x, w) / w


def _rolling_std(x: np.ndarray, w: int, ddof: int = 1) -> np.ndarray:
    """Rolling standard deviation using the two-pass formula."""
    n = len(x)
    result = np.full(n, np.nan)
    for i in range(w - 1, n):
        window = x[i - w + 1: i + 1]
        result[i] = float(np.std(window, ddof=ddof))
    return result


def _ema(x: np.ndarray, span: int) -> np.ndarray:
    """Exponential moving average (Wilder/EMA style, α=2/(span+1))."""
    alpha = 2.0 / (span + 1)
    result = np.full(len(x), np.nan)
    # Find first non-NaN
    start = 0
    while start < len(x) and np.isnan(x[start]):
        start += 1
    if start >= len(x):
        return result
    result[start] = x[start]
    for i in range(start + 1, len(x)):
        if np.isnan(x[i]):
            result[i] = result[i - 1]
        else:
            result[i] = alpha * x[i] + (1 - alpha) * result[i - 1]
    return result


def _rolling_corr(x: np.ndarray, y: np.ndarray, w: int) -> np.ndarray:
    """Rolling Pearson correlation between x and y."""
    n = len(x)
    result = np.full(n, np.nan)
    for i in range(w - 1, n):
        wx = x[i - w + 1: i + 1]
        wy = y[i - w + 1: i + 1]
        if np.std(wx) > 0 and np.std(wy) > 0:
            result[i] = float(np.corrcoef(wx, wy)[0, 1])
        else:
            result[i] = 0.0
    return result


# ---------------------------------------------------------------------------
# DFT energy (matches PineScript dft_params exactly)
# ---------------------------------------------------------------------------

def _dft_energy(ret: np.ndarray, cycle_len: int, win: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes per-bar DFT amplitude (energy) for a single cycle frequency.
    
    PineScript equivalent:
        s = sin(2π * bar_index / len)
        c = cos(2π * bar_index / len)
        a = sum(x*s, win) / sum(s*s, win)
        b = sum(x*c, win) / sum(c*c, win)
        e = sqrt(a² + b²)
    
    Returns (e, a, b, s_arr, c_arr).
    """
    n = len(ret)
    idx = np.arange(n, dtype=np.float64)

    s = np.sin(2.0 * np.pi * idx / cycle_len)
    c = np.cos(2.0 * np.pi * idx / cycle_len)

    xs = _rolling_sum(ret * s, win)
    xc = _rolling_sum(ret * c, win)
    ss = _rolling_sum(s * s, win)
    cc = _rolling_sum(c * c, win)

    with np.errstate(invalid="ignore", divide="ignore"):
        a = np.where(ss != 0.0, xs / ss, 0.0)
        b = np.where(cc != 0.0, xc / cc, 0.0)

    e = np.sqrt(a ** 2 + b ** 2)
    # Where rolling sums are NaN (first win-1 bars), energy is NaN too
    e = np.where(np.isnan(xs), np.nan, e)
    a = np.where(np.isnan(xs), np.nan, a)
    b = np.where(np.isnan(xs), np.nan, b)

    return e, a, b, s, c


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------

def compute(
    bars: List[OHLCVBar],
    params: CryptoraPROParams,
    spx_closes: Optional[np.ndarray] = None,  # SP500 close prices aligned to bars
    vix_closes: Optional[np.ndarray] = None,   # VIX close prices aligned to bars
) -> List[BarResult]:
    """
    Computes all CryptoraPRO indicators for the full bar history.
    Returns one BarResult per bar.
    """
    if len(bars) < 2:
        return []

    n = len(bars)
    p = params

    # --- Bars per day
    if p.tf_type == "Intradia":
        bars_per_day = 1440.0 / p.mins_per_bar
    elif p.tf_type == "Diario":
        bars_per_day = 1.0
    elif p.tf_type == "Semanal":
        bars_per_day = 1.0 / 7.0
    else:  # Mensual
        bars_per_day = 1.0 / 30.0

    def len_from_days(d: float) -> int:
        return max(2, round(d * bars_per_day))

    len1 = len_from_days(p.c1_days)
    len2 = len_from_days(p.c2_days)

    # --- OHLCV arrays
    closes = np.array([b.close for b in bars], dtype=np.float64)
    highs  = np.array([b.high  for b in bars], dtype=np.float64)
    lows   = np.array([b.low   for b in bars], dtype=np.float64)
    opens  = np.array([b.open  for b in bars], dtype=np.float64)
    vols   = np.array([b.volume for b in bars], dtype=np.float64)

    # --- Log returns (nan-safe)
    ret_raw = np.concatenate(([0.0], np.log(closes[1:] / closes[:-1])))
    ret_raw = np.where(np.isfinite(ret_raw), ret_raw, 0.0)
    ret = _ema(ret_raw, p.smooth_len)
    ret = np.where(np.isnan(ret), 0.0, ret)

    # --- Fourier: DFT energies for 2 cycles
    e1, a1, b1, s1, c1_arr = _dft_energy(ret, len1, p.N)
    e2, a2, b2, s2, c2_arr = _dft_energy(ret, len2, p.N)

    # Reconstruction + residual
    recon = np.where(
        np.isnan(a1), 0.0,
        (a1 * s1 + b1 * c1_arr) + (a2 * s2 + b2 * c2_arr)
    )
    resid = ret - recon

    # Residual z-score
    mu_z  = _rolling_mean(resid, p.Nz)
    sig_z = np.array([
        float(np.std(resid[max(0, i - p.Nz + 1): i + 1], ddof=1)) if i >= p.Nz - 1 else np.nan
        for i in range(n)
    ])
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(sig_z != 0.0, (resid - mu_z) / sig_z, 0.0)
    z = np.where(np.isfinite(z), z, 0.0)

    # --- Energy shares
    e_sum = e1 + e2
    with np.errstate(invalid="ignore", divide="ignore"):
        share1 = np.where(e_sum > 0, e1 / e_sum, 0.5)
        share2 = np.where(e_sum > 0, e2 / e_sum, 0.5)
    share1 = np.where(np.isfinite(share1), share1, 0.5)
    share2 = np.where(np.isfinite(share2), share2, 0.5)

    dom_id = np.where(share1 >= share2, 1, 2)
    dom_share = np.where(dom_id == 1, share1, share2)
    dom_energy = np.where(dom_id == 1, e1, e2)
    dom_energy = np.where(np.isfinite(dom_energy), dom_energy, 0.0)

    # Stability: EMA(domEnergy, short) - EMA(domEnergy, long)
    stab_short = _ema(dom_energy, p.st_short)
    stab_long  = _ema(dom_energy, p.st_long)
    stability  = stab_short - stab_long
    stability  = np.where(np.isfinite(stability), stability, 0.0)

    is_strong = dom_share >= p.dom_strong
    is_weak   = dom_share >= p.dom_weak

    # State: 1=TREND, 2=WEAKENING, 3=TRANSITION
    state = np.where(
        is_strong & (stability > 0), 1,
        np.where(is_strong & (stability <= 0), 2,
                 np.where(~is_weak, 3, 2))
    )

    # --- A/D zones
    # retRecent = log(close / close[adPersist])
    ret_recent = np.concatenate((
        np.zeros(p.ad_persist),
        np.log(closes[p.ad_persist:] / closes[:-p.ad_persist])
    ))
    ret_recent = np.where(np.isfinite(ret_recent), ret_recent, 0.0)

    is_anom = np.abs(z) > p.z_thr_ad
    # anomOk: sum of isAnom over last adPersist bars == adPersist (all True)
    anom_sum = _rolling_sum(is_anom.astype(float), p.ad_persist)
    anom_ok  = anom_sum == p.ad_persist

    low_move = np.abs(ret_recent) <= p.max_abs_move

    # Volume z-score for optional confirmation
    vol_mu  = _rolling_mean(vols, p.vol_lookback)
    vol_sig = np.array([
        float(np.std(vols[max(0, i - p.vol_lookback + 1): i + 1], ddof=1))
        if i >= p.vol_lookback - 1 else np.nan
        for i in range(n)
    ])
    with np.errstate(invalid="ignore", divide="ignore"):
        vol_z = np.where(vol_sig != 0.0, (vols - vol_mu) / vol_sig, 0.0)
    vol_z = np.where(np.isfinite(vol_z), vol_z, 0.0)
    vol_ok = np.where(p.use_vol_confirm, vol_z > p.vol_z_thr, True)

    is_accum_base   = p.use_ad_zones & anom_ok & low_move & (ret_recent <= 0)
    is_distrib_base = p.use_ad_zones & anom_ok & low_move & (ret_recent >= 0)
    is_accum   = is_accum_base & vol_ok
    is_distrib = is_distrib_base & vol_ok

    # --- MFI
    typical = (highs + lows + closes) / 3.0
    raw_mf  = typical * vols
    pos_mf  = _rolling_sum(np.where(np.concatenate(([False], typical[1:] > typical[:-1])), raw_mf, 0.0), p.mfi_length)
    neg_mf  = _rolling_sum(np.where(np.concatenate(([False], typical[1:] < typical[:-1])), raw_mf, 0.0), p.mfi_length)
    with np.errstate(invalid="ignore", divide="ignore"):
        mf_ratio = np.where(neg_mf == 0, 999.0, pos_mf / neg_mf)
    mfi = 100.0 - (100.0 / (1.0 + mf_ratio))
    mfi = np.where(np.isfinite(mfi), mfi, 50.0)

    # --- Whale detection
    avg_vol = _rolling_mean(vols, p.whale_lookback)
    is_whale = vols > avg_vol * p.whale_multiplier
    bull_bar = closes > opens
    bear_bar = closes < opens

    whale_buy  = np.zeros(n, dtype=bool)
    whale_sell = np.zeros(n, dtype=bool)
    last_buy_bar  = -1000
    last_sell_bar = -1000

    for i in range(n):
        in_buy_cd  = (i - last_buy_bar)  <= p.cooldown_bars
        in_sell_cd = (i - last_sell_bar) <= p.cooldown_bars
        wb = bool(is_whale[i]) and bool(bull_bar[i]) and not in_sell_cd
        ws = bool(is_whale[i]) and bool(bear_bar[i]) and not in_buy_cd
        whale_buy[i]  = wb
        whale_sell[i] = ws
        if wb:
            last_buy_bar = i
        if ws:
            last_sell_bar = i

    # --- Zone tracking + entry detection (stateful, bar by bar)
    run_acc_high: Optional[float] = None
    run_acc_low:  Optional[float] = None
    run_dis_high: Optional[float] = None
    run_dis_low:  Optional[float] = None
    last_accum_high: Optional[float] = None
    last_accum_low:  Optional[float] = None
    last_distrib_high: Optional[float] = None
    last_distrib_low:  Optional[float] = None
    last_acc_bar = -10000
    last_dis_bar = -10000
    long_fired  = False
    short_fired = False
    last_dom = int(dom_id[0]) if n > 0 else 1

    long_confirmed_arr  = np.zeros(n, dtype=bool)
    short_confirmed_arr = np.zeros(n, dtype=bool)
    enter_accum_arr     = np.zeros(n, dtype=bool)
    enter_distrib_arr   = np.zeros(n, dtype=bool)
    dom_switch_arr      = np.zeros(n, dtype=bool)

    la_high_arr = np.full(n, np.nan)
    la_low_arr  = np.full(n, np.nan)
    ld_high_arr = np.full(n, np.nan)
    ld_low_arr  = np.full(n, np.nan)

    for i in range(n):
        h = highs[i]
        l = lows[i]
        c = closes[i]
        acc  = bool(is_accum[i])
        dis  = bool(is_distrib[i])
        acc_prev = bool(is_accum[i - 1]) if i > 0 else False
        dis_prev = bool(is_distrib[i - 1]) if i > 0 else False
        s1p = float(share1[i]) * 100.0
        s2p = float(share2[i]) * 100.0

        # Dominance switch
        cur_dom = int(dom_id[i])
        if cur_dom != last_dom:
            dom_switch_arr[i] = True
            last_dom = cur_dom

        # Context OK: in transition OR dominant weakening
        in_transition = (s1p >= p.transition_low and s1p <= p.transition_high and
                         s2p >= p.transition_low and s2p <= p.transition_high)
        ds = float(dom_share[i])
        stab = float(stability[i])
        context_ok = in_transition or (ds >= p.dom_strong and stab < 0)

        acc_start = acc and not acc_prev
        acc_end   = not acc and acc_prev
        dis_start = dis and not dis_prev
        dis_end   = not dis and dis_prev

        # --- Accumulation zone tracking
        if acc_start:
            run_acc_high = h
            run_acc_low  = l
            if context_ok:
                enter_accum_arr[i] = True

        if acc and not acc_start:
            if run_acc_high is not None:
                run_acc_high = max(run_acc_high, h)
                run_acc_low  = min(run_acc_low,  l) if run_acc_low is not None else l
            else:
                run_acc_high = h
                run_acc_low  = l

        if acc_end:
            last_accum_high = run_acc_high
            last_accum_low  = run_acc_low
            run_acc_high = None
            run_acc_low  = None
            last_acc_bar = i
            long_fired   = False

        # --- Distribution zone tracking
        if dis_start:
            run_dis_high = h
            run_dis_low  = l
            if context_ok:
                enter_distrib_arr[i] = True

        if dis and not dis_start:
            if run_dis_high is not None:
                run_dis_high = max(run_dis_high, h)
                run_dis_low  = min(run_dis_low,  l) if run_dis_low is not None else l
            else:
                run_dis_high = h
                run_dis_low  = l

        if dis_end:
            last_distrib_high = run_dis_high
            last_distrib_low  = run_dis_low
            run_dis_high = None
            run_dis_low  = None
            last_dis_bar = i
            short_fired  = False

        # Freshness
        acc_fresh = last_accum_high is not None and (i - last_acc_bar) <= p.max_zone_age
        dis_fresh = last_distrib_low is not None and (i - last_dis_bar) <= p.max_zone_age

        # Breakout conditions
        long_break  = acc_fresh and last_accum_high is not None and c > last_accum_high * (1 + p.buffer_pct)
        short_break = dis_fresh and last_distrib_low is not None and c < last_distrib_low * (1 - p.buffer_pct)

        # Confirmed entries (one-shot per zone)
        if context_ok and acc_fresh and long_break and not dis and not long_fired:
            long_confirmed_arr[i] = True
            long_fired = True

        if context_ok and dis_fresh and short_break and not acc and not short_fired:
            short_confirmed_arr[i] = True
            short_fired = True

        la_high_arr[i] = last_accum_high if last_accum_high is not None else np.nan
        la_low_arr[i]  = last_accum_low  if last_accum_low  is not None else np.nan
        ld_high_arr[i] = last_distrib_high if last_distrib_high is not None else np.nan
        ld_low_arr[i]  = last_distrib_low  if last_distrib_low  is not None else np.nan

    # --- Assemble results
    results = []
    for i in range(n):
        br = BarResult(
            time=bars[i].time,
            close=float(closes[i]),
            volume=float(vols[i]),
            share1=float(share1[i]),
            share2=float(share2[i]),
            dom_id=int(dom_id[i]),
            dom_share=float(dom_share[i]),
            state=int(state[i]),
            stability=float(stability[i]),
            recon=float(recon[i]),
            resid=float(resid[i]),
            z=float(z[i]),
            ret=float(ret[i]),
            is_accum=bool(is_accum[i]),
            is_distrib=bool(is_distrib[i]),
            mfi=float(mfi[i]),
            whale_buy=bool(whale_buy[i]),
            whale_sell=bool(whale_sell[i]),
            last_accum_high=float(la_high_arr[i]) if np.isfinite(la_high_arr[i]) else None,
            last_accum_low=float(la_low_arr[i]) if np.isfinite(la_low_arr[i]) else None,
            last_distrib_high=float(ld_high_arr[i]) if np.isfinite(ld_high_arr[i]) else None,
            last_distrib_low=float(ld_low_arr[i]) if np.isfinite(ld_low_arr[i]) else None,
            long_confirmed=bool(long_confirmed_arr[i]),
            short_confirmed=bool(short_confirmed_arr[i]),
            enter_accum_zone=bool(enter_accum_arr[i]),
            enter_distrib_zone=bool(enter_distrib_arr[i]),
            dom_switch=bool(dom_switch_arr[i]),
        )
        results.append(br)

    return results


def to_json_list(results: List[BarResult]) -> List[dict]:
    """Serializes BarResult list to JSON-safe dicts for the API."""
    out = []
    for r in results:
        out.append({
            "time": r.time,
            "close": r.close,
            "volume": r.volume,
            "share1": round(r.share1 * 100, 2),
            "share2": round(r.share2 * 100, 2),
            "dom_id": r.dom_id,
            "dom_share": round(r.dom_share * 100, 2),
            "state": r.state,
            "stability": round(r.stability, 6),
            "z": round(r.z, 4),
            "ret": round(r.ret, 6),
            "is_accum": r.is_accum,
            "is_distrib": r.is_distrib,
            "mfi": round(r.mfi, 2),
            "whale_buy": r.whale_buy,
            "whale_sell": r.whale_sell,
            "last_accum_high": r.last_accum_high,
            "last_accum_low": r.last_accum_low,
            "last_distrib_high": r.last_distrib_high,
            "last_distrib_low": r.last_distrib_low,
            "long_confirmed": r.long_confirmed,
            "short_confirmed": r.short_confirmed,
            "enter_accum_zone": r.enter_accum_zone,
            "enter_distrib_zone": r.enter_distrib_zone,
            "dom_switch": r.dom_switch,
        })
    return out
