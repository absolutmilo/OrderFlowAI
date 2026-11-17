"""Microstructure pattern detection and empirical probabilistic scoring.

Combines HMM-4 regime classifications with real-time order flow microstructure
(CVD divergence, volume acceleration, price delta, and volatility regime)
to evaluate probabilistic Long/Short setups and detect adverse liquidity walls.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np


@dataclass(frozen=True)
class MarketRegimeState:
    dominant_regime: str
    algo_prob: float = 0.0
    iceberg_prob: float = 0.0
    absorption_prob: float = 0.0
    retail_prob: float = 0.0
    iceberg_side: Optional[str] = None
    absorption_side: Optional[str] = None
    persistence_bars: int = 1


@dataclass(frozen=True)
class MicrostructureFeatures:
    price: float
    cvd_usd: float
    cvd_ratio: float
    price_delta_bps: float
    trades_per_second: float
    volume_per_second_usd: float
    inter_arrival_cv: float
    volatility_atr_bps: float = 1.0


@dataclass(frozen=True)
class PatternSignal:
    """Behavior candidate; statistical fields stay empty until calibrated from data."""

    pattern_name: str
    direction: int  # +1 for LONG, -1 for SHORT
    probability_win: Optional[float]
    expectancy_bps: Optional[float]
    confidence: Optional[float]
    recommended_horizon_seconds: Optional[int]
    volatility_regime: str  # "LOW", "NORMAL", "HIGH"
    notes: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class AdverseWallAlert:
    is_adverse: bool
    adverse_regime: Optional[str] = None
    persistence_bars: int = 0
    opposing_direction: int = 0
    reason: str = ""


class MicrostructurePatternEngine:
    """Emits qualitative pattern candidates and checks persistent opposing states."""

    def __init__(
        self,
        min_probability_threshold: float = 55.0,
        adverse_persistence_required: int = 3,
        low_volatility_cutoff_bps: float = 1.0,
        high_volatility_cutoff_bps: float = 3.0,
    ):
        self.min_probability_threshold = min_probability_threshold
        self.adverse_persistence_required = adverse_persistence_required
        self.low_volatility_cutoff_bps = low_volatility_cutoff_bps
        self.high_volatility_cutoff_bps = high_volatility_cutoff_bps

    def classify_volatility(self, volatility_atr_bps: float) -> str:
        if volatility_atr_bps < self.low_volatility_cutoff_bps:
            return "LOW"
        if volatility_atr_bps > self.high_volatility_cutoff_bps:
            return "HIGH"
        return "NORMAL"

    def evaluate_pattern(
        self,
        regime: MarketRegimeState,
        features: MicrostructureFeatures,
        recent_regimes: Sequence[str] = (),
        recent_momentums_bps: Sequence[float] = (),
    ) -> Optional[PatternSignal]:
        vol_regime = self.classify_volatility(features.volatility_atr_bps)
        dom = regime.dominant_regime
        cvd = features.cvd_ratio
        delta = features.price_delta_bps

        # Calculate momentum direction if available
        mom = recent_momentums_bps[-1] if recent_momentums_bps else delta

        # Pattern 1: Passive-liquidity hypothesis + CVD impulse
        if dom == "ICEBERG_COMPRADOR" and cvd > 0.15:
            return PatternSignal(
                pattern_name="PASSIVE_BUYING_CANDIDATE",
                direction=1,
                probability_win=None,
                expectancy_bps=None,
                confidence=None,
                recommended_horizon_seconds=None,
                volatility_regime=vol_regime,
                notes=[
                    "Estado compatible con liquidez pasiva compradora; no confirma reposición ni profundidad.",
                    f"CVD ratio: {cvd * 100:+.1f}% | Volatilidad: {vol_regime}",
                ],
            )

        if dom == "ICEBERG_VENDEDOR" and cvd < -0.15:
            return PatternSignal(
                pattern_name="PASSIVE_SELLING_CANDIDATE",
                direction=-1,
                probability_win=None,
                expectancy_bps=None,
                confidence=None,
                recommended_horizon_seconds=None,
                volatility_regime=vol_regime,
                notes=[
                    "Estado compatible con liquidez pasiva vendedora; no confirma reposición ni profundidad.",
                    f"CVD ratio: {cvd * 100:+.1f}% | Volatilidad: {vol_regime}",
                ],
            )

        # Pattern 2: Low price response followed by a momentum inflection
        is_recent_buy_absorb = (
            len(recent_regimes) >= 2
            and all(r in ("ABSORPTION_BUY", "ICEBERG_COMPRADOR") for r in recent_regimes[-2:])
        )
        if (is_recent_buy_absorb or dom == "ABSORPTION_BUY") and mom > 0.3 and cvd > -0.05:
            return PatternSignal(
                pattern_name="LOW_PRICE_RESPONSE_REVERSAL_LONG",
                direction=1,
                probability_win=None,
                expectancy_bps=None,
                confidence=None,
                recommended_horizon_seconds=None,
                volatility_regime=vol_regime,
                notes=[
                    "Flujo previo compatible con baja respuesta al precio; momentum comprador girando.",
                    f"Estado previo inferido: {dom} | Desbalance CVD: {cvd * 100:+.1f}%",
                ],
            )

        is_recent_sell_absorb = (
            len(recent_regimes) >= 2
            and all(r in ("ABSORPTION_SELL", "ICEBERG_VENDEDOR") for r in recent_regimes[-2:])
        )
        if (is_recent_sell_absorb or dom == "ABSORPTION_SELL") and mom < -0.3 and cvd < 0.05:
            return PatternSignal(
                pattern_name="LOW_PRICE_RESPONSE_REVERSAL_SHORT",
                direction=-1,
                probability_win=None,
                expectancy_bps=None,
                confidence=None,
                recommended_horizon_seconds=None,
                volatility_regime=vol_regime,
                notes=[
                    "Flujo previo compatible con baja respuesta al precio; momentum vendedor girando.",
                    f"Estado previo inferido: {dom} | Desbalance CVD: {cvd * 100:+.1f}%",
                ],
            )

        # Pattern 3: Regular-flow momentum hypothesis
        if regime.algo_prob > 40.0:
            if cvd > 0.25 and mom > 0.5:
                return PatternSignal(
                    pattern_name="REGULAR_FLOW_MOMENTUM_LONG",
                    direction=1,
                    probability_win=None,
                    expectancy_bps=None,
                    confidence=None,
                    recommended_horizon_seconds=None,
                    volatility_regime=vol_regime,
                    notes=[
                        "Cadencia regular inferida por reglas con flujo comprador sostenido; no identifica institución.",
                        f"Score de regularidad: {regime.algo_prob:.1f} | CVD: {cvd * 100:+.1f}%",
                    ],
                )
            if cvd < -0.25 and mom < -0.5:
                return PatternSignal(
                    pattern_name="REGULAR_FLOW_MOMENTUM_SHORT",
                    direction=-1,
                    probability_win=None,
                    expectancy_bps=None,
                    confidence=None,
                    recommended_horizon_seconds=None,
                    volatility_regime=vol_regime,
                    notes=[
                        "Cadencia regular inferida por reglas con flujo vendedor sostenido; no identifica institución.",
                        f"Score de regularidad: {regime.algo_prob:.1f} | CVD: {cvd * 100:+.1f}%",
                    ],
                )

        return None

    def check_adverse_wall(
        self,
        current_position_direction: int,
        recent_regimes: Sequence[str],
    ) -> AdverseWallAlert:
        """Checks whether an opposing inferred state persists against the position."""
        if not current_position_direction or len(recent_regimes) < self.adverse_persistence_required:
            return AdverseWallAlert(is_adverse=False)

        tail = recent_regimes[-self.adverse_persistence_required:]

        # For LONG position: adverse if sellers absorb or stack icebergs
        if current_position_direction == 1:
            adverse_sell_wall = all(
                r in ("ABSORPTION_SELL", "ICEBERG_VENDEDOR")
                for r in tail
            )
            if adverse_sell_wall:
                return AdverseWallAlert(
                    is_adverse=True,
                    adverse_regime=tail[-1],
                    persistence_bars=len(tail),
                    opposing_direction=-1,
                    reason=(
                        f"Estado opuesto inferido de forma persistente ({tail[-1]}) "
                        f"persistente por {len(tail)} snapshots. Sugiere salida protectora."
                    ),
                )

        # For SHORT position: adverse if buyers absorb or stack icebergs
        if current_position_direction == -1:
            adverse_buy_wall = all(
                r in ("ABSORPTION_BUY", "ICEBERG_COMPRADOR")
                for r in tail
            )
            if adverse_buy_wall:
                return AdverseWallAlert(
                    is_adverse=True,
                    adverse_regime=tail[-1],
                    persistence_bars=len(tail),
                    opposing_direction=1,
                    reason=(
                        f"Estado opuesto inferido de forma persistente ({tail[-1]}) "
                        f"persistente por {len(tail)} snapshots. Sugiere salida protectora."
                    ),
                )

        return AdverseWallAlert(is_adverse=False)
