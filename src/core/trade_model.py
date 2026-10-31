"""
Data models for trades, aggregated tape prints, and order flow classifications.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional
import time


class TradeSide(str, Enum):
    BUY = "BUY"    # Buyer was aggressive taker (lifted the offer / ask)
    SELL = "SELL"  # Seller was aggressive taker (hit the bid)


class MarketRegime(str, Enum):
    ALGO_INSTITUTIONAL = "ALGO_INSTITUTIONAL"
    ICEBERG_BUY = "ICEBERG_BUY"
    ICEBERG_SELL = "ICEBERG_SELL"
    ABSORPTION_BUY = "ABSORPTION_BUY"     # Passive limit bids absorbing aggressive sellers
    ABSORPTION_SELL = "ABSORPTION_SELL"   # Passive limit asks absorbing aggressive buyers
    HFT_BURST = "HFT_BURST"
    RETAIL_DOMINANT = "RETAIL_DOMINANT"
    BREAKOUT = "BREAKOUT"


@dataclass
class RawTrade:
    """Represents a single raw execution from an exchange WebSocket."""
    exchange: str
    symbol: str
    trade_id: str
    price: float
    size: float
    usd_value: float
    side: TradeSide
    timestamp_ms: int
    is_liquidation: bool = False

    def to_dict(self) -> dict:
        return {
            "exchange": self.exchange,
            "symbol": self.symbol,
            "trade_id": self.trade_id,
            "price": self.price,
            "size": self.size,
            "usd_value": self.usd_value,
            "side": self.side.value,
            "timestamp_ms": self.timestamp_ms,
            "is_liquidation": self.is_liquidation,
        }


@dataclass
class AggregatedTrade:
    """
    Groups micro-trades occurring inside the aggregation window at the same level.
    Eliminates tape noise (like SignificantTrades / aggr.trade).
    """
    exchange: str
    symbol: str
    first_timestamp_ms: int
    last_timestamp_ms: int
    price: float
    total_size: float
    total_usd_value: float
    side: TradeSide
    trade_count: int
    is_liquidation: bool = False
    detected_patterns: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "exchange": self.exchange,
            "symbol": self.symbol,
            "timestamp_ms": self.last_timestamp_ms,
            "price": round(self.price, 2),
            "size": round(self.total_size, 4),
            "usd_value": round(self.total_usd_value, 2),
            "side": self.side.value,
            "trade_count": self.trade_count,
            "is_liquidation": self.is_liquidation,
            "patterns": self.detected_patterns,
        }


@dataclass
class ClassificationProbabilities:
    """Probabilities computed from microstructural feature analysis."""
    algo_institutional: float  # 0 to 100%
    iceberg: float             # 0 to 100%
    absorption: float          # 0 to 100%
    retail: float              # 0 to 100%
    hft_competition: float     # 0 to 100%
    breakout: float            # 0 to 100%
    dominant_regime: str
    iceberg_side: Optional[str] = None
    absorption_side: Optional[str] = None
    notes: List[str] = field(default_factory=list)
    timestamp_ms: int = field(default_factory=lambda: int(time.time() * 1000))

    def to_dict(self) -> dict:
        return {
            "algo_institutional": round(self.algo_institutional, 1),
            "iceberg": round(self.iceberg, 1),
            "absorption": round(self.absorption, 1),
            "retail": round(self.retail, 1),
            "hft_competition": round(self.hft_competition, 1),
            "breakout": round(self.breakout, 1),
            "dominant_regime": self.dominant_regime,
            "iceberg_side": self.iceberg_side,
            "absorption_side": self.absorption_side,
            "notes": self.notes,
            "timestamp_ms": self.timestamp_ms,
        }


@dataclass
class OrderFlowMetrics:
    """Current rolling metrics calculated from the tape."""
    current_price: float
    cvd_usd: float              # Cumulative Volume Delta (USD) in current window
    cvd_btc: float              # Cumulative Volume Delta (BTC)
    buy_volume_usd: float
    sell_volume_usd: float
    trades_per_second: float
    volume_per_second_usd: float
    price_delta_bps: float
    inter_arrival_cv: float     # Coefficient of variation of trade intervals

    def to_dict(self) -> dict:
        return {
            "current_price": round(self.current_price, 2),
            "cvd_usd": round(self.cvd_usd, 2),
            "cvd_btc": round(self.cvd_btc, 4),
            "buy_volume_usd": round(self.buy_volume_usd, 2),
            "sell_volume_usd": round(self.sell_volume_usd, 2),
            "trades_per_second": round(self.trades_per_second, 1),
            "volume_per_second_usd": round(self.volume_per_second_usd, 2),
            "price_delta_bps": round(self.price_delta_bps, 2),
            "inter_arrival_cv": round(self.inter_arrival_cv, 3),
        }
