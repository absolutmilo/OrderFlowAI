"""
Configuration module for OrderFlowAI.
Defines endpoints, trading pairs, microstructural classification thresholds,
and audio sonification parameters.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class ExchangeConfig:
    name: str = "Binance-Spot + Bybit-Linear"
    symbol: str = "BTCUSDT"
    # Connectors to activate simultaneously (multi-exchange aggregated tape like SignificantTrades / aggr.trade)
    enable_binance_spot: bool = True
    enable_bybit_linear: bool = True
    enable_binance_futures: bool = False  # Disabled by default due to regional IP restrictions on fstream
    reconnect_delay_seconds: float = 2.0
    ping_interval_seconds: float = 20.0


@dataclass
class AggregatorConfig:
    # Buffer time in milliseconds to group micro-executions into a single tape print (like SignificantTrades)
    aggregation_window_ms: int = 50
    # Minimum USD trade value to display on visual tape
    min_tape_display_usd: float = 2500.0
    # Large trade highlight threshold ("Whale" order)
    whale_threshold_usd: float = 100000.0


@dataclass
class ClassifierConfig:
    # Size of the rolling window of trades used for feature extraction
    rolling_window_trades: int = 200
    # Minimum trades before emitting high-confidence classifications
    warmup_trades: int = 25

    # Inference mode: "model" (Gaussian HMM) or "heuristic" (Rule-based sigmoid)
    default_inference_mode: str = "model"
    hmm_model_path: str = "data/hmm_regime_model.json"

    # Algorithmic execution (TWAP/VWAP/Slicing) thresholds:
    algo_inter_arrival_cv_threshold: float = 0.45
    algo_size_cv_threshold: float = 0.50

    # Iceberg concentration threshold
    iceberg_min_volume_ratio: float = 0.22

    # Absorption thresholds
    absorption_min_usd_vol: float = 200000.0
    absorption_min_cvd_ratio: float = 0.35
    absorption_max_price_delta_bps: float = 1.5

    # Iceberg level requirements
    iceberg_min_executions_at_level: int = 6
    iceberg_min_usd_at_level: float = 150000.0

    # HFT competition
    hft_burst_tps_threshold: float = 60.0

    # Breakout detection thresholds
    breakout_min_price_delta_bps: float = 6.0
    breakout_min_cvd_ratio: float = 0.40


@dataclass
class DataCaptureConfig:
    database_path: str = "data/orderflow.sqlite3"
    sample_interval_seconds: int = 1
    label_horizons_seconds: Tuple[int, ...] = (5, 30, 60, 300, 600, 900)
    label_min_move_bps: float = 1.0


@dataclass
class ScalpingConfig:
    symbol: str = "BTCUSDT"
    exchange: str = "Bybit-Linear"
    taker_fee_bps_per_side: float = 6.0
    slippage_bps_per_side: float = 2.0
    min_training_observations: int = 36000
    min_validation_trades: int = 10
    signal_persistence_seconds: int = 3
    max_snapshot_gap_seconds: float = 5.0
    minimum_snapshot_coverage: float = 0.80


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    debug: bool = False
    cors_origins: List[str] = field(default_factory=lambda: ["*"])


@dataclass
class AppConfig:
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    aggregator: AggregatorConfig = field(default_factory=AggregatorConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    data_capture: DataCaptureConfig = field(default_factory=DataCaptureConfig)
    scalping: ScalpingConfig = field(default_factory=ScalpingConfig)
    server: ServerConfig = field(default_factory=ServerConfig)


# Global default configuration instance
CONFIG = AppConfig()
