"""
Order Flow Microstructure Classifier.
Analyzes sequences of trade events over a rolling window to detect institutional execution,
iceberg orders, passive absorption, retail dominance, and HFT competition.
Supports both heuristic rule-based classification and trained Gaussian HMM online inference.
"""

import logging
from collections import deque
from pathlib import Path
from typing import Deque, List, Optional, Tuple
import numpy as np

from src.core.trade_model import (
    AggregatedTrade,
    TradeSide,
    ClassificationProbabilities,
    OrderFlowMetrics,
    MarketRegime,
)
from src.ml.hmm_model import OnlineGaussianHMM
from config import CONFIG

logger = logging.getLogger("OrderFlowAI.Classifier")


class OutOfOrderTradeError(ValueError):
    """Raised when an event would violate a classifier's event-time ordering."""


class MicrostructureClassifier:
    def __init__(
        self,
        warmup_trades: int = CONFIG.classifier.warmup_trades,
        inference_mode: str = getattr(CONFIG.classifier, "default_inference_mode", "model"),
        stream_identity: Optional[Tuple[str, str]] = None,
    ):
        self.warmup_trades = warmup_trades
        self.feature_windows_seconds = tuple(CONFIG.classifier.feature_windows_seconds)
        if (
            len(self.feature_windows_seconds) < 3
            or any(window <= 0 for window in self.feature_windows_seconds)
            or tuple(sorted(set(self.feature_windows_seconds))) != self.feature_windows_seconds
        ):
            raise ValueError(
                "Se requieren al menos tres ventanas positivas, únicas y ascendentes."
            )
        if CONFIG.classifier.primary_feature_window_seconds not in self.feature_windows_seconds:
            raise ValueError("La ventana primaria debe estar incluida en feature_windows_seconds.")
        self.primary_feature_window_seconds = CONFIG.classifier.primary_feature_window_seconds
        self.activity_window_seconds = self.feature_windows_seconds[0]
        self.absorption_window_seconds = self.feature_windows_seconds[1]
        self.rolling_window_seconds = self.feature_windows_seconds[-1]
        self.stream_identity = stream_identity
        self.trades: Deque[AggregatedTrade] = deque()
        self._stream_started_at_ms: Optional[int] = None
        self._last_timestamp_ms: Optional[int] = None
        if inference_mode not in {"heuristic", "model"}:
            raise ValueError(f"Modo de inferencia no válido: {inference_mode}")
        self.inference_mode = inference_mode

        # Initialize HMM online inference engine if model artifact exists
        self.hmm_model: Optional[OnlineGaussianHMM] = None
        self._load_hmm_model()
        if self.inference_mode == "model" and self.hmm_model is None:
            logger.warning("El modo modelo no está disponible; iniciando en modo heurístico.")
            self.inference_mode = "heuristic"

    def _load_hmm_model(self) -> None:
        """Loads trained Gaussian HMM parameters from disk."""
        model_rel_path = getattr(CONFIG.classifier, "hmm_model_path", "data/hmm_regime_model.json")
        model_path = Path(__file__).resolve().parents[2] / model_rel_path
        if model_path.exists():
            try:
                model = OnlineGaussianHMM.load(model_path)
                if model.feature_schema_version != CONFIG.classifier.feature_schema_version:
                    logger.warning(
                        "El artefacto HMM usa un esquema de features incompatible (%s); "
                        "se requiere reentrenarlo para %s.",
                        model.feature_schema_version,
                        CONFIG.classifier.feature_schema_version,
                    )
                    self.hmm_model = None
                    return
                self.hmm_model = model
                logger.info(f"Modelo Gaussian HMM cargado exitosamente desde {model_path}.")
            except Exception as e:
                logger.warning(f"No se pudo cargar el modelo HMM ({e}); usando fallback heurístico.")
                self.hmm_model = None
        else:
            logger.info(f"Artefacto HMM no encontrado en {model_path}; modo heurístico activo.")
            self.hmm_model = None

    def set_inference_mode(self, mode: str) -> str:
        """Toggles between 'heuristic' and 'model' inference modes."""
        valid_modes = {"heuristic", "model"}
        if mode not in valid_modes:
            raise ValueError(f"Modo de inferencia no válido: {mode}")

        if mode == "model" and self.hmm_model is None:
            self._load_hmm_model()
            if self.hmm_model is None:
                raise RuntimeError("No hay un modelo HMM entrenado disponible para cargar.")

        if mode != self.inference_mode:
            self.inference_mode = mode
            if mode == "model" and self.hmm_model is not None:
                self.hmm_model.reset()
        logger.info(f"Modo de inferencia cambiado a: {self.inference_mode}")
        return self.inference_mode

    def toggle_inference_mode(self) -> str:
        """Toggles between heuristic and model modes."""
        new_mode = "model" if self.inference_mode == "heuristic" else "heuristic"
        return self.set_inference_mode(new_mode)

    def add_trade(self, trade: AggregatedTrade):
        """Adds an event to this stream's fixed event-time history."""
        identity = (trade.exchange, trade.symbol)
        if self.stream_identity is None:
            self.stream_identity = identity
        elif identity != self.stream_identity:
            raise ValueError(
                f"El clasificador de {self.stream_identity} no acepta eventos de {identity}."
            )
        timestamp_ms = trade.last_timestamp_ms
        if self._last_timestamp_ms is not None and timestamp_ms < self._last_timestamp_ms:
            raise OutOfOrderTradeError(
                "Los eventos del clasificador deben llegar en orden temporal."
            )
        if (
            self._last_timestamp_ms is not None
            and timestamp_ms - self._last_timestamp_ms
            > CONFIG.classifier.max_trade_gap_seconds * 1000
        ):
            self.trades.clear()
            self._stream_started_at_ms = timestamp_ms
            if self.hmm_model is not None:
                self.hmm_model.reset()
        elif self._stream_started_at_ms is None:
            self._stream_started_at_ms = timestamp_ms

        self.trades.append(trade)
        self._last_timestamp_ms = timestamp_ms
        cutoff_ms = timestamp_ms - self.rolling_window_seconds * 1000
        while self.trades and self.trades[0].last_timestamp_ms <= cutoff_ms:
            self.trades.popleft()

    def is_warmed_up(self) -> bool:
        return (
            len(self.trades) >= self.warmup_trades
            and self._stream_started_at_ms is not None
            and self._last_timestamp_ms is not None
            and self._last_timestamp_ms - self._stream_started_at_ms
            >= self.rolling_window_seconds * 1000
        )

    def get_window_trades(self, window_seconds: int) -> List[AggregatedTrade]:
        if window_seconds not in self.feature_windows_seconds:
            raise ValueError(f"Ventana de features no configurada: {window_seconds}s.")
        if self._last_timestamp_ms is None:
            return []
        cutoff_ms = self._last_timestamp_ms - window_seconds * 1000
        return [
            trade for trade in self.trades
            if trade.last_timestamp_ms > cutoff_ms
        ]

    def get_current_metrics(
        self,
        window_seconds: Optional[int] = None,
    ) -> Optional[OrderFlowMetrics]:
        if window_seconds is None:
            window_seconds = self.primary_feature_window_seconds
        trades = self.get_window_trades(window_seconds)
        if not trades:
            return None

        trades_list = trades
        prices = np.array([t.price for t in trades_list], dtype=np.float64)
        sizes = np.array([t.total_size for t in trades_list], dtype=np.float64)
        usd_values = np.array([t.total_usd_value for t in trades_list], dtype=np.float64)
        sides = [t.side for t in trades_list]
        times = np.array([t.last_timestamp_ms for t in trades_list], dtype=np.int64)

        is_buy = np.array([s == TradeSide.BUY for s in sides])
        buy_usd = float(np.sum(usd_values[is_buy]))
        sell_usd = float(np.sum(usd_values[~is_buy]))
        cvd_usd = buy_usd - sell_usd

        buy_btc = float(np.sum(sizes[is_buy]))
        sell_btc = float(np.sum(sizes[~is_buy]))
        cvd_btc = buy_btc - sell_btc

        tps = len(trades_list) / float(window_seconds)
        vol_per_sec = float(np.sum(usd_values)) / float(window_seconds)

        # Price displacement in basis points (bps)
        current_price = float(prices[-1])
        base_price = float(prices[0])
        price_delta_bps = ((current_price - base_price) / max(1.0, base_price)) * 10000.0

        # Inter-arrival time CV
        delta_times = np.diff(times)
        if len(delta_times) > 0 and np.mean(delta_times) > 0:
            inter_arrival_cv = float(np.std(delta_times) / np.mean(delta_times))
        else:
            inter_arrival_cv = 1.0

        return OrderFlowMetrics(
            current_price=current_price,
            cvd_usd=cvd_usd,
            cvd_btc=cvd_btc,
            buy_volume_usd=buy_usd,
            sell_volume_usd=sell_usd,
            trades_per_second=tps,
            volume_per_second_usd=vol_per_sec,
            price_delta_bps=price_delta_bps,
            inter_arrival_cv=inter_arrival_cv,
            window_seconds=window_seconds,
            event_count=len(trades_list),
        )

    def classify(self, mode: Optional[str] = None) -> ClassificationProbabilities:
        """
        Dispatches classification based on active inference mode (HMM model vs Heuristic).
        """
        if not self.is_warmed_up():
            return self._classify_warming_up()

        active_mode = mode or self.inference_mode
        if active_mode == "model" and self.hmm_model is not None:
            return self._classify_hmm()
        return self._classify_heuristic()

    @staticmethod
    def _classify_warming_up() -> ClassificationProbabilities:
        return ClassificationProbabilities(
            algo_institutional=25.0,
            iceberg=15.0,
            absorption=15.0,
            retail=45.0,
            hft_competition=10.0,
            breakout=10.0,
            dominant_regime=MarketRegime.RETAIL_DOMINANT.value,
            notes=["Calentando ventanas temporales..."],
        )

    def _classify_hmm(self) -> ClassificationProbabilities:
        """
        Calculates probabilities using the learned 4-state Gaussian Hidden Markov Model.
        Replaces static heuristic sigmoid weights with learned transition and emission matrices.
        """
        metrics = self.get_current_metrics(self.primary_feature_window_seconds)
        behavior_metrics = self.get_current_metrics(self.rolling_window_seconds)
        activity_metrics = self.get_current_metrics(self.activity_window_seconds)
        if (
            metrics is None
            or behavior_metrics is None
            or activity_metrics is None
            or self.hmm_model is None
        ):
            return self._classify_heuristic()

        inter_cv = min(5.0, max(0.1, metrics.inter_arrival_cv))
        vol_sec = max(1.0, metrics.volume_per_second_usd)
        log_vol = np.log10(vol_sec)
        tot_vol = max(1.0, metrics.buy_volume_usd + metrics.sell_volume_usd)
        cvd_ratio = min(1.0, abs(metrics.cvd_usd) / tot_vol)
        price_delta = min(30.0, abs(metrics.price_delta_bps))
        tps = max(0.1, min(200.0, metrics.trades_per_second))
        log_tps = np.log10(tps)

        obs = np.array([inter_cv, log_vol, cvd_ratio, price_delta, log_tps], dtype=np.float64)
        alpha = self.hmm_model.step(obs)

        prob_algo = float(alpha[0]) * 100.0
        prob_iceberg = float(alpha[1]) * 100.0
        prob_absorption = float(alpha[2]) * 100.0
        prob_retail = float(alpha[3]) * 100.0

        # Side tags are calculated from the longest configured window.
        trades_list = self.get_window_trades(self.rolling_window_seconds)
        prices = np.array([t.price for t in trades_list], dtype=np.float64)
        usd_values = np.array([t.total_usd_value for t in trades_list], dtype=np.float64)
        is_buy = np.array([t.side == TradeSide.BUY for t in trades_list])

        # Iceberg side calculation
        rounded_prices = np.round(prices, 1)
        level_volumes = {}
        for p, u in zip(rounded_prices, usd_values):
            level_volumes[p] = level_volumes.get(p, 0.0) + u
        max_vol_level = max(level_volumes, key=level_volumes.get) if level_volumes else rounded_prices[-1]
        level_buys = np.sum(usd_values[(rounded_prices == max_vol_level) & is_buy])
        level_sells = np.sum(usd_values[(rounded_prices == max_vol_level) & (~is_buy)])
        iceberg_side = "COMPRADOR" if level_sells > level_buys else "VENDEDOR"

        # Absorption side calculation
        if metrics.cvd_usd > 0:
            absorption_side = "VENTAS (Limit Asks absorben compras agresivas)"
        else:
            absorption_side = "COMPRAS (Limit Bids absorben ventas agresivas)"

        # Activity and directional displacement use their own fixed windows.
        behavior_price_delta = abs(behavior_metrics.price_delta_bps)
        behavior_total_volume = max(
            1.0,
            behavior_metrics.buy_volume_usd + behavior_metrics.sell_volume_usd,
        )
        behavior_cvd_ratio = abs(behavior_metrics.cvd_usd) / behavior_total_volume
        direction_aligned = (
            behavior_metrics.price_delta_bps * behavior_metrics.cvd_usd > 0
        )
        prob_breakout = min(95.0, (behavior_price_delta / 15.0) * 50.0 + (behavior_cvd_ratio * 45.0)) if (
            behavior_price_delta > CONFIG.classifier.breakout_min_price_delta_bps
            and behavior_cvd_ratio > CONFIG.classifier.breakout_min_cvd_ratio
            and direction_aligned
        ) else min(20.0 if not direction_aligned else 35.0, behavior_price_delta * 3.0)

        activity_tps = activity_metrics.trades_per_second
        prob_hft = min(95.0, (activity_tps / 90.0) * 50.0) if activity_tps > CONFIG.classifier.hft_burst_tps_threshold else min(30.0, (activity_tps / 60.0) * 30.0)

        # Dominant state index from HMM belief
        dominant_idx = int(np.argmax(alpha))
        notes = []

        if dominant_idx == 0:
            dominant_regime = MarketRegime.ALGO_INSTITUTIONAL.value
            notes.append(f"HMM Estado: ALGORÍTMICO ({prob_algo:.1f}%) | Cadencia uniforme (CV={inter_cv:.2f}).")
        elif dominant_idx == 1:
            dominant_regime = f"ICEBERG_{iceberg_side}"
            notes.append(f"HMM Estado: ICEBERG ({prob_iceberg:.1f}%) | Concentración en ${max_vol_level:,.1f} ({iceberg_side}).")
        elif dominant_idx == 2:
            dominant_regime = MarketRegime.ABSORPTION_BUY.value if metrics.cvd_usd < 0 else MarketRegime.ABSORPTION_SELL.value
            notes.append(f"HMM Estado: ABSORCIÓN ({prob_absorption:.1f}%) | Desbalance direccional ({cvd_ratio*100:.0f}%) sin desplazamiento.")
        else:
            dominant_regime = MarketRegime.RETAIL_DOMINANT.value
            notes.append(f"HMM Estado: RETAIL ({prob_retail:.1f}%) | Flujo no estructurado y volumen disperso.")

        return ClassificationProbabilities(
            algo_institutional=round(prob_algo, 1),
            iceberg=round(prob_iceberg, 1),
            absorption=round(prob_absorption, 1),
            retail=round(prob_retail, 1),
            hft_competition=round(prob_hft, 1),
            breakout=round(prob_breakout, 1),
            dominant_regime=dominant_regime,
            iceberg_side=iceberg_side,
            absorption_side=absorption_side,
            notes=notes
        )

    def _classify_heuristic(self) -> ClassificationProbabilities:
        """
        Original rule-based heuristic classification using static sigmoid thresholds.
        """
        behavior_trades = self.get_window_trades(self.rolling_window_seconds)
        absorption_metrics = self.get_current_metrics(self.absorption_window_seconds)
        behavior_metrics = self.get_current_metrics(self.rolling_window_seconds)
        activity_metrics = self.get_current_metrics(self.activity_window_seconds)
        if not behavior_trades or absorption_metrics is None or behavior_metrics is None or activity_metrics is None:
            return self._classify_warming_up()

        prices = np.array([t.price for t in behavior_trades], dtype=np.float64)
        usd_values = np.array([t.total_usd_value for t in behavior_trades], dtype=np.float64)
        times = np.array([t.last_timestamp_ms for t in behavior_trades], dtype=np.int64)
        is_buy = np.array([t.side == TradeSide.BUY for t in behavior_trades])

        total_usd = float(np.sum(usd_values))
        buy_usd = float(np.sum(usd_values[is_buy]))
        sell_usd = float(np.sum(usd_values[~is_buy]))
        cvd_usd = buy_usd - sell_usd
        price_delta_bps = abs(behavior_metrics.price_delta_bps)

        notes: List[str] = []

        # 1. Patrón: Algorítmico Heurístico
        delta_times = np.diff(times)
        mean_dt = np.mean(delta_times) if len(delta_times) > 0 else 1.0
        std_dt = np.std(delta_times) if len(delta_times) > 0 else 1.0
        cv_dt = std_dt / max(1.0, mean_dt)

        mean_sz = np.mean(usd_values)
        std_sz = np.std(usd_values)
        cv_sz = std_sz / max(1.0, mean_sz)

        time_regularity_score = 1.0 / (
            1.0 + np.exp(3.5 * (cv_dt - CONFIG.classifier.algo_inter_arrival_cv_threshold))
        )
        size_regularity_score = 1.0 / (
            1.0 + np.exp(3.0 * (cv_sz - CONFIG.classifier.algo_size_cv_threshold))
        )
        prob_algo = (0.60 * time_regularity_score + 0.40 * size_regularity_score) * 100.0
        if prob_algo > 70.0:
            notes.append("Heurística: Cadencia temporal y tamaños regulares.")

        # 2. Patrón: Iceberg Heurístico
        rounded_prices = np.round(prices, 1)
        level_volumes = {}
        for p, u in zip(rounded_prices, usd_values):
            level_volumes[p] = level_volumes.get(p, 0.0) + u

        max_vol_level = max(level_volumes, key=level_volumes.get) if level_volumes else rounded_prices[-1]
        max_level_vol = level_volumes.get(max_vol_level, 0.0)
        level_vol_ratio = max_level_vol / max(1.0, total_usd)
        level_count = np.sum(rounded_prices == max_vol_level)
        iceberg_side = None

        if (
            level_vol_ratio > CONFIG.classifier.iceberg_min_volume_ratio
            and level_count >= CONFIG.classifier.iceberg_min_executions_at_level
            and max_level_vol >= CONFIG.classifier.iceberg_min_usd_at_level
        ):
            level_buys = np.sum(usd_values[(rounded_prices == max_vol_level) & is_buy])
            level_sells = np.sum(usd_values[(rounded_prices == max_vol_level) & (~is_buy)])
            if level_sells > level_buys:
                iceberg_side = "COMPRADOR"
                notes.append(f"Heurística: Concentración ejecutada compradora en ${max_vol_level:,.1f}.")
            else:
                iceberg_side = "VENDEDOR"
                notes.append(f"Heurística: Concentración ejecutada vendedora en ${max_vol_level:,.1f}.")
            prob_iceberg = min(96.0, (level_vol_ratio * 1.8 + (level_count / 20.0)) * 70.0)
        else:
            prob_iceberg = min(50.0, (level_vol_ratio * 2.0) * 50.0)

        # 3. Flujo agresivo con baja respuesta de precio, medido en 5 segundos.
        prob_absorption = 10.0
        absorption_side = None
        absorption_total_usd = (
            absorption_metrics.buy_volume_usd + absorption_metrics.sell_volume_usd
        )
        absorption_cvd_ratio = abs(absorption_metrics.cvd_usd) / max(
            1.0, absorption_total_usd
        )

        if (
            absorption_total_usd >= CONFIG.classifier.absorption_min_usd_vol
            and absorption_cvd_ratio > CONFIG.classifier.absorption_min_cvd_ratio
            and abs(absorption_metrics.price_delta_bps)
            < CONFIG.classifier.absorption_max_price_delta_bps
        ):
            absorption_intensity = absorption_cvd_ratio / max(
                0.1, abs(absorption_metrics.price_delta_bps) + 0.2
            )
            prob_absorption = min(95.0, 45.0 + (absorption_intensity * 25.0))
            if absorption_metrics.cvd_usd > 0:
                absorption_side = "VENTAS (Limit Asks absorben compras agresivas)"
                notes.append("Heurística: Flujo comprador con baja respuesta de precio.")
            else:
                absorption_side = "COMPRAS (Limit Bids absorben ventas agresivas)"
                notes.append("Heurística: Flujo vendedor con baja respuesta de precio.")
        else:
            prob_absorption = min(45.0, absorption_cvd_ratio * 60.0)

        # 4. Burst activity, measured over one second; it does not identify HFT.
        activity_trades = self.get_window_trades(self.activity_window_seconds)
        activity_sides = np.array(
            [trade.side == TradeSide.BUY for trade in activity_trades],
            dtype=bool,
        )
        tps = activity_metrics.trades_per_second
        side_transitions = np.sum(activity_sides[:-1] != activity_sides[1:])
        transition_rate = side_transitions / max(1, len(activity_sides) - 1)

        hft_tps_threshold = CONFIG.classifier.hft_burst_tps_threshold
        if tps > hft_tps_threshold:
            prob_hft = min(95.0, (tps / 90.0) * 50.0 + (transition_rate * 45.0))
            if prob_hft > 75.0:
                notes.append(f"Heurística: Ráfaga de actividad ({tps:.0f} eventos/s).")
        else:
            prob_hft = min(35.0, (tps / max(1.0, hft_tps_threshold)) * 35.0)

        # 5. Directional impulse; require price displacement and signed flow to align.
        cvd_ratio = abs(cvd_usd) / max(1.0, total_usd)
        direction_aligned = behavior_metrics.price_delta_bps * cvd_usd > 0
        if (
            price_delta_bps > CONFIG.classifier.breakout_min_price_delta_bps
            and cvd_ratio > CONFIG.classifier.breakout_min_cvd_ratio
            and direction_aligned
        ):
            prob_breakout = min(94.0, (price_delta_bps / 15.0) * 50.0 + (cvd_ratio * 45.0))
            direction = "alcista" if behavior_metrics.price_delta_bps > 0 else "bajista"
            notes.append(f"Heurística: Impulso direccional {direction}.")
        else:
            prob_breakout = min(
                20.0 if not direction_aligned else 40.0,
                (price_delta_bps / max(1.0, CONFIG.classifier.breakout_min_price_delta_bps))
                * 30.0,
            )

        # 6. Patrón: Retail Dominante
        institutional_pressure = max(prob_algo, prob_iceberg, prob_absorption, prob_hft)
        prob_retail = max(5.0, min(95.0, 100.0 - (institutional_pressure * 0.85)))

        regimes = [
            (prob_algo, MarketRegime.ALGO_INSTITUTIONAL.value),
            (prob_iceberg, f"ICEBERG_{iceberg_side}" if iceberg_side else MarketRegime.ICEBERG_BUY.value),
            (prob_absorption, MarketRegime.ABSORPTION_BUY.value if cvd_usd < 0 else MarketRegime.ABSORPTION_SELL.value),
            (prob_hft, MarketRegime.HFT_BURST.value),
            (prob_breakout, MarketRegime.BREAKOUT.value),
            (prob_retail, MarketRegime.RETAIL_DOMINANT.value)
        ]
        dominant_regime = max(regimes, key=lambda x: x[0])[1]

        return ClassificationProbabilities(
            algo_institutional=round(prob_algo, 1),
            iceberg=round(prob_iceberg, 1),
            absorption=round(prob_absorption, 1),
            retail=round(prob_retail, 1),
            hft_competition=round(prob_hft, 1),
            breakout=round(prob_breakout, 1),
            dominant_regime=dominant_regime,
            iceberg_side=iceberg_side,
            absorption_side=absorption_side,
            notes=notes[:3]
        )
