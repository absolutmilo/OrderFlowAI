"""ATR-based paper backtesting and live scalping setup detection."""

import itertools
import json
import math
import sqlite3
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np


STRATEGIES = {
    "flow_momentum": "Flujo + momentum (base)",
    "fourier_breakout": "Fourier: anomalía + ruptura",
    "whale_trend": "Volumen excepcional + tendencia",
}


@dataclass(frozen=True)
class ScalpParameters:
    horizon_seconds: int
    stop_atr_multiple: float
    target_atr_multiple: float
    imbalance_threshold: float
    momentum_threshold_bps: float
    persistence_seconds: int


class ScalpingEngine:
    def __init__(self, database_path: Path, config):
        self.database_path = database_path
        self.config = config
        self.parameters: Optional[ScalpParameters] = None
        self.history: Deque[Tuple[int, float, float]] = deque(maxlen=120)
        self.last_condition = False
        self.last_signal: Optional[Dict[str, object]] = None
        self.last_backtest: Optional[Dict[str, object]] = None

    def run_backtest(
        self,
        strategy: str = "flow_momentum",
        timeframe_minutes: int = 5,
        direction_mode: str = "original",
    ) -> Dict[str, object]:
        if strategy not in STRATEGIES:
            raise ValueError(f"Estrategia desconocida: {strategy}")
        if timeframe_minutes not in (5, 10, 15):
            raise ValueError("El timeframe debe ser 5, 10 o 15 minutos.")
        if direction_mode not in ("original", "inverse"):
            raise ValueError("El modo de dirección debe ser original o inverse.")

        self.parameters = None
        self.last_condition = False
        self.last_signal = None
        timestamps, prices, imbalance, volume_rate = self._load_observations()
        count = len(timestamps)
        split_index = int(count * 0.7)
        if split_index < self.config.min_training_observations:
            required_observations = math.ceil(self.config.min_training_observations / 0.7)
            result = {
                "status": "collecting_data",
                "message": (
                    f"El 70 % de entrenamiento contiene {split_index} observaciones; "
                    f"se requieren {self.config.min_training_observations}. "
                    f"Captura hasta {required_observations} observaciones totales."
                ),
                "observations": count,
                "training_observations": split_index,
                "required_training_observations": self.config.min_training_observations,
                "required_observations": required_observations,
                "source": self.config.exchange,
                "strategy": strategy,
                "strategy_name": STRATEGIES[strategy],
                "timeframe_minutes": timeframe_minutes,
                "direction_mode": direction_mode,
            }
            self.last_backtest = result
            return result

        gaps_seconds = np.diff(timestamps) / 1000.0
        maximum_allowed_gap = self.config.max_snapshot_gap_seconds
        active_seconds = float(
            np.sum(gaps_seconds[gaps_seconds <= maximum_allowed_gap])
        )
        snapshot_coverage = min(1.0, count / max(1.0, active_seconds))
        maximum_gap_seconds = (
            float(np.max(gaps_seconds))
            if count > 1 else 0.0
        )
        data_quality = {
            "snapshot_coverage_pct": round(snapshot_coverage * 100.0, 2),
            "maximum_snapshot_gap_seconds": round(maximum_gap_seconds, 2),
        }
        minimum_coverage = getattr(self.config, "minimum_snapshot_coverage", 0.80)
        if snapshot_coverage < minimum_coverage:
            result = {
                "status": "insufficient_data_quality",
                "message": (
                    f"La cobertura dentro de los tramos activos es "
                    f"{snapshot_coverage * 100:.1f}%; "
                    f"se requiere al menos {minimum_coverage * 100:.0f}%. "
                    "Los huecos impiden validar entradas y salidas con fiabilidad."
                ),
                "observations": count,
                "training_observations": split_index,
                "source": self.config.exchange,
                "strategy": strategy,
                "strategy_name": STRATEGIES[strategy],
                "timeframe_minutes": timeframe_minutes,
                "direction_mode": direction_mode,
                **data_quality,
            }
            self.parameters = None
            self.last_signal = None
            self.last_backtest = result
            return result

        elapsed_seconds = (timestamps[-1] - timestamps[0]) / 1000.0
        bars = None
        if strategy != "flow_momentum":
            bars = self._aggregate_bars(
                timestamps,
                prices,
                imbalance,
                volume_rate,
                timeframe_minutes,
                max_snapshot_gap_seconds=self.config.max_snapshot_gap_seconds,
            )
            train_bar_count = int(np.searchsorted(bars["snapshot_indices"], split_index))
            minimum_bars = 72 if strategy == "fourier_breakout" else 32
            if train_bar_count < minimum_bars:
                result = {
                    "status": "collecting_timeframe_data",
                    "message": (
                        f"La estrategia requiere al menos {minimum_bars} velas completas "
                        f"de {timeframe_minutes} minutos para entrenamiento; hay "
                        f"{train_bar_count}. Sigue capturando datos."
                    ),
                    "observations": count,
                    "training_observations": split_index,
                    "training_bars": train_bar_count,
                    "required_training_bars": minimum_bars,
                    **data_quality,
                    "source": self.config.exchange,
                    "strategy": strategy,
                    "strategy_name": STRATEGIES[strategy],
                    "timeframe_minutes": timeframe_minutes,
                    "direction_mode": direction_mode,
                }
                self.parameters = None
                self.last_backtest = result
                return result

        train_end_ms = int(timestamps[split_index - 1])
        validation_start_ms = int(timestamps[split_index])
        validation_end_ms = int(timestamps[-1])
        candidates = []
        best_parameters = None
        best_training = None
        horizon_seconds = timeframe_minutes * 60
        atr_by_horizon = {
            horizon_seconds: self._atr_bps(
                timestamps,
                prices,
                horizon_seconds,
                self.config.max_snapshot_gap_seconds,
            )
        }
        signal_cache = {}

        if strategy == "flow_momentum":
            parameter_values = (
                (stop, target, threshold, momentum)
                for stop, target, threshold, momentum in itertools.product(
                    (0.75, 1.0, 1.25),
                    (1.0, 1.5, 2.0),
                    (0.25, 0.40),
                    (0.0, 2.0),
                )
            )
        else:
            parameter_values = (
                (stop, target, 0.25, 0.0)
                for stop, target in itertools.product(
                    (0.75, 1.0, 1.25),
                    (1.0, 1.5, 2.0),
                )
            )

        cached_strategy_signals = None
        strategy_diagnostics = {}
        if strategy != "flow_momentum":
            cached_strategy_signals = self._strategy_signal_indices(
                strategy, bars, strategy_diagnostics
            )
        training_signal_count = sum(
            1 for index, _ in (cached_strategy_signals or [])
            if index < split_index
        )

        max_training_trades = 0
        for stop_multiple, target_multiple, imbalance_threshold, momentum_threshold in parameter_values:
            parameters = ScalpParameters(
                horizon_seconds=horizon_seconds,
                stop_atr_multiple=stop_multiple,
                target_atr_multiple=target_multiple,
                imbalance_threshold=imbalance_threshold,
                momentum_threshold_bps=momentum_threshold,
                persistence_seconds=self.config.signal_persistence_seconds,
            )
            if strategy == "flow_momentum":
                signal_key = (imbalance_threshold, momentum_threshold)
                if signal_key not in signal_cache:
                    signal_cache[signal_key] = self._signal_indices(
                        timestamps, prices, imbalance, parameters, end_index=split_index
                    )
                train_signals = signal_cache[signal_key]
            else:
                train_signals = cached_strategy_signals
            if direction_mode == "inverse":
                train_signals = self._invert_signals(train_signals)
            train_trades = self._simulate(
                timestamps,
                prices,
                atr_by_horizon[horizon_seconds],
                train_signals,
                parameters,
                start_index=60,
                end_index=split_index,
            )
            training = self._summarize(train_trades)
            max_training_trades = max(max_training_trades, int(training["trade_count"]))
            candidates.append((training["expectancy_bps"], parameters, training))
            if (
                training["trade_count"] >= self.config.min_validation_trades
                and (best_training is None or training["expectancy_bps"] > best_training["expectancy_bps"])
            ):
                best_parameters = parameters
                best_training = training

        if best_parameters is None:
            no_signals = strategy != "flow_momentum" and training_signal_count == 0
            result = {
                "status": "no_trainable_setup",
                "message": (
                    (
                        f"La estrategia no generó señales en las "
                        f"{int(np.searchsorted(bars['snapshot_indices'], split_index))} "
                        "velas de entrenamiento; más snapshots no garantizan que "
                        "aparezcan señales."
                    )
                    if no_signals
                    else "No hubo suficientes operaciones simuladas en entrenamiento "
                    "para comparar parámetros. Revisa el diagnóstico de señales."
                ),
                "observations": count,
                "training_observations": split_index,
                "validation_observations": count - split_index,
                **data_quality,
                "training_bars": (
                    int(np.searchsorted(bars["snapshot_indices"], split_index))
                    if bars is not None else None
                ),
                "validation_bars": (
                    int(len(bars["snapshot_indices"]) - np.searchsorted(
                        bars["snapshot_indices"], split_index
                    ))
                    if bars is not None else None
                ),
                "source": self.config.exchange,
                "strategy": strategy,
                "strategy_name": STRATEGIES[strategy],
                "timeframe_minutes": timeframe_minutes,
                "direction_mode": direction_mode,
                "direction_mode_name": (
                    "Invertida (LONG ↔ SHORT)"
                    if direction_mode == "inverse"
                    else "Original"
                ),
                "candidate_count": len(candidates),
                "training_span_days": round(elapsed_seconds / 86400.0, 2),
                "minimum_training_trades": self.config.min_validation_trades,
                "training_signal_count": training_signal_count,
                "max_training_trades": max_training_trades,
                "required_training_bars": 72 if strategy == "fourier_breakout" else (
                    32 if strategy == "whale_trend" else None
                ),
                "signal_diagnostics": strategy_diagnostics,
            }
            self.last_backtest = result
            self.parameters = None
            return result

        validation_start = split_index
        if strategy == "flow_momentum":
            validation_signals = self._signal_indices(
                timestamps, prices, imbalance, best_parameters,
                start_index=validation_start,
            )
        else:
            validation_signals = cached_strategy_signals
        if direction_mode == "inverse":
            validation_signals = self._invert_signals(validation_signals)
        validation_trades = self._simulate(
            timestamps,
            prices,
            atr_by_horizon[horizon_seconds],
            validation_signals,
            best_parameters,
            start_index=validation_start,
            end_index=count,
        )
        validation = self._summarize(validation_trades)
        validated = (
            validation["trade_count"] >= self.config.min_validation_trades
            and validation["expectancy_bps"] > 0
        )
        self.parameters = (
            best_parameters
            if validated and strategy == "flow_momentum"
            else None
        )
        self.last_condition = False
        self.last_signal = None

        result = {
            "status": "validated" if validated else "no_validated_edge",
            "message": (
                "Parámetros habilitados para detectar setups de paper trading."
                if validated and strategy == "flow_momentum"
                else (
                    "Backtest de investigación positivo; la señal live de esta estrategia "
                    "aún no está implementada."
                    if validated and (
                        strategy != "flow_momentum" or direction_mode == "inverse"
                    )
                    else "La estrategia seleccionada no mostró expectativa neta positiva suficiente fuera de muestra; no se activan alertas de entrada."
                )
            ),
            "source": self.config.exchange,
            "symbol": self.config.symbol,
            "strategy": strategy,
            "strategy_name": STRATEGIES[strategy],
            "timeframe_minutes": timeframe_minutes,
            "direction_mode": direction_mode,
            "direction_mode_name": (
                "Invertida (LONG ↔ SHORT)"
                if direction_mode == "inverse"
                else "Original"
            ),
            "observations": count,
            **data_quality,
            "training_observations": split_index,
            "validation_observations": count - split_index,
            "training_bars": (
                int(np.searchsorted(bars["snapshot_indices"], split_index))
                if bars is not None else None
            ),
            "validation_bars": (
                int(len(bars["snapshot_indices"]) - np.searchsorted(
                    bars["snapshot_indices"], split_index
                ))
                if bars is not None else None
            ),
            "training_span_days": round(elapsed_seconds / 86400.0, 2),
            "train_end_utc": datetime.fromtimestamp(train_end_ms / 1000, timezone.utc).isoformat(),
            "validation_start_utc": datetime.fromtimestamp(
                validation_start_ms / 1000, timezone.utc
            ).isoformat(),
            "validation_end_utc": datetime.fromtimestamp(validation_end_ms / 1000, timezone.utc).isoformat(),
            "candidate_count": len(candidates),
            "costs": {
                "taker_fee_bps_per_side": self.config.taker_fee_bps_per_side,
                "slippage_bps_per_side": self.config.slippage_bps_per_side,
                "round_trip_bps": 2 * (
                    self.config.taker_fee_bps_per_side + self.config.slippage_bps_per_side
                ),
            },
            "parameters": asdict(best_parameters),
            "training": best_training,
            "validation": validation,
            "alert_enabled": (
                validated
                and strategy == "flow_momentum"
                and direction_mode == "original"
            ),
        }
        self.last_backtest = result
        return result

    def observe(self, observed_at_ms: int, price: float, features: Dict[str, object]) -> Optional[Dict[str, object]]:
        total_volume = float(features.get("buy_volume_usd", 0.0)) + float(
            features.get("sell_volume_usd", 0.0)
        )
        if total_volume <= 0 or price <= 0:
            self.last_condition = False
            return self.last_signal

        imbalance = float(features.get("cvd_usd", 0.0)) / total_volume
        self.history.append((observed_at_ms, price, imbalance))
        if self.parameters is None:
            self.last_condition = False
            return self.last_signal

        timestamp_array = np.array([item[0] for item in self.history], dtype=np.int64)
        price_array = np.array([item[1] for item in self.history], dtype=np.float64)
        if len(timestamp_array) < 60:
            self.last_condition = False
            return self.last_signal

        momentum, atr = self._current_momentum_and_atr(
            timestamp_array, price_array, self.parameters.horizon_seconds
        )
        if momentum is None or atr is None:
            self.last_condition = False
            return self.last_signal

        threshold = self.parameters.imbalance_threshold
        momentum_threshold = self.parameters.momentum_threshold_bps
        direction = 1 if imbalance >= threshold and momentum >= momentum_threshold else (
            -1 if imbalance <= -threshold and momentum <= -momentum_threshold else 0
        )
        persistence_start = observed_at_ms - self.parameters.persistence_seconds * 1000
        recent = [
            item for item in self.history
            if item[0] >= persistence_start
        ]
        sustained = len(recent) >= self.parameters.persistence_seconds and all(
            (item[2] >= threshold and momentum >= momentum_threshold)
            if direction == 1
            else (item[2] <= -threshold and momentum <= -momentum_threshold)
            for item in recent
        ) if direction else False

        if sustained and not self.last_condition:
            stop_distance = atr * self.parameters.stop_atr_multiple
            target_distance = atr * self.parameters.target_atr_multiple
            self.last_signal = {
                "status": "signal",
                "direction": "LONG" if direction == 1 else "SHORT",
                "entry_price": price,
                "stop_loss": price * (1 - direction * stop_distance / 10000.0),
                "take_profit": price * (1 + direction * target_distance / 10000.0),
                "atr_bps": round(atr, 2),
                "horizon_minutes": self.parameters.horizon_seconds // 60,
                "timestamp_ms": observed_at_ms,
                "source": self.config.exchange,
                "symbol": self.config.symbol,
            }
        self.last_condition = sustained

        if self.last_signal and observed_at_ms - int(self.last_signal["timestamp_ms"]) > (
            self.parameters.horizon_seconds * 1000
        ):
            self.last_signal = None
        return self.last_signal

    def status(self) -> Dict[str, object]:
        result = {
            "source": self.config.exchange,
            "symbol": self.config.symbol,
            "signal": self.last_signal or {"status": "no_setup"},
            "alert_enabled": self.parameters is not None,
            "backtest": self.last_backtest,
            "minimum_training_observations": self.config.min_training_observations,
            "minimum_observations": math.ceil(
                self.config.min_training_observations / 0.7
            ),
        }
        return result

    def _load_observations(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        with sqlite3.connect(self.database_path, timeout=5) as connection:
            rows = connection.execute(
                """
                SELECT observed_at_ms, price, features_json
                FROM market_observations
                WHERE market_source = ? AND symbol = ?
                ORDER BY observed_at_ms ASC
                """,
                (self.config.exchange, self.config.symbol),
            ).fetchall()
        timestamps = []
        prices = []
        imbalances = []
        volume_rates = []
        for observed_at_ms, price, features_json in rows:
            features = json.loads(features_json)
            total_volume = float(features.get("buy_volume_usd", 0.0)) + float(
                features.get("sell_volume_usd", 0.0)
            )
            timestamps.append(int(observed_at_ms))
            prices.append(float(price))
            imbalances.append(
                float(features.get("cvd_usd", 0.0)) / total_volume if total_volume > 0 else 0.0
            )
            volume_rates.append(float(features.get("volume_per_second_usd", 0.0)))
        return (
            np.asarray(timestamps, dtype=np.int64),
            np.asarray(prices, dtype=np.float64),
            np.asarray(imbalances, dtype=np.float64),
            np.asarray(volume_rates, dtype=np.float64),
        )

    @staticmethod
    def _aggregate_bars(
        timestamps: np.ndarray,
        prices: np.ndarray,
        imbalance: np.ndarray,
        volume_rate: np.ndarray,
        timeframe_minutes: int,
        max_snapshot_gap_seconds: float = 5.0,
    ) -> Dict[str, np.ndarray]:
        interval_ms = timeframe_minutes * 60_000
        bucket_ids = timestamps // interval_ms
        boundaries = np.concatenate((
            [0],
            np.flatnonzero(np.diff(bucket_ids)) + 1,
            [len(timestamps)],
        ))
        starts = boundaries[:-1]
        ends = boundaries[1:]
        minimum_snapshots = int(timeframe_minutes * 60 * 0.8)
        maximum_gaps = np.asarray([
            np.max(np.diff(timestamps[start:end])) if end - start > 1 else interval_ms
            for start, end in zip(starts, ends)
        ])
        complete = (
            ((timestamps[ends - 1] - timestamps[starts]) >= int(interval_ms * 0.8))
            & ((ends - starts) >= minimum_snapshots)
            & (maximum_gaps <= max_snapshot_gap_seconds * 1000)
        )
        bars = {
            "timestamps": timestamps[ends - 1],
            "snapshot_indices": ends - 1,
            "open": prices[starts],
            "high": np.asarray([
                np.max(prices[start:end]) for start, end in zip(starts, ends)
            ]),
            "low": np.asarray([
                np.min(prices[start:end]) for start, end in zip(starts, ends)
            ]),
            "close": prices[ends - 1],
            "imbalance": np.asarray([
                np.mean(imbalance[start:end]) for start, end in zip(starts, ends)
            ]),
            "volume_rate": np.asarray([
                np.median(volume_rate[start:end]) for start, end in zip(starts, ends)
            ]),
        }
        return {key: values[complete] for key, values in bars.items()}

    def _strategy_signal_indices(
        self,
        strategy: str,
        bars: Dict[str, np.ndarray],
        diagnostics: Optional[Dict[str, int]] = None,
    ) -> List[Tuple[int, int]]:
        if strategy == "fourier_breakout":
            return self._fourier_breakout_signals(bars, diagnostics)
        if strategy == "whale_trend":
            return self._whale_trend_signals(bars)
        raise ValueError(f"Estrategia sin implementación: {strategy}")

    @staticmethod
    def _invert_signals(
        signals: Sequence[Tuple[int, int]],
    ) -> List[Tuple[int, int]]:
        return [(index, -direction) for index, direction in signals]

    @staticmethod
    def _ema(values: np.ndarray, length: int) -> np.ndarray:
        result = np.empty(len(values), dtype=np.float64)
        if not len(values):
            return result
        alpha = 2.0 / (length + 1)
        result[0] = values[0]
        for index in range(1, len(values)):
            result[index] = alpha * values[index] + (1 - alpha) * result[index - 1]
        return result

    @staticmethod
    def _whale_trend_signals(
        bars: Dict[str, np.ndarray],
    ) -> List[Tuple[int, int]]:
        closes = bars["close"]
        volumes = bars["volume_rate"]
        imbalances = bars["imbalance"]
        ema_fast = ScalpingEngine._ema(closes, 3)
        ema_slow = ScalpingEngine._ema(closes, 12)
        signals = []
        lookback = 30
        for index in range(lookback + 1, len(closes)):
            baseline = float(np.mean(volumes[index - lookback:index]))
            if baseline <= 0 or volumes[index] <= baseline * 2.0:
                continue
            bullish = closes[index] > bars["open"][index]
            bearish = closes[index] < bars["open"][index]
            if bullish and ema_fast[index] > ema_slow[index] and imbalances[index] >= 0.25:
                signals.append((int(bars["snapshot_indices"][index]), 1))
            elif bearish and ema_fast[index] < ema_slow[index] and imbalances[index] <= -0.25:
                signals.append((int(bars["snapshot_indices"][index]), -1))
        return signals

    @staticmethod
    def _fourier_breakout_signals(
        bars: Dict[str, np.ndarray],
        diagnostics: Optional[Dict[str, int]] = None,
    ) -> List[Tuple[int, int]]:
        closes = bars["close"]
        count = len(closes)
        if count < 72:
            if diagnostics is not None:
                diagnostics.update({
                    "complete_bars": count,
                    "required_bars": 72,
                    "raw_anomalies": 0,
                    "persistent_anomaly_bars": 0,
                    "completed_zones": 0,
                    "context_qualified_bars": 0,
                    "breakout_candidates": 0,
                    "signals": 0,
                })
            return []

        returns = np.zeros(count, dtype=np.float64)
        returns[1:] = np.log(closes[1:] / closes[:-1])
        indices = np.arange(count, dtype=np.float64)
        shares = np.full(count, 0.5, dtype=np.float64)
        dominant_energy = np.zeros(count, dtype=np.float64)
        residuals = np.full(count, np.nan, dtype=np.float64)

        for index in range(47, count):
            start = index - 47
            energies = []
            reconstructions = []
            for period in (8.0, 24.0):
                angle = 2.0 * np.pi * indices[start:index + 1] / period
                sine = np.sin(angle)
                cosine = np.cos(angle)
                segment = returns[start:index + 1]
                ss = float(np.dot(sine, sine))
                cc = float(np.dot(cosine, cosine))
                sc = float(np.dot(sine, cosine))
                xs = float(np.dot(segment, sine))
                xc = float(np.dot(segment, cosine))
                determinant = ss * cc - sc * sc
                if determinant <= 1e-12:
                    a = b = 0.0
                else:
                    a = (xs * cc - xc * sc) / determinant
                    b = (xc * ss - xs * sc) / determinant
                energies.append(float(np.hypot(a, b)))
                reconstructions.append(
                    a * sine[-1] + b * cosine[-1]
                )
            total_energy = sum(energies)
            shares[index] = energies[0] / total_energy if total_energy else 0.5
            dominant_energy[index] = max(energies)
            residuals[index] = returns[index] - sum(reconstructions)

        fast_stability = ScalpingEngine._ema(dominant_energy, 3)
        slow_stability = ScalpingEngine._ema(dominant_energy, 8)
        stability = fast_stability - slow_stability
        raw_anomalous = np.zeros(count, dtype=bool)
        for index in range(70, count):
            residual_window = residuals[index - 23:index + 1]
            if not np.all(np.isfinite(residual_window)):
                continue
            deviation = float(np.std(residual_window, ddof=1))
            z_score = (
                (residuals[index] - float(np.mean(residual_window))) / deviation
                if deviation > 0 else 0.0
            )
            recent_move = (
                np.log(closes[index] / closes[index - 2])
                if closes[index - 2] > 0 else 0.0
            )
            raw_anomalous[index] = abs(z_score) > 2.7 and abs(recent_move) <= 0.003
        anomalous = np.zeros(count, dtype=bool)
        anomalous[1:] = raw_anomalous[1:] & raw_anomalous[:-1]

        signals = []
        active_zone = 0
        zone_high = 0.0
        zone_low = 0.0
        last_accumulation = None
        last_distribution = None
        buffer = 0.0005
        max_zone_age = 200
        completed_zones = 0
        context_qualified_bars = 0
        breakout_candidates = 0

        for index in range(70, count):
            recent_move = np.log(closes[index] / closes[index - 2])
            zone_kind = (
                1 if anomalous[index] and recent_move < 0
                else -1 if anomalous[index] and recent_move > 0
                else 0
            )
            if active_zone and zone_kind != active_zone:
                if active_zone == 1:
                    last_accumulation = (zone_high, index - 1)
                else:
                    last_distribution = (zone_low, index - 1)
                completed_zones += 1
                active_zone = 0

            if zone_kind:
                if not active_zone:
                    active_zone = zone_kind
                    zone_high = float(bars["high"][index])
                    zone_low = float(bars["low"][index])
                else:
                    zone_high = max(zone_high, float(bars["high"][index]))
                    zone_low = min(zone_low, float(bars["low"][index]))
                continue

            share1 = shares[index]
            dominant_share = max(share1, 1 - share1)
            context_ok = (
                0.45 <= share1 <= 0.55
                or (dominant_share >= 0.60 and stability[index] < 0)
            )
            if not context_ok:
                continue
            context_qualified_bars += 1

            if (
                last_accumulation is not None
                and index - last_accumulation[1] <= max_zone_age
                and closes[index] > last_accumulation[0] * (1 + buffer)
            ):
                breakout_candidates += 1
                signals.append((int(bars["snapshot_indices"][index]), 1))
                last_accumulation = None
            elif (
                last_distribution is not None
                and index - last_distribution[1] <= max_zone_age
                and closes[index] < last_distribution[0] * (1 - buffer)
            ):
                breakout_candidates += 1
                signals.append((int(bars["snapshot_indices"][index]), -1))
                last_distribution = None
        if diagnostics is not None:
            diagnostics.update({
                "complete_bars": count,
                "required_bars": 72,
                "raw_anomalies": int(np.count_nonzero(raw_anomalous)),
                "persistent_anomaly_bars": int(np.count_nonzero(anomalous)),
                "completed_zones": completed_zones,
                "context_qualified_bars": context_qualified_bars,
                "breakout_candidates": breakout_candidates,
                "signals": len(signals),
            })
        return signals

    @staticmethod
    def _atr_bps(
        timestamps: np.ndarray,
        prices: np.ndarray,
        horizon_seconds: int,
        max_snapshot_gap_seconds: float = 5.0,
    ) -> np.ndarray:
        count = len(prices)
        log_returns_bps = np.zeros(count, dtype=np.float64)
        valid_intervals = np.zeros(count, dtype=bool)
        if count > 1:
            elapsed = np.maximum(0.001, np.diff(timestamps) / 1000.0)
            valid_intervals[1:] = elapsed <= max_snapshot_gap_seconds
            log_returns_bps[1:] = np.where(
                valid_intervals[1:],
                np.diff(np.log(np.maximum(prices, 1e-12)))
                * 10000.0 / np.sqrt(elapsed),
                0.0,
            )

        squared_prefix = np.concatenate(([0.0], np.cumsum(log_returns_bps ** 2)))
        return_prefix = np.concatenate(([0.0], np.cumsum(log_returns_bps)))
        valid_prefix = np.concatenate(([0], np.cumsum(valid_intervals, dtype=np.int64)))
        start_indices = np.searchsorted(timestamps, timestamps - 60_000, side="left")
        indices = np.arange(count)
        samples = np.maximum(
            1,
            valid_prefix[indices + 1] - valid_prefix[start_indices + 1],
        )
        means = (return_prefix[indices + 1] - return_prefix[start_indices + 1]) / samples
        second_moment = (
            squared_prefix[indices + 1] - squared_prefix[start_indices + 1]
        ) / samples
        variance = np.maximum(0.0, second_moment - means ** 2)
        return np.maximum(0.25, np.sqrt(variance * horizon_seconds))

    @staticmethod
    def _momentum_bps(timestamps: np.ndarray, prices: np.ndarray) -> np.ndarray:
        reference = np.searchsorted(timestamps, timestamps - 60_000, side="right") - 1
        valid = reference >= 0
        result = np.zeros(len(timestamps), dtype=np.float64)
        result[valid] = (
            np.log(prices[valid] / prices[reference[valid]]) * 10000.0
        )
        return result

    def _signal_indices(
        self,
        timestamps: np.ndarray,
        prices: np.ndarray,
        imbalance: np.ndarray,
        parameters: ScalpParameters,
        start_index: int = 0,
        end_index: Optional[int] = None,
    ) -> List[Tuple[int, int]]:
        end_index = len(timestamps) if end_index is None else end_index
        momentum = self._momentum_bps(timestamps, prices)
        long_condition = (
            (imbalance >= parameters.imbalance_threshold)
            & (momentum >= parameters.momentum_threshold_bps)
        )
        short_condition = (
            (imbalance <= -parameters.imbalance_threshold)
            & (momentum <= -parameters.momentum_threshold_bps)
        )
        signals = []
        persistence = parameters.persistence_seconds
        long_persisted = np.zeros(len(timestamps), dtype=bool)
        short_persisted = np.zeros(len(timestamps), dtype=bool)
        for index in range(max(start_index, 60 + persistence), end_index):
            if timestamps[index] - timestamps[index - persistence + 1] > persistence * 1500:
                continue
            long_persisted[index] = bool(
                np.all(long_condition[index - persistence + 1:index + 1])
            )
            short_persisted[index] = bool(
                np.all(short_condition[index - persistence + 1:index + 1])
            )
            if long_persisted[index] and not long_persisted[index - 1]:
                signals.append((index, 1))
            elif short_persisted[index] and not short_persisted[index - 1]:
                signals.append((index, -1))
        return signals

    def _simulate(
        self,
        timestamps: np.ndarray,
        prices: np.ndarray,
        atr_bps: np.ndarray,
        signals: Sequence[Tuple[int, int]],
        parameters: ScalpParameters,
        start_index: int,
        end_index: int,
    ) -> List[Dict[str, object]]:
        round_trip_cost = 2 * (
            self.config.taker_fee_bps_per_side + self.config.slippage_bps_per_side
        )
        results = []
        next_entry_index = start_index
        for entry_index, direction in signals:
            if entry_index < next_entry_index or entry_index < start_index:
                continue

            end_time = timestamps[entry_index] + parameters.horizon_seconds * 1000
            horizon_index = int(np.searchsorted(timestamps, end_time, side="right"))
            close_limit = min(end_index, horizon_index)
            if close_limit <= entry_index + 1:
                continue
            horizon_reached = timestamps[end_index - 1] >= end_time

            entry_price = prices[entry_index]
            stop_distance = max(0.25, atr_bps[entry_index]) * parameters.stop_atr_multiple
            target_distance = max(0.25, atr_bps[entry_index]) * parameters.target_atr_multiple
            signed_return = None
            exit_index = close_limit - 1
            exit_reason = "time"
            previous_index = entry_index
            data_gap = False

            for index in range(entry_index + 1, close_limit):
                if (
                    timestamps[index] - timestamps[previous_index]
                    > self.config.max_snapshot_gap_seconds * 1000
                ):
                    data_gap = True
                    break
                previous_index = index
                raw_return = np.log(prices[index] / entry_price) * 10000.0
                signed = direction * raw_return
                if signed <= -stop_distance:
                    signed_return = -stop_distance
                    exit_index = index
                    exit_reason = "stop_loss"
                    break
                if signed >= target_distance:
                    signed_return = target_distance
                    exit_index = index
                    exit_reason = "take_profit"
                    break
            if data_gap:
                continue
            if signed_return is None:
                if not horizon_reached:
                    continue
                signed_return = direction * np.log(prices[exit_index] / entry_price) * 10000.0

            results.append({
                "direction": "LONG" if direction > 0 else "SHORT",
                "gross_return_bps": float(signed_return),
                "net_return_bps": float(signed_return - round_trip_cost),
                "cost_bps": float(round_trip_cost),
                "exit_reason": exit_reason,
                "duration_seconds": float(
                    (timestamps[exit_index] - timestamps[entry_index]) / 1000.0
                ),
            })
            next_entry_index = exit_index + 1
        return results

    @staticmethod
    def _summarize(trades: Sequence[Dict[str, object]]) -> Dict[str, object]:
        summary = ScalpingEngine._summarize_metrics(trades)
        summary["by_direction"] = {
            direction: ScalpingEngine._summarize_metrics(
                [trade for trade in trades if trade["direction"] == direction]
            )
            for direction in ("LONG", "SHORT")
        }
        summary["by_exit_reason"] = {
            reason: ScalpingEngine._summarize_metrics(
                [trade for trade in trades if trade["exit_reason"] == reason]
            )
            for reason in ("stop_loss", "take_profit", "time")
        }
        return summary

    @staticmethod
    def _summarize_metrics(
        trades: Sequence[Dict[str, object]],
    ) -> Dict[str, object]:
        if not trades:
            return {
                "trade_count": 0,
                "gross_expectancy_bps": 0.0,
                "expectancy_bps": 0.0,
                "win_rate_pct": 0.0,
                "profit_factor": 0.0,
                "gross_total_bps": 0.0,
                "total_net_bps": 0.0,
                "total_costs_bps": 0.0,
                "max_drawdown_bps": 0.0,
                "average_duration_seconds": 0.0,
            }
        gross_values = np.asarray(
            [float(trade["gross_return_bps"]) for trade in trades],
            dtype=np.float64,
        )
        net_values = np.asarray(
            [float(trade["net_return_bps"]) for trade in trades],
            dtype=np.float64,
        )
        gains = float(np.sum(net_values[net_values > 0]))
        losses = float(abs(np.sum(net_values[net_values < 0])))
        equity = np.cumsum(net_values)
        drawdown = np.maximum.accumulate(np.concatenate(([0.0], equity))) - np.concatenate(
            ([0.0], equity)
        )
        return {
            "trade_count": int(len(net_values)),
            "gross_expectancy_bps": round(float(np.mean(gross_values)), 3),
            "expectancy_bps": round(float(np.mean(net_values)), 3),
            "win_rate_pct": round(float(np.mean(net_values > 0) * 100.0), 2),
            "profit_factor": round(gains / losses, 3) if losses > 0 else (
                None if gains > 0 else 0.0
            ),
            "gross_total_bps": round(float(np.sum(gross_values)), 3),
            "total_net_bps": round(float(np.sum(net_values)), 3),
            "total_costs_bps": round(
                float(sum(float(trade["cost_bps"]) for trade in trades)), 3
            ),
            "max_drawdown_bps": round(float(np.max(drawdown)), 3),
            "average_duration_seconds": round(
                float(
                    np.mean([
                        float(trade["duration_seconds"])
                        for trade in trades
                    ])
                ),
                2,
            ),
        }

    @staticmethod
    def _current_momentum_and_atr(
        timestamps: np.ndarray,
        prices: np.ndarray,
        horizon_seconds: int,
    ) -> Tuple[Optional[float], Optional[float]]:
        now = timestamps[-1]
        reference = np.searchsorted(timestamps, now - 60_000, side="right") - 1
        if reference < 0 or now - timestamps[reference] > 65_000:
            return None, None
        momentum = float(np.log(prices[-1] / prices[reference]) * 10000.0)
        recent_start = np.searchsorted(timestamps, now - 60_000, side="left")
        recent_prices = prices[recent_start:]
        recent_times = timestamps[recent_start:]
        if len(recent_prices) < 30:
            return None, None
        elapsed = np.maximum(0.001, np.diff(recent_times) / 1000.0)
        returns = np.diff(np.log(np.maximum(recent_prices, 1e-12))) * 10000.0 / np.sqrt(elapsed)
        atr = max(0.25, float(np.std(returns) * np.sqrt(horizon_seconds)))
        return momentum, atr
