"""OHLCV-only research backtests using downloaded Bybit linear klines."""

import math
from datetime import datetime, timezone
from typing import Dict, List, Sequence, Tuple

import numpy as np

from src.core.scalping import ScalpingEngine


class CandleBacktestEngine:
    strategy_name = "EMA trend + momentum + volume (OHLCV)"
    holding_period_bars = 3
    stop_multiples = (0.75, 1.0, 1.25)
    target_multiples = (1.0, 1.5, 2.0)

    def __init__(self, config):
        self.config = config

    def run_backtest(
        self,
        candles: Sequence[Dict[str, float]],
        timeframe_minutes: int,
        direction_mode: str = "original",
        lookback_days: int = 30,
    ) -> Dict[str, object]:
        if timeframe_minutes not in (1, 5, 10, 15):
            raise ValueError("El timeframe debe ser 1, 5, 10 o 15 minutos.")
        if direction_mode not in ("original", "inverse"):
            raise ValueError("El modo de dirección debe ser original o inverse.")

        bars = self._normalize_candles(candles)
        count = len(bars)
        split_index = int(count * 0.7)
        base_result = {
            "source": "Bybit Linear · REST klines",
            "data_source": "bybit_rest_klines",
            "strategy": "candle_ema_momentum",
            "strategy_name": self.strategy_name,
            "timeframe_minutes": timeframe_minutes,
            "direction_mode": direction_mode,
            "direction_mode_name": (
                "Invertida (LONG ↔ SHORT)" if direction_mode == "inverse" else "Original"
            ),
            "observations": count,
            "candle_count": count,
            "training_observations": split_index,
            "validation_observations": count - split_index,
            "lookback_days": lookback_days,
            "alert_enabled": False,
            "early_exit_on_hmm": False,
        }

        if count < 2:
            result = {
                **base_result,
                "status": "collecting_candle_data",
                "message": "Bybit no devolvió suficientes velas completas para validar.",
            }
            return result

        timestamps = np.asarray([int(bar["time_ms"]) for bar in bars], dtype=np.int64)
        expected_interval_ms = timeframe_minutes * 60_000
        gaps = np.diff(timestamps)
        expected_count = int((timestamps[-1] - timestamps[0]) // expected_interval_ms) + 1
        coverage = min(1.0, count / max(1, expected_count))
        maximum_gap_ms = int(np.max(gaps)) if len(gaps) else 0
        data_quality = {
            "snapshot_coverage_pct": round(coverage * 100.0, 2),
            "maximum_snapshot_gap_seconds": round(maximum_gap_ms / 1000.0, 2),
        }
        if np.any(gaps != expected_interval_ms):
            return {
                **base_result,
                **data_quality,
                "status": "insufficient_data_quality",
                "message": (
                    "Las velas descargadas contienen huecos o intervalos duplicados; "
                    "no se valida un backtest que atraviese esos huecos."
                ),
            }

        minimum_training_bars = self.config.min_candle_training_bars
        if split_index < minimum_training_bars:
            required_bars = math.ceil(minimum_training_bars / 0.7)
            return {
                **base_result,
                **data_quality,
                "status": "collecting_candle_data",
                "message": (
                    f"El 70 % de entrenamiento contiene {split_index} velas; "
                    f"se requieren {minimum_training_bars}. Aumenta el periodo descargado."
                ),
                "required_training_bars": minimum_training_bars,
                "required_observations": required_bars,
            }

        opens = np.asarray([bar["open"] for bar in bars], dtype=np.float64)
        highs = np.asarray([bar["high"] for bar in bars], dtype=np.float64)
        lows = np.asarray([bar["low"] for bar in bars], dtype=np.float64)
        closes = np.asarray([bar["close"] for bar in bars], dtype=np.float64)
        volumes = np.asarray([bar["volume"] for bar in bars], dtype=np.float64)
        atr_bps = self._atr_bps(highs, lows, closes)
        signals = self._signal_indices(closes, volumes)
        if direction_mode == "inverse":
            signals = [(index, -direction) for index, direction in signals]

        candidates = []
        best_parameters = None
        best_training = None
        max_training_trades = 0
        for stop_multiple in self.stop_multiples:
            for target_multiple in self.target_multiples:
                training_trades = self._simulate(
                    opens,
                    highs,
                    lows,
                    closes,
                    atr_bps,
                    signals,
                    stop_multiple,
                    target_multiple,
                    start_index=21,
                    end_index=split_index,
                    interval_seconds=timeframe_minutes * 60,
                )
                training = ScalpingEngine._summarize(training_trades)
                max_training_trades = max(max_training_trades, training["trade_count"])
                candidates.append((training["expectancy_bps"], stop_multiple, target_multiple, training))
                if (
                    training["trade_count"] >= self.config.min_validation_trades
                    and (
                        best_training is None
                        or training["expectancy_bps"] > best_training["expectancy_bps"]
                    )
                ):
                    best_parameters = (stop_multiple, target_multiple)
                    best_training = training

        if best_parameters is None:
            return {
                **base_result,
                **data_quality,
                "status": "no_trainable_setup",
                "message": (
                    "La estrategia OHLCV no generó suficientes operaciones en entrenamiento "
                    "para comparar parámetros."
                ),
                "candidate_count": len(candidates),
                "training_bars": split_index,
                "validation_bars": count - split_index,
                "required_training_bars": minimum_training_bars,
                "training_signal_count": sum(index < split_index for index, _ in signals),
                "max_training_trades": max_training_trades,
                "minimum_training_trades": self.config.min_validation_trades,
            }

        stop_multiple, target_multiple = best_parameters
        validation_signals = [
            (index, direction) for index, direction in signals if index >= split_index
        ]
        validation_trades = self._simulate(
            opens,
            highs,
            lows,
            closes,
            atr_bps,
            validation_signals,
            stop_multiple,
            target_multiple,
            start_index=split_index,
            end_index=count,
            interval_seconds=timeframe_minutes * 60,
        )
        validation = ScalpingEngine._summarize(validation_trades)
        validated = (
            validation["trade_count"] >= self.config.min_validation_trades
            and validation["expectancy_bps"] > 0
        )
        interval_ms = expected_interval_ms
        return {
            **base_result,
            **data_quality,
            "status": "validated" if validated else "no_validated_edge",
            "message": (
                "Validación OHLCV positiva; solo investigación/paper trading. "
                "No valida ni activa señales live basadas en flujo de órdenes."
                if validated
                else (
                    "La validación OHLCV no mostró expectativa neta positiva suficiente; "
                    "no activa señales live."
                )
            ),
            "candidate_count": len(candidates),
            "training_bars": split_index,
            "validation_bars": count - split_index,
            "training": best_training,
            "validation": validation,
            "parameters": {
                "stop_atr_multiple": stop_multiple,
                "target_atr_multiple": target_multiple,
                "holding_period_bars": self.holding_period_bars,
                "ema_fast_period": 8,
                "ema_slow_period": 21,
                "volume_lookback_bars": 20,
            },
            "minimum_validation_trades": self.config.min_validation_trades,
            "train_end_utc": datetime.fromtimestamp(
                (int(timestamps[split_index - 1]) + interval_ms) / 1000,
                timezone.utc,
            ).isoformat(),
            "validation_start_utc": datetime.fromtimestamp(
                int(timestamps[split_index]) / 1000, timezone.utc
            ).isoformat(),
            "validation_end_utc": datetime.fromtimestamp(
                (int(timestamps[-1]) + interval_ms) / 1000, timezone.utc
            ).isoformat(),
            "costs": {
                "taker_fee_bps_per_side": self.config.taker_fee_bps_per_side,
                "slippage_bps_per_side": self.config.slippage_bps_per_side,
                "round_trip_bps": 2 * (
                    self.config.taker_fee_bps_per_side
                    + self.config.slippage_bps_per_side
                ),
            },
        }

    @staticmethod
    def _normalize_candles(
        candles: Sequence[Dict[str, float]],
    ) -> List[Dict[str, float]]:
        normalized = []
        for candle in candles:
            try:
                timestamp = int(candle["time_ms"])
                values = {
                    key: float(candle[key])
                    for key in ("open", "high", "low", "close", "volume")
                }
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("A downloaded candle has invalid fields.") from error
            if (
                timestamp < 0
                or not all(math.isfinite(value) for value in values.values())
                or min(values["open"], values["high"], values["low"], values["close"]) <= 0
                or values["volume"] < 0
                or values["high"] < max(values["open"], values["close"], values["low"])
                or values["low"] > min(values["open"], values["close"], values["high"])
            ):
                raise ValueError("A downloaded candle contains invalid OHLCV values.")
            normalized.append({"time_ms": timestamp, **values})

        normalized.sort(key=lambda candle: candle["time_ms"])
        if len({candle["time_ms"] for candle in normalized}) != len(normalized):
            raise ValueError("Downloaded candles contain duplicate timestamps.")
        return normalized

    @staticmethod
    def _signal_indices(
        closes: np.ndarray,
        volumes: np.ndarray,
    ) -> List[Tuple[int, int]]:
        fast_ema = CandleBacktestEngine._ema(closes, 8)
        slow_ema = CandleBacktestEngine._ema(closes, 21)
        volume_average = np.full(len(volumes), np.nan, dtype=np.float64)
        cumulative = np.concatenate(([0.0], np.cumsum(volumes)))
        for index in range(19, len(volumes)):
            volume_average[index] = (cumulative[index + 1] - cumulative[index - 19]) / 20

        long_condition = np.zeros(len(closes), dtype=bool)
        short_condition = np.zeros(len(closes), dtype=bool)
        for index in range(20, len(closes)):
            momentum = closes[index] / closes[index - 3] - 1.0
            volume_confirmed = volumes[index] >= volume_average[index] * 0.8
            long_condition[index] = (
                fast_ema[index] > slow_ema[index]
                and closes[index] > fast_ema[index]
                and momentum > 0
                and volume_confirmed
            )
            short_condition[index] = (
                fast_ema[index] < slow_ema[index]
                and closes[index] < fast_ema[index]
                and momentum < 0
                and volume_confirmed
            )

        signals = []
        for signal_bar in range(20, len(closes) - 1):
            if long_condition[signal_bar] and not long_condition[signal_bar - 1]:
                signals.append((signal_bar + 1, 1))
            elif short_condition[signal_bar] and not short_condition[signal_bar - 1]:
                signals.append((signal_bar + 1, -1))
        return signals

    @staticmethod
    def _ema(values: np.ndarray, period: int) -> np.ndarray:
        result = np.empty(len(values), dtype=np.float64)
        result[0] = values[0]
        alpha = 2.0 / (period + 1)
        for index in range(1, len(values)):
            result[index] = alpha * values[index] + (1 - alpha) * result[index - 1]
        return result

    @staticmethod
    def _atr_bps(
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        period: int = 14,
    ) -> np.ndarray:
        previous_closes = np.concatenate(([closes[0]], closes[:-1]))
        true_ranges = np.maximum(
            highs - lows,
            np.maximum(
                np.abs(highs - previous_closes),
                np.abs(lows - previous_closes),
            ),
        )
        result = np.full(len(closes), np.nan, dtype=np.float64)
        cumulative = np.concatenate(([0.0], np.cumsum(true_ranges)))
        for index in range(period - 1, len(closes)):
            atr = (cumulative[index + 1] - cumulative[index - period + 1]) / period
            result[index] = atr / closes[index] * 10_000.0
        return result

    def _simulate(
        self,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        atr_bps: np.ndarray,
        signals: Sequence[Tuple[int, int]],
        stop_multiple: float,
        target_multiple: float,
        start_index: int,
        end_index: int,
        interval_seconds: int,
    ) -> List[Dict[str, object]]:
        round_trip_cost = 2 * (
            self.config.taker_fee_bps_per_side + self.config.slippage_bps_per_side
        )
        results = []
        next_entry_index = start_index
        for entry_index, direction in signals:
            if (
                entry_index < next_entry_index
                or entry_index < start_index
                or entry_index >= end_index
            ):
                continue
            atr_index = entry_index - 1
            if atr_index < 0 or not math.isfinite(float(atr_bps[atr_index])):
                continue

            entry_price = opens[entry_index]
            distance_bps = max(0.25, float(atr_bps[atr_index]))
            stop_distance = entry_price * distance_bps / 10_000 * stop_multiple
            target_distance = entry_price * distance_bps / 10_000 * target_multiple
            stop_price = entry_price - direction * stop_distance
            target_price = entry_price + direction * target_distance
            exit_limit = min(end_index, entry_index + self.holding_period_bars)
            exit_price = None
            exit_index = entry_index
            exit_reason = "time"

            for index in range(entry_index, exit_limit):
                bar_open = opens[index]
                if direction > 0:
                    stop_hit = bar_open <= stop_price or lows[index] <= stop_price
                    target_hit = bar_open >= target_price or highs[index] >= target_price
                else:
                    stop_hit = bar_open >= stop_price or highs[index] >= stop_price
                    target_hit = bar_open <= target_price or lows[index] <= target_price

                if stop_hit:
                    exit_price = (
                        bar_open
                        if (direction > 0 and bar_open <= stop_price)
                        or (direction < 0 and bar_open >= stop_price)
                        else stop_price
                    )
                    exit_index = index
                    exit_reason = "stop_loss"
                    break
                if target_hit:
                    exit_price = target_price
                    exit_index = index
                    exit_reason = "take_profit"
                    break

            if exit_price is None:
                if exit_limit - entry_index < self.holding_period_bars:
                    continue
                exit_index = exit_limit - 1
                exit_price = closes[exit_index]

            gross_return = direction * math.log(exit_price / entry_price) * 10_000.0
            results.append({
                "direction": "LONG" if direction > 0 else "SHORT",
                "gross_return_bps": float(gross_return),
                "net_return_bps": float(gross_return - round_trip_cost),
                "cost_bps": float(round_trip_cost),
                "exit_reason": exit_reason,
                "duration_seconds": float(
                    (exit_index - entry_index + 1) * interval_seconds
                ),
            })
            next_entry_index = exit_index + 1
        return results


def aggregate_klines(
    candles: Sequence[Dict[str, float]],
    source_interval_minutes: int,
    target_interval_minutes: int,
) -> List[Dict[str, float]]:
    """Aggregate complete, aligned source candles into the requested interval."""
    if (
        source_interval_minutes <= 0
        or target_interval_minutes <= 0
        or target_interval_minutes % source_interval_minutes
    ):
        raise ValueError("The target candle interval must be a multiple of the source interval.")
    if source_interval_minutes == target_interval_minutes:
        return list(candles)

    source_ms = source_interval_minutes * 60_000
    target_ms = target_interval_minutes * 60_000
    required_count = target_interval_minutes // source_interval_minutes
    buckets: Dict[int, List[Dict[str, float]]] = {}
    for candle in candles:
        time_ms = int(candle["time_ms"])
        bucket = time_ms // target_ms * target_ms
        buckets.setdefault(bucket, []).append(candle)

    result = []
    for bucket, rows in sorted(buckets.items()):
        rows.sort(key=lambda candle: int(candle["time_ms"]))
        expected_times = [bucket + source_ms * index for index in range(required_count)]
        if len(rows) != required_count or [
            int(row["time_ms"]) for row in rows
        ] != expected_times:
            continue
        result.append({
            "time_ms": bucket,
            "open": float(rows[0]["open"]),
            "high": max(float(row["high"]) for row in rows),
            "low": min(float(row["low"]) for row in rows),
            "close": float(rows[-1]["close"]),
            "volume": sum(float(row["volume"]) for row in rows),
        })
    return result
