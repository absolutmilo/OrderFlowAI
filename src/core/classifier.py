"""
Order Flow Microstructure Classifier.
Analyzes sequences of trade events over a rolling window to detect institutional execution,
iceberg orders, passive absorption, retail dominance, and HFT competition.
"""

from collections import deque
from typing import Deque, List, Optional, Tuple
import numpy as np

from src.core.trade_model import (
    AggregatedTrade,
    TradeSide,
    ClassificationProbabilities,
    OrderFlowMetrics,
    MarketRegime,
)
from config import CONFIG


class MicrostructureClassifier:
    def __init__(
        self,
        window_size: int = CONFIG.classifier.rolling_window_trades,
        warmup_trades: int = CONFIG.classifier.warmup_trades
    ):
        self.window_size = window_size
        self.warmup_trades = warmup_trades
        self.trades: Deque[AggregatedTrade] = deque(maxlen=window_size)

    def add_trade(self, trade: AggregatedTrade):
        """Pushes a new trade into the rolling window."""
        self.trades.append(trade)

    def is_warmed_up(self) -> bool:
        return len(self.trades) >= self.warmup_trades

    def get_current_metrics(self) -> Optional[OrderFlowMetrics]:
        if not self.trades:
            return None

        trades_list = list(self.trades)
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

        # Time range & throughput
        time_span_ms = max(1, times[-1] - times[0])
        time_span_sec = time_span_ms / 1000.0
        tps = len(trades_list) / max(0.1, time_span_sec)
        vol_per_sec = float(np.sum(usd_values)) / max(0.1, time_span_sec)

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
            inter_arrival_cv=inter_arrival_cv
        )

    def classify(self) -> ClassificationProbabilities:
        """
        Calculates probabilities for each microstructure pattern based on mathematical signatures.
        """
        if not self.is_warmed_up():
            # Return baseline probabilities during warmup
            return ClassificationProbabilities(
                algo_institutional=25.0,
                iceberg=15.0,
                absorption=15.0,
                retail=45.0,
                hft_competition=10.0,
                breakout=10.0,
                dominant_regime=MarketRegime.RETAIL_DOMINANT.value,
                notes=["Calentando ventana de eventos..."]
            )

        trades_list = list(self.trades)
        prices = np.array([t.price for t in trades_list], dtype=np.float64)
        sizes = np.array([t.total_size for t in trades_list], dtype=np.float64)
        usd_values = np.array([t.total_usd_value for t in trades_list], dtype=np.float64)
        times = np.array([t.last_timestamp_ms for t in trades_list], dtype=np.int64)
        is_buy = np.array([t.side == TradeSide.BUY for t in trades_list])

        total_usd = float(np.sum(usd_values))
        buy_usd = float(np.sum(usd_values[is_buy]))
        sell_usd = float(np.sum(usd_values[~is_buy]))
        cvd_usd = buy_usd - sell_usd
        price_spread = float(np.ptp(prices))  # Max price - Min price
        mean_price = float(np.mean(prices))
        price_delta_bps = abs((prices[-1] - prices[0]) / max(1.0, mean_price)) * 10000.0

        notes: List[str] = []

        # -------------------------------------------------------------
        # 1. Patrón: Ejecución Algorítmica Institucional (TWAP / VWAP / Slicing)
        # Signature: Regular equidistant intervals (low delta_t CV) + consistent sizing (low size CV)
        # -------------------------------------------------------------
        delta_times = np.diff(times)
        mean_dt = np.mean(delta_times) if len(delta_times) > 0 else 1.0
        std_dt = np.std(delta_times) if len(delta_times) > 0 else 1.0
        cv_dt = std_dt / max(1.0, mean_dt)

        mean_sz = np.mean(usd_values)
        std_sz = np.std(usd_values)
        cv_sz = std_sz / max(1.0, mean_sz)

        # Scoring: lower CV -> higher algorithmic probability
        time_regularity_score = 1.0 / (
            1.0 + np.exp(3.5 * (cv_dt - CONFIG.classifier.algo_inter_arrival_cv_threshold))
        )
        size_regularity_score = 1.0 / (
            1.0 + np.exp(3.0 * (cv_sz - CONFIG.classifier.algo_size_cv_threshold))
        )
        prob_algo = (0.60 * time_regularity_score + 0.40 * size_regularity_score) * 100.0

        if prob_algo > 70.0:
            notes.append("Cadencia temporal uniforme: Algoritmo de ejecución TWAP/POV activo.")

        # -------------------------------------------------------------
        # 2. Patrón: Iceberg Order (Comprador o Vendedor)
        # Signature: Abnormal accumulation of executed volume at a single tick/price level
        # -------------------------------------------------------------
        # Cluster prices rounded to tick (e.g. 0.10 for BTC)
        rounded_prices = np.round(prices, 1)
        unique_levels, counts = np.unique(rounded_prices, return_counts=True)
        level_volumes = {}
        for p, u in zip(rounded_prices, usd_values):
            level_volumes[p] = level_volumes.get(p, 0.0) + u

        max_vol_level = max(level_volumes, key=level_volumes.get) if level_volumes else rounded_prices[-1]
        max_level_vol = level_volumes.get(max_vol_level, 0.0)
        level_vol_ratio = max_level_vol / max(1.0, total_usd)

        # Count of executions at that dominant price level
        level_count = np.sum(rounded_prices == max_vol_level)
        iceberg_side = None

        prob_iceberg = 10.0
        if (
            level_vol_ratio > CONFIG.classifier.iceberg_min_volume_ratio
            and level_count >= CONFIG.classifier.iceberg_min_executions_at_level
            and max_level_vol >= CONFIG.classifier.iceberg_min_usd_at_level
        ):
            # Check predominant aggressive side hitting this level
            level_buys = np.sum(usd_values[(rounded_prices == max_vol_level) & is_buy])
            level_sells = np.sum(usd_values[(rounded_prices == max_vol_level) & (~is_buy)])

            # If market is selling aggressively into this level and price doesn't break -> Iceberg Comprador
            if level_sells > level_buys:
                iceberg_side = "COMPRADOR"
                notes.append(f"Iceberg Comprador detectado en ${max_vol_level:,.1f} absorbiendo ventas.")
            else:
                iceberg_side = "VENDEDOR"
                notes.append(f"Iceberg Vendedor detectado en ${max_vol_level:,.1f} bloqueando compras.")

            # Sigmoid scoring
            prob_iceberg = min(96.0, (level_vol_ratio * 1.8 + (level_count / 20.0)) * 70.0)
        else:
            prob_iceberg = min(50.0, (level_vol_ratio * 2.0) * 50.0)

        # -------------------------------------------------------------
        # 3. Patrón: Market Maker Absorbiendo (Absorción de Liquidez)
        # Signature: High directional CVD with minimal or negative price displacement
        # -------------------------------------------------------------
        prob_absorption = 10.0
        absorption_side = None
        cvd_ratio = abs(cvd_usd) / max(1.0, total_usd)

        if (
            total_usd >= CONFIG.classifier.absorption_min_usd_vol
            and cvd_ratio > CONFIG.classifier.absorption_min_cvd_ratio
            and price_delta_bps < CONFIG.classifier.absorption_max_price_delta_bps
        ):
            # High delta with suppressed price movement = Passive absorption!
            absorption_intensity = cvd_ratio / max(0.1, price_delta_bps + 0.2)
            prob_absorption = min(95.0, 45.0 + (absorption_intensity * 25.0))

            if cvd_usd > 0:
                absorption_side = "VENTAS (Limit Asks absorben compras agresivas)"
                notes.append("Absorción pasiva en el Ask: Compradores agresivos atrapados sin mover precio.")
            else:
                absorption_side = "COMPRAS (Limit Bids absorben ventas agresivas)"
                notes.append("Absorción pasiva en el Bid: Vendedores agresivos agotados en muro pasivo.")
        else:
            prob_absorption = min(45.0, cvd_ratio * 60.0)

        # -------------------------------------------------------------
        # 4. Patrón: Alta Competencia HFT / Ráfagas Violentas
        # Signature: High trades-per-second (TPS) and rapid aggressor alternation
        # -------------------------------------------------------------
        time_span_sec = max(0.05, (times[-1] - times[0]) / 1000.0)
        tps = len(trades_list) / time_span_sec
        # Side transitions (Buy followed by Sell or vice versa)
        side_transitions = np.sum(is_buy[:-1] != is_buy[1:])
        transition_rate = side_transitions / max(1, len(is_buy) - 1)

        prob_hft = 10.0
        hft_tps_threshold = CONFIG.classifier.hft_burst_tps_threshold
        if tps > hft_tps_threshold:
            prob_hft = min(95.0, (tps / 90.0) * 50.0 + (transition_rate * 45.0))
            if prob_hft > 75.0:
                notes.append(f"Ráfaga de alta frecuencia: {tps:.0f} TPS con alternancia agresiva bid/ask.")
        else:
            prob_hft = min(35.0, (tps / max(1.0, hft_tps_threshold)) * 35.0)

        # -------------------------------------------------------------
        # 5. Patrón: Breakout / Aceleración Unidireccional
        # Signature: High price delta + correlated aggressive volume sweep
        # -------------------------------------------------------------
        prob_breakout = 10.0
        if (
            price_delta_bps > CONFIG.classifier.breakout_min_price_delta_bps
            and cvd_ratio > CONFIG.classifier.breakout_min_cvd_ratio
        ):
            prob_breakout = min(94.0, (price_delta_bps / 15.0) * 50.0 + (cvd_ratio * 45.0))
            direction = "alcista" if cvd_usd > 0 else "bajista"
            notes.append(f"Breakout {direction} en desarrollo: Barrido agresivo de niveles.")
        else:
            prob_breakout = min(
                40.0,
                (price_delta_bps / max(1.0, CONFIG.classifier.breakout_min_price_delta_bps)) * 30.0,
            )

        # -------------------------------------------------------------
        # 6. Patrón: Trading Minorista Dominante (Retail)
        # Signature: Poisson-distributed intervals (high CV), small fragmented sizes, no institutional footprints
        # -------------------------------------------------------------
        institutional_pressure = max(prob_algo, prob_iceberg, prob_absorption, prob_hft)
        prob_retail = max(5.0, min(95.0, 100.0 - (institutional_pressure * 0.85)))

        # Determine dominant regime
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
            notes=notes[:3]  # Keep most relevant notes
        )
