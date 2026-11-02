"""
OrderFlowAI - Gaussian HMM Training Pipeline.
Extracts observations from data/orderflow.sqlite3, trains a 4-state Gaussian Hidden Markov Model
using Baum-Welch (EM) with microstructural warm-start initialization, and exports the model.
"""

import json
import logging
import sqlite3
import sys
from pathlib import Path
from typing import Tuple
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.hmm_model import OnlineGaussianHMM

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("OrderFlowAI.HMMTrainer")

FEATURE_NAMES = [
    "inter_arrival_cv",
    "log_volume_sec",
    "cvd_ratio",
    "price_delta_bps",
    "log_tps",
]

STATE_NAMES = [
    "Algoritmico",
    "Iceberg",
    "Absorcion",
    "Retail",
]


def extract_features_from_db(db_path: Path) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Reads records from SQLite and transforms them into an observation matrix X.
    """
    if not db_path.exists():
        raise FileNotFoundError(f"Base de datos no encontrada en {db_path}")

    logger.info(f"Conectando a {db_path}...")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute(
        """
        SELECT observed_at_ms, features_json
        FROM market_observations
        ORDER BY observed_at_ms ASC
        """
    )
    rows = cur.fetchall()
    conn.close()

    total_rows = len(rows)
    logger.info(f"Observaciones extraídas: {total_rows} filas.")
    if total_rows < 100:
        raise ValueError(f"Insuficientes datos para entrenamiento ({total_rows} filas).")

    timestamps = []
    features_list = []

    for ts, f_json in rows:
        f = json.loads(f_json)

        tot_vol = max(1.0, float(f.get("buy_volume_usd", 0.0) + f.get("sell_volume_usd", 0.0)))
        cvd_usd = float(f.get("cvd_usd", 0.0))
        cvd_ratio = min(1.0, abs(cvd_usd) / tot_vol)

        vol_sec = max(1.0, float(f.get("volume_per_second_usd", 1.0)))
        log_vol = np.log10(vol_sec)

        inter_cv = min(5.0, max(0.1, float(f.get("inter_arrival_cv", 1.0))))
        price_delta = min(30.0, abs(float(f.get("price_delta_bps", 0.0))))

        tps = max(0.1, min(200.0, float(f.get("trades_per_second", 1.0))))
        log_tps = np.log10(tps)

        timestamps.append(ts)
        features_list.append([inter_cv, log_vol, cvd_ratio, price_delta, log_tps])

    X = np.array(features_list, dtype=np.float64)
    ts_array = np.array(timestamps, dtype=np.int64)

    return X, ts_array, total_rows


def train_gaussian_hmm(
    X: np.ndarray,
    n_states: int = 4,
    max_iter: int = 12,
    tol: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Trains a Gaussian HMM using Baum-Welch (EM) with diagonal covariance.
    Uses microstructurally informed warm-start to ensure stable state labeling.
    """
    N, D = X.shape
    K = n_states

    # Z-score standardization
    mu_global = np.mean(X, axis=0)
    std_global = np.maximum(1e-6, np.std(X, axis=0))
    X_norm = (X - mu_global) / std_global

    # Warm-start prior means in normalized space:
    # Feature order: [inter_arrival_cv, log_vol, cvd_ratio, price_delta, log_tps]
    # State 0: Algoritmico (Regular timing: low inter_cv, moderate vol, low price delta)
    # State 1: Iceberg     (Moderate cv, higher vol, low delta, high tps)
    # State 2: Absorcion   (High cvd_ratio, low delta, highest vol)
    # State 3: Retail      (High inter_cv, lower vol, scattered delta, low tps)
    means = np.array([
        [-0.8,  0.0, -0.2, -0.4,  0.0],  # State 0: Algoritmico
        [ 0.0,  0.8,  0.0, -0.2,  0.6],  # State 1: Iceberg
        [ 0.2,  0.7,  1.2, -0.8,  0.5],  # State 2: Absorcion
        [ 0.8, -0.8, -0.4,  0.4, -0.6],  # State 3: Retail
    ], dtype=np.float64)

    vars_diag = np.ones((K, D), dtype=np.float64) * 0.8

    # Prior transition matrix with strong temporal persistence (90% on-diagonal)
    A = np.full((K, K), 0.10 / (K - 1), dtype=np.float64)
    np.fill_diagonal(A, 0.90)

    pi = np.full(K, 1.0 / K, dtype=np.float64)

    def log_gaussian_emissions(X_data, means_arr, vars_arr):
        log_B = np.zeros((N, K), dtype=np.float64)
        for k in range(K):
            diff = X_data - means_arr[k]
            log_det = np.sum(np.log(2.0 * np.pi * vars_arr[k]))
            quad = np.sum((diff ** 2) / vars_arr[k], axis=1)
            log_B[:, k] = -0.5 * (log_det + quad)
        return log_B

    prev_ll = -np.inf
    logger.info(f"Iniciando Baum-Welch EM en {N} observaciones con {K} estados...")

    for it in range(max_iter):
        # 1. Emission log-probabilities
        log_B = log_gaussian_emissions(X_norm, means, vars_diag)

        # 2. Scaled Forward Algorithm
        alpha = np.zeros((N, K), dtype=np.float64)
        c = np.zeros(N, dtype=np.float64)

        b0 = np.exp(log_B[0] - np.max(log_B[0]))
        alpha[0] = pi * b0
        c[0] = np.sum(alpha[0])
        alpha[0] /= max(1e-12, c[0])

        for t in range(1, N):
            b_t = np.exp(log_B[t] - np.max(log_B[t]))
            alpha[t] = (alpha[t - 1] @ A) * b_t
            c[t] = np.sum(alpha[t])
            if c[t] < 1e-12:
                alpha[t] = 1.0 / K
                c[t] = 1.0
            else:
                alpha[t] /= c[t]

        # 3. Scaled Backward Algorithm
        beta = np.zeros((N, K), dtype=np.float64)
        beta[-1] = 1.0

        for t in range(N - 2, -1, -1):
            b_next = np.exp(log_B[t + 1] - np.max(log_B[t + 1]))
            beta[t] = (A @ (b_next * beta[t + 1])) / max(1e-12, c[t + 1])

        # 4. Posterior state probabilities (gamma)
        gamma = alpha * beta
        gamma /= np.maximum(1e-12, np.sum(gamma, axis=1, keepdims=True))

        # 5. Transition counts (xi) & M-step update of A
        b_all = np.exp(log_B[1:] - np.max(log_B[1:], axis=1, keepdims=True))
        term = (b_all * beta[1:]) / np.maximum(1e-12, c[1:, None])
        xi_sum = alpha[:-1].T @ term * A
        A = xi_sum / np.maximum(1e-12, np.sum(xi_sum, axis=1, keepdims=True))

        # Update initial state distribution to stationary marginals across observations
        gamma_sum = np.sum(gamma, axis=0)  # Shape (K,)
        pi = gamma_sum / N

        # Update means and variances
        for k in range(K):
            denom = max(1e-6, gamma_sum[k])
            means[k] = np.sum(gamma[:, k:k + 1] * X_norm, axis=0) / denom
            diff = X_norm - means[k]
            vars_diag[k] = np.sum(gamma[:, k:k + 1] * (diff ** 2), axis=0) / denom
            # Floor variance to prevent collapse onto single points
            vars_diag[k] = np.maximum(0.05, vars_diag[k])

        current_ll = float(np.sum(np.log(np.maximum(1e-12, c))))
        delta_ll = current_ll - prev_ll
        state_distribution = (gamma_sum / N) * 100.0

        logger.info(
            f"Iter {it + 1:02d}/{max_iter:02d} | Log-Likelihood: {current_ll:12.1f} (Δ={delta_ll:8.1f}) | "
            f"Distribución: [{', '.join(f'{s}: {p:.1f}%' for s, p in zip(STATE_NAMES, state_distribution))}]"
        )

        if it > 0 and 0 <= delta_ll < tol:
            logger.info("Convergencia alcanzada.")
            break
        prev_ll = current_ll

    return A, means, vars_diag, pi, mu_global, std_global


def main():
    db_path = PROJECT_ROOT / "data" / "orderflow.sqlite3"
    output_model_path = PROJECT_ROOT / "data" / "hmm_regime_model.json"

    logger.info("=================================================================")
    logger.info("   OrderFlowAI - Pipeline de Entrenamiento Gaussian HMM (Fase 1) ")
    logger.info("=================================================================")

    # 1. Extracción
    X, timestamps, total_rows = extract_features_from_db(db_path)

    # 2. Entrenamiento
    A, means, vars_diag, pi, mu_g, std_g = train_gaussian_hmm(
        X,
        n_states=len(STATE_NAMES),
        max_iter=12,
    )

    # 3. Empaquetar y Guardar
    hmm_model = OnlineGaussianHMM(
        n_states=len(STATE_NAMES),
        state_names=STATE_NAMES,
        feature_names=FEATURE_NAMES,
        transition_matrix=A,
        emission_means=means,
        emission_vars=vars_diag,
        initial_probs=pi,
        scaler_mean=mu_g,
        scaler_std=std_g,
    )

    hmm_model.save(output_model_path)
    logger.info(f"Modelo HMM exportado exitosamente a: {output_model_path}")

    # 4. Imprimir Resumen Microestructural
    print("\n" + "=" * 65)
    print("  MATRIZ DE TRANSICIÓN APRENDIDA (A)")
    print("=" * 65)
    header = f"{'De / A':15s}" + "".join(f"{s:>12s}" for s in STATE_NAMES)
    print(header)
    for i, name in enumerate(STATE_NAMES):
        row_str = "".join(f"{A[i, j] * 100:11.1f}%" for j in range(len(STATE_NAMES)))
        print(f"{name:15s}{row_str}")

    print("\n" + "=" * 65)
    print("  CENTROIDES DE EMISIÓN (Escala Original)")
    print("=" * 65)
    means_original = means * std_g + mu_g
    for i, name in enumerate(STATE_NAMES):
        feats_str = ", ".join(f"{FEATURE_NAMES[j]}={means_original[i, j]:.2f}" for j in range(len(FEATURE_NAMES)))
        print(f"[{name:11s}]: {feats_str}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
