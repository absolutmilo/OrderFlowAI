"""
Gaussian Hidden Markov Model for Order Flow Regime Classification.
Provides online forward filtering (belief state tracking) in pure NumPy for ultra-low latency.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np


class OnlineGaussianHMM:
    """
    Maintains the online filtered belief state alpha_t for streaming inference.
    Computational complexity per tick: O(K^2 + K*D), ~10 microseconds in NumPy.
    """

    def __init__(
        self,
        n_states: int,
        state_names: List[str],
        feature_names: List[str],
        transition_matrix: np.ndarray,
        emission_means: np.ndarray,
        emission_vars: np.ndarray,
        initial_probs: np.ndarray,
        scaler_mean: np.ndarray,
        scaler_std: np.ndarray,
    ):
        self.n_states = n_states
        self.state_names = state_names
        self.feature_names = feature_names
        self.A = np.array(transition_matrix, dtype=np.float64)  # Shape (K, K)
        self.means = np.array(emission_means, dtype=np.float64)  # Shape (K, D)
        self.vars = np.maximum(0.01, np.array(emission_vars, dtype=np.float64))  # Shape (K, D)
        self.pi = np.array(initial_probs, dtype=np.float64)  # Shape (K,)
        self.scaler_mean = np.array(scaler_mean, dtype=np.float64)  # Shape (D,)
        self.scaler_std = np.maximum(1e-6, np.array(scaler_std, dtype=np.float64))  # Shape (D,)

        # Precompute constants for log-Gaussian evaluations
        # log_det: 0.5 * sum(log(2*pi*var)) for each state
        self.log_norm_const = 0.5 * np.sum(np.log(2.0 * np.pi * self.vars), axis=1)  # Shape (K,)

        # Current belief state (filtered probability distribution over states)
        self.alpha: np.ndarray = self.pi.copy()
        self.is_initialized: bool = False

    def reset(self) -> None:
        """Resets belief state back to prior distribution."""
        self.alpha = self.pi.copy()
        self.is_initialized = False

    def transform_features(self, raw_features: np.ndarray) -> np.ndarray:
        """Applies Z-score standardization to raw observation vector."""
        return (raw_features - self.scaler_mean) / self.scaler_std

    def compute_log_emission(self, x_norm: np.ndarray) -> np.ndarray:
        """
        Computes log P(x | S_k) for all states k = 0..K-1.
        x_norm: 1D array of shape (D,)
        Returns: 1D array of shape (K,)
        """
        diff = x_norm - self.means  # Shape (K, D)
        quad = np.sum((diff ** 2) / self.vars, axis=1)  # Shape (K,)
        return - (self.log_norm_const + 0.5 * quad)

    def step(self, raw_features: np.ndarray) -> np.ndarray:
        """
        Online Forward Step:
        alpha_t(j) = P(S_t = j | x_{1:t}) proportional to P(x_t | S_t = j) * sum_i [alpha_{t-1}(i) * A_ij]
        Returns: normalized posterior probability distribution over states (sums to 1.0).
        """
        x_norm = self.transform_features(raw_features)
        log_b = self.compute_log_emission(x_norm)

        # Numerical stability: shift log probabilities by max
        max_log_b = np.max(log_b)
        b = np.exp(log_b - max_log_b)

        if not self.is_initialized:
            # First observation
            unnorm_alpha = self.pi * b
            self.is_initialized = True
        else:
            # Predict step: prior for time t given history up to t-1
            predicted = self.alpha @ self.A  # Shape (K,)
            # Update step: condition on new emission
            unnorm_alpha = predicted * b

        total = np.sum(unnorm_alpha)
        if total > 1e-12:
            self.alpha = unnorm_alpha / total
            # Add small probability floor to prevent dead states during infinite streaming
            self.alpha = np.maximum(1e-4, self.alpha)
            self.alpha /= np.sum(self.alpha)
        else:
            # Fallback in case of extreme numerical underflow
            self.alpha = np.full(self.n_states, 1.0 / self.n_states)

        return self.alpha.copy()

    def get_state_probabilities(self) -> Dict[str, float]:
        """Returns the current state probabilities mapped to state names."""
        return {
            self.state_names[i]: float(self.alpha[i])
            for i in range(self.n_states)
        }

    def save(self, filepath: Path) -> None:
        """Serializes model parameters to a JSON file."""
        data = {
            "model_type": "GaussianHMM",
            "n_states": self.n_states,
            "state_names": self.state_names,
            "feature_names": self.feature_names,
            "scaler_mean": self.scaler_mean.tolist(),
            "scaler_std": self.scaler_std.tolist(),
            "initial_probs": self.pi.tolist(),
            "transition_matrix": self.A.tolist(),
            "emission_means": self.means.tolist(),
            "emission_vars": self.vars.tolist(),
        }
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: Path) -> "OnlineGaussianHMM":
        """Loads and initializes an OnlineGaussianHMM from a JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        return cls(
            n_states=data["n_states"],
            state_names=data["state_names"],
            feature_names=data["feature_names"],
            transition_matrix=np.array(data["transition_matrix"]),
            emission_means=np.array(data["emission_means"]),
            emission_vars=np.array(data["emission_vars"]),
            initial_probs=np.array(data["initial_probs"]),
            scaler_mean=np.array(data["scaler_mean"]),
            scaler_std=np.array(data["scaler_std"]),
        )
