"""Explicit optimal feedback control and probability martingale for the
quantile hedging problem in the Black-Scholes model (Section 4.3, used in
exp_5). These closed-form objects exist because, in Black-Scholes, the event
{Z_T g(X_T) <= q_p} is equivalent to a half-line event for the driving
Brownian motion.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def terminal_threshold(T: float, p: float) -> float:
    """c_p such that P_T^* = 1_{W_T >= c_p}, with c_p = -sqrt(T) * Phi^{-1}(p)."""
    return -np.sqrt(T) * norm.ppf(p)


def optimal_probability_process(w: np.ndarray, t: np.ndarray, T: float, p: float) -> np.ndarray:
    """P_t^* = Phi((W_t - c_p) / sqrt(T - t)) for 0 <= t < T (Section 4.3).

    ``w`` has shape (n_steps, batch) with w[k] = W_{t_k}; ``t`` has shape
    (n_steps,) with t[k] = t_k < T. Returns the same shape as ``w``.
    """
    c_p = terminal_threshold(T, p)
    denom = np.sqrt(np.clip(T - t, 1e-12, None))[:, None]
    return norm.cdf((w - c_p) / denom)


def optimal_feedback(t: np.ndarray, p: np.ndarray, T: float) -> np.ndarray:
    """alpha*(t, p) = phi(Phi^{-1}(p)) / sqrt(T - t) for t < T (Section 4.3),

    extended by 0 at p = 0 and p = 1.
    """
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    denom = np.sqrt(np.clip(T - t, 1e-12, None))
    feedback = norm.pdf(norm.ppf(p)) / denom
    feedback = np.where((p <= 1e-12) | (p >= 1.0 - 1e-12), 0.0, feedback)
    return feedback
