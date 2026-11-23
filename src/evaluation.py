"""Shared evaluation utilities: loss-function factories for Algorithm 1,
RMSE computation, and the log-log convergence fit (Figure 4).
"""

from __future__ import annotations

import numpy as np
import torch

from src.simulation import simulate_financial, simulate_toy


def toy_loss_fn(p0: float, N: int, T: float, M: float, bound_mode: str, eta_stab: float | None):
    """Batch loss for the toy model: mean(P_N) estimates v(p) = p (g=1, Z=1)."""

    def _loss(net, batch_size: int, generator: torch.Generator) -> torch.Tensor:
        result = simulate_toy(
            net, p0, N, T, M, batch_size,
            bound_mode=bound_mode, eta_stab=eta_stab, generator=generator,
        )
        return result.p_final

    return _loss


def financial_loss_fn(
    p0: float, x0: float, mu: float, vol_fn, lambda_fn, g_fn,
    N: int, T: float, M: float, bound_mode: str, eta_stab: float | None, s_x: float,
):
    """Batch loss for the financial models: Z_N g(X_N) P_N^theta (equation (23))."""

    def _loss(net, batch_size: int, generator: torch.Generator) -> torch.Tensor:
        result = simulate_financial(
            net, p0, x0, mu, vol_fn, lambda_fn, g_fn, N, T, M, batch_size,
            bound_mode=bound_mode, eta_stab=eta_stab, s_x=s_x, generator=generator,
        )
        return result.z_final * g_fn(result.x_final) * result.p_final

    return _loss


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    """Root mean squared difference between two equally-shaped arrays."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def loglog_fit(n_values: np.ndarray, errors: np.ndarray) -> tuple[float, float]:
    """Slope and intercept of log(error) = intercept + slope * log(N)."""
    log_n = np.log(np.asarray(n_values, dtype=float))
    log_e = np.log(np.asarray(errors, dtype=float))
    slope, intercept = np.polyfit(log_n, log_e, 1)
    return float(slope), float(intercept)
