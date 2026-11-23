"""Reference quantile hedging values via the closed-form / Monte Carlo
formula (26) of Section 4.3:

    v(0, x, p) = E[Z_T g(X_T) 1_{Z_T g(X_T) < q_p}] + q_p (p - P(Z_T g(X_T) < q_p)),

where q_p is the p-quantile of Z_T g(X_T).
"""

from __future__ import annotations

import numpy as np

from data.generate_synthetic import brownian_increments_numpy, standard_normal_samples


def quantile_hedging_value(y_samples: np.ndarray, p: float) -> float:
    """Evaluate formula (26) from samples of Y = Z_T g(X_T)."""
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must lie in [0, 1]")
    if p == 0.0:
        return 0.0
    if p == 1.0:
        return float(np.mean(y_samples))
    q_p = np.quantile(y_samples, p)
    mask = y_samples < q_p
    term1 = np.mean(y_samples * mask)
    prob_below = np.mean(mask)
    return float(term1 + q_p * (p - prob_below))


def black_scholes_terminal_samples(
    x0: float,
    mu: float,
    sigma: float,
    T: float,
    strike: float,
    n_samples: int,
    seed: int,
) -> np.ndarray:
    """Y = Z_T g(X_T) in closed form (Section 4.3), g a put payoff.

    X_T = x0 * exp((mu - sigma^2/2) T + sigma W_T),
    Z_T = exp(lambda W_T - lambda^2 T / 2), lambda = -mu/sigma,
    with W_T ~ N(0, T) sampled directly (no time discretization).
    """
    xi = standard_normal_samples(n_samples, seed)
    w_T = np.sqrt(T) * xi
    lam = -mu / sigma
    x_T = x0 * np.exp((mu - 0.5 * sigma**2) * T + sigma * w_T)
    z_T = np.exp(lam * w_T - 0.5 * lam**2 * T)
    payoff = np.maximum(strike - x_T, 0.0)
    return z_T * payoff


def local_vol_fn_numpy(base: float, slope: float, center: float = 100.0):
    """Numpy counterpart of ``src.simulation.local_vol_fn`` for the Monte
    Carlo reference: ell(x) = base + slope * min(|x/center - 1|, 1)."""

    def _vol(x: np.ndarray) -> np.ndarray:
        return base + slope * np.clip(np.abs(x / center - 1.0), None, 1.0)

    return _vol


def local_vol_terminal_samples(
    x0: float,
    mu: float,
    ell_fn,
    strike: float,
    T: float,
    n_paths: int,
    n_steps: int,
    seed: int,
) -> np.ndarray:
    """Y = Z_T g(X_T) by Euler simulation of the local-volatility model
    (Section 4.3): dX = mu X dt + X ell(X) dW, dZ = Z lambda(X) dW with
    lambda(x) = -mu/ell(x), b = 0.
    """
    rng = np.random.default_rng(seed)
    h = T / n_steps
    sqrt_h = np.sqrt(h)
    x = np.full(n_paths, float(x0))
    z = np.ones(n_paths)
    for _ in range(n_steps):
        dW = brownian_increments_numpy(n_paths, sqrt_h, rng)
        ell = ell_fn(x)
        lam = -mu / ell
        x_next = x + mu * x * h + x * ell * dW
        z = z * np.exp(-0.5 * lam**2 * h + lam * dW)
        x = x_next
    payoff = np.maximum(strike - x, 0.0)
    return z * payoff
