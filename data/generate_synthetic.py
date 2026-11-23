"""Synthetic-data generation for the quantile hedging replication.

Every experiment in this repository is driven entirely by simulated Brownian
motion: there is no external market data (Section 4, "Numerical
experiments"). This module is the single place that draws the underlying
random numbers (Gaussian increments), both as torch tensors (used inside the
projected neural scheme of src/simulation.py and src/training.py) and as
numpy arrays (used by the closed-form / Monte Carlo references of
src/reference.py).
"""

from __future__ import annotations

import numpy as np
import torch


def brownian_increments(
    batch_size: int,
    sqrt_h: float,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """One time-step of Brownian increments, shape (batch_size,).

    ``dW_n = sqrt(h) * xi``, with xi standard normal (independent across the
    batch and across calls when a persistent ``generator`` is supplied).
    """
    xi = torch.randn(batch_size, generator=generator, device=device)
    return xi * sqrt_h


def standard_normal_samples(n_samples: int, seed: int) -> np.ndarray:
    """i.i.d. standard normal sample, used to build Y = Z_T g(X_T) directly
    (Black-Scholes closed-form reference) or to drive an Euler scheme
    (local-volatility Monte Carlo reference)."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n_samples)


def brownian_increments_numpy(n_paths: int, sqrt_h: float, rng: np.random.Generator) -> np.ndarray:
    """One Euler time-step of Brownian increments, shape (n_paths,)."""
    return rng.standard_normal(n_paths) * sqrt_h
