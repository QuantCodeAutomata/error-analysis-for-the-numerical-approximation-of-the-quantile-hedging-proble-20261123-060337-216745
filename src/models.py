"""Model specifications shared by the Black-Scholes and local-volatility
experiments (Section 4.3): drift, volatility, Girsanov kernel, payoff, and
the matching closed-form / Monte Carlo reference sampler.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

import src.config as cfg
from src.reference import (
    black_scholes_terminal_samples,
    local_vol_fn_numpy,
    local_vol_terminal_samples,
)
from src.simulation import (
    black_scholes_lambda_fn,
    black_scholes_vol_fn,
    local_vol_fn,
    local_vol_lambda_fn,
    put_payoff,
)


@dataclass
class ModelSpec:
    name: str
    x0: float
    mu: float
    vol_fn: Callable
    lambda_fn: Callable
    g_fn: Callable
    reference_sampler: Callable[[int, int], np.ndarray]  # (n_samples, seed) -> Y


def _bs_sampler(n_samples: int, seed: int) -> np.ndarray:
    return black_scholes_terminal_samples(
        cfg.BS_X0, cfg.BS_MU, cfg.BS_SIGMA, cfg.BS_T, cfg.BS_STRIKE, n_samples, seed
    )


def _lv_sampler_factory(n_steps: int):
    ell_np = local_vol_fn_numpy(cfg.LV_ELL_BASE, cfg.LV_ELL_SLOPE)

    def _sampler(n_paths: int, seed: int) -> np.ndarray:
        return local_vol_terminal_samples(
            cfg.BS_X0, cfg.BS_MU, ell_np, cfg.BS_STRIKE, cfg.BS_T, n_paths, n_steps, seed
        )

    return _sampler


def black_scholes_spec() -> ModelSpec:
    """Black-Scholes put (Section 4.3): constant volatility and Girsanov
    kernel lambda = -mu/sigma. The reference uses the closed-form sample of
    Y = Z_T g(X_T) (formula (26)), evaluated without time discretization."""
    return ModelSpec(
        name="black_scholes",
        x0=cfg.BS_X0,
        mu=cfg.BS_MU,
        vol_fn=black_scholes_vol_fn(cfg.BS_SIGMA),
        lambda_fn=black_scholes_lambda_fn(cfg.BS_MU, cfg.BS_SIGMA),
        g_fn=put_payoff(cfg.BS_STRIKE),
        reference_sampler=_bs_sampler,
    )


def local_vol_spec(reference_steps: int | None = None) -> ModelSpec:
    """Lipschitz local-volatility put (Section 4.3): piecewise-linear
    volatility ell(x) = 0.16 + 0.10 min(|x/100-1|, 1), Girsanov kernel
    lambda(x) = -mu/ell(x). The reference is a Monte Carlo Euler simulation
    with ``reference_steps`` time steps (formula (26))."""
    steps = reference_steps or cfg.LV_REFERENCE_STEPS
    ell = local_vol_fn(cfg.LV_ELL_BASE, cfg.LV_ELL_SLOPE)
    return ModelSpec(
        name="local_vol",
        x0=cfg.BS_X0,
        mu=cfg.BS_MU,
        vol_fn=ell,
        lambda_fn=local_vol_lambda_fn(cfg.BS_MU, ell),
        g_fn=put_payoff(cfg.BS_STRIKE),
        reference_sampler=_lv_sampler_factory(steps),
    )
