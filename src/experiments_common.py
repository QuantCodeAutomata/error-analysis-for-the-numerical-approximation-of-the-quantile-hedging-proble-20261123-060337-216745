"""Shared experiment drivers used by exp_3/exp_6 (fixed vs adaptive bound,
Figure 3) and exp_4/exp_7 (convergence under grid refinement, Figure 4), so
that the Black-Scholes and local-volatility cases of Section 4.3 share one
implementation of Algorithm 1 and of the reference computation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

import src.config as cfg
from src.evaluation import financial_loss_fn, loglog_fit
from src.models import ModelSpec
from src.networks import SharedControlNet
from src.reference import quantile_hedging_value
from src.simulation import simulate_financial
from src.training import train_control


@dataclass
class FixedVsAdaptiveResult:
    model_name: str
    p_grid: tuple
    reference: list
    fixed_estimate: list
    fixed_stderr: list
    adaptive_estimate: list
    adaptive_stderr: list
    N: int
    M: float
    eta_stab: float


def train_single(
    spec: ModelSpec, p0: float, N: int, M: float, bound_mode: str, eta_stab,
    n_iterations: int, batch_size: int, val_size: int, test_size: int,
    validation_frequency: int, seed: int,
):
    """Train one network for a given model, control bound and p0 (used
    directly by exp_3/exp_6 and by exp_5's feedback comparison)."""
    net = SharedControlNet(input_dim=3)
    loss_fn = financial_loss_fn(
        p0, spec.x0, spec.mu, spec.vol_fn, spec.lambda_fn, spec.g_fn,
        N, cfg.BS_T, M, bound_mode, eta_stab, cfg.S_X,
    )
    return train_control(
        net, loss_fn, n_iterations=n_iterations, batch_size=batch_size,
        val_size=val_size, test_size=test_size, learning_rate=cfg.ADAM_LR_FINANCIAL,
        grad_clip_norm=cfg.GRAD_CLIP_NORM, validation_frequency=validation_frequency,
        seed=seed,
    )


def run_fixed_vs_adaptive(
    spec: ModelSpec,
    p_grid: tuple,
    N: int,
    reference_samples: int,
    n_iterations: int,
    batch_size: int,
    val_size: int,
    test_size: int,
    validation_frequency: int = cfg.VALIDATION_FREQUENCY,
    eta_stab: float = cfg.FIN_ETA_STAB,
    seed: int = cfg.RANDOM_SEED,
) -> FixedVsAdaptiveResult:
    """Figure 3: quantile hedging value under a fixed vs an adaptive control
    bound, compared against the Monte Carlo / closed-form reference (26)."""
    h = cfg.BS_T / N
    M = h ** (-1.0 / 8.0)

    y_ref = spec.reference_sampler(reference_samples, seed)
    reference = [quantile_hedging_value(y_ref, p) for p in p_grid]

    fixed_est, fixed_se, adapt_est, adapt_se = [], [], [], []
    for i, p in enumerate(p_grid):
        if p in (0.0, 1.0):
            # At p=0 the quantile hedging value is 0 and at p=1 it is the full
            # super-replication price E[Z_T g(X_T)]: both are trivial cases of
            # (26) that coincide with the already-computed reference, not with
            # the literal probability p (that identity only holds in the toy
            # model of Section 4.2, where v(p)=p).
            fixed_est.append(reference[i]); fixed_se.append(0.0)
            adapt_est.append(reference[i]); adapt_se.append(0.0)
            continue
        res_fixed = train_single(
            spec, p, N, M, "fixed", None, n_iterations, batch_size, val_size,
            test_size, validation_frequency, seed=seed + 100 * i,
        )
        res_adapt = train_single(
            spec, p, N, M, "adaptive", eta_stab, n_iterations, batch_size, val_size,
            test_size, validation_frequency, seed=seed + 100 * i,
        )
        fixed_est.append(res_fixed.test_estimate); fixed_se.append(res_fixed.test_stderr)
        adapt_est.append(res_adapt.test_estimate); adapt_se.append(res_adapt.test_stderr)

    return FixedVsAdaptiveResult(
        model_name=spec.name, p_grid=p_grid, reference=reference,
        fixed_estimate=fixed_est, fixed_stderr=fixed_se,
        adaptive_estimate=adapt_est, adaptive_stderr=adapt_se,
        N=N, M=M, eta_stab=eta_stab,
    )


@dataclass
class ConvergenceResult:
    model_name: str
    grids: tuple
    reference_value: float
    mean_abs_error: list = field(default_factory=list)
    std_abs_error: list = field(default_factory=list)
    mean_martingale_defect: list = field(default_factory=list)
    slope: float = 0.0
    intercept: float = 0.0


def run_convergence_study(
    spec: ModelSpec,
    grids: tuple,
    x: float,
    p: float,
    reference_value: float,
    n_chains: int,
    n_iterations: int,
    batch_size: int,
    val_size: int,
    test_size: int,
    validation_frequency: int = cfg.VALIDATION_FREQUENCY,
    eta_stab: float = cfg.CONV_ETA_STAB,
    seed: int = cfg.RANDOM_SEED,
) -> ConvergenceResult:
    """Figure 4: mean absolute pricing error and martingale defect
    E[P_N]-p as a function of N, using only the adaptive bound (25), with
    warm-started chains (the shared architecture does not depend on N)."""
    result = ConvergenceResult(model_name=spec.name, grids=grids, reference_value=reference_value)

    chain_states = [None] * n_chains
    for N in grids:
        h = cfg.BS_T / N
        M = N ** (1.0 / 8.0)
        errors, defects = [], []
        for c in range(n_chains):
            net = SharedControlNet(input_dim=3)
            if chain_states[c] is not None:
                net.load_state_dict(chain_states[c])
            loss_fn = financial_loss_fn(
                p, spec.x0, spec.mu, spec.vol_fn, spec.lambda_fn, spec.g_fn,
                N, cfg.BS_T, M, "adaptive", eta_stab, cfg.S_X,
            )
            res = train_control(
                net, loss_fn, n_iterations=n_iterations, batch_size=batch_size,
                val_size=val_size, test_size=test_size, learning_rate=cfg.ADAM_LR_FINANCIAL,
                grad_clip_norm=cfg.GRAD_CLIP_NORM, validation_frequency=validation_frequency,
                seed=seed + 1000 * c + N,
            )
            chain_states[c] = res.net.state_dict()
            errors.append(abs(res.test_estimate - reference_value))

            # Martingale defect E[P_N] - p, estimated on an independent sample.
            gen = torch.Generator().manual_seed(seed + 5000 * c + N)
            with torch.no_grad():
                sim = simulate_financial(
                    res.net, p, spec.x0, spec.mu, spec.vol_fn, spec.lambda_fn, spec.g_fn,
                    N, cfg.BS_T, M, test_size, bound_mode="adaptive", eta_stab=eta_stab,
                    s_x=cfg.S_X, generator=gen,
                )
            defects.append(float(sim.p_final.mean()) - p)

        result.mean_abs_error.append(float(np.mean(errors)))
        result.std_abs_error.append(float(np.std(errors, ddof=1)) if n_chains > 1 else 0.0)
        result.mean_martingale_defect.append(float(np.mean(defects)))

    slope, intercept = loglog_fit(np.array(grids), np.array(result.mean_abs_error))
    result.slope, result.intercept = slope, intercept
    return result
