"""Unit tests for src.evaluation (Algorithm 1 losses, RMSE, log-log fit)."""

from __future__ import annotations

import numpy as np
import torch

from src.evaluation import loglog_fit, rmse, toy_loss_fn
from src.networks import SharedControlNet


def test_rmse_zero_for_identical_arrays():
    a = np.array([1.0, 2.0, 3.0])
    assert rmse(a, a) == 0.0


def test_rmse_known_value():
    a = np.array([0.0, 0.0])
    b = np.array([3.0, 4.0])
    assert abs(rmse(a, b) - np.sqrt((9 + 16) / 2)) < 1e-9


def test_loglog_fit_recovers_known_power_law():
    n_values = np.array([10, 20, 30, 50, 75, 100])
    true_slope, true_const = -0.5, 2.0
    errors = true_const * n_values.astype(float) ** true_slope
    slope, intercept = loglog_fit(n_values, errors)
    assert abs(slope - true_slope) < 1e-6
    assert abs(np.exp(intercept) - true_const) < 1e-6


def test_toy_loss_fn_returns_per_path_tensor():
    torch.manual_seed(0)
    net = SharedControlNet(input_dim=2)
    loss_fn = toy_loss_fn(0.3, N=10, T=1.0, M=1.0, bound_mode="fixed", eta_stab=None)
    gen = torch.Generator().manual_seed(1)
    out = loss_fn(net, batch_size=128, generator=gen)
    assert out.shape == (128,)
    # Zero control: the loss (= P_N) must equal p0 for every path.
    assert torch.allclose(out, torch.full_like(out, 0.3))
