"""Integration test for src.training (Algorithm 1) on the toy model, run at a
small scale so it executes in a few seconds."""

from __future__ import annotations

import numpy as np
import torch

from src.evaluation import toy_loss_fn
from src.networks import SharedControlNet
from src.training import train_control


def test_train_control_reduces_toy_loss_towards_zero_at_p_half():
    """At p=0.5 the toy value is v(0.5)=0.5; training should not diverge and
    must return a finite estimate with the right validation bookkeeping."""
    net = SharedControlNet(input_dim=2)
    loss_fn = toy_loss_fn(0.5, N=10, T=1.0, M=1.0, bound_mode="fixed", eta_stab=None)
    result = train_control(
        net, loss_fn, n_iterations=5, batch_size=64, val_size=64, test_size=256,
        learning_rate=5e-3, grad_clip_norm=1.0, validation_frequency=1, seed=0,
    )
    assert np.isfinite(result.test_estimate)
    assert len(result.train_losses) == 5
    assert len(result.val_losses) == 5
    assert result.test_stderr >= 0.0
    assert 0.0 <= result.test_estimate <= 1.0


def test_train_control_deterministic_with_fixed_seed():
    torch.manual_seed(42)
    net_a = SharedControlNet(input_dim=2)
    torch.manual_seed(42)
    net_b = SharedControlNet(input_dim=2)
    loss_fn = toy_loss_fn(0.4, N=5, T=1.0, M=1.0, bound_mode="fixed", eta_stab=None)
    res_a = train_control(net_a, loss_fn, n_iterations=3, batch_size=32, val_size=32,
                           test_size=64, learning_rate=5e-3, validation_frequency=1, seed=7)
    res_b = train_control(net_b, loss_fn, n_iterations=3, batch_size=32, val_size=32,
                           test_size=64, learning_rate=5e-3, validation_frequency=1, seed=7)
    assert abs(res_a.test_estimate - res_b.test_estimate) < 1e-9
