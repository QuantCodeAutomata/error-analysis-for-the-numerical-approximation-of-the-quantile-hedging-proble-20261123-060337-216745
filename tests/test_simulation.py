"""Unit tests for src.simulation (Section 4, equations (19)-(22))."""

from __future__ import annotations

import torch

from src.networks import SharedControlNet
from src.simulation import (
    _project_step,
    black_scholes_lambda_fn,
    black_scholes_vol_fn,
    local_vol_fn,
    put_payoff,
    simulate_financial,
    simulate_toy,
)


def test_zero_control_toy_is_martingale_in_expectation():
    """With the zero-initialized network (alpha = 0), P_n never moves, so
    P_N = p0 exactly (no projection can occur)."""
    torch.manual_seed(0)
    net = SharedControlNet(input_dim=2)
    result = simulate_toy(net, p0=0.37, N=20, T=1.0, M=1.0, batch_size=512)
    assert torch.allclose(result.p_final, torch.full_like(result.p_final, 0.37))


def test_toy_boundary_values_are_trivial():
    torch.manual_seed(0)
    net = SharedControlNet(input_dim=2)
    for p0 in (0.0, 1.0):
        result = simulate_toy(net, p0=p0, N=10, T=1.0, M=1.0, batch_size=64)
        assert torch.allclose(result.p_final, torch.full_like(result.p_final, p0))


def test_projected_process_stays_in_unit_interval():
    torch.manual_seed(3)
    net = SharedControlNet(input_dim=2)
    with torch.no_grad():
        net.net[-1].weight.add_(2.0)
    result = simulate_toy(net, p0=0.5, N=50, T=1.0, M=5.0, batch_size=4096,
                           bound_mode="fixed")
    assert torch.all(result.p_final >= 0.0) and torch.all(result.p_final <= 1.0)


def test_project_step_absorbs_and_freezes():
    active = torch.tensor([True, True, True])
    p_current = torch.tensor([0.5, 0.5, 0.5])
    candidate = torch.tensor([1.2, -0.3, 0.6])
    p_next, active_next = _project_step(candidate, active, p_current)
    assert torch.allclose(p_next, torch.tensor([1.0, 0.0, 0.6]))
    assert active_next.tolist() == [False, False, True]
    # Once inactive, the state must freeze at the next call.
    p_frozen, active_frozen = _project_step(torch.tensor([0.9, 0.9, 0.9]), active_next, p_next)
    assert torch.allclose(p_frozen, torch.tensor([1.0, 0.0, 0.9]))
    assert active_frozen.tolist() == [False, False, True]


def test_simulate_financial_no_lookahead_control_uses_past_state():
    """The control at step n must depend only on (t_n, X_n, P_n); perturbing
    the terminal payoff must not retroactively change recorded controls."""
    torch.manual_seed(4)
    net = SharedControlNet(input_dim=3)
    with torch.no_grad():
        net.net[-1].weight.add_(1.0)
    gen1 = torch.Generator().manual_seed(123)
    res1 = simulate_financial(
        net, p0=0.5, x0=100.0, mu=0.1, vol_fn=black_scholes_vol_fn(0.2),
        lambda_fn=black_scholes_lambda_fn(0.1, 0.2), g_fn=put_payoff(100.0),
        N=10, T=1.0, M=1.0, batch_size=256, bound_mode="fixed",
        generator=gen1, record_path=True,
    )
    gen2 = torch.Generator().manual_seed(123)
    res2 = simulate_financial(
        net, p0=0.5, x0=100.0, mu=0.1, vol_fn=black_scholes_vol_fn(0.2),
        lambda_fn=black_scholes_lambda_fn(0.1, 0.2), g_fn=put_payoff(999.0),  # different strike
        N=10, T=1.0, M=1.0, batch_size=256, bound_mode="fixed",
        generator=gen2, record_path=True,
    )
    # The control path (which only depends on (t, X, P), not on g) must match
    # exactly even though the terminal payoff differs.
    assert torch.allclose(res1.path["alpha"], res2.path["alpha"])
    assert torch.allclose(res1.path["x"], res2.path["x"])


def test_local_vol_fn_matches_definition():
    x = torch.tensor([0.0, 50.0, 100.0, 150.0, 300.0])
    ell = local_vol_fn(0.16, 0.10, center=100.0)(x)
    expected = torch.tensor([0.16 + 0.10, 0.16 + 0.05, 0.16, 0.16 + 0.05, 0.16 + 0.10])
    assert torch.allclose(ell, expected, atol=1e-6)
