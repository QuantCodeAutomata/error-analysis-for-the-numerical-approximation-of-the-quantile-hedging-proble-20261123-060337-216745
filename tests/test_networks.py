"""Unit tests for src.networks (Section 4.1, equations (21)-(22))."""

from __future__ import annotations

import torch

from src.networks import SharedControlNet, zero_control_state_dict


def test_zero_initial_control():
    """The network must start at the feasible zero control (Section 4.1)."""
    torch.manual_seed(0)
    net = SharedControlNet(input_dim=2)
    x = torch.randn(32, 2)
    out = net(x)
    assert torch.allclose(out, torch.zeros_like(out))


def test_output_bounded_in_unit_interval():
    torch.manual_seed(1)
    net = SharedControlNet(input_dim=3)
    # Perturb the last layer so the output is not trivially zero.
    with torch.no_grad():
        net.net[-1].weight.add_(0.5)
        net.net[-1].bias.add_(0.1)
    x = torch.randn(256, 3) * 5.0
    out = net(x)
    assert torch.all(out >= -1.0 - 1e-6) and torch.all(out <= 1.0 + 1e-6)


def test_shared_architecture_across_input_dims():
    net2 = SharedControlNet(input_dim=2)
    net3 = SharedControlNet(input_dim=3)
    assert net2.net[0].in_features == 2
    assert net3.net[0].in_features == 3
    for net in (net2, net3):
        assert net.net[0].out_features == 32
        assert net.net[2].out_features == 32


def test_zero_control_state_dict_zeros_last_layer():
    torch.manual_seed(2)
    net = SharedControlNet(input_dim=2)
    with torch.no_grad():
        net.net[-1].weight.add_(1.0)
    state = zero_control_state_dict(net)
    for key, value in state.items():
        if "net.4." in key:
            assert torch.all(value == 0.0)
