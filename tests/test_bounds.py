"""Unit tests for src.bounds (Section 4.2, equation (25))."""

from __future__ import annotations

import math

import pytest
import torch

from src.bounds import bound_value


def test_fixed_bound_is_constant():
    p = torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0])
    out = bound_value(p, M=1.7783, h=0.01, mode="fixed")
    assert torch.allclose(out, torch.full_like(p, 1.7783))


def test_adaptive_bound_shrinks_near_boundary():
    h = 0.01
    M = 1.7783
    p = torch.tensor([0.001, 0.5, 0.999])
    out = bound_value(p, M=M, h=h, mode="adaptive", eta_stab=0.6)
    # Near the boundary the bound must be strictly smaller than at p=0.5,
    # and never exceed M (equation 25).
    assert out[0] < out[1]
    assert out[2] < out[1]
    assert torch.all(out <= M + 1e-9)
    assert torch.all(out >= 0.0)


def test_adaptive_bound_equals_fixed_when_far_from_boundary():
    # For p far enough from {0, 1}, eta_stab * d(p) / sqrt(h) >= M, so the
    # adaptive bound saturates at the fixed level M.
    h = 0.01
    M = 0.001
    p = torch.tensor([0.5])
    out = bound_value(p, M=M, h=h, mode="adaptive", eta_stab=0.6)
    assert math.isclose(float(out[0]), M, rel_tol=1e-6)


def test_adaptive_requires_eta_stab():
    with pytest.raises(ValueError):
        bound_value(torch.tensor([0.5]), M=1.0, h=0.01, mode="adaptive")


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        bound_value(torch.tensor([0.5]), M=1.0, h=0.01, mode="bogus")
