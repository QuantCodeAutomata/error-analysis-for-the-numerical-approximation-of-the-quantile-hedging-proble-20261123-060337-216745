"""Unit tests for src.reference (Section 4.3, formula (26))."""

from __future__ import annotations

import numpy as np

from src.reference import (
    black_scholes_terminal_samples,
    local_vol_fn_numpy,
    local_vol_terminal_samples,
    quantile_hedging_value,
)


def test_quantile_hedging_value_boundary_cases():
    y = np.array([0.0, 0.0, 1.0, 2.0, 5.0])
    assert quantile_hedging_value(y, 0.0) == 0.0
    assert quantile_hedging_value(y, 1.0) == float(np.mean(y))


def test_quantile_hedging_value_monotone_in_p():
    rng = np.random.default_rng(0)
    y = rng.exponential(size=5000)
    values = [quantile_hedging_value(y, p) for p in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)]
    assert all(b >= a - 1e-9 for a, b in zip(values, values[1:]))


def test_quantile_hedging_value_rejects_out_of_range_p():
    y = np.array([1.0, 2.0])
    try:
        quantile_hedging_value(y, 1.5)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for p outside [0, 1]")


def test_black_scholes_samples_are_nonnegative_put_payoff_times_density():
    y = black_scholes_terminal_samples(100.0, 0.1, 0.2, 1.0, 100.0, 20000, seed=1)
    assert np.all(y >= 0.0)
    assert y.mean() > 0.0


def test_quantile_hedging_matches_super_replication_price():
    """At p=1, formula (26) must reduce to the super-replication price
    E[Z_T g(X_T)], the full cost of a sure hedge (Section 4.3)."""
    y = black_scholes_terminal_samples(100.0, 0.1, 0.2, 1.0, 100.0, 200000, seed=2)
    assert abs(quantile_hedging_value(y, 1.0) - y.mean()) < 1e-9


def test_local_vol_terminal_samples_consistent_with_payoff_bound():
    ell = local_vol_fn_numpy(0.16, 0.10)
    y = local_vol_terminal_samples(100.0, 0.1, ell, 100.0, 1.0, 2000, 50, seed=3)
    assert np.all(y >= 0.0)
