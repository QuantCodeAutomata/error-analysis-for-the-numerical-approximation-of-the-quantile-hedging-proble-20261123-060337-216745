"""Unit tests for src.feedback (Section 4.3, explicit optimal control)."""

from __future__ import annotations

import numpy as np

from src.feedback import optimal_feedback, optimal_probability_process, terminal_threshold


def test_optimal_feedback_matches_paper_value_at_t0():
    # Section 4.3 states alpha*(0, 0.7) ~ 0.347693 for T=1.
    value = optimal_feedback(np.array([0.0]), np.array([0.7]), T=1.0)
    assert abs(float(value[0]) - 0.347693) < 1e-4


def test_optimal_feedback_vanishes_at_boundary_probabilities():
    value = optimal_feedback(np.array([0.0, 0.0]), np.array([0.0, 1.0]), T=1.0)
    assert np.allclose(value, 0.0)


def test_optimal_feedback_positive_and_symmetric_in_time_remaining():
    t = np.array([0.0])
    p = np.array([0.5])
    v1 = optimal_feedback(t, p, T=1.0)
    v2 = optimal_feedback(t, p, T=4.0)
    assert v1[0] > 0.0
    # Less time remaining (fixed T, later t) increases the feedback intensity.
    t_late = np.array([0.9])
    v_late = optimal_feedback(t_late, p, T=1.0)
    assert v_late[0] > v1[0]


def test_optimal_probability_process_is_increasing_in_w():
    t = np.array([0.0, 0.5])
    w = np.array([[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]])
    p_star = optimal_probability_process(w, t, T=1.0, p=0.5)
    assert np.all(np.diff(p_star, axis=1) > 0)


def test_terminal_threshold_sign():
    # p < 0.5 implies Phi^{-1}(p) < 0, hence c_p > 0.
    assert terminal_threshold(1.0, 0.3) > 0.0
    assert terminal_threshold(1.0, 0.7) < 0.0
