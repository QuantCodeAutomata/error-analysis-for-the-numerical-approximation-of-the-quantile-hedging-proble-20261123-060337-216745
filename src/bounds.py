"""Fixed and adaptive control bounds (Section 4.2, equation (25); Figure 2).

Fixed bound: |alpha| <= M.
Adaptive bound: |alpha| <= m_{h,M}^{eta_stab}(p) = min(M, eta_stab * d(p) / sqrt(h)),
with d(p) = min(p, 1-p) the distance of p to the boundary {0, 1}. The bound
shrinks near the boundary, preventing one-step jumps large enough to exit
(0, 1) and be absorbed, which is the mechanism responsible for the projection
bias quantified by equation (24).
"""

from __future__ import annotations

import math

import torch


def bound_value(
    p: torch.Tensor,
    M: float,
    h: float,
    mode: str,
    eta_stab: float | None = None,
) -> torch.Tensor:
    """Elementwise control bound at state ``p`` for the given ``mode``.

    Parameters
    ----------
    p: current probability state P_n, any shape.
    M: fixed truncation level.
    h: mesh size T/N.
    mode: "fixed" or "adaptive".
    eta_stab: stabilization parameter (required for "adaptive").
    """
    if mode == "fixed":
        return torch.full_like(p, float(M))
    if mode == "adaptive":
        if eta_stab is None:
            raise ValueError("eta_stab is required for the adaptive bound")
        d = torch.minimum(p, 1.0 - p)
        adaptive = eta_stab * d / math.sqrt(h)
        return torch.clamp(adaptive, max=float(M))
    raise ValueError(f"unknown bound mode: {mode!r}")
