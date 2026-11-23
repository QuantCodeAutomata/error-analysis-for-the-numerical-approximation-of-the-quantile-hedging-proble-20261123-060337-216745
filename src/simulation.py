"""Fully discrete projected scheme for the controlled probability process
(Section 4, equations (19)-(22)) and the uncontrolled diffusion models used
in the numerical experiments (Section 4.3).

All simulations are vectorized over the batch dimension with torch tensors;
no state carries look-ahead information (the control at step n only uses
quantities known at t_n).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from data.generate_synthetic import brownian_increments
from src.bounds import bound_value
from src.networks import SharedControlNet


@dataclass
class SimulationResult:
    """Container for a simulated batch.

    p_final: P_N, shape (batch,).
    x_final: X_N (financial models only), shape (batch,) or None.
    z_final: Z_N (financial models only), shape (batch,) or None.
    path: optional dict of recorded trajectories when record_path=True,
        with keys "t", "p", "alpha", "x" (x only for financial models),
        each a tensor of shape (N, batch) (or (N,) for "t").
    """

    p_final: torch.Tensor
    x_final: torch.Tensor | None = None
    z_final: torch.Tensor | None = None
    path: dict | None = None


def _project_step(
    p_tilde_candidate: torch.Tensor,
    active: torch.Tensor,
    p_current: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One step of the projection/absorption rule, equations (19)-(20).

    ``active`` marks paths with n < tau (not yet absorbed). Returns the
    updated (p_next, active_next).
    """
    exited = active & ((p_tilde_candidate <= 0.0) | (p_tilde_candidate >= 1.0))
    absorbed_at_one = exited & (p_tilde_candidate >= 1.0)
    p_next = torch.where(active, p_tilde_candidate, p_current)
    p_next = torch.where(absorbed_at_one, torch.ones_like(p_next), p_next)
    p_next = torch.where(exited & ~absorbed_at_one, torch.zeros_like(p_next), p_next)
    active_next = active & ~exited
    return p_next, active_next


def simulate_toy(
    net: SharedControlNet,
    p0: float,
    N: int,
    T: float,
    M: float,
    batch_size: int,
    bound_mode: str = "fixed",
    eta_stab: float | None = None,
    generator: torch.Generator | None = None,
    record_path: bool = False,
) -> SimulationResult:
    """Simulate the toy model (g=1, b=0, lambda=0, so Z=1) of Section 4.2.

    The controlled process evolves as dP = alpha dW with the projection and
    absorption rule of equations (19)-(20); the shared network takes
    (t_n/T, P_n) as input.
    """
    h = T / N
    device = next(net.parameters()).device
    if p0 in (0.0, 1.0):
        p_final = torch.full((batch_size,), float(p0), device=device)
        return SimulationResult(p_final=p_final)

    p = torch.full((batch_size,), float(p0), device=device)
    active = torch.ones((batch_size,), dtype=torch.bool, device=device)
    path = {"t": [], "p": [p.clone()], "alpha": []} if record_path else None

    sqrt_h = h**0.5
    for n in range(N):
        t_norm = torch.full((batch_size,), n / N, device=device)
        net_in = torch.stack([t_norm, p], dim=-1)
        raw = net(net_in)
        bound = bound_value(p, M, h, bound_mode, eta_stab)
        alpha = bound * raw
        dW = brownian_increments(batch_size, sqrt_h, generator=generator, device=device)
        p_tilde = p + alpha * dW
        p, active = _project_step(p_tilde, active, p)
        if record_path:
            path["t"].append(n * h)
            path["p"].append(p.clone())
            path["alpha"].append(alpha.clone())

    if record_path:
        path["p"] = torch.stack(path["p"], dim=0)
        path["alpha"] = torch.stack(path["alpha"], dim=0)
        path["t"] = torch.tensor(path["t"] + [N * h])
    return SimulationResult(p_final=p, path=path)


def simulate_financial(
    net: SharedControlNet,
    p0: float,
    x0: float,
    mu: float,
    vol_fn,
    lambda_fn,
    g_fn,
    N: int,
    T: float,
    M: float,
    batch_size: int,
    bound_mode: str = "fixed",
    eta_stab: float | None = None,
    s_x: float = 0.2,
    generator: torch.Generator | None = None,
    record_path: bool = False,
) -> SimulationResult:
    """Simulate quantile hedging in a 1-D diffusion model (Section 4.3).

    dX_t = mu X_t dt + X_t vol_fn(X_t) dW_t,
    dZ_t = Z_t lambda_fn(X_t) dW_t   (b=0, risk-neutral discounting absent),
    with the shared network taking (t_n/T, X_hat_n, P_n) as input, where
    X_hat_n = log(max(X_n, 1e-6)/x0) / s_x (Section 4.1).
    """
    h = T / N
    device = next(net.parameters()).device
    x = torch.full((batch_size,), float(x0), device=device)
    z = torch.ones((batch_size,), device=device)

    if p0 in (0.0, 1.0):
        p = torch.full((batch_size,), float(p0), device=device)
        active = torch.zeros((batch_size,), dtype=torch.bool, device=device)
    else:
        p = torch.full((batch_size,), float(p0), device=device)
        active = torch.ones((batch_size,), dtype=torch.bool, device=device)

    sqrt_h = h**0.5
    w = torch.zeros((batch_size,), device=device)
    path = {"t": [], "p": [p.clone()], "alpha": [], "x": [x.clone()], "w": [w.clone()]} if record_path else None

    for n in range(N):
        dW = brownian_increments(batch_size, sqrt_h, generator=generator, device=device)
        w = w + dW

        ell = vol_fn(x)
        lam = lambda_fn(x)
        x_next = x + mu * x * h + x * ell * dW
        z_next = z * torch.exp(-0.5 * lam**2 * h + lam * dW)

        if active.any():
            x_hat = torch.log(torch.clamp(x, min=1e-6) / x0) / s_x
            t_norm = torch.full((batch_size,), n / N, device=device)
            net_in = torch.stack([t_norm, x_hat, p], dim=-1)
            raw = net(net_in)
            bound = bound_value(p, M, h, bound_mode, eta_stab)
            alpha = bound * raw
            p_tilde = p + alpha * dW
            p, active = _project_step(p_tilde, active, p)
        else:
            alpha = torch.zeros((batch_size,), device=device)

        x, z = x_next, z_next
        if record_path:
            path["t"].append(n * h)
            path["p"].append(p.clone())
            path["alpha"].append(alpha.clone())
            path["x"].append(x.clone())
            path["w"].append(w.clone())

    g_x = g_fn(x)
    if record_path:
        path["p"] = torch.stack(path["p"], dim=0)
        path["alpha"] = torch.stack(path["alpha"], dim=0)
        path["x"] = torch.stack(path["x"], dim=0)
        path["w"] = torch.stack(path["w"], dim=0)
        path["t"] = torch.tensor(path["t"] + [N * h])
    return SimulationResult(p_final=p, x_final=x, z_final=z, path=path)


def put_payoff(strike: float):
    """g(x) = (K - x)^+ (Section 4.3)."""

    def _payoff(x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(strike - x, min=0.0)

    return _payoff


def black_scholes_vol_fn(sigma: float):
    def _vol(x: torch.Tensor) -> torch.Tensor:
        return torch.full_like(x, float(sigma))

    return _vol


def black_scholes_lambda_fn(mu: float, sigma: float):
    lam = -mu / sigma

    def _lambda(x: torch.Tensor) -> torch.Tensor:
        return torch.full_like(x, float(lam))

    return _lambda


def local_vol_fn(base: float, slope: float, center: float = 100.0):
    """ell(x) = base + slope * min(|x/center - 1|, 1) (Section 4.3)."""

    def _vol(x: torch.Tensor) -> torch.Tensor:
        return base + slope * torch.clamp(torch.abs(x / center - 1.0), max=1.0)

    return _vol


def local_vol_lambda_fn(mu: float, vol_fn):
    def _lambda(x: torch.Tensor) -> torch.Tensor:
        return -mu / vol_fn(x)

    return _lambda
