"""Shared neural-network control (Section 4.1, equations (21)-(22)).

A single time-shared network N^N(s, x_hat, p, theta) parametrizes the control
at every decision date. It has two hidden ReLU layers of width 32 and a
tanh output (Section 4.1): N^N = Psi^M(N~^N), with Psi^M(z) = M * tanh(z) in
dimension one (q=1). The network itself returns the unscaled tanh output in
[-1, 1]; the bound (fixed or adaptive) is applied by the caller so that the
same architecture serves both the toy and the financial examples.
"""

from __future__ import annotations

import torch
from torch import nn

from src.config import HIDDEN_WIDTH


class SharedControlNet(nn.Module):
    """Shared network mapping (s, [x_hat], p) to a raw control in [-1, 1].

    Parameters
    ----------
    input_dim:
        Number of scalar inputs: 2 for the toy model (t_n/T, P_n), 3 for the
        financial models (t_n/T, X_hat_n, P_n).
    hidden_width:
        Width of each hidden layer (Section 4.1: 32).
    """

    def __init__(self, input_dim: int, hidden_width: int = HIDDEN_WIDTH) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_width),
            nn.ReLU(),
            nn.Linear(hidden_width, hidden_width),
            nn.ReLU(),
            nn.Linear(hidden_width, 1),
        )
        # "Unless warm-started, the network is initialized with a zero last
        # layer, so the initial candidate is the feasible zero control"
        # (Section 4.1).
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the raw tanh output, shape (...,)."""
        raw = torch.tanh(self.net(inputs))
        return raw.squeeze(-1)


def zero_control_state_dict(net: SharedControlNet) -> dict:
    """State dict reproducing the zero control (last layer zeroed)."""
    state = {k: v.clone() for k, v in net.state_dict().items()}
    for key in state:
        if "net.4." in key:  # final Linear layer
            state[key].zero_()
    return state
