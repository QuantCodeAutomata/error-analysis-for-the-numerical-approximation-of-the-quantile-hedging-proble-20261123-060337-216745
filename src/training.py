"""Algorithm 1 (Section 4.1): neural approximation of the fully discrete
projected problem, with validation-based model selection and a single
evaluation of the selected model on an independent test sample.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import torch

from src.networks import SharedControlNet


@dataclass
class TrainingResult:
    net: SharedControlNet
    best_val_loss: float
    test_estimate: float
    test_stderr: float
    train_losses: list = field(default_factory=list)
    val_losses: list = field(default_factory=list)


def train_control(
    net: SharedControlNet,
    loss_fn,
    n_iterations: int,
    batch_size: int,
    val_size: int,
    test_size: int,
    learning_rate: float,
    grad_clip_norm: float = 1.0,
    validation_frequency: int = 1,
    seed: int = 0,
) -> TrainingResult:
    """Run Algorithm 1.

    ``loss_fn(net, batch_size, generator)`` must simulate the controlled
    process for ``batch_size`` independent paths and return the per-path
    loss tensor Z_N g(X_N) P_N^theta (or P_N for the toy model), so that its
    mean is the batch loss (23).
    """
    device = next(net.parameters()).device
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    train_generator = torch.Generator(device=device).manual_seed(seed)
    val_generator = torch.Generator(device=device).manual_seed(seed + 1)
    test_generator = torch.Generator(device=device).manual_seed(seed + 2)

    best_val_loss = float("inf")
    best_state = copy.deepcopy(net.state_dict())
    result = TrainingResult(net=net, best_val_loss=best_val_loss, test_estimate=0.0, test_stderr=0.0)

    for r in range(n_iterations):
        optimizer.zero_grad()
        per_path = loss_fn(net, batch_size, train_generator)
        train_loss = per_path.mean()
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip_norm)
        optimizer.step()
        result.train_losses.append(float(train_loss.detach()))

        if (r + 1) % validation_frequency == 0 or r == n_iterations - 1:
            with torch.no_grad():
                val_per_path = loss_fn(net, val_size, val_generator)
                val_loss = float(val_per_path.mean())
            result.val_losses.append(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(net.state_dict())

    net.load_state_dict(best_state)
    with torch.no_grad():
        test_per_path = loss_fn(net, test_size, test_generator)
    test_estimate = float(test_per_path.mean())
    test_stderr = float(test_per_path.std(unbiased=True) / (test_size**0.5))

    result.best_val_loss = best_val_loss
    result.test_estimate = test_estimate
    result.test_stderr = test_stderr
    return result
