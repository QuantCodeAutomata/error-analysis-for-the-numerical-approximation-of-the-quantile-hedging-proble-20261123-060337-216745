"""exp_5: Black-Scholes put, comparison of the learned feedback with the
explicit optimal control (Section 4.3, Figures 5-6).

Trains the network at p=0.7 with both the fixed and the adaptive bound,
replays independent test paths, and compares the neural control/probability
process with the exact optimal feedback alpha*(t,p) and the exact optimal
martingale P_t^* = Phi((W_t - c_p)/sqrt(T-t)), evaluated along the same
Brownian path (Section 4.3).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import src.config as cfg
from src.experiments_common import train_single
from src.feedback import optimal_feedback, optimal_probability_process
from src.models import black_scholes_spec
from src.simulation import simulate_financial

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def _replay_and_compare(net, spec, p, N, M, bound_mode, eta_stab, test_size, seed):
    """Replay ``net`` on independent test paths and compare with the exact
    optimal feedback/probability process evaluated on the same Brownian path."""
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        sim = simulate_financial(
            net, p, spec.x0, spec.mu, spec.vol_fn, spec.lambda_fn, spec.g_fn,
            N, cfg.BS_T, M, test_size, bound_mode=bound_mode, eta_stab=eta_stab,
            s_x=cfg.S_X, generator=gen, record_path=True,
        )
    t = sim.path["t"].numpy()  # (N+1,)
    p_path = sim.path["p"].numpy()  # (N+1, batch)
    alpha_path = sim.path["alpha"].numpy()  # (N, batch)
    w_path = sim.path["w"].numpy()  # (N+1, batch)

    t_pre = t[:-1]  # decision dates t_0..t_{N-1}, all < T
    p_pre = p_path[:-1]  # P_k^NN at the decision dates
    t_grid = t_pre[:, None] * np.ones_like(p_pre)
    alpha_star_at_learned = optimal_feedback(t_grid, p_pre, cfg.BS_T)

    p_star = optimal_probability_process(w_path[:-1], t_pre, cfg.BS_T, p)

    feedback_rmse = float(np.sqrt(np.mean((alpha_path - alpha_star_at_learned) ** 2)))
    probability_rmse = float(np.sqrt(np.mean((p_path[:-1] - p_star) ** 2)))
    return {
        "feedback_rmse": feedback_rmse,
        "probability_rmse": probability_rmse,
        "t": t, "p_path": p_path, "alpha_path": alpha_path,
        "alpha_star": alpha_star_at_learned, "p_star": p_star,
    }


def run(n_iterations: int, batch_size: int, val_size: int, test_size: int,
        validation_frequency: int, seed: int = cfg.RANDOM_SEED) -> dict:
    spec = black_scholes_spec()
    N, p = cfg.FEEDBACK_N, cfg.FEEDBACK_P
    h = cfg.BS_T / N
    M_fixed = h ** (-1.0 / 8.0)

    out = {}
    for bound_mode, M in (("fixed", M_fixed), ("adaptive", M_fixed)):
        eta_stab = cfg.FEEDBACK_ETA_STAB if bound_mode == "adaptive" else None
        res = train_single(
            spec, p, N, M, bound_mode, eta_stab, n_iterations, batch_size,
            val_size, test_size, validation_frequency, seed=seed,
        )
        comparison = _replay_and_compare(
            res.net, spec, p, N, M, bound_mode, eta_stab, test_size, seed=seed + 777,
        )
        out[bound_mode] = comparison
    return out


def _plot(out: dict) -> None:
    n_traj = cfg.N_SAMPLE_TRAJECTORIES
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for col, bound_mode in enumerate(("fixed", "adaptive")):
        res = out[bound_mode]
        t_pre = res["t"][:-1]
        ax = axes[0, col]
        for i in range(n_traj):
            ax.plot(t_pre, res["alpha_path"][:, i], color=f"C{i}", alpha=0.8, lw=1.0)
            ax.plot(t_pre, res["alpha_star"][:, i], color=f"C{i}", ls="--", alpha=0.6, lw=1.0)
        ax.set_title(f"Feedback control, {bound_mode} bound\n"
                     f"RMSE={res['feedback_rmse']:.3f}")
        ax.set_ylabel("alpha_k")

        ax = axes[1, col]
        for i in range(n_traj):
            ax.plot(res["t"], res["p_path"][:, i], color=f"C{i}", alpha=0.8, lw=1.0)
            ax.plot(t_pre, res["p_star"][:, i], color=f"C{i}", ls="--", alpha=0.6, lw=1.0)
        ax.set_title(f"Probability process, {bound_mode} bound\n"
                     f"RMSE={res['probability_rmse']:.3f}")
        ax.set_xlabel("t")
        ax.set_ylabel("P_t")
    fig.suptitle("Learned feedback/probability (solid) vs exact optimal (dashed), "
                 "Black-Scholes put, p=0.7 (Section 4.3, Figures 5-6)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp5_feedback_vs_optimal.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                         help="use the paper's validation/test sizes instead of the reduced "
                              "configuration used to fit the available computation time")
    args = parser.parse_args()

    batch_size = cfg.BATCH_SIZE
    val_size = cfg.VAL_SIZE if args.full else cfg.QUICK_VAL_SIZE
    test_size = cfg.TEST_SIZE if args.full else cfg.QUICK_TEST_SIZE
    validation_frequency = cfg.VALIDATION_FREQUENCY if args.full else cfg.QUICK_VALIDATION_FREQUENCY

    out = run(cfg.FEEDBACK_N_ITERATIONS, batch_size, val_size, test_size, validation_frequency)
    RESULTS_DIR.mkdir(exist_ok=True)
    _plot(out)

    summary = {
        "experiment": "exp_5", "p": cfg.FEEDBACK_P, "N": cfg.FEEDBACK_N,
        "n_iterations": cfg.FEEDBACK_N_ITERATIONS,
        "batch_size": batch_size, "val_size": val_size, "test_size": test_size,
        "fixed_feedback_rmse": out["fixed"]["feedback_rmse"],
        "fixed_probability_rmse": out["fixed"]["probability_rmse"],
        "adaptive_feedback_rmse": out["adaptive"]["feedback_rmse"],
        "adaptive_probability_rmse": out["adaptive"]["probability_rmse"],
    }
    out_path = RESULTS_DIR / "exp5_feedback_vs_optimal.json"
    out_path.write_text(json.dumps(summary, indent=2))

    print(f"fixed:    feedback RMSE={summary['fixed_feedback_rmse']:.4f}  "
          f"probability RMSE={summary['fixed_probability_rmse']:.4f}")
    print(f"adaptive: feedback RMSE={summary['adaptive_feedback_rmse']:.4f}  "
          f"probability RMSE={summary['adaptive_probability_rmse']:.4f}")
    print(f"Saved {out_path} and results/exp5_feedback_vs_optimal.png")


if __name__ == "__main__":
    main()
