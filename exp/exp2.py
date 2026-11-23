"""exp_2: Toy example, projection bias comparison with fixed vs adaptive
control bounds (Section 4.2, Figure 2).

Compares a fixed bound M = h^(-1/8) against the adaptive state-dependent
bound m_{h,M}^{eta_stab}(p) = min(M, eta_stab * d(p) / sqrt(h)) (equation 25),
at two grid resolutions N in {30, 100}, against the exact value v(p) = p.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import src.config as cfg
from src.evaluation import toy_loss_fn
from src.networks import SharedControlNet
from src.plotting import plot_projection_bias
from src.training import train_control

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def _train_one(p: float, N: int, M: float, bound_mode: str, eta_stab,
                n_iterations: int, batch_size: int, val_size: int, test_size: int,
                validation_frequency: int, seed: int) -> dict:
    if p in (0.0, 1.0):
        return {"estimate": p, "stderr": 0.0}
    net = SharedControlNet(input_dim=2)
    loss_fn = toy_loss_fn(p, N, cfg.EXP2_T, M, bound_mode, eta_stab)
    res = train_control(
        net, loss_fn, n_iterations=n_iterations, batch_size=batch_size,
        val_size=val_size, test_size=test_size, learning_rate=cfg.ADAM_LR_TOY,
        grad_clip_norm=cfg.GRAD_CLIP_NORM, validation_frequency=validation_frequency,
        seed=seed,
    )
    return {"estimate": res.test_estimate, "stderr": res.test_stderr}


def run(batch_size: int, val_size: int, test_size: int, validation_frequency: int,
        seed: int = cfg.RANDOM_SEED) -> dict:
    grids_out = []
    for N in cfg.EXP2_GRIDS:
        h = cfg.EXP2_T / N
        M = h ** (-1.0 / 8.0)
        rows = []
        for i, p in enumerate(cfg.P_GRID):
            # Same base seed for the fixed/adaptive pair at a given (N, p), so
            # both schemes are trained with the same Brownian samples.
            base_seed = seed + 100 * i + N
            fixed = _train_one(p, N, M, "fixed", None, cfg.EXP2_N_ITERATIONS,
                                batch_size, val_size, test_size, validation_frequency, base_seed)
            adaptive = _train_one(p, N, M, "adaptive", cfg.EXP2_ETA_STAB, cfg.EXP2_N_ITERATIONS,
                                   batch_size, val_size, test_size, validation_frequency, base_seed)
            rows.append({
                "p": p, "exact": p,
                "fixed_estimate": fixed["estimate"], "fixed_stderr": fixed["stderr"],
                "adaptive_estimate": adaptive["estimate"], "adaptive_stderr": adaptive["stderr"],
            })
        grids_out.append({"N": N, "h": h, "M": M, "eta_stab": cfg.EXP2_ETA_STAB, "rows": rows})
    RESULTS_DIR.mkdir(exist_ok=True)
    plot_projection_bias(grids_out, RESULTS_DIR / "exp2_toy_projection_bias.png")
    return {
        "experiment": "exp_2",
        "n_iterations": cfg.EXP2_N_ITERATIONS,
        "batch_size": batch_size, "val_size": val_size, "test_size": test_size,
        "grids": grids_out,
    }


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

    output = run(batch_size, val_size, test_size, validation_frequency)
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "exp2_toy_projection_bias.json"
    out_path.write_text(json.dumps(output, indent=2))

    for grid in output["grids"]:
        print(f"N={grid['N']}  M={grid['M']:.4f}")
        for row in grid["rows"]:
            print(f"  p={row['p']:.1f}  exact={row['exact']:.2f}  "
                  f"fixed={row['fixed_estimate']:.4f}  adaptive={row['adaptive_estimate']:.4f}")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
