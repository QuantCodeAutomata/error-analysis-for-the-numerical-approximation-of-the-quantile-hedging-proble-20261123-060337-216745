"""exp_1: Toy example, small-control consistency check (Section 4.2, Figure 1).

In the toy model g=1, b=0, lambda=0 (so Z=1), the exact quantile hedging value
is v(p) = p. This experiment trains the projected neural scheme with a small
control bound M = eps * sqrt(N / (2 T log(2N))) (Proposition 4.1) and checks
that the test-set estimate reproduces v(p) = p for every p in {0, 0.2, ..., 1}.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import src.config as cfg
from src.evaluation import toy_loss_fn
from src.networks import create_shared_control_net
from src.plotting import plot_exact_vs_estimate
from src.training import train_control

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def run(batch_size: int, val_size: int, test_size: int, validation_frequency: int,
        seed: int = cfg.RANDOM_SEED) -> dict:
    rows = []
    for i, p in enumerate(cfg.P_GRID):
        if p in (0.0, 1.0):
            rows.append({"p": p, "exact": p, "estimate": p, "stderr": 0.0, "abs_error": 0.0})
            continue
        net = create_shared_control_net(input_dim=2, seed=seed + i)
        loss_fn = toy_loss_fn(p, cfg.EXP1_N, cfg.EXP1_T, cfg.EXP1_M, "fixed", None)
        res = train_control(
            net, loss_fn,
            n_iterations=cfg.EXP1_N_ITERATIONS,
            batch_size=batch_size, val_size=val_size, test_size=test_size,
            learning_rate=cfg.ADAM_LR_TOY, grad_clip_norm=cfg.GRAD_CLIP_NORM,
            validation_frequency=validation_frequency, seed=seed + i,
        )
        rows.append({
            "p": p, "exact": p, "estimate": res.test_estimate, "stderr": res.test_stderr,
            "abs_error": abs(res.test_estimate - p),
        })
    return {
        "experiment": "exp_1",
        "M": cfg.EXP1_M, "N": cfg.EXP1_N, "n_iterations": cfg.EXP1_N_ITERATIONS,
        "batch_size": batch_size, "val_size": val_size, "test_size": test_size,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                         help="use the paper's validation/test sizes (2**15/2**17) and "
                              "validation frequency instead of the reduced configuration used "
                              "to fit the available computation time")
    args = parser.parse_args()

    batch_size = cfg.BATCH_SIZE
    val_size = cfg.VAL_SIZE if args.full else cfg.QUICK_VAL_SIZE
    test_size = cfg.TEST_SIZE if args.full else cfg.QUICK_TEST_SIZE
    validation_frequency = cfg.VALIDATION_FREQUENCY if args.full else cfg.QUICK_VALIDATION_FREQUENCY

    output = run(batch_size, val_size, test_size, validation_frequency)
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "exp1_toy_consistency.json"
    out_path.write_text(json.dumps(output, indent=2))

    p_grid = [row["p"] for row in output["rows"]]
    exact = [row["exact"] for row in output["rows"]]
    estimate = [row["estimate"] for row in output["rows"]]
    plot_exact_vs_estimate(
        p_grid, exact, {"neural estimate (small M)": estimate},
        f"Toy example: small-control consistency check (M={cfg.EXP1_M:.4f})",
        RESULTS_DIR / "exp1_toy_consistency.png",
    )

    print(f"M = {cfg.EXP1_M:.6f}")
    for row in output["rows"]:
        print(f"p={row['p']:.1f}  exact={row['exact']:.4f}  "
              f"estimate={row['estimate']:.4f}  stderr={row['stderr']:.5f}")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
