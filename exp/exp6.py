"""exp_6: Lipschitz local-volatility put, quantile hedging value comparison
with fixed vs adaptive control bounds (Section 4.3, Figure 3b).

Same as exp_3 but with the piecewise-linear local-volatility model
ell(x) = 0.16 + 0.10*min(|x/100-1|, 1), whose value function is not Lipschitz
at p=1 (Section 4.3). The reference is a Monte Carlo approximation of
formula (26) using Euler paths of the local-volatility model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import src.config as cfg
from src.experiments_common import run_fixed_vs_adaptive
from src.models import local_vol_spec
from src.plotting import plot_fixed_vs_adaptive
from src.reference import quantile_hedging_value

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def run(batch_size: int, val_size: int, test_size: int, validation_frequency: int,
        reference_paths: int, reference_steps: int, seed: int = cfg.RANDOM_SEED) -> dict:
    spec = local_vol_spec(reference_steps=reference_steps)
    result = run_fixed_vs_adaptive(
        spec, cfg.P_GRID, cfg.FIN_N, reference_paths, cfg.FIN_N_ITERATIONS,
        batch_size, val_size, test_size, validation_frequency,
        eta_stab=cfg.FIN_ETA_STAB, seed=seed,
    )

    # Reference-convergence check (Section 4.3): compare the reference curve
    # computed with reference_steps against a coarser grid.
    spec_check = local_vol_spec(reference_steps=cfg.LV_REFERENCE_STEPS_CHECK)
    y_check = spec_check.reference_sampler(reference_paths, seed)
    reference_check = [quantile_hedging_value(y_check, p) for p in cfg.P_GRID]
    max_ref_diff = max(abs(a - b) for a, b in zip(result.reference, reference_check))

    return {
        "experiment": "exp_6", "model": result.model_name, "N": result.N, "M": result.M,
        "eta_stab": result.eta_stab, "reference_paths": reference_paths,
        "reference_steps": reference_steps, "reference_steps_check": cfg.LV_REFERENCE_STEPS_CHECK,
        "max_reference_grid_difference": max_ref_diff,
        "n_iterations": cfg.FIN_N_ITERATIONS,
        "batch_size": batch_size, "val_size": val_size, "test_size": test_size,
        "rows": [
            {"p": p, "reference": result.reference[i],
             "fixed_estimate": result.fixed_estimate[i], "fixed_stderr": result.fixed_stderr[i],
             "adaptive_estimate": result.adaptive_estimate[i], "adaptive_stderr": result.adaptive_stderr[i]}
            for i, p in enumerate(result.p_grid)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                         help="use the paper's validation/test/reference sizes instead of the "
                              "reduced configuration used to fit the available computation time")
    args = parser.parse_args()

    batch_size = cfg.BATCH_SIZE
    val_size = cfg.VAL_SIZE if args.full else cfg.QUICK_VAL_SIZE
    test_size = cfg.TEST_SIZE if args.full else cfg.QUICK_TEST_SIZE
    validation_frequency = cfg.VALIDATION_FREQUENCY if args.full else cfg.QUICK_VALIDATION_FREQUENCY
    reference_paths = cfg.LV_REFERENCE_PATHS if args.full else cfg.LV_REFERENCE_PATHS_RUN
    reference_steps = cfg.LV_REFERENCE_STEPS if args.full else cfg.LV_REFERENCE_STEPS_CHECK

    output = run(batch_size, val_size, test_size, validation_frequency,
                 reference_paths, reference_steps)
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "exp6_local_vol_fixed_vs_adaptive.json"
    out_path.write_text(json.dumps(output, indent=2))

    plot_fixed_vs_adaptive(
        output["rows"], "Local-volatility ATM put: quantile hedging value (Section 4.3, Figure 3b)",
        RESULTS_DIR / "exp6_local_vol_fixed_vs_adaptive.png",
    )

    print(f"Local-volatility put, N={output['N']}, M={output['M']:.4f}")
    for row in output["rows"]:
        print(f"p={row['p']:.1f}  reference={row['reference']:.4f}  "
              f"fixed={row['fixed_estimate']:.4f}  adaptive={row['adaptive_estimate']:.4f}")
    print(f"max reference grid difference = {output['max_reference_grid_difference']:.3e}")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
