"""exp_3: Black-Scholes put, quantile hedging value comparison with fixed vs
adaptive control bounds (Section 4.3, Figure 3a).

Computes v(0, x, p) for an at-the-money put across p in {0, 0.2, ..., 1} with
a fixed bound M = h^(-1/8) and the adaptive bound (25), against the
closed-form reference (26) evaluated on 2**20 standard-normal samples.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import src.config as cfg
from src.experiments_common import run_fixed_vs_adaptive
from src.models import black_scholes_spec
from src.plotting import plot_fixed_vs_adaptive

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def run(batch_size: int, val_size: int, test_size: int, validation_frequency: int,
        reference_samples: int, seed: int = cfg.RANDOM_SEED) -> dict:
    spec = black_scholes_spec()
    result = run_fixed_vs_adaptive(
        spec, cfg.P_GRID, cfg.FIN_N, reference_samples, cfg.FIN_N_ITERATIONS,
        batch_size, val_size, test_size, validation_frequency,
        eta_stab=cfg.FIN_ETA_STAB, seed=seed,
    )
    return {
        "experiment": "exp_3", "model": result.model_name, "N": result.N, "M": result.M,
        "eta_stab": result.eta_stab, "reference_samples": reference_samples,
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
    reference_samples = cfg.BS_REFERENCE_SAMPLES if args.full else cfg.QUICK_TEST_SIZE * 4

    output = run(batch_size, val_size, test_size, validation_frequency, reference_samples)
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "exp3_black_scholes_fixed_vs_adaptive.json"
    out_path.write_text(json.dumps(output, indent=2))

    plot_fixed_vs_adaptive(
        output["rows"], "Black-Scholes ATM put: quantile hedging value (Section 4.3, Figure 3a)",
        RESULTS_DIR / "exp3_black_scholes_fixed_vs_adaptive.png",
    )

    print(f"Black-Scholes put, N={output['N']}, M={output['M']:.4f}")
    for row in output["rows"]:
        print(f"p={row['p']:.1f}  reference={row['reference']:.4f}  "
              f"fixed={row['fixed_estimate']:.4f}  adaptive={row['adaptive_estimate']:.4f}")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
