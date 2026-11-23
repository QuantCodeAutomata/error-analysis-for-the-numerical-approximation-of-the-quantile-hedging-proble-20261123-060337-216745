"""exp_4: Black-Scholes put, convergence of quantile hedging errors under
grid refinement (Section 4.3, Figure 4a).

Trains only the adaptive bound (25) at (x, p) = (100, 0.7) for N in
{10, 20, 30, 50, 75, 100}, warm-starting 3 independent chains across grids,
and fits a log-log regression of the mean absolute error against N.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import src.config as cfg
from src.experiments_common import run_convergence_study
from src.models import black_scholes_spec
from src.plotting import plot_convergence
from src.reference import quantile_hedging_value

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def run(grids: tuple, n_chains: int, batch_size: int, val_size: int, test_size: int,
        validation_frequency: int, reference_samples: int, seed: int = cfg.RANDOM_SEED) -> dict:
    spec = black_scholes_spec()
    y_ref = spec.reference_sampler(reference_samples, seed)
    reference_value = quantile_hedging_value(y_ref, cfg.CONV_P)

    result = run_convergence_study(
        spec, grids, cfg.CONV_X, cfg.CONV_P, reference_value, n_chains,
        cfg.CONV_N_ITERATIONS, batch_size, val_size, test_size, validation_frequency,
        eta_stab=cfg.CONV_ETA_STAB, seed=seed,
    )
    return {
        "experiment": "exp_4", "model": result.model_name, "x": cfg.CONV_X, "p": cfg.CONV_P,
        "reference_value": result.reference_value, "reference_samples": reference_samples,
        "n_chains": n_chains, "n_iterations": cfg.CONV_N_ITERATIONS,
        "batch_size": batch_size, "val_size": val_size, "test_size": test_size,
        "grids": list(result.grids),
        "mean_abs_error": result.mean_abs_error, "std_abs_error": result.std_abs_error,
        "mean_martingale_defect": result.mean_martingale_defect,
        "loglog_slope": result.slope, "loglog_intercept": result.intercept,
        "theoretical_pcpt_exponent": -1.0 / 8.0,
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
    # The closed-form reference (26) is a cheap vectorised Monte Carlo evaluation,
    # so it is always computed at the paper's full precision (see exp3.py).
    reference_samples = cfg.BS_REFERENCE_SAMPLES
    grids = cfg.CONV_GRIDS_PAPER if args.full else cfg.CONV_GRIDS_RUN

    output = run(grids, cfg.CONV_N_CHAINS, batch_size, val_size, test_size,
                 validation_frequency, reference_samples)
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "exp4_black_scholes_convergence.json"
    out_path.write_text(json.dumps(output, indent=2))

    plot_convergence(
        output["grids"], output["mean_abs_error"], output["std_abs_error"],
        output["loglog_slope"], output["loglog_intercept"],
        "Black-Scholes put: pricing-error convergence under grid refinement (Figure 4a)",
        RESULTS_DIR / "exp4_black_scholes_convergence.png",
    )

    print(f"Black-Scholes put, reference v(0,100,0.7) = {output['reference_value']:.4f}")
    for N, err, std, defect in zip(output["grids"], output["mean_abs_error"],
                                    output["std_abs_error"], output["mean_martingale_defect"]):
        print(f"N={N:4d}  mean|err|={err:.4f}  std={std:.4f}  E[P_N]-p={defect:.5f}")
    print(f"log-log slope = {output['loglog_slope']:.4f} "
          f"(theoretical PCPT exponent = {output['theoretical_pcpt_exponent']:.4f})")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
