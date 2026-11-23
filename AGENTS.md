# Repository guide for coding agents

This repository replicates selected numerical experiments from Section 4
("Numerical experiments") of *Error analysis for the numerical approximation
of the quantile hedging problem*. It implements the neural-network-based,
fully discrete, projected Monte Carlo scheme for the quantile hedging value
v(0, x, p), the shared time-indexed control network and Algorithm 1 training
procedure of Section 4.1, the toy-model consistency and projection-bias
studies of Section 4.2, and the Black-Scholes / Lipschitz local-volatility
studies of Section 4.3 (fixed-vs-adaptive control bounds and convergence
under grid refinement).

No market data is used: every experiment is driven by simulated Brownian
motion. `data/generate_synthetic.py` is the only place that draws the
underlying Gaussian increments; `data/data_loader.py` is not part of this
repository because no experiment requires external market data.

## Layout

```
src/                      flat library modules (no subpackages)
  config.py               every experiment parameter, grouped by experiment,
                           each with a comment on its meaning, unit and source
  networks.py              shared time-indexed control network (Section 4.1)
  bounds.py                fixed and adaptive control bounds (equation 25)
  simulation.py            projected scheme for the toy and financial models (eq. 19-22)
  training.py              Algorithm 1 (train / validate / test)
  evaluation.py            loss-function factories, RMSE, log-log convergence fit
  reference.py             closed-form / Monte Carlo reference value (equation 26)
  feedback.py              exact optimal feedback and probability martingale (Section 4.3)
  models.py                Black-Scholes / local-volatility model specifications
  experiments_common.py    shared drivers for the fixed-vs-adaptive and convergence studies
  plotting.py               shared plotting helpers (matplotlib)
exp/                       one script per experiment (exp1.py .. exp7.py), each
                           calling only the src/ functions above with their
                           documented signatures
data/
  generate_synthetic.py    Brownian increments / standard normal samples
tests/                     pytest unit tests for src/ (synthetic fixtures only)
results/                   JSON metrics, PNG figures and RESULTS.md, all written
                           directly by exp/*, no per-experiment subdirectories
```

## Conventions

- All experiment parameters live in `src/config.py` as module-level
  UPPER_CASE variables, grouped by experiment under comment headers; no
  classes or parameter dicts. Do not hardcode a parameter value in `exp/` or
  `src/` modules that already has a `config.py` entry.
- Data preparation code lives only under `data/`. Since every experiment uses
  simulated paths, `data/generate_synthetic.py` is the single entry point for
  drawing Gaussian increments; do not add a second random-number source in
  `src/` or `exp/`.
- `exp/*.py` scripts call `src/` functions with their documented argument
  names and return keys; they do not re-implement simulation, training or
  evaluation logic locally.
- Every experiment script accepts `--full` to switch from the reduced
  Monte Carlo configuration (smaller validation/test/reference sample sizes,
  chosen to fit the available computation time) to the paper's exact sample
  sizes. The reduced and full values are both defined in `src/config.py`
  (e.g. `VAL_SIZE`, `TEST_SIZE`, `BS_REFERENCE_SAMPLES`, `LV_REFERENCE_PATHS`)
  and selected at run time, never hardcoded separately in `exp/`.

## Commands

Setup:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tests (synthetic fixtures only, no network access needed):

```bash
python -m pytest
python -m pytest -m "not market_data"   # equivalent here: no market_data tests exist
```

Experiments (reduced Monte Carlo configuration by default; add `--full` for
the paper's exact sample sizes):

```bash
python -m exp.exp1
python -m exp.exp2
python -m exp.exp3 [--full]
python -m exp.exp4 [--full]
python -m exp.exp5 [--full]
python -m exp.exp6 [--full]
python -m exp.exp7 [--full]
```

`reproduce.sh` runs the test suite and every experiment in this order with
the same commands as README.md.

## Notes for future changes

- After rerunning any experiment, its command overwrites the corresponding
  files in `results/`. The numbers and verdicts written by hand in
  `results/RESULTS.md` must then be re-checked against the new files and
  updated manually; they are not regenerated automatically.
- No API keys are required anywhere in this repository. If a future change
  introduces real market data, add it only through `data/data_loader.py`
  following the QuantX Dataloader / Massive REST convention, read keys from
  an uncommitted `.env`, and never commit downloaded data.
- Keep deviations from the paper's methodology (reduced sample sizes,
  substituted details for points the paper leaves open, identified causes of
  residual gaps) documented in `results/RESULTS.md`, not in `src/config.py`
  comments or elsewhere in the code.
