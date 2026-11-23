# Error analysis for the numerical approximation of the quantile hedging problem

Replication of selected numerical experiments from Section 4 ("Numerical
experiments") of *Error analysis for the numerical approximation of the
quantile hedging problem*.

The paper studies a neural-network-based, fully discrete, projected
Monte Carlo scheme for the quantile hedging value

    v(0, x, p) = sup { P(V_T >= g(X_T)) : V_0 <= x, V admissible, P(V_T >= g(X_T)) >= p }

via its probabilistic representation as a controlled martingale problem for
a probability process P_t in [0, 1]. Section 4.1 describes the shared,
time-indexed neural control network and Algorithm 1 (the training
procedure); Section 4.2 studies a toy model in which the exact value is
v(p) = p; Section 4.3 studies quantile hedging of an at-the-money put in a
Black-Scholes model and in a Lipschitz local-volatility model.

No market data is used anywhere in this repository: every experiment is
driven by simulated Brownian motion (Section 4, "Numerical experiments").
`data/generate_synthetic.py` is the single place that draws the underlying
Gaussian increments.

## Repository layout

```
src/                      shared library code (flat modules)
  config.py               every experiment parameter, grouped by experiment
  networks.py             shared time-indexed control network (Section 4.1)
  bounds.py               fixed and adaptive control bounds (equation 25)
  simulation.py           projected scheme for the toy and financial models (eq. 19-22)
  training.py             Algorithm 1 (train / validate / test)
  evaluation.py           loss-function factories, RMSE, log-log convergence fit
  reference.py            closed-form / Monte Carlo reference value (equation 26)
  feedback.py             exact optimal feedback and probability martingale (Section 4.3)
  models.py               Black-Scholes / local-volatility model specifications
  experiments_common.py   shared drivers for the fixed-vs-adaptive and convergence studies
  plotting.py             shared plotting helpers
exp/                      one script per experiment (exp1.py .. exp7.py)
data/
  generate_synthetic.py   Brownian increments / standard normal samples (torch and numpy)
tests/                    pytest unit tests for src/
results/                  JSON metrics, PNG figures and RESULTS.md (written by exp/*)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

No API keys or external data are required.

## Tests

```bash
python -m pytest
```

## Running the experiments

Each experiment can be run directly as a module from the repository root.
By default every script uses a reduced Monte Carlo configuration (smaller
validation/test sample sizes and, for Section 4.3, a reduced reference
sample) chosen to fit the available computation time; pass `--full` to use
the paper's exact sample sizes (`VAL_SIZE = 2**15`, `TEST_SIZE = 2**17`,
`BS_REFERENCE_SAMPLES = 2**20`, `LV_REFERENCE_PATHS = 2**18`). See
`results/RESULTS.md`, section "Deviations from the paper", for the executed
configuration and the estimated time of the full configuration.

```bash
python -m exp.exp1   # Toy example: small-control consistency check (Figure 1)
python -m exp.exp2   # Toy example: projection bias, fixed vs adaptive bound (Figure 2)
python -m exp.exp3   # Black-Scholes put: fixed vs adaptive bound (Figure 3a)
python -m exp.exp4   # Black-Scholes put: convergence under grid refinement (Figure 4a)
python -m exp.exp5   # Black-Scholes put: learned feedback vs exact optimal control (Figures 5-6)
python -m exp.exp6   # Local-volatility put: fixed vs adaptive bound (Figure 3b)
python -m exp.exp7   # Local-volatility put: convergence under grid refinement (Figure 4b)
```

`reproduce.sh` runs the test suite and every experiment in this order.

## Results

All generated tables, metrics (JSON) and figures (PNG) are written directly
to `results/`. `results/RESULTS.md` compares each experiment's output with
the values reported in the paper, states a verdict for every compared
metric, and lists every deviation from the paper and every paper result that
is not reproduced here.
