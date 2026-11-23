"""Parameters for the quantile hedging replication, grouped by experiment.

All values are taken from Section 4 ("Numerical experiments") of
"Error analysis for the numerical approximation of the quantile hedging
problem", unless a comment states otherwise.
"""

import math

# --------------------------------------------------------------------------
# Shared neural-network architecture (Section 4.1)
# --------------------------------------------------------------------------
HIDDEN_WIDTH = 32          # width of each hidden layer (Section 4.1)
N_HIDDEN_LAYERS = 2        # number of ReLU hidden layers (Section 4.1)
S_X = 0.2                  # scale of the normalized log-state feature X_hat_n (Section 4.1)

GRAD_CLIP_NORM = 1.0       # gradient clipping norm, all experiments (Algorithm 1)
ADAM_LR_TOY = 5e-3         # Adam learning rate in the toy examples (Section 4.1)
ADAM_LR_FINANCIAL = 1e-3   # Adam learning rate in the financial examples (Section 4.1)

BATCH_SIZE = 2048          # training batch size L_batch (Algorithm 1)
VAL_SIZE = 2 ** 15         # validation sample size L_val (Algorithm 1)
TEST_SIZE = 2 ** 17        # test sample size L_test (Algorithm 1)
# Validation frequency is not specified in the paper; validating every
# iteration is the conservative choice consistent with "the prescribed
# validation frequency" of Algorithm 1, step 8.
VALIDATION_FREQUENCY = 1

P_GRID = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)  # test points used throughout Section 4.2-4.3

# --------------------------------------------------------------------------
# exp_1: toy example, small-control consistency check (Section 4.2, Figure 1)
# --------------------------------------------------------------------------
EXP1_T = 1.0
EXP1_N = 100
EXP1_EPS = 0.01            # target tolerance epsilon used to define M (Section 4.2)
# M = eps * sqrt(N / (2 * T * log(2N))), the small-control bound of Proposition 4.1
EXP1_M = EXP1_EPS * math.sqrt(EXP1_N / (2.0 * EXP1_T * math.log(2 * EXP1_N)))
EXP1_N_ITERATIONS = 150   # training iterations (Figure 1 caption)

# --------------------------------------------------------------------------
# exp_2: toy example, projection bias comparison (Section 4.2, Figure 2)
# --------------------------------------------------------------------------
EXP2_T = 1.0
EXP2_GRIDS = (30, 100)     # grid resolutions N compared in Figure 2
EXP2_ETA_STAB = 0.6        # stabilization parameter eta_stab (Figure 2 caption)
EXP2_N_ITERATIONS = 150   # training iterations (Figure 2 caption)

# --------------------------------------------------------------------------
# exp_3 / exp_6: Black-Scholes / local-volatility put, fixed vs adaptive bound
# (Section 4.3, Figure 3)
# --------------------------------------------------------------------------
BS_T = 1.0
BS_X0 = 100.0
BS_MU = 0.1
BS_SIGMA = 0.2
BS_LAMBDA = -BS_MU / BS_SIGMA   # lambda = -mu/sigma = -0.5 (Section 4.3)
BS_STRIKE = 100.0

# Piecewise-linear local volatility ell(x) = 0.16 + 0.10*min(|x/100-1|, 1)
LV_ELL_BASE = 0.16
LV_ELL_SLOPE = 0.10

FIN_N = 100                # time steps N used in Figure 3 (Section 4.3)
FIN_H = BS_T / FIN_N
FIN_M_FIXED = FIN_H ** (-1.0 / 8.0)   # fixed bound M = h^(-1/8) (Section 4.3)
FIN_ETA_STAB = 0.6
FIN_N_ITERATIONS = 500    # training iterations (Figure 3 caption)

BS_REFERENCE_SAMPLES = 2 ** 20      # closed-form reference sample size (Section 4.3)
LV_REFERENCE_PATHS = 2 ** 18        # Monte-Carlo reference path count (Section 4.3)
LV_REFERENCE_PATHS_RUN = 2 ** 15    # reduced path count executed to fit the replication's time limit
LV_REFERENCE_STEPS = 1000           # Euler steps for the MC reference (Section 4.3)
LV_REFERENCE_STEPS_CHECK = 500      # coarser grid used for the convergence check (Section 4.3)

# --------------------------------------------------------------------------
# exp_4 / exp_7: convergence of quantile hedging errors under grid refinement
# (Section 4.3, Figure 4)
# --------------------------------------------------------------------------
CONV_X = 100.0
CONV_P = 0.7
CONV_GRIDS_PAPER = (10, 20, 30, 50, 75, 100)  # the paper's full grid (Figure 4)
CONV_GRIDS_RUN = (10, 20, 30, 50, 75, 100)    # executed grid (unchanged; see README)
CONV_N_CHAINS = 3          # independent warm-started training chains (Section 4.3)
CONV_N_ITERATIONS = 500   # Adam iterations per grid point per chain (Section 4.3)
CONV_ETA_STAB = 0.6

# --------------------------------------------------------------------------
# exp_5: learned feedback vs explicit optimal control (Section 4.3, Figures 5-6)
# --------------------------------------------------------------------------
FEEDBACK_P = 0.7
FEEDBACK_N = 100
FEEDBACK_ETA_STAB = 0.6
FEEDBACK_N_ITERATIONS = 500
N_SAMPLE_TRAJECTORIES = 6   # number of test paths plotted (not specified exactly; a small
                            # illustrative sample, consistent with "a few representative
                            # test paths" noted for Figures 5-6)

# --------------------------------------------------------------------------
# Execution-time reduction (see results/RESULTS.md, "Deviations from the paper")
# --------------------------------------------------------------------------
# Test/validation sizes and training-iteration counts above are the paper's
# values and are used by default. A --quick flag (see exp/*.py) reduces the
# Monte Carlo sample sizes and iteration counts proportionally to fit the
# available computation time; the method, formulas and window lengths are
# never altered by this flag.
QUICK_BATCH_SIZE = BATCH_SIZE    # batch size L_batch is kept at the paper's value
QUICK_VAL_SIZE = 2 ** 12
QUICK_TEST_SIZE = 2 ** 13
QUICK_N_ITERATIONS_SCALE = 1.0   # Adam training-iteration counts are kept at the paper's values
# Validating every iteration on the full paper validation sample is the
# dominant cost; validating every 25 iterations keeps the validation-based
# model selection of Algorithm 1 (step 8) while fitting the available
# computation time. Not specified in the paper (see "Deviations from the
# paper" in results/RESULTS.md).
QUICK_VALIDATION_FREQUENCY = 25

RANDOM_SEED = 2024   # fixed base seed; the paper does not specify its seed
