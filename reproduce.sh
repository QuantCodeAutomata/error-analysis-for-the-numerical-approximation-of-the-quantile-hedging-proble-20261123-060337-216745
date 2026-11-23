#!/usr/bin/env bash
# Runs the test suite and every experiment of Section 4 ("Numerical
# experiments") with the reduced sample sizes used to fit the available
# computation time. Pass --full as an argument to this script to run every
# experiment with the paper's exact sample sizes instead.
set -uo pipefail
cd "$(dirname "$0")"

FLAG=""
if [ "${1:-}" = "--full" ]; then
  FLAG="--full"
fi

python -m pytest -q || exit 1

for exp in exp1 exp2 exp3 exp4 exp5 exp6 exp7; do
  echo "=== python -m exp.${exp} ${FLAG} ==="
  python -m exp.${exp} ${FLAG} || exit 1
done
