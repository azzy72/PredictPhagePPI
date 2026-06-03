#!/bin/bash
# Submit tune_ffnn.sh as a job array.
# Grid size must match the arrays inside tune_ffnn.sh, and %N limits how many run concurrently (adjust to your cluster's capacity).

set -euo pipefail

# directory of this script (works even if run from another cwd)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )"

N_COMBOS=81

# %8 caps concurrent running tasks; drop it to let SLURM run them all at once.
sbatch --array=0-$((N_COMBOS - 1))%8 "${SCRIPT_DIR}/tune_ffnn.sh"