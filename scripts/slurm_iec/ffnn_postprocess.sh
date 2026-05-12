#!/bin/bash
# scripts/slurm_iec/ffnn_postprocess.sh
# Aggregates results from all training tasks and runs collect_iterres.py.
# Submitted by submit_iter_excl.sh with --dependency=afterok:<array_job_id>.
# Do not run directly until all training tasks have finished.
#
#SBATCH --job-name=IterExclPostProc
#SBATCH --partition=cpu      # no GPU needed here — adjust to your cluster
#SBATCH --nodes=1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=/home/projects/s215045/PredictPhagePPI/tmp/%j-%x.out
#SBATCH --error=/home/projects/s215045/PredictPhagePPI/tmp/%j-%x.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s215045@student.dtu.dk

set -euo pipefail

ROOT_DIR=$(git rev-parse --show-toplevel)
DIR_IN_NN_RUN="$ROOT_DIR/nn_runs/iter_excl_PFI_parallel/"
ACC_DIR="$ROOT_DIR/tmp/accuracies_par"

# ── 3. Post-Processing: Average Accuracies ────────────────────────────────────
echo "-------------------------------------------------------"
echo "Calculating Average Test Accuracy..."
echo "-------------------------------------------------------"

# Collect all per-pair accuracy files written by the array tasks
accuracies=$(cat "$ACC_DIR"/*.txt 2>/dev/null || true)

if [[ -z "$accuracies" ]]; then
    echo "WARNING: no accuracy files found in $ACC_DIR — skipping average."
else
    average=$(echo "$accuracies" | awk '
        NF > 0 { sum += $1; count++ }
        END     { if (count > 0) print sum / count; else print "0" }
    ')
    total_runs=$(echo "$accuracies" | grep -c '[0-9]' || true)

    echo "Total Runs Analyzed:        $total_runs"
    echo "Global Average Test Accuracy: $average"
fi

# ── 4. Collecting results ─────────────────────────────────────────────────────
echo ""
echo "📊 Collecting results..."
python3 "$ROOT_DIR/scripts/collect_iterres.py" --base_dir "$DIR_IN_NN_RUN"
