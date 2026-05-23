#!/bin/bash
# submit_all_iter_excl.sh
# Submit jobs for all combinations of N, K, and DOWNDIR values.
# Usage: bash submit_all_iter_excl.sh

set -euo pipefail

# Define the arrays of values to iterate through
#N_VALUES=(300 500 1000)
#K_VALUES=(9 12 18 24 32 51)
N_VALUES=(500)
K_VALUES=(12 24)
DOWNDIR_VALUES=("SM_sketches_allphages" "encoded_sketches_allphages" "encoded_sketches_data2_allphages" "SM_sketches_data2_allphages")

ROOT_DIR=$(git rev-parse --show-toplevel)

# Counter for total jobs submitted
total_jobs=0
total_pp_jobs=0

# Function to submit jobs for a given N, K, DOWNDIR combination
submit_for_combination() {
    local N="$1"
    local K="$2"
    local DOWNDIR="$3"
    
    echo "=========================================="
    echo "Submitting jobs for: N=$N, K=$K, DOWNDIR=$DOWNDIR"
    echo "=========================================="
    
    DATA_DIR="$ROOT_DIR/data_prod"
    BACT_CLUSTER_FILE="$DATA_DIR/$DOWNDIR/sim_matrices/combined_bact_clusters_n${N}_k${K}.csv"
    PHAGE_CLUSTER_FILE="$DATA_DIR/$DOWNDIR/sim_matrices/combined_phage_clusters_n${N}_k${K}.csv"

    # Check if files exist
    if [[ ! -f "$BACT_CLUSTER_FILE" ]]; then
        echo "⚠ Warning: $BACT_CLUSTER_FILE not found. Skipping."
        echo ""
        return 1
    fi
    if [[ ! -f "$PHAGE_CLUSTER_FILE" ]]; then
        echo "⚠ Warning: $PHAGE_CLUSTER_FILE not found. Skipping."
        echo ""
        return 1
    fi

    # ── 1. Enumerate every (bpartition, ppartition) pair and write a task-map file ───
    TASK_MAP="$ROOT_DIR/tmp/IterExcl_Taskmap_${DOWNDIR}_${N}_${K}.txt"
    mkdir -p "$ROOT_DIR/tmp"
    rm -f "$TASK_MAP"

    bpartitions=$(tail -n +2 "$BACT_CLUSTER_FILE" | awk -F',' '{print $3}' | sort -nu)
    ppartitions=$(tail -n +2 "$PHAGE_CLUSTER_FILE" | awk -F',' '{print $4}' | sort -u)

    for b in $bpartitions; do
        for p in $ppartitions; do
            echo "$b $p" >> "$TASK_MAP"
        done
    done

    total_tasks=$(wc -l < "$TASK_MAP")
    echo "Task map written: $total_tasks pairs → $TASK_MAP"

    # ── 2. Submit the array job (1-indexed to match $SLURM_ARRAY_TASK_ID) ────────
    array_job_id=$(sbatch \
        --array="1-${total_tasks}" \
        --parsable \
        "$ROOT_DIR/scripts/slurm_iec/ffnn_train_task.sh" \
        "$N" "$K" "$DOWNDIR")

    echo "✓ Submitted training array job: $array_job_id  (${total_tasks} tasks)"
    ((total_jobs++))

    # ── 3. Submit post-processing, gated on the whole array finishing cleanly ─────
    pp_job_id=$(sbatch \
        --dependency="afterany:${array_job_id}" \
        --parsable \
        --kill-on-invalid-dep=yes \
        "$ROOT_DIR/scripts/slurm_iec/ffnn_postprocess.sh" \
        "$N" "$K" "$DOWNDIR")

    echo "✓ Submitted post-processing job: $pp_job_id  (runs after $array_job_id)"
    ((total_pp_jobs++))
    echo ""
    
    return 0
}

# ──────────────────────────────────────────────────────────────────────────────
# MAIN: Iterate through all combinations
# ──────────────────────────────────────────────────────────────────────────────
echo "Submitting jobs for all N, K, DOWNDIR combinations..."
echo "N values: ${N_VALUES[@]}"
echo "K values: ${K_VALUES[@]}"
echo "DOWNDIR values: ${DOWNDIR_VALUES[@]}"
echo ""
echo "Total combinations to attempt: $((${#N_VALUES[@]} * ${#K_VALUES[@]} * ${#DOWNDIR_VALUES[@]}))"
echo ""

for N in "${N_VALUES[@]}"; do
    for K in "${K_VALUES[@]}"; do
        for DOWNDIR in "${DOWNDIR_VALUES[@]}"; do
            submit_for_combination "$N" "$K" "$DOWNDIR" || true
        done
    done
done

echo "=========================================="
echo "Submission Summary"
echo "=========================================="
echo "Training array jobs submitted: $total_jobs"
echo "Post-processing jobs submitted: $total_pp_jobs"
echo ""
echo "Monitor all jobs with:"
echo "  squeue -u \$USER"
