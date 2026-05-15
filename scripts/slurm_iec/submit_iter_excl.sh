#!/bin/bash
# submit_iter_excl.sh
# Run this script on the login node to submit all jobs.
# Usage: bash submit_iter_excl.sh

set -euo pipefail

if [[ $# -gt 3 ]]; then
    echo "Usage: bash submit_iter_excl.sh [N] [K] [DOWNDIR]" >&2
    exit 1
fi

N="${1:-500}"
K="${2:-12}"
DOWNDIR="${3:-encoded_sketches}" # "SM_sketches", "encoded_sketches", "encoded_sketches_data2", "SM_sketches_data2"
ROOT_DIR=$(git rev-parse --show-toplevel)
DATA_DIR="$ROOT_DIR/data_prod"
BACT_CLUSTER_FILE="$DATA_DIR/$DOWNDIR/sim_matrices/combined_bact_clusters_n${N}_k${K}.csv"
PHAGE_CLUSTER_FILE="$DATA_DIR/$DOWNDIR/sim_matrices/combined_phage_clusters_n${N}_k${K}.csv"

# ── 1. Enumerate every (bcluster, pcluster) pair and write a task-map file ───
# Each line of the task map = one array task index → "bcluster_num pcluster_num"
TASK_MAP="$ROOT_DIR/tmp/iter_excl_task_map_${N}_${K}.txt"
mkdir -p "$ROOT_DIR/tmp"
rm -f "$TASK_MAP"

bclusters=$(tail -n +2 "$BACT_CLUSTER_FILE" | awk -F',' '{print $2}' | sort -nu)
pclusters=$(tail -n +2 "$PHAGE_CLUSTER_FILE" | awk -F',' '{print $3}' | sort -u)

for b in $bclusters; do
    for p in $pclusters; do
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

echo "Submitted training array job: $array_job_id  (${total_tasks} tasks)"

# ── 3. Submit post-processing, gated on the whole array finishing cleanly ─────
pp_job_id=$(sbatch \
    --dependency="afterok:${array_job_id}" \
    --parsable \
    --kill-on-invalid-dep=yes \
    "$ROOT_DIR/scripts/slurm_iec/ffnn_postprocess.sh" \
    "$N" "$K")

echo "Submitted post-processing job: $pp_job_id  (runs after $array_job_id)"
echo ""
echo "Monitor with:"
echo "  squeue -j ${array_job_id},${pp_job_id}"
