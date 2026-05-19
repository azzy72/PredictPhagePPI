#!/bin/bash
# scripts/slurm_iec/ffnn_train_task.sh
# One Slurm array task = one (bcluster, pcluster) training run.
# Submitted by submit_iter_excl.sh — do not run directly.

#SBATCH --job-name=IterExclTrain
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --mem=50G
#SBATCH --cpus-per-task=25
#SBATCH --gres=shard:1
#SBATCH --time=01:00:00
#SBATCH --output=/home/projects/s215045/PredictPhagePPI/tmp/parallel_tmp/%A_%a-%x.out
#SBATCH --error=/home/projects/s215045/PredictPhagePPI/tmp/parallel_tmp/%A_%a-%x.err
#SBATCH --mail-type=FAIL,ARRAY_TASKS
#SBATCH --mail-user=s215045@student.dtu.dk

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
if [[ $# -gt 3 ]]; then
    echo "Usage: sbatch ffnn_train_task.sh [N] [K] [DOWNDIR]" >&2
    exit 1
fi

N="${1:-500}"
K="${2:-12}"
DOWNDIR="${3:-encoded_sketches}" # "SM_sketches", "encoded_sketches", "encoded_sketches_data2", "SM_sketches_data2"
ROOT_DIR=$(git rev-parse --show-toplevel)
DATA_DIR="$ROOT_DIR/data_prod"
CUSTOM_PARENT_DIR="IterExcl_${DOWNDIR}_n${N}_k${K}" # for organizing outputs by config
BACT_CLUSTER_FILE="$DATA_DIR/$DOWNDIR/sim_matrices/combined_bact_clusters_n${N}_k${K}.csv"
PHAGE_CLUSTER_FILE="$DATA_DIR/$DOWNDIR/sim_matrices/combined_phage_clusters_n${N}_k${K}.csv"
TASK_MAP="$ROOT_DIR/tmp/IterExcl_Taskmap_${DOWNDIR}_${N}_${K}.txt"

# ── Resolve this task's (bcluster, pcluster) pair from the task map ───────────
# SLURM_ARRAY_TASK_ID is 1-indexed; sed line numbers are also 1-indexed.
pair=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TASK_MAP")
bcluster_num=$(echo "$pair" | awk '{print $1}')
pcluster_num=$(echo "$pair" | awk '{print $2}')

echo "Array task $SLURM_ARRAY_TASK_ID → bcluster=$bcluster_num  pcluster=$pcluster_num"

# ── Resolve strain names from the cluster CSVs ────────────────────────────────
bact_strains=$(tail -n +2 "$BACT_CLUSTER_FILE" \
    | awk -F',' -v c="$bcluster_num" '$3==c {print $1}' \
    | paste -sd ',' -)
bact_strains=$(echo "$bact_strains" | sed 's/_reoriented//g')

phage_strains=$(tail -n +2 "$PHAGE_CLUSTER_FILE" \
    | awk -F',' -v c="$pcluster_num" '$4==c {print $1}' \
    | paste -sd ',' -)

echo "Bact strains  : $bact_strains"
echo "Phage strains : $phage_strains"

# ── Training run ──────────────────────────────────────────────────────────────
CUSTOM_OUT="$CUSTOM_PARENT_DIR/cluster_b${bcluster_num}_p${pcluster_num}"

#if "encoded_sketches" in "$DOWNDIR"; then add --use_encoded argument to FFNN_inner.py
if [[ "$DOWNDIR" == *"encoded_sketches"* ]]; then
    ENCODED_FLAG="--use_encoded"
else
    ENCODED_FLAG=""
fi

#if "data2" in "$DOWNDIR"; then add --data2 argument to FFNN_inner.py
if [[ "$DOWNDIR" == *"data2"* ]]; then
    DATA2_FLAG="--data2"
else
    DATA2_FLAG=""
fi

python3 "$ROOT_DIR/scripts/FFNN_inner.py" \
    --nk "$N" "$K" \
    --cv \
    --kf_n_splits 4 \
    --exclude_clusters \
    --exclude_bact_clusters "$bact_strains" \
    --exclude_phage_clusters "$phage_strains" \
    --test_on_excluded \
    --perform_pfi \
    --top_kmers_num 200 \
    --out "$CUSTOM_OUT" \
    --logging \
    $DATA2_FLAG \
    $ENCODED_FLAG

# ── Extract and persist accuracy for this pair ────────────────────────────────
# Write to a per-pair file so the post-processing job can gather them all
# without any race conditions.
ACC_FILE="$ROOT_DIR/tmp/accuracies_${DOWNDIR}_n${N}_k${K}/b${bcluster_num}_p${pcluster_num}.txt"
mkdir -p "$(dirname "$ACC_FILE")"

acc=$(find "$ROOT_DIR/nn_runs/${CUSTOM_OUT}_run1/log_run1.txt" \
        -exec grep "Final test loss:" {} + \
    | awk -F'test accuracy: ' '{print $2}')

if [[ -n "$acc" ]]; then
    echo "$acc" > "$ACC_FILE"
    echo "Accuracy written to $ACC_FILE: $acc"
else
    echo "WARNING: no accuracy value found for b${bcluster_num}_p${pcluster_num}"
fi
