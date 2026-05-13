#!/bin/bash
# scripts/slurm_iec/extract_accuracy.sh
# Extracts the final test accuracy from a completed training run and writes it to a file.
# This is indepdentn of the pipeline

N=500
K=18
DOWNDIR="encoded_sketches" # "SM_sketches", "encoded_sketches", "encoded_sketches_data2", "SM_sketches_data2"
ROOT_DIR=$(git rev-parse --show-toplevel)
DATA_DIR="$ROOT_DIR/data_prod"
CUSTOM_PARENT_DIR="iter_excl_PFI_parallel_n${N}_k${K}" # for organizing outputs by config
BACT_CLUSTER_FILE="$DATA_DIR/$DOWNDIR/sim_matrices/combined_bact_clusters_n${N}_k${K}.csv"
PHAGE_CLUSTER_FILE="$DATA_DIR/$DOWNDIR/sim_matrices/combined_phage_clusters_n${N}_k${K}.csv"
TASK_MAP="$ROOT_DIR/tmp/iter_excl_task_map_${N}_${K}.txt"

# ── Resolve this task's (bcluster, pcluster) pair from the task map ───────────
# SLURM_ARRAY_TASK_ID is 1-indexed; sed line numbers are also 1-indexed.
pair=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TASK_MAP")
bcluster_num=$(echo "$pair" | awk '{print $1}')
pcluster_num=$(echo "$pair" | awk '{print $2}')

echo "Array task $SLURM_ARRAY_TASK_ID → bcluster=$bcluster_num  pcluster=$pcluster_num"

# ── Resolve strain names from the cluster CSVs ────────────────────────────────
bact_strains=$(tail -n +2 "$BACT_CLUSTER_FILE" \
    | awk -F',' -v c="$bcluster_num" '$2==c {print $1}' \
    | paste -sd ',' -)
bact_strains=$(echo "$bact_strains" | sed 's/_reoriented//g')

phage_strains=$(tail -n +2 "$PHAGE_CLUSTER_FILE" \
    | awk -F',' -v c="$pcluster_num" '$3==c {print $1}' \
    | paste -sd ',' -)

echo "Bact strains  : $bact_strains"
echo "Phage strains : $phage_strains"

# ── Extract and persist accuracy for this pair ────────────────────────────────
# Write to a per-pair file so the post-processing job can gather them all
# without any race conditions.
ACC_FILE="$ROOT_DIR/tmp/accuracies_n${N}_k${K}/b${bcluster_num}_p${pcluster_num}.txt"
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
