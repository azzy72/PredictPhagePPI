#!/bin/bash
# run_iter_excl_local.sh
# Local equivalent of submit_iter_excl.sh + ffnn_train_task.sh + ffnn_postprocess.sh.
# Runs all (bcluster, pcluster) training pairs in parallel (capped at MAX_PARALLEL),
# then runs post-processing once all pairs are done.
#
# Usage: bash run_iter_excl_local.sh [N] [K] [DOWNDIR] [MAX_PARALLEL]
#   N            : sketch size          (default: 500)
#   K            : k-mer length         (default: 12)
#   DOWNDIR      : data subdirectory    (default: encoded_sketches)
#   MAX_PARALLEL : max concurrent jobs  (default: number of CPU cores)
#
# Example: bash run_iter_excl_local.sh 500 12 encoded_sketches 8

set -euo pipefail

# ── Arguments ─────────────────────────────────────────────────────────────────
if [[ $# -gt 4 ]]; then
    echo "Usage: bash run_iter_excl_local.sh [N] [K] [DOWNDIR] [MAX_PARALLEL]" >&2
    exit 1
fi

N="${1:-500}"
K="${2:-12}"
DOWNDIR="${3:-encoded_sketches}"
MAX_PARALLEL="${4:-1}"

# ── Paths (mirrors the slurm scripts) ─────────────────────────────────────────
ROOT_DIR=$(git rev-parse --show-toplevel)
DATA_DIR="$ROOT_DIR/data_prod"
BACT_CLUSTER_FILE="$DATA_DIR/$DOWNDIR/sim_matrices/combined_bact_clusters_n${N}_k${K}.csv"
PHAGE_CLUSTER_FILE="$DATA_DIR/$DOWNDIR/sim_matrices/combined_phage_clusters_n${N}_k${K}.csv"
TASK_MAP="$ROOT_DIR/tmp/IterExcl_Taskmap_${DOWNDIR}_${N}_${K}.txt"
ACC_DIR="$ROOT_DIR/tmp/accuracies_${DOWNDIR}_n${N}_k${K}"
DIR_IN_NN_RUN="$ROOT_DIR/nn_runs/IterExcl_${DOWNDIR}_n${N}_k${K}"
LOG_DIR="$ROOT_DIR/tmp/local_logs_${DOWNDIR}_n${N}_k${K}"
CUSTOM_PARENT_DIR="IterExcl_${DOWNDIR}_n${N}_k${K}"

mkdir -p "$ROOT_DIR/tmp" "$ACC_DIR" "$LOG_DIR"

# ── Python interpreter ────────────────────────────────────────────────────────
# Defaults to the 'python' on your PATH (i.e. the one 'which python' shows).
# Override by setting the env var before running:
#   PYTHON=/usr/bin/python3.9 bash run_iter_excl_local.sh
PYTHON="${PYTHON:-$(which python3)}"
echo "Using Python: $PYTHON  ($(${PYTHON} --version 2>&1))"

# ── Flags derived from DOWNDIR (mirrors ffnn_train_task.sh) ───────────────────
ENCODED_FLAG=""
DATA2_FLAG=""
[[ "$DOWNDIR" == *"encoded_sketches"* ]] && ENCODED_FLAG="--use_encoded"
[[ "$DOWNDIR" == *"data2"* ]]            && DATA2_FLAG="--data2"

# ── 1. Build task map ─────────────────────────────────────────────────────────
echo "Building task map..."
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
echo "Running up to $MAX_PARALLEL tasks in parallel..."
echo ""

# ── 2. Training: run all pairs in parallel, capped at MAX_PARALLEL ────────────
run_training_task() {
    local task_id="$1"
    local bcluster_num="$2"
    local pcluster_num="$3"
    local log_file="$LOG_DIR/task_${task_id}_b${bcluster_num}_p${pcluster_num}.log"

    {
        echo "=== Task $task_id: bcluster=$bcluster_num  pcluster=$pcluster_num ==="

        # Resolve strain names
        local bact_strains phage_strains
        bact_strains=$(tail -n +2 "$BACT_CLUSTER_FILE" \
            | awk -F',' -v c="$bcluster_num" '$3==c {print $1}' \
            | paste -sd ',' -)
        bact_strains=$(echo "$bact_strains" | sed 's/_reoriented//g')

        phage_strains=$(tail -n +2 "$PHAGE_CLUSTER_FILE" \
            | awk -F',' -v c="$pcluster_num" '$4==c {print $1}' \
            | paste -sd ',' -)

        echo "Bact strains  : $bact_strains"
        echo "Phage strains : $phage_strains"

        local CUSTOM_OUT="$CUSTOM_PARENT_DIR/cluster_b${bcluster_num}_p${pcluster_num}"

        $PYTHON "$ROOT_DIR/scripts/FFNN_inner.py" \
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

        # Extract and persist accuracy
        local ACC_FILE="$ACC_DIR/b${bcluster_num}_p${pcluster_num}.txt"
        local acc
        acc=$(find "$ROOT_DIR/nn_runs/${CUSTOM_OUT}_run1/log_run1.txt" \
                -exec grep "Final test loss:" {} + 2>/dev/null \
            | awk -F'test accuracy: ' '{print $2}')

        if [[ -n "$acc" ]]; then
            echo "$acc" > "$ACC_FILE"
            echo "Accuracy: $acc → $ACC_FILE"
        else
            echo "WARNING: no accuracy value found for b${bcluster_num}_p${pcluster_num}"
        fi

        echo "=== Task $task_id done ==="
    } > "$log_file" 2>&1

    # Print a one-liner to stdout so progress is visible while tasks run silently
    echo "  [done] task $task_id (b${bcluster_num} / p${pcluster_num}) → $log_file"
}

# Concurrency gate: spawn up to MAX_PARALLEL background jobs at a time
declare -a pids=()
task_id=0

while IFS=' ' read -r bcluster_num pcluster_num; do
    ((task_id++))

    # If we've hit the cap, wait for any one job to finish before spawning more
    while [[ ${#pids[@]} -ge $MAX_PARALLEL ]]; do
        # Wait for the oldest job; remove it from the list
        wait "${pids[0]}"
        pids=("${pids[@]:1}")
    done

    echo "  [start] task $task_id / $total_tasks (b${bcluster_num} / p${pcluster_num})"
    run_training_task "$task_id" "$bcluster_num" "$pcluster_num" &
    pids+=($!)

done < "$TASK_MAP"

# Wait for all remaining background jobs
echo "Waiting for last $(( ${#pids[@]} )) task(s) to finish..."
for pid in "${pids[@]}"; do
    wait "$pid"
done

echo ""
echo "All $total_tasks training tasks complete."
echo ""

# ── 3. Post-Processing: Average Accuracies ────────────────────────────────────
echo "-------------------------------------------------------"
echo "Calculating Average Test Accuracy..."
echo "-------------------------------------------------------"

shopt -s nullglob
acc_files=("$ACC_DIR"/*.txt)

if [[ ${#acc_files[@]} -eq 0 ]]; then
    echo "WARNING: no accuracy files found in $ACC_DIR — skipping average."
else
    accuracies=$(cat "${acc_files[@]}")
    average=$(echo "$accuracies" | awk 'NF > 0 { sum += $1; count++ } END { if (count > 0) print sum / count; else print "0" }')
    total_runs=$(echo "$accuracies" | grep -c '[0-9]' || true)
    echo "Total Runs Analyzed:          $total_runs"
    echo "Global Average Test Accuracy: $average"
fi

# ── 4. Collecting results ─────────────────────────────────────────────────────
echo ""
echo "📊 Collecting results..."
$PYTHON "$ROOT_DIR/scripts/collect_iterres.py" \
    --base_dir "$DIR_IN_NN_RUN" \
    --out_dir "$DATA_DIR/IterExclClus_${DOWNDIR}_n${N}_k${K}/" \
    --show_cm_bar_percentage \
    --weight_pfi \
    --highlight_multi \
    --network_top_kmers 40 \
    --top_kmers 200

echo ""
echo "📊 Collecting results with harsh filtering..."
$PYTHON "$ROOT_DIR/scripts/collect_iterres.py" \
    --base_dir "$DIR_IN_NN_RUN" \
    --out_dir "$DATA_DIR/IterExclClus_${DOWNDIR}_n${N}_k${K}_harsh/" \
    --show_cm_bar_percentage \
    --weight_pfi \
    --highlight_multi \
    --network_top_kmers 40 \
    --top_kmers 200 \
    --filter_harsh

echo ""
echo "✅ All done."