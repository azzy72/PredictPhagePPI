#!/bin/bash
#SBATCH --job-name=IterECn500k12
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --mem=50G
#SBATCH --cpus-per-task=25
#SBATCH --gres=shard:1
#SBATCH --time=72:00:00
#!SBATCH --begin=15:20:00
#SBATCH --output=/home/projects/s215045/PredictPhagePPI/tmp/%j-%x.out
#SBATCH --error=/home/projects/s215045/PredictPhagePPI/tmp/%j-%x.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s215045@student.dtu.dk

# ── Argument parsing ──────────────────────────────────────────────────────────
ROOT_DIR=$(git rev-parse --show-toplevel)
DATA_DIR="$ROOT_DIR/data_prod/"

# Defaults
DRY_RUN=false
DIR_IN_NN_RUN="$ROOT_DIR/nn_runs/iter_excl_PFI/"
N=500
K=12
DOWNDIR = "encoded_sketches" # "SM_sketches", "encoded_sketches", "encoded_sketches_data2", "SM_sketches_data2"
ROOT_DIR=$(git rev-parse --show-toplevel)
DATA_DIR="$ROOT_DIR/data_prod"
BACT_CLUSTER_FILE="$DATA_DIR/$DOWNDIR/sim_matrices/combined_bact_clusters_n${N}_k${K}.csv"
PHAGE_CLUSTER_FILE="$DATA_DIR/$DOWNDIR/sim_matrices/combined_phage_clusters_n${N}_k${K}.csv"
NK_VALS="${N} ${K}"

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --dir-in-nn-run PATH      Output directory for NN runs (default: $DIR_IN_NN_RUN)"
    echo "  --bact-cluster-file PATH  Bacteria cluster CSV (default: $BACT_CLUSTER_FILE)"
    echo "  --phage-cluster-file PATH Phage cluster CSV (default: $PHAGE_CLUSTER_FILE)"
    echo "  --nk-vals 'N K'           n and k values as a quoted pair (default: '$NK_VALS')"
    echo "  --dry-run                 Print commands without executing them"
    echo "  -h, --help                Show this help message"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)            DRY_RUN=true;              shift ;;
        --dir-in-nn-run)      DIR_IN_NN_RUN="$2";        shift 2 ;;
        --bact-cluster-file)  BACT_CLUSTER_FILE="$2";    shift 2 ;;
        --phage-cluster-file) PHAGE_CLUSTER_FILE="$2";   shift 2 ;;
        --nk-vals)            NK_VALS="$2";              shift 2 ;;
        -h|--help)            usage ;;
        *) echo "[ERROR] Unknown argument: $1" >&2; usage ;;
    esac
done

# Validate required files exist (skip in dry-run)
if ! $DRY_RUN; then
    for f in "$BACT_CLUSTER_FILE" "$PHAGE_CLUSTER_FILE"; do
        if [[ ! -f "$f" ]]; then
            echo "[ERROR] File not found: $f" >&2
            exit 1
        fi
    done
fi

if $DRY_RUN; then
    echo "[DRY-RUN] No commands will be executed."
fi

echo "Configuration:"
echo "  DIR_IN_NN_RUN      = $DIR_IN_NN_RUN"
echo "  BACT_CLUSTER_FILE  = $BACT_CLUSTER_FILE"
echo "  PHAGE_CLUSTER_FILE = $PHAGE_CLUSTER_FILE"
echo "  NK_VALS            = $NK_VALS"

# ── Helper ────────────────────────────────────────────────────────────────────
run() {
    # Prints the command, then executes it unless --dry-run was passed.
    echo "+ $*"
    if ! $DRY_RUN; then
        "$@"
    fi
}

# Remaining configuration
RAW_DIR="$ROOT_DIR/raw_data/phagehost_KU/"

# 1. Collect cluster groups from CSV file
# Extract unique cluster numbers from the CSV (skip header)
bclusters=$(tail -n +2 "$BACT_CLUSTER_FILE" | awk -F',' '{print $2}' | sort -nu)
echo "Recognized these bact clusters: $bclusters"

pclusters=$(tail -n +3 "$PHAGE_CLUSTER_FILE" | awk -F',' '{print $3}' | sort -u)
echo "Recognized these phage clusters: $pclusters"

# Calculate totals for the progress bar
bcluster_count=$(echo "$bclusters" | wc -w)
pcluster_count=$(echo "$pclusters" | wc -w)
total_tasks=$((bcluster_count * pcluster_count))
current_task=0
echo "Starting training for $total_tasks cluster pairs..."

# 2. Training Loop
for bcluster_num in $bclusters; do
    bact_strains=$(tail -n +2 "$BACT_CLUSTER_FILE" | awk -F',' -v c="$bcluster_num" '$2==c {print $1}' | paste -sd ',' -)
    bact_strains=$(echo "$bact_strains" | sed 's/_reoriented//g')
    echo "Cluster $bcluster_num contains strains: $bact_strains"

    for pcluster_num in $pclusters; do
        phage_strains=$(tail -n +3 "$PHAGE_CLUSTER_FILE" | awk -F',' -v c="$pcluster_num" '$3==c {print $1}' | paste -sd ',' -)
        echo "Cluster $pcluster_num contains phages: $phage_strains"
        ((current_task++))

        # --- Progress Bar Logic ---
        percent=$((current_task * 100 / total_tasks))
        filled=$((percent / 4))
        empty=$((25 - filled))

        printf -v bar_str "%${filled}s" ""; bar_str=${bar_str// /#}
        printf -v space_str "%${empty}s" ""; space_str=${space_str// /-}

        printf "\rProgress: [%s%s] %d%% (%d/%d) | Current: cluster_%s/%s " \
               "$bar_str" "$space_str" "$percent" "$current_task" "$total_tasks" "$bcluster_num" "$pcluster_num"

        CUSTOM_OUT="iter_excl_PFI/cluster_b${bcluster_num}_p${pcluster_num}"
        run python3 "$ROOT_DIR/scripts/FFNN_inner.py" \
            --nk $NK_VALS \
            --cv \
            --kf_n_splits 4 \
            --exclude_clusters \
            --exclude_bact_clusters $bact_strains \
            --exclude_phage_clusters $phage_strains \
            --test_on_excluded \
            --perform_pfi \
            --out "$CUSTOM_OUT" \
            --logging

        accuracies=""
        echo "Extracting accuracy for pair: $bcluster_num / $pcluster_num"
        if ! $DRY_RUN; then
            acc=$(find "$ROOT_DIR/nn_runs/${CUSTOM_OUT}_run*/log_run*.txt" -exec grep "Final test loss:" {} + | awk -F'test accuracy: ' '{print $2}')
            if [[ -n "$acc" ]]; then
                accuracies="${accuracies}"$'\n'"$acc"
            fi
            echo "$accuracies"
        fi
    echo ""  # Newline after each cluster pair for better readability
    done
done

# 3. Post-Processing: Average Accuracies
echo "-------------------------------------------------------"
echo "Calculating Average Test Accuracy..."
echo "-------------------------------------------------------"
if ! $DRY_RUN; then
    average=$(echo "$accuracies" | awk '
        { sum += $1; count++ }
        END { if (count > 0) print sum / count; else print "0" }
    ')
    total_runs=$(echo "$accuracies" | wc -l)
    echo "Total Runs Analyzed: $total_runs"
    echo "Global Average Test Accuracy: $average"
fi

# 4. Collecting results
echo "📊 Collecting results..."
run python3 "$ROOT_DIR/scripts/collect_iterres.py" --base_dir "$DIR_IN_NN_RUN"