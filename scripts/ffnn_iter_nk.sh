#!/bin/bash
#SBATCH --job-name=IterNK_PredictPhage
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --mem=50G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu
#!SBATCH --time=00:00:00
#!SBATCH --begin=15:20:00
#SBATCH --output=/home/projects/s215045/PredictPhagePPI/tmp/%j-%x.out
#SBATCH --error=/home/projects/s215045/PredictPhagePPI/tmp/%j-%x.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s215045@student.dtu.dk

# Configuration
ROOT_DIR=$(git rev-parse --show-toplevel)

# Function to parse either "1-3" or "500,800,1200"
parse_input() {
    local input=$1
    if [[ "$input" == *","* ]]; then
        # If it contains a comma, replace commas with spaces
        echo "${input//,/ }"
    elif [[ "$input" == *"-"* ]]; then
        # If it contains a hyphen, use seq (replaces - with space for seq)
        seq ${input/-/ }
    else
        # If it's just a single number
        echo "$input"
    fi
}

# Check for collection-only flag
COLLECTION_ONLY=false
for arg in "$@"; do
    if [ "$arg" == "--collection-only" ]; then
        COLLECTION_ONLY=true
        break
    fi
done

# Validation
if [ "$1" != "--nk" ]; then
    echo "Usage: $0 --nk <n_input> <k_input> [--collection-only]"
    echo "Example: $0 --nk 500,800,1200 1-3"
    echo "Example (collection only): $0 --nk 500,800,1200 1-3 --collection-only"
    exit 1
fi

# Expand inputs into arrays
n_values=($(parse_input "$2"))
k_values=($(parse_input "$3"))

# Calculate totals
total_n=${#n_values[@]}
total_k=${#k_values[@]}
total_tasks=$(( total_n * total_k ))
current_step=0

echo "🔭 Project Root: $ROOT_DIR"

if [ "$COLLECTION_ONLY" = true ]; then
    echo "⚡ Running in collection-only mode (skipping downsampling and execution)..."
    #Check if DIR_IN_NN_RUNS exists
    if [ ! -d "$ROOT_DIR/nn_runs/IterNK/" ]; then
        echo "❌ Error: Directory $ROOT_DIR/nn_runs/IterNK/ does not exist. Please run the full script without --collection-only first to generate results."
        exit 1
    fi
    DIR_IN_NN_RUNS="$ROOT_DIR/nn_runs/IterNK/"
else
    echo "🔭 Submitted combinations..."
    echo "--------------------------------"

    for n in "${n_values[@]}"; do
        for k in "${k_values[@]}"; do
            echo "Running: n=$n, k=$k"
        done
    done

    echo "--------------------------------"
    echo "🔭 Running $total_tasks combinations..."
    echo "--------------------------------"

    # --- Phase 1: Downsampling ---
    echo "🧹 Beginning downsampling..."
    for n in "${n_values[@]}"; do
        for k in "${k_values[@]}"; do
            echo "Downsampling: n=$n, k=$k"
            # FIXED: Added missing backslash after 'minhash'
            if [ -n "$METHOD" ]; then
                python3 "$ROOT_DIR/scripts/downsampling.py" \
                --nk "$n" "$k" \
                --method "$METHOD"
            else
                echo "No method specified, running downsampling with sourmash."
                python3 "$ROOT_DIR/scripts/downsampling.py" \
                --nk "$n" "$k"
            fi
        done
    done
    echo "✅ Downsampling complete."

    # --- Phase 2: Execution ---
    echo "🚀 Beginning execution..."
    mkdir -p "$DIR_IN_NN_RUNS"
    for n in "${n_values[@]}"; do
        for k in "${k_values[@]}"; do
            ((current_step++))
            echo "[$current_step/$total_tasks] Processing: n=$n, k=$k"
            
            # FIXED: Ensured paths and arguments are quoted
            python3 "$ROOT_DIR/scripts/FFNN_inner.py" \
                --nk "$n" "$k" \
                --out "IterNK/IterNK_n${n}_k${k}_$METHOD" \
                --logging
        done
    done
    echo "✅ FFNN iteration complete."
fi

# --- Phase 2: Collection ---
echo "📊 Collecting results and generating plots..."
python3 "$ROOT_DIR/scripts/collect_iternk_res.py" \
    --base_dir "$DIR_IN_NN_RUNS" 
echo "✅ Collection and plotting complete and can be found in $ROOT_DIR/data_prod/iterNK/"

