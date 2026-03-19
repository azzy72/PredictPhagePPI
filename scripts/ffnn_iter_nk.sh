#!/bin/bash
#SBATCH --job-name=IterNK_PredictPhage
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --mem=50G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu
#SBATCH --output=/home/projects/s215045/PredictPhagePPI/tmp/%j-%x.out
#SBATCH --error=/home/projects/s215045/PredictPhagePPI/tmp/%j-%x.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s215045@student.dtu.dk

# Configuration
ROOT_DIR=$(git rev-parse --show-toplevel)
DIR_IN_NN_RUNS="$ROOT_DIR/nn_runs/IterNK/"
COLLECTION_ONLY=false
PASSTHROUGH_ARGS=()
METHOD=""

# Function to parse ranges/lists
parse_input() {
    local input=$1
    if [[ "$input" == *","* ]]; then
        echo "${input//,/ }"
    elif [[ "$input" == *"-"* ]]; then
        seq ${input/-/ }
    else
        echo "$input"
    fi
}

# --- Robust Argument Parsing ---
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --nk)
            # Bash-only: Used to define the loop ranges
            N_INPUT="$2"
            K_INPUT="$3"
            shift 3
            ;;
        --collection-only)
            # Bash-only: Prevents the execution of Phase 1 and 2
            COLLECTION_ONLY=true
            shift
            ;;
        --method)
            # Shared: Used in downsampling.py and the output path
            METHOD="$2"
            shift 2
            ;;
        --*)
            # Passthrough: Any other flag starting with -- is sent to FFNN_inner.py
            # This handles --cv, --smote, --n_epochs, --learning_rate, etc.
            PASSTHROUGH_ARGS+=("$1")
            if [[ -n "$2" && ! "$2" =~ ^-- ]]; then
                PASSTHROUGH_ARGS+=("$2")
                shift 2
            else
                shift 1
            fi
            ;;
        *)
            shift
            ;;
    esac
done

# Validation
if [[ -z "$N_INPUT" || -z "$K_INPUT" ]]; then
    echo "❌ Error: Missing --nk <n_range> <k_range>"
    exit 1
fi

n_values=($(parse_input "$N_INPUT"))
k_values=($(parse_input "$K_INPUT"))

if [ "$COLLECTION_ONLY" = true ]; then
    echo "⚡ Mode: Collection Only"
else
    # --- Phase 1: Downsampling ---
    echo "🧹 Downsampling..."
    for n in "${n_values[@]}"; do
        for k in "${k_values[@]}"; do
            python3 "$ROOT_DIR/scripts/downsampling.py" \
                --nk "$n" "$k" \
                ${METHOD:+--method "$METHOD"}
        done
    done

    # --- Phase 2: Execution ---
    echo "🚀 Executing FFNN Iterations..."
    mkdir -p "$DIR_IN_NN_RUNS"
    for n in "${n_values[@]}"; do
        for k in "${k_values[@]}"; do
            echo "Running n=$n, k=$k..."
            
            # The script provides its own --nk and --out inside the loop
            # Everything else from PASSTHROUGH_ARGS is appended at the end
            python3 "$ROOT_DIR/scripts/FFNN_inner.py" \
                --nk "$n" "$k" \
                --out "IterNK/IterNK_n${n}_k${k}_${METHOD:-sourmash}" \
                --logging \
                "${PASSTHROUGH_ARGS[@]}"
        done
    done
fi

# --- Phase 3: Collection ---
echo "📊 Collecting results..."
python3 "$ROOT_DIR/scripts/collect_iterres.py" --base_dir "$DIR_IN_NN_RUNS"