#!/bin/bash
#SBATCH --job-name=IterNK_PredictPhage
#SBATCH --partition=gpu
#SBATCH --nodes=4
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

# Validation
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <n_input> <k_input>"
    echo "Example: $0 500,800,1200 1-3"
    exit 1
fi

# Expand inputs into arrays
n_values=($(parse_input "$1"))
k_values=($(parse_input "$2"))

# Calculate totals
total_n=${#n_values[@]}
total_k=${#k_values[@]}
total_tasks=$(( total_n * total_k ))
current_step=0

echo "🔭 Project Root: $ROOT_DIR"
echo "🔭 Submitted combinations..."
echo "--------------------------------"

for n in "${n_values[@]}"; do
    for k in "${k_values[@]}"; do
        echo "Running: n=$n, k=$k"
        # Place your command here, for example:
        # python3 experiment.py --n "$n" --k "$k"
    done
done

echo "--------------------------------"
echo "🔭 Running $total_tasks combinations..."
echo "--------------------------------"

# # --- Phase 1: Downsampling ---
# echo "🧹 Beginning downsampling..."
# for n in "${n_values[@]}"; do
#     for k in "${k_values[@]}"; do
#         echo "Downsampling: n=$n, k=$k"
#         # FIXED: Added missing backslash after 'minhash'
#         python3 "$ROOT_DIR/scripts/downsampling.py" \
#             --nk "$n $k" \
#             --method "minhash"
#     done
# done
# echo "✅ Downsampling complete."

# --- Phase 2: Execution ---
echo "🚀 Beginning execution..."
for n in "${n_values[@]}"; do
    for k in "${k_values[@]}"; do
        ((current_step++))
        echo "[$current_step/$total_tasks] Processing: n=$n, k=$k"
        
        # FIXED: Ensured paths and arguments are quoted
        python3 "$ROOT_DIR/scripts/FFNN_inner.py" \
            --nk " $n $k" \
            --logging
    done
done
echo "✅ Execution complete."