#!/bin/bash
#SBATCH --job-name=IterExcl_PredictPhage
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
DATA_DIR="$ROOT_DIR/data_prod/"
RAW_DIR="$ROOT_DIR/raw_data/phagehost_KU/"
BACTA_FILE="$RAW_DIR/bacteriaKU_cleaned.fasta"
PHAGE_FILE="$RAW_DIR/phage_cleaned.fasta"
NK_VALS="500 12"

# 1. Collect names
bact_names=$(grep ">" "$BACTA_FILE" | awk '{print $2}' | sort -u)
echo "Recognized these bact names: $bact_names"
phage_names=$(grep ">" "$PHAGE_FILE" | grep -v "training" | awk -F'_' '{print $NF}' | sort -u)
echo "Recognized these phage names: $phage_names"

# Calculate totals for the progress bar
bact_count=$(echo "$bact_names" | wc -w)
phage_count=$(echo "$phage_names" | wc -w)
total_tasks=$((bact_count * phage_count))
current_task=0
echo "Starting training for $total_tasks pairs..."

# 2. Training Loop
for bact in $bact_names; do
    for phage in $phage_names; do
        ((current_task++))
        
        # --- Progress Bar Logic ---
        percent=$((current_task * 100 / total_tasks))
        filled=$((percent / 4)) # Bar length of 25 characters
        empty=$((25 - filled))
        
        # Create strings for the bar
        printf -v bar_str "%${filled}s" ""; bar_str=${bar_str// /#}
        printf -v space_str "%${empty}s" ""; space_str=${space_str// /-}
        
        # Print the progress bar (\r keeps it on the same line)
        printf "\rProgress: [%s%s] %d%% (%d/%d) | Current: %s/%s " \
               "$bar_str" "$space_str" "$percent" "$current_task" "$total_tasks" "$bact" "$phage"

        CUSTOM_OUT="iter_excl/${bact}_${phage}"
        python3 "$ROOT_DIR/scripts/FFNN_inner.py" \
            --nk $NK_VALS \
            --cv \
            --kf_n_splits 4 \
            --exclude_noninteractions \
            --exclude_bacts "$bact" \
            --exclude_phages "$phage" \
            --out "$CUSTOM_OUT" \
            --logging
    done
done

# 3. Post-Processing: Average Accuracies
echo "-------------------------------------------------------"
echo "Calculating Average Test Accuracy..."
echo "-------------------------------------------------------"

# Extract accuracy values from all log_run*.txt files produced in this session
# The regex looks for 'test accuracy:' followed by the numerical value
accuracies=$(find "$ROOT_DIR/nn_runs/${CUSTOM_OUT}_run*/log_run*.txt" -exec grep "Final test loss:" {} + | awk -F'test accuracy: ' '{print $2}')

for bact in $bact_names; do
    for phage in $phage_names; do
        echo "Extracting accuracy for pair: $bact / $phage"
        #echo "$PROJ_DIR/nn_runs/excl_${bact}_${phage}_run1/log_run1.txt"
        acc=$(find "$PROJ_DIR/nn_runs/excl_${bact}_${phage}_run1/log_run1.txt" -exec grep "Final test loss:" {} + | awk -F'test accuracy: ' '{print $2}')
        if [[ -n "$acc" ]]; then
            accuracies="${accuracies}"$'\n'"$acc"
        fi
        echo $accuracies
    done
done
# Perform the average using awk
average=$(echo "$accuracies" | awk '
    { sum += $1; count++ } 
    END { if (count > 0) print sum / count; else print "0" }
')

total_runs=$(echo "$accuracies" | wc -l)

echo "Total Runs Analyzed: $total_runs"
echo "Global Average Test Accuracy: $average"

# 4. Collecting results
echo "📊 Collecting results..."
python3 "$ROOT_DIR/scripts/collect_iterres.py" --base_dir "$DIR_IN_NN_RUN