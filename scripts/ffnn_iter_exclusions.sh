#!/bin/bash
#SBATCH --job-name=PredictPhage
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --mem=20G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu
#!SBATCH --time=00:00:30
#SBATCH --output=/home/projects/s215045/PredictPhagePPI/tmp/%j-%x.out
#SBATCH --error=/home/projects/s215045/PredictPhagePPI/tmp/%j-%x.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=s215045@student.dtu.dk

# Configuration
PROJ_DIR="/home/projects/s215045/PredictPhagePPI/"
DATA_DIR="$PROJ_DIR/data_prod/"
RAW_DIR="$PROJ_DIR/raw_data/phagehost_KU/"
BACTA_FILE="$RAW_DIR/bacteriaKU_cleaned.fasta"
PHAGE_FILE="$RAW_DIR/phage_cleaned.fasta"
NK_VALS="500 12"

# 1. Extract Names
bact_names=$(grep ">" "$BACTA_FILE" | awk '{print $2}')
phage_names=$(grep ">" "$PHAGE_FILE" | awk -F'_' '{print $NF}')

# 2. Training Loop
for bact in $bact_names; do
    for phage in $phage_names; do
        CUSTOM_OUT="excl_${bact}_${phage}"
        
        python3 "$PROJ_DIR/scripts/FFNN_inner.py" \
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
accuracies=$(find "$PROJ_DIR/nn_runs/${CUSTOM_OUT}_run*/log_run*.txt" -exec grep "Final test loss:" {} + | awk -F'test accuracy: ' '{print $2}')

# Perform the average using awk
average=$(echo "$accuracies" | awk '
    { sum += $1; count++ } 
    END { if (count > 0) print sum / count; else print "0" }
')

total_runs=$(echo "$accuracies" | wc -l)

echo "Total Runs Analyzed: $total_runs"
echo "Global Average Test Accuracy: $average"