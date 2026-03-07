#!/bin/bash

# Configuration
BACTA_FILE="bacteriaKU_cleaned.fasta"
PHAGE_FILE="phage_cleaned.fasta"
NK_VALS="500 12"

# 1. Extract Names
bact_names=$(grep ">" "$BACTA_FILE" | awk '{print $2}')
phage_names=$(grep ">" "$PHAGE_FILE" | awk -F'_' '{print $NF}')

# 2. Training Loop
for bact in $bact_names; do
    for phage in $phage_names; do
        CUSTOM_OUT="excl_${bact}_${phage}"
        
        python3 FFNN_inner.py \
            --nk $NK_VALS \
            --exclude_noninteractions \
            --exclude_bacts "$bact" \
            --exclude_phages "$phage" \
            --out "$CUSTOM_OUT" \
            --use_encoded \
            --logging
    done
done

# 3. Post-Processing: Average Accuracies
echo "-------------------------------------------------------"
echo "Calculating Average Test Accuracy..."
echo "-------------------------------------------------------"

# Extract accuracy values from all log_run*.txt files produced in this session
# The regex looks for 'test accuracy:' followed by the numerical value
accuracies=$(find nn_runs/ -name "log_run*.txt" -exec grep "Final test loss:" {} + | awk -F'test accuracy: ' '{print $2}')

# Perform the average using awk
average=$(echo "$accuracies" | awk '
    { sum += $1; count++ } 
    END { if (count > 0) print sum / count; else print "0" }
')

total_runs=$(echo "$accuracies" | wc -l)

echo "Total Runs Analyzed: $total_runs"
echo "Global Average Test Accuracy: $average"