#!/bin/bash
#SBATCH --job-name=IterExclClusPFI
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --mem=50G
#SBATCH --cpus-per-task=25
#SBATCH --gres=shard:1
##SBATCH --time=00:00:00
#!SBATCH --begin=15:20:00
#SBATCH --output=/home/projects/s215045/PredictPhagePPI/tmp/%j-%x.out
#SBATCH --error=/home/projects/s215045/PredictPhagePPI/tmp/%j-%x.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s215045@student.dtu.dk

# Configuration
ROOT_DIR=$(git rev-parse --show-toplevel)
DATA_DIR="$ROOT_DIR/data_prod/"
RAW_DIR="$ROOT_DIR/raw_data/phagehost_KU/"
DIR_IN_NN_RUN="$ROOT_DIR/nn_runs/iter_excl_PFI_old/"
BACT_CLUSTER_FILE="$DATA_DIR/bact_clusters_with_genus.csv"
PHAGE_CLUSTER_FILE="$DATA_DIR/phage_clusters.csv"
#PHAGE_FILE="$RAW_DIR/phage_cleaned.fasta"
NK_VALS="500 12"

# 1. Collect cluster groups from CSV file
# Extract unique cluster numbers from the CSV (skip header)
bclusters=$(tail -n +2 "$BACT_CLUSTER_FILE" | awk -F',' '{print $2}' | sort -nu)
echo "Recognized these bact clusters: $bclusters"

pclusters=$(tail -n +2 "$PHAGE_CLUSTER_FILE" | awk -F',' '{print $2}' | sort -u)
echo "Recognized these phage clusters: $pclusters"

# Calculate totals for the progress bar
bcluster_count=$(echo "$bclusters" | wc -w)
pcluster_count=$(echo "$pclusters" | wc -w)
total_tasks=$((bcluster_count * pcluster_count))
current_task=0
echo "Starting training for $total_tasks cluster pairs..."

# 2. Training Loop
for bcluster_num in $bclusters; do
    # Get all strain names for this cluster (first column where second column matches cluster_num)
    bact_strains=$(tail -n +2 "$BACT_CLUSTER_FILE" | awk -F',' -v c="$bcluster_num" '$2==c {print $1}' | paste -sd ',' -)
    bact_strains=$(echo "$bact_strains" | sed 's/_reoriented//g') # if "_reoriented" is in the strain name, remove it
    echo "Cluster $bcluster_num contains strains: $bact_strains"
    
    for pcluster_num in $pclusters; do
        phage_strains=$(tail -n +2 "$PHAGE_CLUSTER_FILE" | awk -F',' -v c="$pcluster_num" '$2==c {print $1}' | paste -sd ',' -)
        echo "Cluster $pcluster_num contains phages: $phage_strains"
        ((current_task++))
        
        # --- Progress Bar Logic ---
        percent=$((current_task * 100 / total_tasks))
        filled=$((percent / 4)) # Bar length of 25 characters
        empty=$((25 - filled))
        
        # Create strings for the bar
        printf -v bar_str "%${filled}s" ""; bar_str=${bar_str// /#}
        printf -v space_str "%${empty}s" ""; space_str=${space_str// /-}
        
        # Print the progress bar (\r keeps it on the same line)
        printf "\rProgress: [%s%s] %d%% (%d/%d) | Current: cluster_%s/%s " \
               "$bar_str" "$space_str" "$percent" "$current_task" "$total_tasks" "$bcluster_num" "$pcluster_num"

        CUSTOM_OUT="iter_excl_PFI_old/cluster_b${bcluster_num}_p${pcluster_num}"
        python3 "$ROOT_DIR/scripts/FFNN_inner.py" \
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

        # Extract accuracy values from all log_run*.txt files produced in this session
        # The regex looks for 'test accuracy:' followed by the numerical value
        accuracies=""
        echo "Extracting accuracy for pair: $bcluster_num / $pcluster_num"
        acc=$(find "$ROOT_DIR/nn_runs/${CUSTOM_OUT}_run*/log_run*.txt" -exec grep "Final test loss:" {} + | awk -F'test accuracy: ' '{print $2}')
        if [[ -n "$acc" ]]; then
            accuracies="${accuracies}"$'\n'"$acc"
        fi
        echo $accuracies

    done
done

# 3. Post-Processing: Average Accuracies
echo "-------------------------------------------------------"
echo "Calculating Average Test Accuracy..."
echo "-------------------------------------------------------"
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
python3 "$ROOT_DIR/scripts/collect_iterres.py" --base_dir "$DIR_IN_NN_RUN"