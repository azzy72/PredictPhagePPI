#!/bin/bash
#SBATCH --job-name=ProkkaBatch
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=50G
#SBATCH --time=04:00:00
#SBATCH --output=/home/projects/s215045/PredictPhagePPI/tmp/%j-%x.out
#SBATCH --error=/home/projects/s215045/PredictPhagePPI/tmp/%j-%x.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s215045@student.dtu.dk

# 1. Setup Directories
ROOT_DIR=$(git rev-parse --show-toplevel)
# This is where your merged .fasta files are located from the previous step
INPUT_DIR="concatenated_bacteria" 
# This is the parent directory for all Prokka results
BASE_OUTDIR="$ROOT_DIR/data_prod/prokka_results"

mkdir -p "$BASE_OUTDIR"

# 2. Loop through the merged fasta files
for file in "$INPUT_DIR"/*.fasta; do
    
    # Extract the base name (e.g., "E_coli_merged" or "E_coli")
    filename=$(basename "$file" .fasta)
    
    # Define a specific output folder for THIS bacteria
    # Prokka requires a unique, non-existent folder or --force
    SPECIFIC_OUTDIR="$BASE_OUTDIR/$filename"

    echo "Starting annotation for: $filename"

    prokka --outdir "$SPECIFIC_OUTDIR" \
           --prefix "$filename" \
           --cpus 12 \
           --metagenome \
           --addgenes \
           --force \
           --quiet \
           "$file"

done

echo "All annotations complete."