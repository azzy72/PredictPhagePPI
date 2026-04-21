#!/bin/bash
#SBATCH --job-name=phage_annot
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu
#SBATCH --time=04:00:00          
#!SBATCH --begin=15:20:00
#SBATCH --output=/home/projects/s215045/PredictPhagePPI/tmp/%j-%x.out
#SBATCH --error=/home/projects/s215045/PredictPhagePPI/tmp/%j-%x.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s215045@student.dtu.dk

# --- SETUP ---
# Configuration
ROOT_DIR=$(git rev-parse --show-toplevel)
DATA_DIR="$ROOT_DIR/data_prod/"
RAW_DIR="$ROOT_DIR/raw_data/phagehost_KU/"

INPUT_DIR="$RAW_DIR/phage_genomes/" # Path to your split fasta files
DB_DIR="$ROOT_DIR/../phold/pharokka_db" # Path to Pharokka Database
source $(conda info --base)/etc/profile.d/conda.sh # Activate your environment
conda activate pharokkaENV

# --- LOGIC TO GET CURRENT FILE ---
# Create an array of files to process
FILES=($INPUT_DIR/*.fasta)
# SLURM_ARRAY_TASK_ID starts at 1, so we subtract 1 for bash index
INDEX=$((SLURM_ARRAY_TASK_ID - 1))
INPUT_FASTA=${FILES[$INDEX]}

# Extract filename without extension for naming outputs
PHAGE_NAME=$(basename "$INPUT_FASTA" .fasta)

echo "Starting processing for: $PHAGE_NAME"
echo "Input file: $INPUT_FASTA"

# Create specific output directories
PHAROKKA_OUT="${DATA_DIR}/pharokka/${PHAGE_NAME}_pharokka_output"
PHOLD_OUT="${DATA_DIR}/phold/${PHAGE_NAME}_phold_output"
PLOT_OUT="${DATA_DIR}/phold_plots/${PHAGE_NAME}_phold_plot"

# --- STEP 1: PHAROKKA ---
# Run annotation to generate Genbank file
pharokka.py \
    -i "$INPUT_FASTA" \
    -d "$DB_DIR" \
    -o "$PHAROKKA_OUT" \
    -t $SLURM_CPUS_PER_TASK \
    --fast

# --- STEP 2: PHOLD ---
# Run Phold using the Genbank output from Pharokka
# Note: pharokka usually names the gbk file 'pharokka.gbk' inside the output folder
conda activate pholdENV
phold run \
    -i "${PHAROKKA_OUT}/pharokka.gbk" \
    -o "$PHOLD_OUT" \
    -t $SLURM_CPUS_PER_TASK \
    -p "$PHAGE_NAME" \
    --foldseek_gpu

# --- STEP 3: PHOLD PLOT ---
# Generate the annotation map plot
# -i: Uses the .gbk file produced in the previous step (located inside PHOLD_OUT)
# -t: Uses the PHAGE_NAME as the title on the plot
phold plot \
    -i "${PHOLD_OUT}/${PHAGE_NAME}.gbk" \
    -o "$PLOT_OUT" \
    -t "$PHAGE_NAME"

echo "Finished processing $PHAGE_NAME"