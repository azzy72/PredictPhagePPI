#!/bin/bash
#SBATCH --job-name=pharo_step1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu
#SBATCH --time=04:00:00          
#SBATCH --output=/home/projects/s215045/PredictPhagePPI/tmp/%j-%x.out
#SBATCH --error=/home/projects/s215045/PredictPhagePPI/tmp/%j-%x.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s215045@student.dtu.dk

# --- SETUP ---
ROOT_DIR=$(git rev-parse --show-toplevel)
DATA_DIR="$ROOT_DIR/data_prod/"
RAW_DIR="$ROOT_DIR/raw_data/phagehost_KU/"
INPUT_DIR="$RAW_DIR/phage_genomes/" 
DB_DIR="$ROOT_DIR/../phold/pharokka_db" 

source $(conda info --base)/etc/profile.d/conda.sh
conda activate pharokkaENV

# --- LOGIC ---
FILES=($INPUT_DIR/*.fasta)
INDEX=$((SLURM_ARRAY_TASK_ID - 1))
INPUT_FASTA=${FILES[$INDEX]}
PHAGE_NAME=$(basename "$INPUT_FASTA" .fasta)

mkdir -p "${DATA_DIR}/pharokka"
PHAROKKA_OUT="${DATA_DIR}/pharokka/${PHAGE_NAME}_pharokka_output"

echo "Running Pharokka for: $PHAGE_NAME"

pharokka.py \
    -i "$INPUT_FASTA" \
    -d "$DB_DIR" \
    -o "$PHAROKKA_OUT" \
    -t $SLURM_CPUS_PER_TASK \
    --fast

echo "Finished Pharokka for $PHAGE_NAME"