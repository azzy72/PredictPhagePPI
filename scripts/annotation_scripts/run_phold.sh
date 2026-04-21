#!/bin/bash
#SBATCH --job-name=phold_step2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu
#SBATCH --time=04:00:00          
#SBATCH --output=/home/projects/s215045/PredictPhagePPI/tmp/%j-%x.out
#SBATCH --error=/home/projects/s215045/PredictPhagePPI/tmp/%j-%x.err

# --- SETUP ---
ROOT_DIR=$(git rev-parse --show-toplevel)
DATA_DIR="$ROOT_DIR/data_prod/"
RAW_DIR="$ROOT_DIR/raw_data/phagehost_KU/"
INPUT_DIR="$RAW_DIR/phage_genomes/" 

source $(conda info --base)/etc/profile.d/conda.sh
conda activate pholdENV

# --- LOGIC ---
FILES=($INPUT_DIR/*.fasta)
INDEX=$((SLURM_ARRAY_TASK_ID - 1))
INPUT_FASTA=${FILES[$INDEX]}
PHAGE_NAME=$(basename "$INPUT_FASTA" .fasta)

PHAROKKA_OUT="${DATA_DIR}/pharokka/${PHAGE_NAME}_pharokka_output"
mkdir -p "${DATA_DIR}/phold"
PHOLD_OUT="${DATA_DIR}/phold/${PHAGE_NAME}_phold_output"

# Check if Pharokka output exists first
if [ ! -f "${PHAROKKA_OUT}/pharokka.gbk" ]; then
    echo "ERROR: Pharokka Genbank file not found for $PHAGE_NAME"
    exit 1
fi

echo "Running Phold Run for: $PHAGE_NAME"

phold run \
    -i "${PHAROKKA_OUT}/pharokka.gbk" \
    -o "$PHOLD_OUT" \
    -t $SLURM_CPUS_PER_TASK \
    -p "$PHAGE_NAME" \
    --foldseek_gpu

echo "Finished Phold Run for $PHAGE_NAME"