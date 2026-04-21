#!/bin/bash
#SBATCH --job-name=pholdplot_step3
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=shard:1
#SBATCH --time=01:00:00          
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

PHOLD_OUT="${DATA_DIR}/phold/${PHAGE_NAME}_phold_output"
mkdir -p "${DATA_DIR}/phold_plots"
PLOT_OUT="${DATA_DIR}/phold_plots/${PHAGE_NAME}_phold_plot"

# Check if Phold GBK exists
if [ ! -f "${PHOLD_OUT}/${PHAGE_NAME}.gbk" ]; then
    echo "ERROR: Phold Genbank file not found for $PHAGE_NAME"
    exit 1
fi

echo "Running Phold Plot for: $PHAGE_NAME"

phold plot \
    -i "${PHOLD_OUT}/${PHAGE_NAME}.gbk" \
    -o "$PLOT_OUT" \
    -t "$PHAGE_NAME"

echo "Finished Plotting for $PHAGE_NAME"