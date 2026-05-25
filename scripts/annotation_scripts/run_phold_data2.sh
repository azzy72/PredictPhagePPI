#!/bin/bash
#SBATCH --job-name=phold_d2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=shard:1
#SBATCH --time=04:00:00
#SBATCH --output=/home/projects/s215045/PredictPhagePPI/tmp/%j-%x.out
#SBATCH --error=/home/projects/s215045/PredictPhagePPI/tmp/%j-%x.err

# Submit as array after pharokka has finished:
#   sbatch --array=1-54 run_phold_data2.sh

ROOT_DIR=$(git rev-parse --show-toplevel)
INPUT_DIR="$ROOT_DIR/raw_data/phagehost_KU/data2_phages"
PHAROKKA_OUTDIR="$ROOT_DIR/data_prod/pharokka"
PHOLD_OUTDIR="$ROOT_DIR/data_prod/phold"

FILES=($INPUT_DIR/*.fasta)
INDEX=$((SLURM_ARRAY_TASK_ID - 1))
INPUT_FASTA=${FILES[$INDEX]}
PHAGE_NAME=$(basename "$INPUT_FASTA" .fasta)

PHAROKKA_GBK="$PHAROKKA_OUTDIR/${PHAGE_NAME}_pharokka_output/pharokka.gbk"

if [ ! -f "$PHAROKKA_GBK" ]; then
    echo "ERROR: Pharokka GBK not found for $PHAGE_NAME — run pharokka first" >&2
    exit 1
fi

echo "[$SLURM_ARRAY_TASK_ID] Running Phold for: $PHAGE_NAME"

mkdir -p "$PHOLD_OUTDIR"

phold run \
    -i "$PHAROKKA_GBK" \
    -o "$PHOLD_OUTDIR/${PHAGE_NAME}_phold_output" \
    -t $SLURM_CPUS_PER_TASK \
    -p "$PHAGE_NAME" \
    --foldseek_gpu

echo "Finished: $PHAGE_NAME"
