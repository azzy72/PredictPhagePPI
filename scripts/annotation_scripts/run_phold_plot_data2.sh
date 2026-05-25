#!/bin/bash
#SBATCH --job-name=pholdplot_d2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=shard:1
#SBATCH --time=01:00:00
#SBATCH --output=/home/projects/s215045/PredictPhagePPI/tmp/%j-%x.out
#SBATCH --error=/home/projects/s215045/PredictPhagePPI/tmp/%j-%x.err

# Submit as array after phold has finished:
#   sbatch --array=1-54 run_phold_plot_data2.sh

ROOT_DIR=$(git rev-parse --show-toplevel)
INPUT_DIR="$ROOT_DIR/raw_data/phagehost_KU/data2_phages"
PHOLD_OUTDIR="$ROOT_DIR/data_prod/phold"
PLOT_OUTDIR="$ROOT_DIR/data_prod/phold_plots"

mkdir -p "$PLOT_OUTDIR"

for INPUT_FASTA in "$INPUT_DIR"/*.fasta; do
    PHAGE_NAME=$(basename "$INPUT_FASTA" .fasta)
    PHOLD_GBK="$PHOLD_OUTDIR/${PHAGE_NAME}_phold_output/${PHAGE_NAME}.gbk"

    if [ ! -f "$PHOLD_GBK" ]; then
        echo "ERROR: Phold GBK not found for $PHAGE_NAME — skipping" >&2
        continue
    fi

    echo "Plotting: $PHAGE_NAME"

    phold plot \
        -i "$PHOLD_GBK" \
        -o "$PLOT_OUTDIR/${PHAGE_NAME}_phold_plot" \
        -t "$PHAGE_NAME" \
        -f

    echo "Finished: $PHAGE_NAME"
done