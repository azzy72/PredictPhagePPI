#!/bin/bash
ROOT_DIR=$(git rev-parse --show-toplevel)
INPUT_DIR="$ROOT_DIR/raw_data/phagehost_KU/data2_phages"
DB_DIR="$ROOT_DIR/../phold/pharokka_db"
PHAROKKA_OUTDIR="$ROOT_DIR/data_prod/pharokka"

mkdir -p "$PHAROKKA_OUTDIR"

for INPUT_FASTA in "$INPUT_DIR"/*.fasta; do
    PHAGE_NAME=$(basename "$INPUT_FASTA" .fasta)
    echo "Running Pharokka for: $PHAGE_NAME"

    pharokka.py \
        -i "$INPUT_FASTA" \
        -d "$DB_DIR" \
        -o "$PHAROKKA_OUTDIR/${PHAGE_NAME}_pharokka_output" \
        -f \
        --fast

    echo "Finished: $PHAGE_NAME"
done
