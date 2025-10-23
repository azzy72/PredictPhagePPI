#!/bin/bash

######################   Constants   ######################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/../data"
RAW_DATA_DIR="$DATA_DIR/raw_data"
HOBOHM_DATA_DIR="$DATA_DIR/hobohm_data"
FFNN_DIR="$DATA_DIR/ffnn_out"
NNALIGN_DIR="$DATA_DIR/nnalign_out"
HOBOHM_DIR_NAME="hobohm_data_out"
RAW_DIR_NAME="raw_data_out"
PEARSON_DIR="$DATA_DIR/pearson_out"

#######################   Create Output Directory   ######################
mkdir -p "$PEARSON_DIR"


#######################   Xycorr for FFNN   ######################
for ALLELE_IN_DIR in $(find "$RAW_DATA_DIR" -mindepth 1 -maxdepth 1 -type d); do
    ALLELE_NAME=$(basename "$ALLELE_IN_DIR")
    echo "Processing allele: $ALLELE_NAME"

    FFNN_RAW_PRED=$(find "$FFNN_DIR/$ALLELE_NAME/$RAW_DIR_NAME" -mindepth 1 -maxdepth 1 -type f -name "*_predictions.txt")
    FFNN_HOBOHM_PRED=$(find "$FFNN_DIR/$ALLELE_NAME/$HOBOHM_DIR_NAME" -mindepth 1 -maxdepth 1 -type f -name "*_predictions.txt")
    NNALIGN_RAW_PRED=$(find "$NNALIGN_DIR/$ALLELE_NAME/$RAW_DIR_NAME" -mindepth 1 -maxdepth 1 -type f -name "*_predictions.txt")
    NNALIGN_HOBOHM_PRED=$(find "$NNALIGN_DIR/$ALLELE_NAME/$HOBOHM_DIR_NAME" -mindepth 1 -maxdepth 1 -type f -name "*_predictions.txt")

    # Error handling for FFNN_RAW_PRED
    if [[ -z "$FFNN_RAW_PRED" ]]; then
        echo "Warning: No _predictions.txt file found for FFNN_RAW in $ALLELE_NAME. Skipping to next allele."
        continue
    fi
    # Error handling for FFNN_HOBOHM_PRED
    if [[ -z "$FFNN_HOBOHM_PRED" ]]; then
        echo "Warning: No _predictions.txt file found for FFNN_HOBOHM in $ALLELE_NAME. Skipping to next allele."
        continue
    fi
    # Error handling for NNALIGN_RAW_PRED
    if [[ -z "$NNALIGN_RAW_PRED" ]]; then
        echo "Warning: No _predictions.txt file found for NNALIGN_RAW in $ALLELE_NAME. Skipping to next allele."
        continue
    fi
    # Error handling for NNALIGN_HOBOHM_PRED
    if [[ -z "$NNALIGN_HOBOHM_PRED" ]]; then
        echo "Warning: No _predictions.txt file found for NNALIGN_HOBOHM in $ALLELE_NAME. Skipping to next allele."
        continue
    fi

    cat "$FFNN_RAW_PRED" | awk -F',' '{print $2,$3}' | "$SCRIPT_DIR/xycorr" > "$PEARSON_DIR/${ALLELE_NAME}_ffnn_raw_pearson.txt"
    cat "$FFNN_HOBOHM_PRED" | awk -F',' '{print $2,$3}' | "$SCRIPT_DIR/xycorr" > "$PEARSON_DIR/${ALLELE_NAME}_ffnn_hobohm_pearson.txt"
    cat "$NNALIGN_RAW_PRED" | awk -F',' '{print $2,$3}' | "$SCRIPT_DIR/xycorr" > "$PEARSON_DIR/${ALLELE_NAME}_nnalign_raw_pearson.txt"
    cat "$NNALIGN_HOBOHM_PRED" | awk -F',' '{print $2,$3}' | "$SCRIPT_DIR/xycorr" > "$PEARSON_DIR/${ALLELE_NAME}_nnalign_hobohm_pearson.txt"
done