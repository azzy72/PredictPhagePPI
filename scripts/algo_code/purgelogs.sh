#!/bin/bash

######################   Constants   ######################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/../data"
RAW_DATA_DIR="$DATA_DIR/raw_data"
NNALIGN_DIR="$DATA_DIR/nnalign_out"
FFNN_DIR="$DATA_DIR/ffnn_out"

######################   Iterating over all HLA alleles - Purging files ending in ".log"   ######################
for ALLELE_IN_DIR in $(find "$RAW_DATA_DIR" -mindepth 1 -maxdepth 1 -type d | grep -v "_out"); do
    ALLELE_NAME=$(basename "$ALLELE_IN_DIR")
    # Clear logs from nnalign_out
    ALLELE_OUT_DIR="$NNALIGN_DIR/$ALLELE_NAME"
    for logfile in $(find $ALLELE_OUT_DIR -type f -name "*.log"); do
        rm $logfile
    done
    # Clear logs from ffnn_out
    ALLELE_OUT_DIR="$NNALIGN_DIR/$ALLELE_NAME"
    for logfile in $(find $ALLELE_OUT_DIR -type f -name "*.log"); do
        rm $logfile
    done
done
