#!/bin/bash

######################   Constants   ######################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/../data"
RAW_DATA_DIR="$DATA_DIR/raw_data"
HOBOHM_DATA_DIR="$DATA_DIR/hobohm_data"
NNALIGN_DIR="$DATA_DIR/nnalign_out"


######################   Ensure data directories exists   ######################
mkdir -p "$NNALIGN_DIR"


#######################   Training NNALIGN on raw data for each HLA allele   ######################
echo "Training NNALIGN on raw data for each HLA allele..."
START_TIME_RAW_TRAINING=$(date +%s)
for RAW_ALLELE_IN_DIR in $(find "$RAW_DATA_DIR" -mindepth 1 -maxdepth 1 -type d | grep -v "_out"); do
    ALLELE_NAME=$(basename "$RAW_ALLELE_IN_DIR")
    ALLELE_OUT_DIR="$NNALIGN_DIR/$ALLELE_NAME"
    NNALIGN_OUT_DIR="$ALLELE_OUT_DIR/nnalign_out"
    RAW_DATA_OUT_NNALIGN_DIR="$ALLELE_OUT_DIR/raw_data_out"
    mkdir -p "$ALLELE_OUT_DIR"
    mkdir -p "$RAW_DATA_OUT_NNALIGN_DIR"
    nohup python3 "$SCRIPT_DIR/train_nnalign.py" -dir "$RAW_ALLELE_IN_DIR" -savepath "$RAW_DATA_OUT_NNALIGN_DIR" > "$RAW_DATA_OUT_NNALIGN_DIR/train_stdout.log" 2>&1 &
done

#######################   Wait for all NNALIGN training to finish   ######################
echo "Waiting for all NNALIGN training on raw data to finish..."
wait
END_TIME_RAW_TRAINING=$(date +%s)
DURATION_RAW_TRAINING=$((END_TIME_RAW_TRAINING - START_TIME_RAW_TRAINING))
echo "Time taken for NNALIGN training on raw data: ${DURATION_RAW_TRAINING} seconds."



#####################   Training NNALIGN on hobohm data for each HLA allele   ######################
echo "Training NNALIGN on hobohm data for each HLA allele..."
START_TIME_HOBOHM_TRAINING=$(date +%s)
for HOBOHM_ALLELE_IN_DIR in $(find "$HOBOHM_DATA_DIR" -mindepth 1 -maxdepth 1 -type d | grep -v "_out"); do
    ALLELE_NAME=$(basename "$HOBOHM_ALLELE_IN_DIR")
    ALLELE_OUT_DIR="$NNALIGN_DIR/$ALLELE_NAME"
    HOBOHM_DATA_OUT_NNALIGN_DIR="$ALLELE_OUT_DIR/hobohm_data_out"
    mkdir -p "$ALLELE_OUT_DIR"
    mkdir -p "$HOBOHM_DATA_OUT_NNALIGN_DIR"
    nohup python3 "$SCRIPT_DIR/train_nnalign.py" -dir "$HOBOHM_ALLELE_IN_DIR" -savepath "$HOBOHM_DATA_OUT_NNALIGN_DIR" > "$HOBOHM_DATA_OUT_NNALIGN_DIR/train_stdout.log" 2>&1 &
done

######################   Wait for all NNALIGN training to finish   ######################
echo "Waiting for all NNALIGN training on hobohm data to finish..."
wait
END_TIME_HOBOHM_TRAINING=$(date +%s)
DURATION_HOBOHM_TRAINING=$((END_TIME_HOBOHM_TRAINING - START_TIME_HOBOHM_TRAINING))
echo "Time taken for NNALIGN training on hobohm data: ${DURATION_HOBOHM_TRAINING} seconds."



#######################   NNALIGN testing on raw data   ######################
echo "Testing NNALIGN on raw data for each HLA allele..."
START_TIME_RAW_TESTING=$(date +%s)
for RAW_ALLELE_IN_DIR in $(find "$RAW_DATA_DIR" -mindepth 1 -maxdepth 1 -type d | grep -v "_out"); do
    ALLELE_NAME=$(basename "$RAW_ALLELE_IN_DIR")
    ALLELE_OUT_DIR="$NNALIGN_DIR/$ALLELE_NAME"
    NNALIGN_OUT_DIR="$ALLELE_OUT_DIR/nnalign_out"
    RAW_DATA_OUT_NNALIGN_DIR="$ALLELE_OUT_DIR/raw_data_out"
    nohup python3 "$SCRIPT_DIR/test_nnalign.py" -dir "$RAW_ALLELE_IN_DIR" -savepath "$RAW_DATA_OUT_NNALIGN_DIR" > "$RAW_DATA_OUT_NNALIGN_DIR/train_stdout.log" 2>&1 &
done

#######################   Wait for all NNALIGN tests on raw data to finish   ######################
echo "Waiting for all NNALIGN tests on raw data to finish..."
wait
END_TIME_RAW_TESTING=$(date +%s)
DURATION_RAW_TESTING=$((END_TIME_RAW_TESTING - START_TIME_RAW_TESTING))
echo "Time taken for NNALIGN testing on raw data: ${DURATION_RAW_TESTING} seconds."



#######################   NNALIGN testing on hobohm data   ######################
echo "Testing NNALIGN on hobohm data for each HLA allele..."
START_TIME_HOBOHM_TESTING=$(date +%s)
for HOBOHM_ALLELE_IN_DIR in $(find "$HOBOHM_DATA_DIR" -mindepth 1 -maxdepth 1 -type d | grep -v "_out"); do
    ALLELE_NAME=$(basename "$HOBOHM_ALLELE_IN_DIR")
    ALLELE_OUT_DIR="$NNALIGN_DIR/$ALLELE_NAME"
    HOBOHM_DATA_OUT_NNALIGN_DIR="$ALLELE_OUT_DIR/hobohm_data_out"
    nohup python3 "$SCRIPT_DIR/test_nnalign.py" -dir "$HOBOHM_ALLELE_IN_DIR" -savepath "$HOBOHM_DATA_OUT_NNALIGN_DIR" > "$HOBOHM_DATA_OUT_NNALIGN_DIR/train_stdout.log" 2>&1 &
done

#######################   Final message   ######################
wait
END_TIME_HOBOHM_TESTING=$(date +%s)
DURATION_HOBOHM_TESTING=$((END_TIME_HOBOHM_TESTING - START_TIME_HOBOHM_TESTING))
echo "Time taken for NNALIGN testing on hobohm data: ${DURATION_HOBOHM_TESTING} seconds."

echo "All NNALIGN tests completed successfully!"
echo "Results can be found in the $NNALIGN_DIR directory."