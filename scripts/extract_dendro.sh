#!/bin/bash

# Configuration
BASE_DIR="./data_prod"
STAGING_DIR="./data_prod/staging_download"

mkdir -p "$STAGING_DIR"

FOLDERS=(
  "encoded_sketches"
  "encoded_sketches_data2"
  "encoded_sketches_allphages"
  "encoded_sketches_data2_allphages"
  "SM_sketches"
  "SM_sketches_data2"
  "SM_sketches_allphages"
  "SM_sketches_data2_allphages"
)

for FOLDER in "${FOLDERS[@]}"; do

  # enc vs SM
  if [[ "$FOLDER" == encoded_* ]]; then
    TYPE="_enc"
  else
    TYPE="_SM"
  fi

  # D1 vs D2
  if [[ "$FOLDER" == *data2* ]]; then
    DATASET="_D2"
  else
    DATASET="_D1"
  fi

  # allphages flag
  if [[ "$FOLDER" == *allphages* ]]; then
    ALLPHAGES=true
  else
    ALLPHAGES=false
  fi

  SUFFIX="${TYPE}${DATASET}"
  SIM_DIR="${BASE_DIR}/${FOLDER}/sim_matrices"

  echo "Processing $FOLDER..."

  for FILEPATH in "${SIM_DIR}"/BactDendro*.png "${SIM_DIR}"/PhageDendro*.png; do
    # Skip if glob matched nothing
    [ -f "$FILEPATH" ] || continue

    BASENAME=$(basename "$FILEPATH")

    # Replace nX with _allphages if applicable
    if $ALLPHAGES; then
      NEW_NAME=$(echo "$BASENAME" | sed 's/n[0-9]\+/_allphages/g')
    else
      NEW_NAME="$BASENAME"
    fi

    # Insert suffix before .png
    NEW_NAME="${NEW_NAME%.png}${SUFFIX}.png"

    cp "$FILEPATH" "${STAGING_DIR}/${NEW_NAME}"
    echo "  $BASENAME -> $NEW_NAME"
  done

done

echo "Done. Staging folder ready at: $STAGING_DIR"