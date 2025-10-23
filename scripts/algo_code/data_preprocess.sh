#!/bin/bash




######################   Constants   ######################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/../data"
RAW_DATA_DIR="$DATA_DIR/raw_data"
FILE_NAME="NetMHCpan_train.tar.gz"
TEMP_DIR_NAME="NetMHCpan_train"
PARENT_DIR="$(realpath "$SCRIPT_DIR/..")"
HOBOHM_DIR="$DATA_DIR/hobohm_data"



######################   Changing permissions for all files   ######################
# Apply permissions recursively to the parent directory of the script
chmod -R a+rwx "$PARENT_DIR"

# Confirmation message showing the resolved path
echo "Permissions applied to directory: $PARENT_DIR"




######################   Downloading the data   ######################
echo "Downloading NetMHCpan data..."

# Ensure data directory exists
mkdir -p "$DATA_DIR"
mkdir -p "$RAW_DATA_DIR"

# Downloading the data and removing irrelevant files
curl -o "$RAW_DATA_DIR/$FILE_NAME" "https://services.healthtech.dtu.dk/suppl/immunology/NAR_NetMHCpan_NetMHCIIpan/$FILE_NAME"
tar -xzf "$RAW_DATA_DIR/$FILE_NAME" -C "$RAW_DATA_DIR"
mv "$RAW_DATA_DIR/$TEMP_DIR_NAME"/* "$RAW_DATA_DIR"
rmdir "$RAW_DATA_DIR/$TEMP_DIR_NAME"
#rm "$RAW_DATA_DIR"/*_el "$FILE_NAME"




######################   Subsetting data by HLA allele   ######################
echo "Subsetting data by HLA allele..."
# Extract HLA IDs from allelelist
HLA_IDS=$(awk '{print $1}' "$RAW_DATA_DIR/allelelist" | grep "HLA-" | cut -c5-)

# Loop over each allele ID
for ALLELE in $HLA_IDS; do
    DIR_NAME=$(echo "$ALLELE" | sed 's/://g')
    ALLELE_DIR="$RAW_DATA_DIR/${DIR_NAME}"
    mkdir -p "$ALLELE_DIR"

    valid=true  # Tracks whether all output files meet the line count threshold

    for file in "$RAW_DATA_DIR"/*_ba; do
        if [ -f "$file" ]; then
            base_file=$(basename "$file")
            output_file="${ALLELE_DIR}/${base_file}_clean"
            grep -E "${ALLELE}\$" "$file" | cut -d' ' -f1,2 > "$output_file"

            line_count=$(wc -l < "$output_file")
            if [ "$line_count" -lt 400 ]; then
                valid=false
            fi
        fi
    done

    if [ "$valid" = false ]; then
        rm -rf "$ALLELE_DIR"
    fi
done




######################   Final Cleanup   ######################
echo "Cleaning up temporary files..."
# Remove the original files
find "$RAW_DATA_DIR" -maxdepth 1 -type f -exec rm -f {} \;





#########################   Compiling hobohm code   ######################
echo "Compiling HOBOHM code..."
gcc "$SCRIPT_DIR/hobohm.c" -o "$SCRIPT_DIR/hobohm" -lm



########################   Generate HOBOHM data   ######################
echo "Creating HOBOHM data in the background..."
# Ensure data directory exists
mkdir -p "$HOBOHM_DIR"

# Iterating over all folders in the raw_data directory
for ALLELE_IN_DIR in $(find "$RAW_DATA_DIR" -mindepth 1 -maxdepth 1 -type d | grep -v "_out"); do
    ALLELE_NAME=$(basename "$ALLELE_IN_DIR")
    ALLELE_OUT_DIR="$HOBOHM_DIR/$ALLELE_NAME"
    mkdir -p "$ALLELE_OUT_DIR"

    # Iterating over all data files (except c000_*) and running.
    for DATA_FILE in "$ALLELE_IN_DIR"/*; do
        DATA_FILE_NAME=$(basename "$DATA_FILE")
        if [[ "$DATA_FILE_NAME" != c000_* ]]; then
            nohup "$SCRIPT_DIR/hobohm" -f "$DATA_FILE" -o "$ALLELE_OUT_DIR/" >/dev/null 2>&1 & #> "$ALLELE_OUT_DIR/${DATA_FILE_NAME}_hobohm1_stdout.log" &
        else
            cp "$DATA_FILE" "$ALLELE_OUT_DIR/"
        fi
    done
done

wait


#########################   Final Cleanup   ######################
echo "Cleaning up compiled scripts..."
rm -f "$SCRIPT_DIR/hobohm"







#########################   Completion Message   ######################
echo "All done!"

