#!/bin/bash

# FASTA Concatenation Script
# Usage: ./ConcatenateFasta.sh <input_folder_path> <output_file_path>
#
# This script combines all files ending with .fasta or .fna (case-insensitive)
# from the input folder into a single output file, renaming headers using
# the SOURCE FILE NAME as the prefix.

# --- Argument Validation ---
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_folder_path> <output_file_path>"
    echo "Example: $0 ./phage_genomes/ all_phages.fasta"
    exit 1
fi

INPUT_DIR="$1"
OUTPUT_FILE="$2"

# Remove trailing slash from directory path if present
INPUT_DIR_FULL=${INPUT_DIR%/}

if [ ! -d "$INPUT_DIR_FULL" ]; then
    echo "Error: Input directory not found at '$INPUT_DIR_FULL'"
    exit 1
fi

# --- Concatenation Logic ---
echo "Starting concatenation of FASTA files from: $INPUT_DIR_FULL/"
echo "New prefix for all entries will be the SOURCE FILE NAME."
echo "Output file: $OUTPUT_FILE"

# Prepare the output file to be empty
> "$OUTPUT_FILE"

# Enable case-insensitive globbing for file matching
shopt -s nocaseglob

# Define the find command with correct escaped parentheses
# NOTE: We are now finding files using -iname \*.fasta -o -iname \*.fna
FIND_CMD="find \"$INPUT_DIR_FULL\" -maxdepth 1 -type f \( -iname \*.fasta -o -iname \*.fna \)"

# Use the find command to iterate over files
while IFS= read -r FASTA_FILE; do
    # Skip if no files were found
    if [ -z "$FASTA_FILE" ]; then
        continue
    fi
    
    # 🌟 NEW LOGIC: Extract the file name (basename) and remove the extension
    FILE_BASENAME=$(basename "$FASTA_FILE ")
    ENTRY_PREFIX=${FILE_BASENAME%%.*} # Removes everything after the first dot (the extension)

    echo "Processing: $FASTA_FILE (Prefix: >$ENTRY_PREFIX)"
    
    # Use sed to rename all header lines (starting with '>')
    # by replacing the existing header with the file's base name and a pipe.
    # sed command: substitute '>' with '>' + $ENTRY_PREFIX + '|'
    sed 's/^>/>'$ENTRY_PREFIX' | /' "$FASTA_FILE" >> "$OUTPUT_FILE"
    
done < <(eval $FIND_CMD)

# Disable case-insensitive globbing
shopt -u nocaseglob

# --- Completion ---
if [ $? -eq 0 ]; then
    COUNT=$(grep -c "^>" "$OUTPUT_FILE")
    echo "--------------------------------------------------------"
    echo "Concatenation complete. File saved to $OUTPUT_FILE."
    echo "Total records concatenated: $COUNT"
    echo "--------------------------------------------------------"
else
    echo "An error occurred during concatenation."
fi
