#!/bin/bash

# FASTA Concatenation Script
# Usage: ./ConcatenateFasta.sh <input_folder_path> <output_file_path>
#
# This script combines all files ending with .fasta (case-insensitive)
# from the input folder into a single output file.

# --- Argument Validation ---
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_folder_path> <output_file_path>"
    echo "Example: $0 ./phage_genomes/ all_phages.fasta"
    exit 1
fi

INPUT_DIR="$1"
OUTPUT_FILE="$2"

# Remove trailing slash from directory path if present
INPUT_DIR=${INPUT_DIR%/}

if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory not found at '$INPUT_DIR'"
    exit 1
fi

# --- Concatenation Logic ---
echo "Starting concatenation of FASTA files from: $INPUT_DIR/"
echo "Output file: $OUTPUT_FILE"

# Find all .fasta or .FASTA files and concatenate them.
# The 'cat' command handles the joining; wildcards ensure all files are included.
# Using 'shopt -s nocaseglob' makes the matching case-insensitive.
shopt -s nocaseglob
cat "$INPUT_DIR"/*.fna > "$OUTPUT_FILE"
shopt -u nocaseglob # Turn off case-insensitive matching

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

