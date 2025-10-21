#!/bin/bash

# --- Usage and Validation ---
# Check if a directory was provided
if [ -z "$1" ]; then
    echo "Usage: $0 <bact_dir>"
    echo "Example: $0 raw_data/phagehost_KU/bacteria_fasta/"
    exit 1
fi

# Assign the first argument to a variable
INPUT_DIR="$1"

# Define the Python script location (assuming it's in the same directory)
PYTHON_SCRIPT="./push_kmer_comp.py"

# --- Execution ---
# Execute the Python script, passing the directory as the first argument.
# The output (bact_kmer_df CSV) is captured by the variable 'bact_kmer_df'.
# NOTE: This variable will hold the *string* (CSV) representation of the DataFrame.
bact_kmer_df=$(python3 "$PYTHON_SCRIPT" "$INPUT_DIR")

# Check the exit status of the Python script
if [ $? -ne 0 ]; then
    echo "Error: Python script failed." >&2
    # The Python script has already printed a detailed error to stderr
    exit 1
fi

# --- Output ---
# The variable bact_kmer_df now contains the CSV data from the DataFrame.
# You can now 'output' it by printing it to stdout, saving it to a file, etc.

# 1. To print the final result (the DataFrame contents) to stdout:
echo "$bact_kmer_df"

# 2. Alternatively, to save it to a file:
# echo "$bact_kmer_df" > "bact_kmer_output.csv"
# echo "DataFrame saved to bact_kmer_output.csv" >&2

# Success
exit 0