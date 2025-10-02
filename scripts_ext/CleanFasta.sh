#!/bin/bash

# FASTA Cleaning Script
# Usage: ./CleanFasta <fasta_in_path> <fasta_out_path>
#
# This script performs two cleaning operations on a FASTA file:
# 1. Ensures every header starts with '> ' (adds a space after >).
# 2. Inserts a blank line after every record.

# --- Argument Validation ---
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <fasta_in_path> <fasta_out_path>"
    exit 1
fi

FASTA_IN="$1"
FASTA_OUT="$2"

if [ ! -f "$FASTA_IN" ]; then
    echo "Error: Input file not found at $FASTA_IN"
    exit 1
fi

echo "Starting FASTA cleaning for: $FASTA_IN"

# --- Cleaning Pipeline ---
cat "$FASTA_IN" | \
# 1. Add space after '>' if not present (or ensure only one space)
sed 's/^>/> /' | \
# 2. Use awk to insert a blank line before every header line.
awk '
/^> / { 
    # If it is a header, and it is not the very first line (NR > 1), 
    # print a blank line to separate the record that just finished.
    if (NR > 1) { 
        print "" 
    } 
} 
{ 
    # Print the current line (header or sequence)
    print 
}
' | \
# 3. Remove any leading blank line introduced by the awk command on the first line
sed '1{/^$/d;}' > "$FASTA_OUT"

# --- Completion ---
if [ $? -eq 0 ]; then
    echo "Successfully cleaned and saved to $FASTA_OUT"
else
    echo "An error occurred during cleaning."
fi

