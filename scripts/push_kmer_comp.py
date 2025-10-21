#!/usr/bin/env python

# NOTE: This script assumes 'manipulations' and 'analysis' modules are available.

import sys
import os
import pandas as pd
# Assuming these modules are available on the remote server
from manipulations import fasta_to_kmerdf 
from analysis import perform_pca 

def main():
    # 1. Check for the input directory argument
    if len(sys.argv) < 2:
        # Exit with error if no directory is provided
        sys.stderr.write("Error: Missing input argument for 'bact_dir'.\n")
        sys.exit(1)
        
    bact_dir = sys.argv[1] # Get the directory from the first command-line argument

    # Ensure the directory path ends with a slash for clean concatenation
    if not bact_dir.endswith(os.sep):
        bact_dir += os.sep

    # 2. Find and select fasta files (limiting to 1 file as in your original code)
    try:
        bact_fastas = [bact_dir + file for file in os.listdir(bact_dir) 
                       if file.endswith('.fasta') or file.endswith(".fna")][:1]
    except FileNotFoundError:
        sys.stderr.write(f"Error: Directory not found: {bact_dir}\n")
        sys.exit(1)

    if not bact_fastas:
        sys.stderr.write(f"Warning: No matching fasta/fna files found in {bact_dir}\n")
        # Proceed, but kmer_df will likely be empty or cause errors later.
        
    # 3. Define parameters and perform k-mer conversion
    k = 100
    try:
        # The result DataFrame 'bact_kmer_df'
        bact_kmer_df = fasta_to_kmerdf(bact_fastas, k=k, quiet=False, sparse=False, relative=True)
    except Exception as e:
        sys.stderr.write(f"Error during k-mer conversion: {e}\n")
        sys.exit(1)

    # 4. Output the DataFrame.
    # We use 'to_csv' to print a structured representation of the DataFrame to stdout,
    # which can then be captured by the calling bash script.
    # Set 'index=False' if you don't want the DataFrame index in the output.
    # For simple viewing/piping, printing the __str__ representation is often used:
    # print(bact_kmer_df) 
    
    # But for a reliable structured output that can be read back into a DataFrame:
    bact_kmer_df.to_csv(sys.stdout, index=True) # Output with index to stdout
    
    # Print diagnostic info to stderr (so it doesn't pollute the stdout data stream)
    sys.stderr.write(f"Bacteria k-mer DataFrame shape: {bact_kmer_df.shape}\n")

if __name__ == "__main__":
    main()