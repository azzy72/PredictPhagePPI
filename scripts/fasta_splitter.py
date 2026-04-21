#!/usr/bin/python3

import os
import sys

def split_fasta(input_file, output_dir):
    """
    Splits a multi-fasta file into individual files.
    Files are named based on the first word of the fasta header.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    current_file = None
    try:
        with open(input_file, 'r') as f:
            for line in f:
                if line.startswith('>'):
                    if current_file:
                        current_file.close()
                    
                    # Get the header name, removing '>' and taking the first word
                    header = line[1:].strip().split()[0]
                    # Sanitize header for filename (remove characters like / or :)
                    safe_header = "".join([c for c in header if c.isalnum() or c in ('_','-')]).rstrip()
                    
                    output_path = os.path.join(output_dir, f"{safe_header}.fasta")
                    current_file = open(output_path, 'w')
                    current_file.write(">".join(line.split("> ")))
                else:
                    if current_file:
                        current_file.write(line)
        
        if current_file:
            current_file.close()
        print(f"Successfully split into {output_dir}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 split_fasta.py <input.fasta> <output_directory>")
    else:
        split_fasta(sys.argv[1], sys.argv[2])