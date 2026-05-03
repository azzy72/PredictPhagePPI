#!/bin/bash

# Check if correct number of arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_directory> <output_directory>"
    exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="$2"

# Validate input directory
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory '$INPUT_DIR' does not exist."
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Define output filenames with path
BACT_OUTPUT="$OUTPUT_DIR/BactSim_n500_k12_clusters.csv"
PHAGE_OUTPUT="$OUTPUT_DIR/PhageSim_n500_k12_clusters.csv"

echo "Starting merging process with unique cluster assignments..."
echo "Input Dir:  $INPUT_DIR"
echo "Output Dir: $OUTPUT_DIR"

# Function to merge files and assign unique cluster IDs
merge_files() {
    local prefix=$1
    local output=$2
    
    echo "Processing $prefix files..."
    
    # Get files in natural version order
    files=$(ls -v "$INPUT_DIR"/${prefix}*.mat.[0-9]*.csv 2>/dev/null)
    
    if [ -z "$files" ]; then
        echo "No files found for $prefix in $INPUT_DIR"
        return
    fi

    # Write the header to the output file
    echo ",Cluster,genus,Cluster_Genus" > "$output"
    
    cluster_id=0
    for f in $files; do
        # 1. Skip the header row of the individual file (sed '1d')
        # 2. Extract the identifier (assuming it's in the 3rd column based on your sample, 
        #    or change $3 to the correct column index if needed)
        
        # Using awk to handle the CSV parsing more cleanly
        # We skip the first line (NR>1) and append our cluster metadata
        awk -F',' -v cid="$cluster_id" 'NR>1 {
            # Clean up potential carriage returns from Windows-style line endings
            gsub(/\r/, "", $0);
            
            # Use column 3 (label/name) as the primary identifier
            # Adjust the $3 below if the name is in a different column
            print $3 "," cid ",Unknown,Unknown"
        }' "$f" >> "$output"
        
        # Increment cluster ID for the next file
        ((cluster_id++))
    done
    
    echo "Done. Merged $(echo "$files" | wc -l) files into $output with $cluster_id unique clusters."
}

# Execute for Bacteria
merge_files "BactSim_n500_k12" "$BACT_OUTPUT"

# Execute for Phage
merge_files "PhageSim_n500_k12" "$PHAGE_OUTPUT"

echo "---------------------------------------"
echo "Consolidation complete."
ls -lh "$BACT_OUTPUT" "$PHAGE_OUTPUT"