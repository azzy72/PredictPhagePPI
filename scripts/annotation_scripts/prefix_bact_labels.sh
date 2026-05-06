#!/bin/bash

# Check if correct number of arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <labels_csv> <lookup_csv>"
    exit 1
fi

LABELS_FILE=$1
LOOKUP_FILE=$2

# Check if files exist
if [[ ! -f "$LABELS_FILE" ]]; then
    echo "Error: Labels file '$LABELS_FILE' not found."
    exit 1
fi

if [[ ! -f "$LOOKUP_FILE" ]]; then
    echo "Error: Lookup file '$LOOKUP_FILE' not found."
    exit 1
fi

# Use awk to process files and preserve the structure/formatting
awk -F',' '
BEGIN {
    OFS = ","
}
# Pass 1: Read lookup file
(NR==FNR) {
    if (FNR > 1) {
        id = $1
        # Strip suffixes from lookup IDs to match the label file format
        sub(/_reoriented$/, "", id)
        sub(/_KMC$/, "", id)
        
        # Get species name from the 4th column (Cluster_Genus)
        species = $4
        # Remove any trailing carriage returns from lookup
        gsub(/\r/, "", species)
        
        if (species != "") {
            # Shorten "Pectobacterium punjabense" to "P. punjabense"
            # split species by space
            n = split(species, parts, " ")
            if (n >= 2) {
                # Get first char of genus, add dot, then rest of name
                prefix = substr(parts[1], 1, 1) ". "
                for (i=2; i<=n; i++) {
                    prefix = prefix parts[i] (i==n ? "" : " ")
                }
            } else {
                # If it is only one word, just use it
                prefix = species
            }
            
            # Replace spaces with underscores for the prefix string
            gsub(/ /, "_", prefix)
            map[id] = prefix "_"
        }
    }
    next
}
# Pass 2: Process labels file
{
    # Remove existing carriage return for processing
    had_cr = sub(/\r$/, "", $0)

    if (FNR == 1) {
        # Identify columns dynamically
        for (i=1; i<=NF; i++) {
            if ($i == "label") label_idx = i
            if ($i == "name") name_idx = i
        }
        # Print header
        line = $0
    } else {
        # Check the label column (e.g., J45_21) against our map
        key = $label_idx
        sub(/_reoriented.*$/, "", key)   # strip _reoriented and everything after
        sub(/_KMC.*$/, "", key)          # or strip _KMC and everything after
        if (key in map) {
            $label_idx = map[key] $label_idx
            $name_idx = map[key] $name_idx
        }
        line = $0
    }
    
    # Append \r back if original file had it to maintain exact file similarity
    if (had_cr) {
        printf "%s\r\n", line
    } else {
        print line
    }
}
' "$LOOKUP_FILE" "$LABELS_FILE"