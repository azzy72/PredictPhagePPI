#!/bin/bash
#SBATCH --job-name=MergeFasta
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --mem=20G
#SBATCH --time=00:30:00
#!SBATCH --begin=15:20:00
#SBATCH --output=/home/projects/s215045/PredictPhagePPI/tmp/%j-%x.out
#SBATCH --error=/home/projects/s215045/PredictPhagePPI/tmp/%j-%x.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s215045@student.dtu.dk

# Create the output directory if it doesn't exist
mkdir -p concatenated_bacteria

for file in bacteria_fasta/*.fna; do
    
    # Extract the filename without path or extension
    filename=$(basename "$file" .fna)
    output_file="concatenated_bacteria/${filename}_merged.fasta"
    
    # 1. Write the new header
    echo ">$filename" > "$output_file"
    
    # 2. Process the sequence:
    # - grep -v '^>' : Remove old headers
    # - tr -d '\n'   : Remove all existing newlines to create one long string
    # - fold -w 60   : Wrap that string into lines of exactly 60 characters
    grep -v '^>' "$file" | tr -d '\n' | fold -w 60 >> "$output_file"

    echo "Processed $filename"
done

echo "Done! Files are formatted with 60-character line widths."