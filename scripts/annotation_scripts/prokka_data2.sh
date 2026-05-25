#!/bin/bash
#SBATCH --job-name=prokka_data2
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=50G
#SBATCH --time=04:00:00
#SBATCH --output=/home/projects/s215045/PredictPhagePPI/tmp/%j-%x.out
#SBATCH --error=/home/projects/s215045/PredictPhagePPI/tmp/%j-%x.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s215045@student.dtu.dk

# Prokka annotation for dataset 2 bacteria (data2_klebbacts/)
#
# Submit as a single job (loops internally — 14 genomes, fast enough):
#   sbatch prokka_data2.sh
#
# Naming note
# -----------
# Input files are named: Kp_KU1.autocycler_medaka_polypolish.fasta
# The clean prefix (everything before the first '.') is used for prokka
# output, e.g. Kp_KU1. This keeps output directories and TSV filenames
# consistent with how the rest of the pipeline refers to these strains.
#
# Duplicate note
# --------------
# Kp_KU6 has two assemblies:
#   Kp_KU6.autocycler_medaka_polypolish.fasta           (single contig)
#   Kp_KU6.autocycler_medaka_polypolish_concatenated.fasta (linear+circular concatenated)
# Only the plain (non-concatenated) file is processed below.
# If you want to use the concatenated version instead, swap the glob or
# add an explicit exclusion of the non-concatenated KU6.

# --- SETUP ---
ROOT_DIR=$(git rev-parse --show-toplevel)
DATA_DIR="$ROOT_DIR/data_prod"
INPUT_DIR="$ROOT_DIR/raw_data/phagehost_KU/data2_klebbacts_renamed"
BASE_OUTDIR="$DATA_DIR/prokka_bacts"   # same parent dir as dataset 1 results

mkdir -p "$BASE_OUTDIR"

# --- LOOP OVER GENOMES ---
for file in "$INPUT_DIR"/*.fasta; do

    # Skip the _concatenated KU6 variant (would give the same clean name as
    # the plain KU6, causing a collision in the output directory)
    if [[ "$file" == *_concatenated.fasta ]]; then
        echo "Skipping concatenated variant: $(basename "$file")"
        continue
    fi

    # Strip the assembly-method suffix so prokka names stay clean:
    # Kp_KU1.autocycler_medaka_polypolish.fasta  ->  Kp_KU1
    raw_name=$(basename "$file" .fasta)
    GENOME_NAME="${raw_name%%.*}"   # everything before the first '.'

    SPECIFIC_OUTDIR="$BASE_OUTDIR/$GENOME_NAME"

    echo "Annotating: $GENOME_NAME  ($file)"

    prokka --outdir "$SPECIFIC_OUTDIR" \
           --prefix "$GENOME_NAME" \
           --cpus 12 \
           --metagenome \
           --addgenes \
           --force \
           --quiet \
           "$file"

    echo "Done: $GENOME_NAME"

done

echo "All dataset 2 bacteria annotations complete."