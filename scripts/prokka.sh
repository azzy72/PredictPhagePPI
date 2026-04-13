#!/bin/bash
#SBATCH --job-name=PROKKA
#SBATCH --partition=cpu
#SBATCH --nodes=2
#SBATCH --mem=50G
#SBATCH --time=00:30:00
#!SBATCH --begin=15:20:00
#SBATCH --output=/home/projects/s215045/PredictPhagePPI/tmp/%j-%x.out
#SBATCH --error=/home/projects/s215045/PredictPhagePPI/tmp/%j-%x.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s215045@student.dtu.dk

ROOT_DIR=$(git rev-parse --show-toplevel)
DATA_DIR="$ROOT_DIR/data_prod/"
RAW_DIR="$ROOT_DIR/raw_data/phagehost_KU/"

prokka --outdir "$DATA_DIR/prokka_output" \
       --prefix prokka_annotated \
       --metagenome \
       --addgenes \
       --force \
       --quiet \
       $1