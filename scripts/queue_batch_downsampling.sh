#!/bin/bash
#SBATCH --job-name=BatchDownsample
#SBATCH --partition=cpu
#SBATCH --nodes=4
#SBATCH --mem=20G
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --output=/home/projects/s215045/PredictPhagePPI/tmp/%j-%x.out
#SBATCH --error=/home/projects/s215045/PredictPhagePPI/tmp/%j-%x.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=s215045@student.dtu.dk

if [[ "$#" -eq 0 ]]; then
    echo "Error: No python script provided to sbatch."
    exit 1
fi

echo "With arguments: $@"
echo "Job ID: $SLURM_JOB_ID"

# Use "$@" to preserve arguments exactly as passed
python3 -u "$@"
