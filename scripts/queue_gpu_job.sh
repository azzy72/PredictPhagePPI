#!/bin/bash
#SBATCH --job-name=PredictPhage
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --mem=10G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu
#SBATCH --output=/home/projects/s215045/PredictPhagePPI/tmp/%j-%x.out
#SBATCH --error=/home/projects/s215045/PredictPhagePPI/tmp/%j-%x.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=s215045@student.dtu.dk

# Capture the script path passed from convert_and_queue.sh
#PYTHON_SCRIPT=$1

if [[ "$#" -eq 0 ]]; then
    echo "Error: No python script provided to sbatch."
    exit 1
fi

echo "With arguments: $@"
echo "Job ID: $SLURM_JOB_ID"

# Use "$@" to preserve arguments exactly as passed
python3 -u "$@"
