#!/bin/bash
#SBATCH --job-name=PredictPhage
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --mem=20G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu
#!SBATCH --time=00:00:30
#SBATCH --output=/home/projects/s215045/PredictPhagePPI/tmp/%j-%x.out
#SBATCH --error=/home/projects/s215045/PredictPhagePPI/tmp/%j-%x.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=s215045@student.dtu.dk

# Capture the script path passed from convert_and_queue.sh
PYTHON_SCRIPT=$1

# Basic check to ensure a file was actually passed
if [ -z "$PYTHON_SCRIPT" ]; then
    echo "Error: No python script provided to sbatch."
    exit 1
fi

echo "Running script: $PYTHON_SCRIPT"
echo "With arguments: $@"
echo "Job ID: $SLURM_JOB_ID"

python3 -u "$PYTHON_SCRIPT" "$@"
