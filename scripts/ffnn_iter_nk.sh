#!/bin/bash
#SBATCH --job-name=PredictPhage
#SBATCH --partition=gpu
#SBATCH --nodes=4
#SBATCH --mem=50G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu
#!SBATCH --time=00:00:00
#SBATCH --begin=15:20:00
#SBATCH --output=/home/projects/s215045/PredictPhagePPI/tmp/%j-%x.out
#SBATCH --error=/home/projects/s215045/PredictPhagePPI/tmp/%j-%x.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s215045@student.dtu.dk

