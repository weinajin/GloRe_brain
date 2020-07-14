#!/bin/bash 
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --mem=16G 
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=6
#SBATCH --time=3-05:00 
#SBATCH --output=GloRe-plain.out
#SBATCH --array=0-4
#SBATCH --account=def-hamarneh
#SBATCH --mail-user=weinaj@sfu.ca
#SBATCH --mail-type=ALL
#SBATCH --job-name GloRe-plain
python train_brats.py --fold=$SLURM_ARRAY_TASK_ID
