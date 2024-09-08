#!/bin/sh

#SBATCH --job-name=mis_eval
#SBATCH --output=logs/slurm-%j.out  # Standard output of the script (Can be abs$
#SBATCH --error=logs/slurm-%j.err  # Standard error of the script
#SBATCH --gres=gpu:1  # Number of GPUs if needed
#SBATCH --cpus-per-task=12  # Number of CPUs (Don't use more than 24 per GPU)
#SBATCH --mem=126G  # Memory in GB (Don't use more than 126G per GPU)

source /home/guests/premt_cara/miniconda3/etc/profile.d/conda.sh
conda activate n_myenv

# python eval_hf.py
python eval_mistral_hit.py