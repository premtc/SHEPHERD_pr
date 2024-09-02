#!/bin/sh

#SBATCH --job-name=plm_shepherd
#SBATCH --output=logs/slurm-%j.out  # Standard output of the script (Can be abs$
#SBATCH --error=logs/slurm-%j.err  # Standard error of the script
#SBATCH --gres=gpu:1  # Number of GPUs if needed
#SBATCH --cpus-per-task=12  # Number of CPUs (Don't use more than 24 per GPU)
#SBATCH --mem=126G  # Memory in GB (Don't use more than 126G per GPU)

source /home/guests/premt_cara/miniconda3/etc/profile.d/conda.sh
conda activate n_myenv

python train.py \
        --edgelist KG_edgelist_mask.txt \
        --node_map KG_node_map.txt \
        --patient_data disease_simulated \
        --run_type patients_like_me \
        --saved_node_embeddings_path checkpoints/pretrain_2_2.ckpt \
        --sparse_sample 300 \
        --lr 5e-05 \
        --upsample_cand 1 \
        --neighbor_sampler_size 15 \
        --lmbda 0.7 \
        --kappa 0.09000000000000001 \
        --do_inference \
        --best_ckpt checkpoints/plm2.ckpt