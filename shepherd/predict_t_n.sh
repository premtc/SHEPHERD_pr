#!/bin/sh

#SBATCH --job-name=pred_shepherd
#SBATCH --output=logs/slurm-%j.out  # Standard output of the script (Can be abs$
#SBATCH --error=logs/slurm-%j.err  # Standard error of the script
#SBATCH --gres=gpu:1  # Number of GPUs if needed
#SBATCH --cpus-per-task=12  # Number of CPUs (Don't use more than 24 per GPU)
#SBATCH --mem=126G  # Memory in GB (Don't use more than 126G per GPU)

source /home/guests/premt_cara/miniconda3/etc/profile.d/conda.sh
conda activate n_myenv

python predict.py \
--run_type causal_gene_discovery \
--patient_data disease_simulated \
--edgelist KG_edgelist_mask.txt \
--node_map KG_node_map.txt \
--saved_node_embeddings_path checkpoints/pretrain_2_2.ckpt \
--best_ckpt checkpoints/casual.ckpt

python predict.py \
--run_type patients_like_me \
--patient_data disease_simulated \
--edgelist KG_edgelist_mask.txt \
--node_map KG_node_map.txt \
--saved_node_embeddings_path checkpoints/pretrain_2_2.ckpt \
--best_ckpt checkpoints/plm2.ckpt

python predict.py \
--run_type disease_characterization \
--patient_data disease_simulated \
--edgelist KG_edgelist_mask.txt \
--node_map KG_node_map.txt \
--saved_node_embeddings_path checkpoints/pretrain_2_2.ckpt \
--best_ckpt checkpoints/dcp.ckpt