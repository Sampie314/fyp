#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --nodelist=TC1N04
#SBATCH --gres=gpu:0
#SBATCH --mem=64G
#SBATCH --time=360
#SBATCH --job-name=HDBSCAN_cluster_6
#SBATCH --output=%x_output_%j.out
#SBATCH --error=%x_error_%j.err

module load anaconda
source activate fyp1
python HDBSCAN_cluster_6.py

