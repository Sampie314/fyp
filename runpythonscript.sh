#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --nodelist=TC1N04
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=360
#SBATCH --job-name=agglo_res
#SBATCH --output=%x_output_%j.out
#SBATCH --error=%x_error_%j.err

module load anaconda
source activate fyp1
python Agglomerative_res.py

