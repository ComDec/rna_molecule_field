#!/bin/bash
#SBATCH --partition=GPU_3090
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

python runSys.py --input ./data/ --output