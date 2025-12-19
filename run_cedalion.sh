#!/bin/bash
#SBATCH --job-name=cedalion_gpu
#SBATCH --partition=gpu-2d
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --ntasks-per-node=1
#SBATCH --error=/home/orabe/fNIRS_sparseToDense/logs/cedalion_jobs_error-%j.err
#SBATCH --output=/home/orabe/fNIRS_sparseToDense/logs/cedalion_jobs_output-%j.out

# run script with apptainer
apptainer run --nv --bind `pwd`/xkb:/var/lib/xkb,`pwd`/cedalion:/app cedalion_20251207.sif jupyter notebook --ip 0.0.0.0 --no-browser

# apptainer exec --nv /home/space/ibs/datasets/cedalion.sif bash

# sbatch run_cedalion2.sh
#  ssh -L 8888:head022:8888 -o ServerAliveInterval=60 orabe@hydra.ml.tu-berlin.de

# srun --partition=gpu-2h --mem=64G --gpus-per-node=1 --ntasks-per-node=1 --pty bash
# srun --partition=gpu-5h --mem=128G --pty  bash
# srun --partition=cpu-2h --pty bash
