#!/bin/bash
#SBATCH --job-name=cedalion_gpu
#SBATCH --partition=gpu-2d
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --mem=128G
#SBATCH --ntasks-per-node=1
#SBATCH --error=/home/orabe/fNIRS_sparseToDense/logs/cedalion_jobs_error-%j.err
#SBATCH --output=/home/orabe/fNIRS_sparseToDense/logs/cedalion_jobs_output-%j.out

# run script with apptainer
apptainer run --nv --bind `pwd`/xkb:/var/lib/xkb,`pwd`/cedalion:/app /home/orabe/fNIRS_sparseToDense/cedalion_20251207.sif jupyter notebook --ip 0.0.0.0 --no-browser

# apptainer exec --nv /home/space/ibs/datasets/cedalion.sif bash

# sbatch run_cedalion2.sh
#  ssh -L 8888:head022:8888 -o ServerAliveInterval=60 orabe@hydra.ml.tu-berlin.de

# srun --partition=gpu-2d --mem=256G --gres=gpu:1 --pty bash
# srun --partition=gpu-5h --mem=128G --pty  bash
# srun --partition=cpu-2h --pty bash
