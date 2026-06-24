#!/bin/bash
#SBATCH --job-name=PyTorch_GPU
#SBATCH --partition=gpu-2d
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --ntasks-per-node=1
#SBATCH --error=/home/orabe/fNIRS_sparseToDense/logs/pytorch_jobs_error-%j.err
#SBATCH --output=/home/orabe/fNIRS_sparseToDense/logs/pytorch_jobs_output-%j.out

apptainer run --nv /home/orabe/containers/pytorch_container.sif python src/mixed_batch_train_image_space_processing_segments_V2.py

# apptainer run --nv /home/orabe/sparse_to_dense_fnirs/pytorch_container.sif python src/plot_training_results.py

# apptainer run --nv /home/orabe/sparse_to_dense_fnirs/pytorch_container.sif python src/export_sparse_dense_results_table.py



# apptainer run --nv /home/orabe/sparse_to_dense_fnirs/pytorch_container.sif python src/train_image_space_segments.py


# apptainer run --nv /home/orabe/sparse_to_dense_fnirs/pytorch_container.sif python src/subset/train_parcel_vae.py


# apptainer run --nv pytorch_container.sif jupyter notebook --ip 0.0.0.0 --no-browser
# apptainer run --nv pytorch_container.sif python src/analyze_results.py

# srun --partition=gpu-2d --gpus=1 --pty bash
# srun --partition=gpu-2d --gpus-per-node=1 --ntasks-per-node=1 --pty bash

# srun --partition=gpu-2d --gres=gpu:1 --mem=128G --pty  --constraint="80gb|40gb" bash
# srun --partition=gpu-2h --gres=gpu:1 --gpus-per-node=1 --ntasks-per-node=1 --mem=96G --pty bash


# apptainer run --nv /home/orabe/sparse_to_dense_fnirs/pytorch_container.sif jupyter notebook --ip 0.0.0.0 --no-browser