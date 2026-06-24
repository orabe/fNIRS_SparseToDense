#!/bin/bash
#SBATCH --job-name=gradcpt_dot_prep
#SBATCH --partition=cpu-2d
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --array=0-19
#SBATCH --output=logs/gradcpt-%A_%a.out

SUBJECT=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" gradcpt_subjects.txt)

echo "Running subject: $SUBJECT"

apptainer run --nv -B /home/space/ibs/datasets/raw_data/:/ibs_data --bind /home/smoradi/cedalion:/app /home/smoradi/cedalion.sif bash -c "
python -u dot_ch2par.py $SUBJECT
" 