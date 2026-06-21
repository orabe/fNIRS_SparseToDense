#!/bin/bash
#SBATCH --job-name=recon_param_prep
#SBATCH --partition=gpu-2d
#SBATCH --gres=gpu:1
#SBATCH --mem=256G
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-44%10
#SBATCH --error=/home/orabe/fNIRS_sparseToDense/logs/err/recon-param-%A_%a.err
#SBATCH --output=/home/orabe/fNIRS_sparseToDense/logs/out/recon-param-%A_%a.out

set -euo pipefail

: "${DATASET:?DATASET is required. Example: sbatch --export=ALL,DATASET=vfc_hd,SUBSET=full --array=0-16%10 scripts/sbatch_recon_param_preprocessing.sh}"
SUBSET=${SUBSET:-full}
REPO_DIR=${REPO_DIR:-/home/orabe/fNIRS_sparseToDense}
FILE_LIST=${FILE_LIST:-${REPO_DIR}/datasets/pre_processed/imageRecon_params/${DATASET}/${SUBSET}/recon_param_files.txt}
CONTAINER=${CONTAINER:-${REPO_DIR}/cedalion_20251207.sif}

if [[ ! -f "${FILE_LIST}" ]]; then
  echo "File list not found: ${FILE_LIST}" >&2
  echo "Create it first, for example:" >&2
  echo "  python src/subset/make_recon_param_job_list.py --dataset ${DATASET} --subset ${SUBSET}" >&2
  exit 1
fi

FILE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$FILE_LIST")
if [[ -z "${FILE}" ]]; then
  echo "No file found for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} in ${FILE_LIST}" >&2
  exit 1
fi

echo "dataset: ${DATASET}"
echo "subset: ${SUBSET}"
echo "task: ${SLURM_ARRAY_TASK_ID}"
echo "file: ${FILE}"

apptainer exec --nv \
  --bind "${REPO_DIR}/xkb:/var/lib/xkb,${REPO_DIR}/cedalion:/app,${REPO_DIR}:${REPO_DIR}" \
  "${CONTAINER}" \
  bash -c "cd ${REPO_DIR} && python -u src/subset/recon_param_preprocessing.py --dataset '${DATASET}' --subset '${SUBSET}' --file '${FILE}'"
