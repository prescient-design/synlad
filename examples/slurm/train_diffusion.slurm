#!/bin/bash
#
# Single-GPU latent diffusion training. Replace the ${...} placeholders with
# values for your cluster.
#
# Set output/error paths via env vars before submission, e.g.:
#   export SBATCH_OUTPUT=$LOG_DIR/train_diffusion_%j.out
#   export SBATCH_ERROR=$LOG_DIR/train_diffusion_%j.err

#SBATCH --job-name=train_diffusion
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=72:00:00
#SBATCH --partition=${PARTITION}
#SBATCH --gres=gpu:${GPU_TYPE}:1
#SBATCH --account=${ACCOUNT}
#SBATCH --qos=${QOS}

source ~/.bashrc
set -euo pipefail
source ${WORK_DIR}/.venv/bin/activate

# WANDB_API_KEY should be set in the environment (e.g. via ~/.netrc) before submission.
export WANDB_DIR=${WANDB_DIR}

cd ${WORK_DIR}/src/synlad
python train_diffusion.py experiment=train_diffusion_ph4_cond
