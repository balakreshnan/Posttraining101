#!/bin/bash
#SBATCH --job-name=rlvr-demo
#SBATCH --partition=REPLACE_WITH_PARTITION
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/rlvr-%j.out
#SBATCH --error=logs/rlvr-%j.err

set -euo pipefail

# --- Edit these for your cluster ---
PROJECT_DIR="${PROJECT_DIR:-$HOME/Posttraining101/rlvr}"
VENV_DIR="${VENV_DIR:-$HOME/Posttraining101/.venv314}"
PYTHON_MODULE="${PYTHON_MODULE:-python/3.14}"   # module name varies by cluster
# ------------------------------------

mkdir -p "$PROJECT_DIR/logs"
cd "$PROJECT_DIR"

module load "$PYTHON_MODULE" 2>/dev/null || true

if [ ! -d "$VENV_DIR" ]; then
    python3.14 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
pip install --upgrade pip -q
# requirements.txt pins torch==2.11.0+cu128 -- if the cluster's driver
# doesn't support CUDA 12.8, edit requirements.txt's --extra-index-url
# to match (e.g. .../whl/cu124) before this install.
pip install -r requirements.txt -q

python train_rlvr.py \
    --model "${RLVR_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}" \
    --iterations "${RLVR_ITERATIONS:-500}" \
    --batch-size "${RLVR_BATCH_SIZE:-32}" \
    --max-new-tokens "${RLVR_MAX_NEW_TOKENS:-64}" \
    --lr "${RLVR_LR:-1e-5}" \
    --eval-n "${RLVR_EVAL_N:-50}" \
    --save-dir "${RLVR_SAVE_DIR:-out/rlvr-run-$SLURM_JOB_ID}"
