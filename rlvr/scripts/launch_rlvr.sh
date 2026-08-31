#!/bin/bash
# Runs INSIDE the pyxis/enroot container on the compute node (see submit_hecate.sh).
# The container already bundles a CUDA-matched PyTorch, so we don't touch torch here
# -- only install the lightweight deps train_rlvr.py needs on top of it.
set -e

PROJECT_DIR="/lustre/fsw/general_sa/bbalakreshna"

echo "$(hostname): Installing RLVR dependencies..."
pip install --quiet -r "$PROJECT_DIR/requirements-cluster.txt"

export HF_HOME="/lustre/fsw/general_sa/bbalakreshna/hf_cache"
mkdir -p "$HF_HOME"

echo "$(hostname): Launching RLVR training on ${RLVR_NPROC_PER_NODE:-4} GPU(s) (DDP via torchrun)..."
torchrun --standalone --nproc_per_node="${RLVR_NPROC_PER_NODE:-4}" \
  "$PROJECT_DIR/train_rlvr.py" \
  --model "${RLVR_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}" \
  --iterations "${RLVR_ITERATIONS:-1000}" \
  --batch-size "${RLVR_BATCH_SIZE:-16}" \
  --max-new-tokens "${RLVR_MAX_NEW_TOKENS:-64}" \
  --lr "${RLVR_LR:-1e-5}" \
  --eval-n "${RLVR_EVAL_N:-50}" \
  --save-dir "$PROJECT_DIR/out/rlvr-hecate-run1"
