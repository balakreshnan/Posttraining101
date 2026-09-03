#!/bin/bash
# Runs INSIDE the pyxis/enroot container on the compute node.
# EXPERIMENTAL -- see train_rlvr_qwen38.py's docstring for the full history
# of what has been observed on hecate and why the defaults are what they are.
set -e

LUSTRE_DIR="/lustre/fsw/general_sa/bbalakreshna"

echo "$(hostname): Installing RLVR + Qwen3.8-Flash-Next dependencies..."
pip install --quiet -r "$LUSTRE_DIR/requirements-qwen38.txt"

export HF_HOME="$LUSTRE_DIR/hf_cache"
mkdir -p "$HF_HOME"

# Off by default now: the script guards logits, gradients and parameters
# itself, so the CUDA assert should no longer fire, and synchronous kernel
# launches slow every step. Set CUDA_LAUNCH_BLOCKING=1 in the environment
# if you need a precise stack trace for a new failure.
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"

echo "$(hostname): Launching RLVR fine-tuning for Qwen3.8-Flash-Next (single process, device_map=auto)..."
python "$LUSTRE_DIR/train_rlvr_qwen38.py" \
  --model "${RLVR_MODEL:-Qwen/Qwen3.8-Flash-Next}" \
  --iterations "${RLVR_ITERATIONS:-10}" \
  --batch-size "${RLVR_BATCH_SIZE:-1}" \
  --grad-accum "${RLVR_GRAD_ACCUM:-4}" \
  --max-new-tokens "${RLVR_MAX_NEW_TOKENS:-64}" \
  --lr "${RLVR_LR:-5e-6}" \
  --eval-n "${RLVR_EVAL_N:-6}" \
  --lora-r "${RLVR_LORA_R:-16}" \
  --seed "${RLVR_SEED:-0}" \
  --quant "${RLVR_QUANT:-none}" \
  --max-memory-per-gpu "${RLVR_MAX_MEMORY_PER_GPU:-200GiB}" \
  --skip-quant-modules "${RLVR_SKIP_QUANT_MODULES:-}" \
  --save-dir "$LUSTRE_DIR/out/rlvr-qwen38-run1"
