#!/bin/bash
# Runs INSIDE the pyxis/enroot container on the compute node.
# EXPERIMENTAL -- see train_rlvr_qwen38.py's docstring for why this is a
# different (LoRA + 4-bit + single-process device_map="auto") setup than
# launch_rlvr.sh's DDP full-replica training.
set -e

LUSTRE_DIR="/lustre/fsw/general_sa/bbalakreshna"

echo "$(hostname): Installing RLVR + Qwen3.8-Flash-Next dependencies..."
pip install --quiet -r "$LUSTRE_DIR/requirements-qwen38.txt"

export HF_HOME="$LUSTRE_DIR/hf_cache"
mkdir -p "$HF_HOME"

echo "$(hostname): Launching RLVR fine-tuning for Qwen3.8-Flash-Next (single process, device_map=auto)..."
python "$LUSTRE_DIR/train_rlvr_qwen38.py" \
  --model "${RLVR_MODEL:-Qwen/Qwen3.8-Flash-Next}" \
  --iterations "${RLVR_ITERATIONS:-10}" \
  --batch-size "${RLVR_BATCH_SIZE:-2}" \
  --max-new-tokens "${RLVR_MAX_NEW_TOKENS:-32}" \
  --lr "${RLVR_LR:-5e-6}" \
  --eval-n "${RLVR_EVAL_N:-6}" \
  --lora-r "${RLVR_LORA_R:-16}" \
  --save-dir "$LUSTRE_DIR/out/rlvr-qwen38-run1"
