#!/bin/bash
# Runs INSIDE the pyxis/enroot container on the compute node.
# See train_rlvr_nemotron.py's docstring for the design and defaults.
set -e

LUSTRE_DIR="/lustre/fsw/general_sa/bbalakreshna"

echo "$(hostname): Installing RLVR + Nemotron dependencies..."
pip install --quiet -r "$LUSTRE_DIR/requirements-nemotron.txt"

export HF_HOME="$LUSTRE_DIR/hf_cache"
mkdir -p "$HF_HOME"

# Guards in the script make the CUDA assert unreachable; keep launches async for speed.
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"

# Task: 'arith' (task.py synthetic arithmetic -- this model already scores 100%, so it
# only learns brevity) or 'gsm8k' (real word problems, held-out test-split eval).
TASK="${RLVR_TASK:-gsm8k}"
if [ "$TASK" = "gsm8k" ]; then
  DEFAULT_TOKENS=384   # step-by-step reasoning needs room; truncated solutions score 0.1
  RUN_NAME="${RLVR_RUN_NAME:-rlvr-nemotron-gsm8k-run1}"
else
  DEFAULT_TOKENS=64
  RUN_NAME="${RLVR_RUN_NAME:-rlvr-nemotron-run1}"
fi

COMMON_ARGS=(
  --model "${RLVR_MODEL:-nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16}"
  --task "$TASK"
  --iterations "${RLVR_ITERATIONS:-10}"
  --batch-size "${RLVR_BATCH_SIZE:-1}"
  --grad-accum "${RLVR_GRAD_ACCUM:-4}"
  --max-new-tokens "${RLVR_MAX_NEW_TOKENS:-$DEFAULT_TOKENS}"
  --lr "${RLVR_LR:-1e-5}"
  --eval-n "${RLVR_EVAL_N:-30}"
  --lora-r "${RLVR_LORA_R:-16}"
  --lora-target-modules "${RLVR_LORA_TARGETS:-q_proj,k_proj,v_proj,o_proj,in_proj}"
  --seed "${RLVR_SEED:-0}"
  --save-dir "$LUSTRE_DIR/out/$RUN_NAME"
)
# Optional: a system prompt for templates that toggle reasoning that way (e.g. "/no_think").
if [ -n "${RLVR_SYSTEM_PROMPT:-}" ]; then
  COMMON_ARGS+=(--system-prompt "$RLVR_SYSTEM_PROMPT")
fi

NPROC="${RLVR_NPROC:-1}"
if [ "$NPROC" -gt 1 ]; then
  # Data-parallel: one full bf16 replica per GPU (~60GB each). Guards are synchronised
  # across ranks. Not yet exercised on multi-GPU -- complete a single-process run first.
  echo "$(hostname): Launching RLVR ($TASK) for Nemotron-3.5-Lightning via torchrun x$NPROC (DDP)..."
  torchrun --standalone --nproc_per_node="$NPROC" "$LUSTRE_DIR/train_rlvr_nemotron.py" "${COMMON_ARGS[@]}"
else
  echo "$(hostname): Launching RLVR ($TASK) for Nemotron-3.5-Lightning (single process, one GPU)..."
  python "$LUSTRE_DIR/train_rlvr_nemotron.py" --device-map "${RLVR_DEVICE_MAP:-single}" "${COMMON_ARGS[@]}"
fi
