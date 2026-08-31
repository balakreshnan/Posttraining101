#!/bin/bash
# Run this FROM hecate's login node home directory (~), after the code has
# been placed on Lustre (see scripts/README_cluster.md). Mirrors the
# account's existing srun/pyxis submission pattern. Backgrounds the srun
# call so your terminal stays free -- training output goes to
# $PROJECT_DIR/out/hecate_run1.log, and start/end timestamps + total
# elapsed time are recorded to $PROJECT_DIR/out/hecate_run1.timing once
# the job finishes.
set -e

export ACCOUNT="${ACCOUNT:-general_sa}"
export LUSTRE_DIR="${LUSTRE_DIR:-/lustre/fsw/$ACCOUNT/$USER}"
export PROJECT_DIR="${PROJECT_DIR:-$LUSTRE_DIR}"
mkdir -p "$PROJECT_DIR/out"

LOG_FILE="$PROJECT_DIR/out/hecate_run1.log"
TIMING_FILE="$PROJECT_DIR/out/hecate_run1.timing"

{
  START_TS=$(date +%s)
  echo "Job started: $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$TIMING_FILE"

  srun --account=general_sa \
       --partition=batch-xdr \
       --nodes=1 \
       --ntasks-per-node=1 \
       --time=1:30:00 \
       --job-name=general_sa-rlvr.qwen15b \
       --container-image=gitlab-master.nvidia.com/dl/dgx/pytorch:main-py3-devel \
       --container-mount-home \
       --container-mounts=/lustre:/lustre \
       --no-container-remap-root \
       --mpi=pmix \
       --export=ALL \
       "$PROJECT_DIR/scripts/launch_rlvr.sh" > "$LOG_FILE" 2>&1

  END_TS=$(date +%s)
  ELAPSED=$((END_TS - START_TS))
  {
    echo "Job ended: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "Elapsed seconds: $ELAPSED"
    printf "Elapsed (h:m:s): %02d:%02d:%02d\n" $((ELAPSED/3600)) $((ELAPSED%3600/60)) $((ELAPSED%60))
  } >> "$TIMING_FILE"
} &
disown

sleep 2
squeue -u "$USER"
echo "Log: $LOG_FILE"
echo "Timing (written once the job finishes): $TIMING_FILE"
