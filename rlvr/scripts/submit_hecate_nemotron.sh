#!/bin/bash
# Run FROM hecate's login node (~) after train_rlvr_nemotron.py,
# requirements-nemotron.txt and scripts/launch_rlvr_nemotron.sh are on Lustre.
# Backgrounds the srun call; output -> $LUSTRE_DIR/out/hecate_nemotron_run1.log,
# start/end timestamps + elapsed -> $LUSTRE_DIR/out/hecate_nemotron_run1.timing.
set -e

export ACCOUNT="${ACCOUNT:-general_sa}"
export LUSTRE_DIR="${LUSTRE_DIR:-/lustre/fsw/$ACCOUNT/$USER}"
mkdir -p "$LUSTRE_DIR/out"

LOG_FILE="$LUSTRE_DIR/out/hecate_nemotron_run1.log"
TIMING_FILE="$LUSTRE_DIR/out/hecate_nemotron_run1.timing"

{
  START_TS=$(date +%s)
  echo "Job started: $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$TIMING_FILE"

  srun --account=general_sa \
       --partition=batch-xdr \
       --nodes=1 \
       --ntasks-per-node=1 \
       --time=5:00:00 \
       --job-name=general_sa-rlvr.nemotron35 \
       --container-image=gitlab-master.nvidia.com/dl/dgx/pytorch:main-py3-devel \
       --container-mount-home \
       --container-mounts=/lustre:/lustre \
       --no-container-remap-root \
       --mpi=pmix \
       --export=ALL \
       "$LUSTRE_DIR/scripts/launch_rlvr_nemotron.sh" > "$LOG_FILE" 2>&1

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
