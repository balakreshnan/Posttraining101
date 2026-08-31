#!/bin/bash
# Run this FROM hecate's login node home directory (~), after the code has
# been placed on Lustre (see scripts/README_cluster.md). Mirrors the
# account's existing srun/pyxis submission pattern. Backgrounds the srun
# call (via nohup + disown) so your terminal stays free -- output goes to
# $PROJECT_DIR/out/hecate_run1.log instead of the terminal.
set -e

export ACCOUNT="${ACCOUNT:-general_sa}"
export LUSTRE_DIR="${LUSTRE_DIR:-/lustre/fsw/$ACCOUNT/$USER}"
export PROJECT_DIR="${PROJECT_DIR:-$LUSTRE_DIR/rlvr-posttraining101}"
mkdir -p "$PROJECT_DIR/out"

nohup srun --account=general_sa \
     --partition=batch-xdr \
     --nodes=1 \
     --ntasks-per-node=1 \
     --time=1:00:00 \
     --job-name=general_sa-rlvr.qwen15b \
     --container-image=gitlab-master.nvidia.com/dl/dgx/pytorch:main-py3-devel \
     --container-mount-home \
     --container-mounts=/lustre:/lustre \
     --no-container-remap-root \
     --mpi=pmix \
     --export=ALL \
     "$PROJECT_DIR/scripts/launch_rlvr.sh" \
     > "$PROJECT_DIR/out/hecate_run1.log" 2>&1 &
disown

sleep 2
squeue -u "$USER"
echo "Log: $PROJECT_DIR/out/hecate_run1.log"
