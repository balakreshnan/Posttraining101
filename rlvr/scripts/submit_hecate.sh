#!/bin/bash
# Run this FROM hecate's login node home directory (~), after the code has
# been placed on Lustre (see scripts/README_cluster.md). Mirrors the
# account's existing srun/pyxis submission pattern.
set -e

srun --account=general_sa \
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
     /lustre/fsw/general_sa/bbalakreshna/rlvr-posttraining101/scripts/launch_rlvr.sh
