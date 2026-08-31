#!/usr/bin/env bash
# Sync the rlvr/ project to a remote cluster over SSH/rsync.
#
# Usage:
#   CLUSTER_HOST=myuser@cluster.example.com CLUSTER_DIR=~/Posttraining101 ./sync_to_cluster.sh
#
# Requires CLUSTER_HOST to be set (no default — this script refuses to
# guess a remote host). CLUSTER_DIR defaults to ~/Posttraining101.

set -euo pipefail

: "${CLUSTER_HOST:?Set CLUSTER_HOST=user@hostname before running this script}"
CLUSTER_DIR="${CLUSTER_DIR:-~/Posttraining101}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"  # rlvr/

echo "Syncing $SCRIPT_DIR -> $CLUSTER_HOST:$CLUSTER_DIR/rlvr"
rsync -avz --exclude '__pycache__' --exclude 'out/' --exclude 'logs/' \
    "$SCRIPT_DIR/" "$CLUSTER_HOST:$CLUSTER_DIR/rlvr/"

echo "Done. To submit the job:"
echo "  ssh $CLUSTER_HOST 'cd $CLUSTER_DIR/rlvr && sbatch scripts/slurm_rlvr.sbatch'"
