# Running RLVR on hecate (SLURM + pyxis/enroot)

Validated end-to-end: code + model weights + HF cache all live on Lustre
at `/lustre/fsw/general_sa/bbalakreshna/rlvr-posttraining101`, submitted
from the hecate login node home directory via the account's existing
`srun --container-image=...` pattern. A 150-step run took ~3 minutes and
went from 54% -> 100% greedy accuracy on the toy arithmetic task.

Nothing here runs automatically — every block below is meant to be
pasted into your own already-authenticated hecate terminal.

## 1. Write the code to Lustre

Paste the full heredoc block that creates `task.py`, `train_rlvr.py`,
`requirements-cluster.txt`, and `scripts/launch_rlvr.sh` under
`$PROJECT_DIR` (see chat history for the exact block, or recreate it
from the files in this repo — the content must match exactly).

```bash
export ACCOUNT="general_sa"
export LUSTRE_DIR="/lustre/fsw/$ACCOUNT/$USER"
export PROJECT_DIR="$LUSTRE_DIR/rlvr-posttraining101"
mkdir -p "$PROJECT_DIR/scripts" "$PROJECT_DIR/out" "$LUSTRE_DIR/hf_cache"
# ... cat > "$PROJECT_DIR/task.py" << 'PYEOF' ... (etc, one heredoc per file)
```

Why no `python3.14` venv here: hecate's login/compute nodes only have
system `/usr/bin/python3` (currently 3.12), no 3.14 module. Rather than
build a custom Python from source, we lean on the account's existing
container image instead (see step 2), which already bundles a
CUDA-matched PyTorch — `launch_rlvr.sh` only `pip install`s the
lightweight extras (`transformers`, `datasets`, `numpy`, `tqdm`) from
`requirements-cluster.txt` on top of it, deliberately *not*
reinstalling torch.

## 2. Submit the training job

```bash
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
```

No `--gres=gpu:N` needed — hecate's batch-xdr/batch-spx nodes expose 4
GPUs per node by default; our single-process job just uses GPU 0.

Override the run's hyperparameters via env vars before the `srun` line
(they're read by `launch_rlvr.sh`), e.g.:

```bash
RLVR_ITERATIONS=300 RLVR_BATCH_SIZE=32 RLVR_MODEL=Qwen/Qwen2.5-3B-Instruct \
  nohup srun --account=general_sa ... # (same command as above)
```

## 3. Check status / view output

```bash
# Queued or running?
squeue -u $USER

# Final state once it's left the queue (COMPLETED / FAILED / TIMEOUT, exit code)
sacct -u $USER --format=JobID,JobName,State,Elapsed,ExitCode -j <JOBID>

# Live output
tail -f "$PROJECT_DIR/out/hecate_run1.log"
# or, if $PROJECT_DIR isn't set in this shell:
tail -f /lustre/fsw/general_sa/bbalakreshna/rlvr-posttraining101/out/hecate_run1.log
```

Look for `Using device: cuda (...)` near the top, `step N/150 | ...`
progress lines, and `Accuracy after training: ...` / `Saved fine-tuned
model to ...` at the end. A Python traceback instead of that last line
means it failed — `sacct`'s ExitCode will be non-zero.

Checkpoint lands at `$PROJECT_DIR/out/rlvr-hecate-run1`; HF model
downloads are cached under `$LUSTRE_DIR/hf_cache` (via `HF_HOME`, set
inside `launch_rlvr.sh`) — nothing touches `$HOME`.

## 4. Push the fine-tuned model to Hugging Face

The login node's Python is externally managed (PEP 668), so use a small
venv (kept on Lustre, not `$HOME`) just for the upload -- no GPU needed:

```bash
python3 -m venv "$PROJECT_DIR/.venv-upload"
source "$PROJECT_DIR/.venv-upload/bin/activate"

pip install --quiet -U huggingface_hub

hf auth login   # paste a HF token with "write" scope; input is hidden

hf upload Balab2021/rlvr-qwen2.5-1.5b-instruct \
  "$PROJECT_DIR/out/rlvr-hecate-run1" \
  --repo-type model

deactivate
```

`hf upload` creates the repo if it doesn't exist. Don't paste the HF
token itself anywhere it'd be echoed/logged -- `hf auth login`'s prompt
hides input.

---

## Other clusters (generic SLURM, no container / no pyxis)

If you're pointing this at a different cluster that isn't
container-based, use the local Python 3.14 venv approach instead:
`scripts/slurm_rlvr.sbatch` + `scripts/sync_to_cluster.sh` +
`requirements.txt` (the CUDA-pinned one, not `requirements-cluster.txt`).
See that sbatch script's header comments for the knobs to adjust
(partition, module names, CUDA wheel tag).
