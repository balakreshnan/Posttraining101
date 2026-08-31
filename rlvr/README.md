# RLVR Sample (Reinforcement Learning with Verifiable Rewards)

A minimal, self-contained example of RLVR post-training: a causal LM
(`Qwen/Qwen2.5-1.5B-Instruct` by default, swap with `--model`) is
fine-tuned with REINFORCE against a **deterministic verifier** (no
reward model, no human preference data).

The task is synthetic arithmetic (`task.py`), with per-operator operand
ranges (addition/subtraction up to 999, multiplication up to 50 -- 3-digit
multiplication is unreasonably hard for a small LM in one shot). The
model is asked to reason briefly and end with `Final Answer: <number>`
via its chat template; `verify_reward` parses that line (falling back to
the last integer in the completion) and checks it against ground truth,
returning 1.0 / 0.1 / 0.0. Allowing a short chain-of-thought budget
(`--max-new-tokens 64`) before the graded answer is what makes this task
learnable via RL instead of stuck at whatever the base model gets in one
token.

For a smaller/faster CPU-only smoke test, pass `--model distilgpt2`
(then expect much lower absolute accuracy — it's a 82M-param base model).

## 1. Environment setup (local, Python 3.14)

From the repo root:

```bash
py -3.14 -m venv .venv314
.venv314\Scripts\activate      # Windows
# source .venv314/bin/activate # Linux/macOS
pip install --upgrade pip
pip install -r rlvr/requirements.txt
```

`requirements.txt` pins a CUDA 12.8 build of torch (`torch==2.11.0+cu128`)
via `--extra-index-url`. If your machine has no NVIDIA GPU / matching
driver, edit that line to plain `torch>=2.4` (CPU-only) instead -- the
training script auto-detects `torch.cuda.is_available()` and falls back
to CPU either way, this only affects which wheel gets installed.

## 2. Validate

```bash
cd rlvr
python -c "from task import sample_batch, verify_reward; import random; print(sample_batch(random.Random(0), 3))"
```

## 3. Run training

```bash
python train_rlvr.py --iterations 150 --batch-size 16 --max-new-tokens 64 --lr 1e-5 --eval-n 50 --save-dir out/rlvr-run1
```

With Qwen2.5-1.5B-Instruct, expect greedy-eval accuracy to start around
40-55% and climb toward 75-85%+ over ~100-150 steps on GPU (a few
minutes on a 24GB card). On CPU-only the same run takes much longer --
drop `--iterations`/`--batch-size` for a quicker smoke test
(e.g. `--iterations 40 --batch-size 6`). This is a teaching example of
the RLVR loop, not a push for state-of-the-art arithmetic accuracy.

Useful flags:
- `--model` swap in any causal LM (e.g. `gpt2`, `Qwen/Qwen2.5-0.5B`)
- `--save-dir out/` save the fine-tuned checkpoint
- `--iterations`, `--batch-size`, `--lr` scale up for real runs

## 4. Running on the hecate cluster (SLURM + pyxis/enroot)

Validated: a single-GPU 150-step run on hecate took ~3 minutes and went
from 54% to 100% greedy accuracy, with code/weights/HF cache all kept on
Lustre, submitted from the login-node home directory via the account's
existing `srun --container-image=...` pattern (no Python 3.14 needed
there — hecate's container already bundles a CUDA-matched PyTorch).
`train_rlvr.py` also supports multi-GPU data-parallel training (DDP) via
`torchrun`; the hecate scripts now launch with `--nproc_per_node=4` and
1000 iterations by default to use the full node. Code/weights live
directly under `/lustre/fsw/general_sa/bbalakreshna` (`PROJECT_DIR` ==
`LUSTRE_DIR`) for the multi-GPU setup.

See [`scripts/README_cluster.md`](scripts/README_cluster.md) for the
full copy-paste command blocks: writing the code to Lustre, submitting
the job, checking status/output, and pushing the fine-tuned model to
Hugging Face (`hf upload`). Nothing runs automatically against the
cluster — every command is meant to be pasted into your own
already-authenticated terminal.

## 5. Other clusters over SSH (generic SLURM, no container)

If you're pointing this at a plain SLURM cluster without pyxis/enroot
containers, use `scripts/slurm_rlvr.sbatch` + `scripts/sync_to_cluster.sh`
instead — see the "Other clusters" section at the bottom of
[`scripts/README_cluster.md`](scripts/README_cluster.md).
