# Hecate execute shell — multi-GPU / DDP (copy/paste blocks)

These are the exact commands to stand up and run the RLVR sample on
hecate using **4-GPU data-parallel training (DDP via `torchrun`)** --
each GPU holds its own model replica, samples its own batch (seeded
per-rank), and gradients are all-reduced across ranks every step.
Effective global batch = 4 x `--batch-size`. Defaults to 1000
iterations. Paste each block as-is into your already-authenticated
hecate terminal, in order. See [`README_cluster.md`](README_cluster.md)
for the narrative walkthrough, or
[`executeshell-singlegpu.md`](executeshell-singlegpu.md) for the
validated single-GPU (150-iteration) version.

## Block 1 — write the RLVR code to Lustre

Writes `task.py`, `train_rlvr.py`, `requirements-cluster.txt`, and
`scripts/launch_rlvr.sh` directly under
`/lustre/fsw/general_sa/bbalakreshna` (PROJECT_DIR == LUSTRE_DIR, no
subfolder).

```bash
export ACCOUNT="general_sa"
export LUSTRE_DIR="/lustre/fsw/$ACCOUNT/$USER"
export PROJECT_DIR="$LUSTRE_DIR"
mkdir -p "$PROJECT_DIR/scripts" "$PROJECT_DIR/out" "$LUSTRE_DIR/hf_cache"

cat > "$PROJECT_DIR/task.py" << 'PYEOF'
"""Synthetic verifiable-reward task for the RLVR sample.

The task is deliberately simple (two-operand arithmetic) so the whole
pipeline can be trained and verified on a laptop CPU in a couple of
minutes. Swap this module out for a harder verifiable task (GSM8K,
code execution, unit tests, etc.) once the pipeline is validated.

Per-operator operand ranges are tuned so the task is hard enough to
leave real room for RL to improve a ~1.5B instruct model, but not so
hard (e.g. raw 3-digit x 3-digit multiplication) that it's essentially
unsolvable without much longer chain-of-thought budgets.
"""

import random
import re

OPS = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
}

# (min_operand, max_operand) per operator -- multiplication gets a
# smaller range since it's much harder for a small LM to do reliably.
OPERAND_RANGES = {
    "+": (10, 999),
    "-": (10, 999),
    "*": (2, 50),
}

PROMPT_TEMPLATE = "Question: What is {a} {op} {b}?\nAnswer: The result is"

_QUESTION_TEMPLATE = (
    "What is {a} {op} {b}? "
    "You may reason briefly, but end your reply with exactly one line "
    "in the form 'Final Answer: <number>'."
)

# Prefer the number following "Final Answer" if the model followed the
# requested format; otherwise fall back to the last integer anywhere in
# the completion (covers models/formats that skip the exact phrase).
_FINAL_ANSWER_RE = re.compile(r"Final Answer:\s*(-?\d+)", re.IGNORECASE)
_ANSWER_RE = re.compile(r"-?\d+")


def sample_problem(rng: random.Random, max_operand: int | None = None) -> dict:
    op = rng.choice(list(OPS.keys()))
    lo, hi = OPERAND_RANGES[op]
    if max_operand is not None:
        hi = min(hi, max_operand)
    a = rng.randint(lo, hi)
    b = rng.randint(lo, hi)
    answer = OPS[op](a, b)
    question = _QUESTION_TEMPLATE.format(a=a, op=op, b=b)
    prompt = PROMPT_TEMPLATE.format(a=a, op=op, b=b)
    return {"question": question, "prompt": prompt, "answer": answer, "op": op}


def sample_batch(rng: random.Random, batch_size: int, max_operand: int | None = None) -> list[dict]:
    return [sample_problem(rng, max_operand) for _ in range(batch_size)]


def extract_answer(completion_text: str) -> int | None:
    final = _FINAL_ANSWER_RE.search(completion_text)
    if final:
        return int(final.group(1))
    matches = _ANSWER_RE.findall(completion_text)
    if not matches:
        return None
    try:
        return int(matches[-1])
    except ValueError:
        return None


def verify_reward(completion_text: str, gold_answer: int) -> float:
    """The verifier: a pure function, no learned reward model involved.

    Returns 1.0 for an exact correct answer, 0.1 for producing *some*
    parseable integer (partial credit for following the format), else 0.0.
    """
    predicted = extract_answer(completion_text)
    if predicted is None:
        return 0.0
    if predicted == gold_answer:
        return 1.0
    return 0.1
PYEOF

cat > "$PROJECT_DIR/train_rlvr.py" << 'PYEOF'
"""Minimal RLVR (Reinforcement Learning with Verifiable Rewards) demo.

Algorithm: REINFORCE with a running-mean baseline, applied to a causal
LM (Qwen2.5-1.5B-Instruct by default) on a synthetic arithmetic task
whose reward is computed by a deterministic verifier (task.verify_reward)
-- no learned reward model, no human preference data.

Defaults are sized for GPU (auto-detected via torch.cuda.is_available());
on a 24GB card, 150 steps at batch size 16 takes a few minutes and
reliably pushes greedy-eval accuracy from ~50-60% to 85-95%+. The same
run works on CPU, just much slower -- pass smaller --iterations/--batch-size
for a CPU smoke test.

Multi-GPU: launch with torchrun for data-parallel training (DDP) -- each
rank holds its own model replica on its own GPU, samples its own batch
(seeded per-rank so ranks see different problems), and DDP all-reduces
gradients every step so all replicas stay in sync. Only rank 0 logs,
evaluates, and saves the checkpoint.

Usage:
    python train_rlvr.py --iterations 150 --batch-size 16 --save-dir out/run1
    torchrun --standalone --nproc_per_node=4 train_rlvr.py \\
        --iterations 1000 --batch-size 16 --save-dir out/run1
"""

import argparse
import os
import random
from collections import defaultdict

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoModelForCausalLM, AutoTokenizer

from task import OPS, sample_batch, verify_reward

OP_NAMES = list(OPS.keys())


def build_prompt(tokenizer, problem: dict) -> str:
    """Use the tokenizer's chat template for instruct models, else the raw prompt."""
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": problem["question"]}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return problem["prompt"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RLVR training demo")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--iterations", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=16, help="Per-GPU (per-rank) batch size")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-dir", default=None, help="Optional dir to save the fine-tuned model")
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--eval-n", type=int, default=50, help="Number of held-out problems for greedy eval")
    p.add_argument(
        "--require-gpu",
        action="store_true",
        help="Fail fast instead of silently falling back to CPU if no CUDA device is found",
    )
    return p.parse_args()


@torch.no_grad()
def _greedy_eval(model, tokenizer, device, rng, n=20, max_new_tokens=8):
    """Returns (overall_accuracy, {op: (correct, total)})."""
    problems = sample_batch(rng, n)
    correct = 0
    per_op = {op: [0, 0] for op in OP_NAMES}  # op -> [correct, total]
    for prob in problems:
        inputs = tokenizer(build_prompt(tokenizer, prob), return_tensors="pt").to(device)
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        completion = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        op = prob["op"]
        per_op[op][1] += 1
        if verify_reward(completion, prob["answer"]) == 1.0:
            correct += 1
            per_op[op][0] += 1
    per_op_result = {op: (c, t) for op, (c, t) in per_op.items()}
    return correct / n, per_op_result


def _print_per_op_accuracy(label: str, per_op: dict) -> None:
    parts = []
    for op in OP_NAMES:
        correct, total = per_op.get(op, (0, 0))
        pct = f"{correct / total:.2%}" if total else "n/a"
        parts.append(f"{op}: {pct} ({correct}/{total})")
    print(f"  {label} by operator -- " + ", ".join(parts))


def main() -> None:
    args = parse_args()

    # torchrun sets these; plain `python train_rlvr.py` leaves them unset (single process).
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    distributed = world_size > 1
    is_main = rank == 0

    # Each rank samples different problems -- avoids every GPU training on
    # identical batches, which would waste the extra compute.
    rng = random.Random(args.seed + rank)
    torch.manual_seed(args.seed + rank)

    if distributed:
        if not torch.cuda.is_available():
            raise SystemExit("Distributed (torchrun) launch requires CUDA GPUs.")
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        if args.require_gpu and not torch.cuda.is_available():
            raise SystemExit(
                "No CUDA device found but --require-gpu was set. "
                "Check `nvidia-smi` and that torch was installed from the cu12x index "
                "(see requirements.txt), or drop --require-gpu to run on CPU."
            )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if is_main:
        if device.type == "cuda":
            suffix = f" x{world_size} ranks" if distributed else ""
            print(f"Using device: cuda ({torch.cuda.get_device_name(device)}){suffix}")
        else:
            print("Using device: cpu (no CUDA device found -- this will be slow for these defaults)")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # required for batched generation

    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    model.train()

    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank])
    # generate() only exists on the underlying HF model, not the DDP wrapper.
    generator = model.module if distributed else model

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    if is_main:
        print("Baseline accuracy before training:")
        baseline_acc, baseline_per_op = _greedy_eval(generator, tokenizer, device, rng, n=args.eval_n, max_new_tokens=args.max_new_tokens)
        print(f"  greedy accuracy = {baseline_acc:.2%}")
        _print_per_op_accuracy("baseline", baseline_per_op)
    if distributed:
        dist.barrier()

    # Per-operator training reward, accumulated locally on this rank across
    # all steps -- reduced (summed) across ranks after training so the final
    # report reflects every GPU's data, not just rank 0's local shard.
    op_reward_sum = defaultdict(float)
    op_reward_count = defaultdict(int)

    running_baseline = 0.0
    for step in range(1, args.iterations + 1):
        problems = sample_batch(rng, args.batch_size)
        prompts = [build_prompt(tokenizer, p) for p in problems]

        enc = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        prompt_len = enc["input_ids"].shape[1]

        with torch.no_grad():
            gen_out = generator.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_k=0,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Rewards from the verifier (the "VR" in RLVR) -- no reward model.
        rewards = []
        for i, prob in enumerate(problems):
            completion_ids = gen_out[i][prompt_len:]
            completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
            r = verify_reward(completion_text, prob["answer"])
            rewards.append(r)
            op_reward_sum[prob["op"]] += r
            op_reward_count[prob["op"]] += 1
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)

        # Running-mean baseline reduces variance of the REINFORCE gradient.
        # Local to each rank (computed from that rank's own batch only) --
        # DDP's gradient all-reduce is what keeps the ranks' models in sync.
        batch_mean = rewards_t.mean().item()
        advantages = rewards_t - running_baseline
        running_baseline = 0.9 * running_baseline + 0.1 * batch_mean

        # Teacher-forced forward pass to get log-probs of the tokens the
        # model actually sampled during rollout.
        attention_mask = (gen_out != tokenizer.pad_token_id).long()
        logits = model(input_ids=gen_out, attention_mask=attention_mask).logits
        log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
        target_ids = gen_out[:, 1:]
        token_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

        # Only count log-probs for generated tokens (mask out prompt + padding).
        gen_mask = torch.zeros_like(target_ids, dtype=torch.float32)
        gen_mask[:, prompt_len - 1:] = 1.0
        gen_mask *= attention_mask[:, 1:].float()

        seq_log_prob = (token_log_probs * gen_mask).sum(dim=1) / gen_mask.sum(dim=1).clamp(min=1)

        loss = -(advantages * seq_log_prob).mean()

        optimizer.zero_grad()
        loss.backward()  # DDP all-reduces gradients across ranks here
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if is_main and step % args.log_every == 0:
            print(
                f"step {step:4d}/{args.iterations} | "
                f"mean_reward={batch_mean:.3f} | baseline={running_baseline:.3f} | loss={loss.item():.4f}"
            )

    if distributed:
        dist.barrier()

    # Sum per-operator training reward across all ranks so the report covers
    # every GPU's data, not just rank 0's local shard.
    op_sum_t = torch.tensor([op_reward_sum[op] for op in OP_NAMES], dtype=torch.float32, device=device)
    op_count_t = torch.tensor([op_reward_count[op] for op in OP_NAMES], dtype=torch.float32, device=device)
    if distributed:
        dist.reduce(op_sum_t, dst=0)
        dist.reduce(op_count_t, dst=0)

    if is_main:
        print("\nTraining reward by operator (all ranks, all steps):")
        for i, op in enumerate(OP_NAMES):
            count = int(op_count_t[i].item())
            mean = (op_sum_t[i] / op_count_t[i]).item() if count else float("nan")
            print(f"  {op}: mean_reward={mean:.3f} over {count} samples")

        print("\nAccuracy after training:")
        final_acc, final_per_op = _greedy_eval(generator, tokenizer, device, rng, n=args.eval_n, max_new_tokens=args.max_new_tokens)
        print(f"  greedy accuracy = {final_acc:.2%} (baseline was {baseline_acc:.2%})")
        _print_per_op_accuracy("final", final_per_op)

        if args.save_dir:
            generator.save_pretrained(args.save_dir)
            tokenizer.save_pretrained(args.save_dir)
            print(f"Saved fine-tuned model to {args.save_dir}")

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
PYEOF

cat > "$PROJECT_DIR/requirements-cluster.txt" << 'REQEOF'
# For use inside the gitlab-master.nvidia.com/dl/dgx/pytorch:main-py3-devel
# container on hecate -- that image already bundles a CUDA-matched torch,
# so we deliberately don't pin/install torch here.
transformers>=4.44
datasets>=2.20
numpy>=1.26
tqdm>=4.66
REQEOF

cat > "$PROJECT_DIR/scripts/launch_rlvr.sh" << 'SHEOF'
#!/bin/bash
# Runs INSIDE the pyxis/enroot container on the compute node.
# The container already bundles a CUDA-matched PyTorch, so we don't touch
# torch here -- only install the lightweight deps train_rlvr.py needs.
set -e

PROJECT_DIR="/lustre/fsw/general_sa/bbalakreshna"

echo "$(hostname): Installing RLVR dependencies..."
pip install --quiet -r "$PROJECT_DIR/requirements-cluster.txt"

export HF_HOME="/lustre/fsw/general_sa/bbalakreshna/hf_cache"
mkdir -p "$HF_HOME"

echo "$(hostname): Launching RLVR training on ${RLVR_NPROC_PER_NODE:-4} GPU(s) (DDP via torchrun)..."
torchrun --standalone --nproc_per_node="${RLVR_NPROC_PER_NODE:-4}" \
  "$PROJECT_DIR/train_rlvr.py" \
  --model "${RLVR_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}" \
  --iterations "${RLVR_ITERATIONS:-1000}" \
  --batch-size "${RLVR_BATCH_SIZE:-16}" \
  --max-new-tokens "${RLVR_MAX_NEW_TOKENS:-64}" \
  --lr "${RLVR_LR:-1e-5}" \
  --eval-n "${RLVR_EVAL_N:-50}" \
  --save-dir "$PROJECT_DIR/out/rlvr-hecate-run1"
SHEOF
chmod +x "$PROJECT_DIR/scripts/launch_rlvr.sh"

echo "--- Files written ---"
ls -la "$PROJECT_DIR" "$PROJECT_DIR/scripts"
wc -l "$PROJECT_DIR/task.py" "$PROJECT_DIR/train_rlvr.py"
```

## Block 2 — write and run `submit_hecate.sh`

Writes the job launcher to `$LUSTRE_DIR` (top level, not under the
project's `scripts/`) and submits it in the background -- training
output goes to `$PROJECT_DIR/out/hecate_run1.log`, and start/end
timestamps + total elapsed time are recorded to
`$PROJECT_DIR/out/hecate_run1.timing` once the job finishes.

```bash
cat > "$LUSTRE_DIR/submit_hecate.sh" << 'SHEOF'
#!/bin/bash
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
SHEOF
chmod +x "$LUSTRE_DIR/submit_hecate.sh"

bash "$LUSTRE_DIR/submit_hecate.sh"
```

## Checking on it afterward

```bash
squeue -u $USER
sacct -u $USER --format=JobID,JobName,State,Elapsed,ExitCode -j <JOBID>
cat "$PROJECT_DIR/out/hecate_run1.timing"   # start/end timestamps + total elapsed time
tail -f "$PROJECT_DIR/out/hecate_run1.log"
```

## Pushing the result to Hugging Face

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

## Checking GPU usage for a running job

Replace `509861` with your job's actual ID (from `squeue -u $USER`).

```bash
# 1. Find which node(s) the job is running on
scontrol show job 509861 | grep -i nodelist
# or
squeue -j 509861

# 2. Attach to the running job's allocation and run nvidia-smi there
#    (--overlap lets you add a step without stealing the job's resources)
srun --jobid=509861 --overlap --pty nvidia-smi

# 3. Live-monitoring, refreshing every 2s
srun --jobid=509861 --overlap --pty watch -n 2 nvidia-smi
```

With the 4-GPU DDP run you should see 4 `python`/`torchrun` processes,
one per GPU, each using roughly the same memory as the single-GPU run
(~18GB for the 1.5B model + AdamW optimizer states), with utilization
spiking during each step's generate/forward/backward phases.

## Generating insights from the run

`analyze_run.py` parses the training log's `step N/M | mean_reward=...`
lines plus the baseline/final accuracy lines and prints a summary
(reward/loss trend, accuracy delta, throughput from the `.timing` file).
It only uses the Python standard library, so it runs fine with the login
node's system `python3` directly -- no venv, no `pip install`, no PEP 668
issue.

```bash
cat > "$LUSTRE_DIR/analyze_run.py" << 'PYEOF'
"""Summarize an RLVR training run from its log (and optional .timing file).

Parses the step-by-step `step N/M | mean_reward=... | baseline=... | loss=...`
lines plus the baseline/final greedy-eval accuracy lines that train_rlvr.py
prints, and reports the key numbers you'd otherwise have to eyeball out of
the raw log: reward trend, loss trend, accuracy delta, and (if a .timing
file is given) wall-clock time and derived throughput.

Usage:
    python analyze_run.py out/hecate_run1.log
    python analyze_run.py out/hecate_run1.log --timing out/hecate_run1.timing
    python analyze_run.py out/hecate_run1.log --plot out/hecate_run1.png
"""

import argparse
import re
import statistics

STEP_RE = re.compile(
    r"step\s+(\d+)/(\d+)\s*\|\s*mean_reward=([\d.]+)\s*\|\s*baseline=([\d.]+)\s*\|\s*loss=(-?[\d.]+)"
)
ACCURACY_RE = re.compile(r"greedy accuracy\s*=\s*([\d.]+)%")
DEVICE_RE = re.compile(r"Using device:\s*(.+)")
SAVED_RE = re.compile(r"Saved fine-tuned model to (.+)")
PER_OP_EVAL_RE = re.compile(r"(baseline|final) by operator -- (.+)")
PER_OP_EVAL_ENTRY_RE = re.compile(r"([+\-*]):\s*([\d.]+)%\s*\((\d+)/(\d+)\)")
PER_OP_TRAIN_RE = re.compile(r"^\s*([+\-*]):\s*mean_reward=([\d.]+)\s*over\s*(\d+)\s*samples", re.MULTILINE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize an RLVR training run")
    p.add_argument("log", help="Path to the training log (e.g. out/hecate_run1.log)")
    p.add_argument("--timing", default=None, help="Path to the matching .timing file")
    p.add_argument("--plot", default=None, help="Optional path to save a reward/loss curve PNG (requires matplotlib)")
    p.add_argument("--head-frac", type=float, default=0.1, help="Fraction of steps counted as 'early' vs 'late' for trend comparison")
    return p.parse_args()


def _fmt_pct(x: float) -> str:
    return f"{x:.2f}%"


def main() -> None:
    args = parse_args()

    with open(args.log, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    steps, rewards, baselines, losses = [], [], [], []
    for m in STEP_RE.finditer(text):
        steps.append(int(m.group(1)))
        rewards.append(float(m.group(3)))
        baselines.append(float(m.group(4)))
        losses.append(float(m.group(5)))

    accuracies = [float(x) for x in ACCURACY_RE.findall(text)]
    device_match = DEVICE_RE.search(text)
    saved_match = SAVED_RE.search(text)

    if not steps:
        print("No 'step N/M | mean_reward=...' lines found -- is this a train_rlvr.py log?")
        return

    n = len(steps)
    head_n = max(1, int(n * args.head_frac))
    tail_n = max(1, int(n * args.head_frac))

    print("=" * 60)
    print(f"RLVR run summary: {args.log}")
    print("=" * 60)

    if device_match:
        print(f"Device:            {device_match.group(1)}")
    print(f"Total steps:       {n} (log claims {steps[-1]}/{STEP_RE.search(text).group(2)} at first match)")

    print()
    print("-- Reward --")
    print(f"  Mean (all steps): {statistics.mean(rewards):.3f}")
    print(f"  Early avg (first {head_n}): {statistics.mean(rewards[:head_n]):.3f}")
    print(f"  Late avg  (last {tail_n}):  {statistics.mean(rewards[-tail_n:]):.3f}")
    print(f"  Max / Min:         {max(rewards):.3f} / {min(rewards):.3f}")
    print(f"  Running baseline:  {baselines[0]:.3f} -> {baselines[-1]:.3f}")

    print()
    print("-- Loss --")
    print(f"  Early avg (first {head_n}): {statistics.mean(losses[:head_n]):.4f}")
    print(f"  Late avg  (last {tail_n}):  {statistics.mean(losses[-tail_n:]):.4f}")
    print(f"  Std dev (all steps): {statistics.pstdev(losses):.4f}")

    if len(accuracies) >= 2:
        baseline_acc, final_acc = accuracies[0], accuracies[-1]
        print()
        print("-- Greedy-eval accuracy --")
        print(f"  Baseline: {_fmt_pct(baseline_acc)}")
        print(f"  Final:    {_fmt_pct(final_acc)}")
        print(f"  Delta:    {'+' if final_acc >= baseline_acc else ''}{final_acc - baseline_acc:.2f} pts")
    elif accuracies:
        print()
        print(f"-- Greedy-eval accuracy -- only one reading found: {_fmt_pct(accuracies[0])} (run may be incomplete)")

    per_op_train = PER_OP_TRAIN_RE.findall(text)
    if per_op_train:
        print()
        print("-- Training reward by operator (all ranks, all steps) --")
        for op, mean_reward, count in per_op_train:
            print(f"  {op}: mean_reward={float(mean_reward):.3f} over {count} samples")

    per_op_evals = PER_OP_EVAL_RE.findall(text)
    if per_op_evals:
        print()
        print("-- Accuracy by operator --")
        for label, rest in per_op_evals:
            entries = ", ".join(
                f"{op}={pct}% ({c}/{t})" for op, pct, c, t in PER_OP_EVAL_ENTRY_RE.findall(rest)
            )
            print(f"  {label}: {entries}")

    if saved_match:
        print()
        print(f"Checkpoint saved to: {saved_match.group(1).strip()}")

    if args.timing:
        with open(args.timing, "r", encoding="utf-8", errors="replace") as f:
            timing_text = f.read()
        print()
        print("-- Timing --")
        print(timing_text.strip())
        elapsed_m = re.search(r"Elapsed seconds:\s*(\d+)", timing_text)
        if elapsed_m:
            elapsed_s = int(elapsed_m.group(1))
            print(f"  Throughput: {elapsed_s / n:.2f} sec/step ({n} steps)")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("\n(--plot requested but matplotlib isn't installed: pip install matplotlib)")
        else:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
            ax1.plot(steps, rewards, label="mean_reward", alpha=0.6)
            ax1.plot(steps, baselines, label="running_baseline", linewidth=2)
            ax1.set_ylabel("reward")
            ax1.legend()
            ax1.set_title(f"RLVR training curve -- {args.log}")

            ax2.plot(steps, losses, color="tab:red", alpha=0.7)
            ax2.set_ylabel("loss")
            ax2.set_xlabel("step")

            fig.tight_layout()
            fig.savefig(args.plot, dpi=150)
            print(f"\nSaved plot to {args.plot}")


if __name__ == "__main__":
    main()
PYEOF

python3 "$LUSTRE_DIR/analyze_run.py" "$LUSTRE_DIR/out/hecate_run1.log" --timing "$LUSTRE_DIR/out/hecate_run1.timing"
```

Add `--plot "$LUSTRE_DIR/out/hecate_run1.png"` for a reward/loss curve PNG --
that flag needs `matplotlib`, which isn't in `requirements-cluster.txt`
(only needed if you want the plot; install it into the `.venv-upload`
venv from the Hugging Face upload step above, or a new one, if so).

## Interactive HTML insights dashboard

`generate_dashboard.py` parses the same log lines as `analyze_run.py`
and renders a single self-contained HTML file: stat cards (accuracy
delta, throughput, mean reward), a reward/baseline curve, a loss curve,
and per-operator accuracy/reward bar charts -- all interactive (hover
tooltips, toggleable legend) via Chart.js. Only the standard library is
used to *generate* the file (runs fine with hecate's system `python3`,
no venv needed); Chart.js itself loads from a CDN, so viewing the file
needs internet access in whatever browser you open it in (not on the
cluster).

```bash
cat > "$LUSTRE_DIR/generate_dashboard.py" << 'PYEOF'
"""Generate a self-contained interactive HTML dashboard from an RLVR run.

Parses the same log lines as analyze_run.py (step/reward/loss, baseline/
final accuracy, per-operator breakdowns, .timing) and renders them as an
interactive HTML file: reward/loss curves, before/after accuracy, and
per-operator bar charts, all with hover tooltips and toggleable legends.

Only the Python standard library is used to *generate* the file (no
pip install needed -- runs fine with hecate's system python3). Chart.js
is loaded from a CDN in the generated HTML, so viewing it needs an
internet connection in the browser you open it in (not on the cluster).

Usage:
    python3 generate_dashboard.py out/hecate_run1.log \\
        --timing out/hecate_run1.timing \\
        --output out/hecate_run1_dashboard.html
"""

import argparse
import html
import json
import re

STEP_RE = re.compile(
    r"step\s+(\d+)/(\d+)\s*\|\s*mean_reward=([\d.]+)\s*\|\s*baseline=([\d.]+)\s*\|\s*loss=(-?[\d.]+)"
)
ACCURACY_RE = re.compile(r"greedy accuracy\s*=\s*([\d.]+)%")
DEVICE_RE = re.compile(r"Using device:\s*(.+)")
SAVED_RE = re.compile(r"Saved fine-tuned model to (.+)")
PER_OP_EVAL_RE = re.compile(r"(baseline|final) by operator -- (.+)")
PER_OP_EVAL_ENTRY_RE = re.compile(r"([+\-*]):\s*([\d.]+)%\s*\((\d+)/(\d+)\)")
PER_OP_TRAIN_RE = re.compile(r"^\s*([+\-*]):\s*mean_reward=([\d.]+)\s*over\s*(\d+)\s*samples", re.MULTILINE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate an interactive HTML dashboard for an RLVR run")
    p.add_argument("log", help="Path to the training log (e.g. out/hecate_run1.log)")
    p.add_argument("--timing", default=None, help="Path to the matching .timing file")
    p.add_argument("--output", default=None, help="Output HTML path (default: <log>.dashboard.html)")
    return p.parse_args()


def parse_log(log_path: str) -> dict:
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    steps, rewards, baselines, losses = [], [], [], []
    for m in STEP_RE.finditer(text):
        steps.append(int(m.group(1)))
        rewards.append(float(m.group(3)))
        baselines.append(float(m.group(4)))
        losses.append(float(m.group(5)))

    accuracies = [float(x) for x in ACCURACY_RE.findall(text)]
    device_match = DEVICE_RE.search(text)
    saved_match = SAVED_RE.search(text)

    per_op_eval = {}  # label -> {op: (correct, total)}
    for label, rest in PER_OP_EVAL_RE.findall(text):
        per_op_eval[label] = {
            op: (int(c), int(t)) for op, pct, c, t in PER_OP_EVAL_ENTRY_RE.findall(rest)
        }

    per_op_train = {
        op: (float(mean_reward), int(count))
        for op, mean_reward, count in PER_OP_TRAIN_RE.findall(text)
    }

    return {
        "steps": steps,
        "rewards": rewards,
        "baselines": baselines,
        "losses": losses,
        "accuracies": accuracies,
        "device": device_match.group(1).strip() if device_match else None,
        "saved_to": saved_match.group(1).strip() if saved_match else None,
        "per_op_eval": per_op_eval,
        "per_op_train": per_op_train,
    }


def parse_timing(timing_path: str) -> dict:
    with open(timing_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    result = {"raw": text.strip()}
    for key, pattern in [
        ("started", r"Job started:\s*(.+)"),
        ("ended", r"Job ended:\s*(.+)"),
        ("elapsed_seconds", r"Elapsed seconds:\s*(\d+)"),
        ("elapsed_hms", r"Elapsed \(h:m:s\):\s*(.+)"),
    ]:
        m = re.search(pattern, text)
        if m:
            result[key] = m.group(1).strip()
    return result


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>RLVR Run Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  :root {
    --bg: #0f1117; --panel: #171a24; --border: #262b3a;
    --text: #e6e8ef; --muted: #9aa1b2;
    --accent: #6ea8fe; --good: #4ade80; --bad: #f87171; --warn: #fbbf24;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 24px; background: var(--bg); color: var(--text);
    font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
  }
  h1 { font-size: 1.4rem; margin: 0 0 4px; }
  .subtitle { color: var(--muted); margin-bottom: 24px; font-size: 0.9rem; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin-bottom: 24px; }
  .stat { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; }
  .stat .label { color: var(--muted); font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em; }
  .stat .value { font-size: 1.6rem; font-weight: 600; margin-top: 4px; }
  .stat .delta { font-size: 0.85rem; margin-top: 2px; }
  .delta.pos { color: var(--good); }
  .delta.neg { color: var(--bad); }
  .panels { display: grid; grid-template-columns: 1fr; gap: 20px; }
  .panel { background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 18px; }
  .panel h2 { font-size: 1rem; margin: 0 0 12px; color: var(--text); }
  .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  @media (max-width: 800px) { .two-col { grid-template-columns: 1fr; } }
  canvas { max-height: 340px; }
  footer { margin-top: 24px; color: var(--muted); font-size: 0.78rem; }
  code { background: #1f2330; padding: 1px 6px; border-radius: 4px; }
</style>
</head>
<body>
  <h1>RLVR Run Dashboard</h1>
  <div class="subtitle">__SUBTITLE__</div>

  <div class="grid" id="stat-grid"></div>

  <div class="panels">
    <div class="panel">
      <h2>Reward &amp; running baseline over training</h2>
      <canvas id="rewardChart"></canvas>
    </div>
    <div class="panel">
      <h2>Loss over training</h2>
      <canvas id="lossChart"></canvas>
    </div>
    <div class="two-col">
      <div class="panel">
        <h2>Accuracy by operator (baseline vs final)</h2>
        <canvas id="perOpAccChart"></canvas>
      </div>
      <div class="panel">
        <h2>Training reward by operator (all steps)</h2>
        <canvas id="perOpRewardChart"></canvas>
      </div>
    </div>
  </div>

  <footer>Generated from <code>__LOGPATH__</code> by generate_dashboard.py</footer>

<script>
const DATA = __DATA_JSON__;

Chart.defaults.color = "#9aa1b2";
Chart.defaults.borderColor = "#262b3a";

// -- Stat cards --
const statGrid = document.getElementById("stat-grid");
function statCard(label, value, delta) {
  const div = document.createElement("div");
  div.className = "stat";
  let deltaHtml = "";
  if (delta !== undefined && delta !== null) {
    const cls = delta >= 0 ? "pos" : "neg";
    const sign = delta >= 0 ? "+" : "";
    deltaHtml = `<div class="delta ${cls}">${sign}${delta.toFixed(2)} pts</div>`;
  }
  div.innerHTML = `<div class="label">${label}</div><div class="value">${value}</div>${deltaHtml}`;
  statGrid.appendChild(div);
}

if (DATA.device) statCard("Device", DATA.device);
statCard("Total steps", DATA.steps.length);
if (DATA.accuracies.length >= 2) {
  const base = DATA.accuracies[0], fin = DATA.accuracies[DATA.accuracies.length - 1];
  statCard("Baseline accuracy", base.toFixed(1) + "%");
  statCard("Final accuracy", fin.toFixed(1) + "%", fin - base);
}
if (DATA.timing && DATA.timing.elapsed_hms) {
  statCard("Elapsed", DATA.timing.elapsed_hms);
}
if (DATA.timing && DATA.timing.elapsed_seconds && DATA.steps.length) {
  const perStep = DATA.timing.elapsed_seconds / DATA.steps.length;
  statCard("Throughput", perStep.toFixed(2) + " sec/step");
}
if (DATA.rewards.length) {
  const mean = DATA.rewards.reduce((a, b) => a + b, 0) / DATA.rewards.length;
  statCard("Mean reward", mean.toFixed(3));
}

// -- Reward chart --
new Chart(document.getElementById("rewardChart"), {
  type: "line",
  data: {
    labels: DATA.steps,
    datasets: [
      { label: "mean_reward (per step)", data: DATA.rewards, borderColor: "#6ea8fe", backgroundColor: "transparent", pointRadius: 0, borderWidth: 1.5, tension: 0.15 },
      { label: "running_baseline", data: DATA.baselines, borderColor: "#4ade80", backgroundColor: "transparent", pointRadius: 0, borderWidth: 2, tension: 0.15 },
    ],
  },
  options: {
    responsive: true,
    interaction: { mode: "index", intersect: false },
    scales: { x: { title: { display: true, text: "step" } }, y: { min: 0, max: 1.05 } },
  },
});

// -- Loss chart --
new Chart(document.getElementById("lossChart"), {
  type: "line",
  data: {
    labels: DATA.steps,
    datasets: [
      { label: "loss", data: DATA.losses, borderColor: "#f87171", backgroundColor: "transparent", pointRadius: 0, borderWidth: 1.5, tension: 0.15 },
    ],
  },
  options: {
    responsive: true,
    interaction: { mode: "index", intersect: false },
    scales: { x: { title: { display: true, text: "step" } } },
  },
});

// -- Per-operator accuracy --
const opOrder = ["+", "-", "*"];
const perOpEval = DATA.per_op_eval || {};
function accFor(label) {
  return opOrder.map((op) => {
    const entry = (perOpEval[label] || {})[op];
    return entry ? (100 * entry[0] / entry[1]) : null;
  });
}
new Chart(document.getElementById("perOpAccChart"), {
  type: "bar",
  data: {
    labels: opOrder,
    datasets: [
      { label: "baseline", data: accFor("baseline"), backgroundColor: "#fbbf24aa" },
      { label: "final", data: accFor("final"), backgroundColor: "#4ade80aa" },
    ],
  },
  options: {
    responsive: true,
    scales: { y: { min: 0, max: 100, title: { display: true, text: "accuracy %" } } },
  },
});

// -- Per-operator training reward --
const perOpTrain = DATA.per_op_train || {};
new Chart(document.getElementById("perOpRewardChart"), {
  type: "bar",
  data: {
    labels: opOrder,
    datasets: [
      {
        label: "mean training reward",
        data: opOrder.map((op) => (perOpTrain[op] ? perOpTrain[op][0] : null)),
        backgroundColor: "#6ea8feaa",
      },
    ],
  },
  options: {
    responsive: true,
    plugins: {
      tooltip: {
        callbacks: {
          afterLabel: (ctx) => {
            const op = opOrder[ctx.dataIndex];
            const entry = perOpTrain[op];
            return entry ? `n = ${entry[1]} samples` : "";
          },
        },
      },
    },
    scales: { y: { min: 0, max: 1.05, title: { display: true, text: "mean reward" } } },
  },
});
</script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    parsed = parse_log(args.log)

    if not parsed["steps"]:
        raise SystemExit("No 'step N/M | mean_reward=...' lines found -- is this a train_rlvr.py log?")

    data = dict(parsed)
    data["timing"] = parse_timing(args.timing) if args.timing else None

    output_path = args.output or (args.log.rsplit(".", 1)[0] + ".dashboard.html")

    subtitle_bits = [html.escape(args.log)]
    if data["device"]:
        subtitle_bits.append(html.escape(data["device"]))
    subtitle = " | ".join(subtitle_bits)

    out_html = (
        HTML_TEMPLATE
        .replace("__SUBTITLE__", subtitle)
        .replace("__LOGPATH__", html.escape(args.log))
        .replace("__DATA_JSON__", json.dumps(data))
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(out_html)

    print(f"Wrote dashboard to {output_path}")


if __name__ == "__main__":
    main()
PYEOF

python3 "$LUSTRE_DIR/generate_dashboard.py" "$LUSTRE_DIR/out/hecate_run1.log" \
  --timing "$LUSTRE_DIR/out/hecate_run1.timing" \
  --output "$LUSTRE_DIR/out/hecate_run1_dashboard.html"
```

Then copy `hecate_run1_dashboard.html` off Lustre (`scp`, or download via the
Hugging Face repo/whatever transfer path you already use) and open it in a
browser -- it's a single file, no server needed.
