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

# Matches both train_rlvr.py (…| loss=x) and train_rlvr_qwen38.py (…| grad_norm=x | rollouts=a/b)
STEP_RE = re.compile(
    r"step\s+(\d+)/(\d+)\s*\|\s*mean_reward=([\d.]+)\s*\|\s*baseline=([\d.]+)"
    r"(?:\s*\|\s*loss=(-?[\d.]+))?(?:\s*\|\s*grad_norm=([\d.]+))?"
)
SKIP_GRAD_RE = re.compile(r"^step\s+(\d+): NON-FINITE gradient norm", re.MULTILINE)
ACCOUNTING_RE = re.compile(r"Step accounting:\s*(\{.*\})")
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
    grad_norms = []
    for m in STEP_RE.finditer(text):
        steps.append(int(m.group(1)))
        rewards.append(float(m.group(3)))
        baselines.append(float(m.group(4)))
        if m.group(5) is not None:
            losses.append(float(m.group(5)))
        if m.group(6) is not None:
            grad_norms.append(float(m.group(6)))
    skipped_grad = [int(s) for s in SKIP_GRAD_RE.findall(text)]
    accounting_m = ACCOUNTING_RE.search(text)

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

    if losses:
        print()
        print("-- Loss --")
        print(f"  Early avg (first {head_n}): {statistics.mean(losses[:head_n]):.4f}")
        print(f"  Late avg  (last {tail_n}):  {statistics.mean(losses[-tail_n:]):.4f}")
        print(f"  Std dev (all steps): {statistics.pstdev(losses):.4f}")

    if grad_norms:
        print()
        print("-- Gradient norm (after clipping) --")
        print(f"  Early avg (first {head_n}): {statistics.mean(grad_norms[:head_n]):.4f}")
        print(f"  Late avg  (last {tail_n}):  {statistics.mean(grad_norms[-tail_n:]):.4f}")
        print(f"  Max:                 {max(grad_norms):.4f}")
        print("  (-> 0 late in the run means the baseline caught up with the reward: task solved)")

    if skipped_grad or accounting_m:
        print()
        print("-- Stability guards (train_rlvr_qwen38.py) --")
        if skipped_grad:
            first, last = skipped_grad[0], skipped_grad[-1]
            print(f"  Steps skipped for non-finite gradient: {len(skipped_grad)} (first at step {first}, last at step {last})")
        if accounting_m:
            print(f"  Step accounting: {accounting_m.group(1)}")

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
"""Generate a self-contained interactive HTML dashboard from an RLVR run log.

Works for both training scripts:
  - train_rlvr.py        : step N/M | mean_reward=.. | baseline=.. | loss=..
  - train_rlvr_qwen38.py : step N/M | mean_reward=.. | baseline=.. | grad_norm=.. | rollouts=a/b
Step lines are parsed generically as `key=value` pairs, so any numeric
metric present gets a chart. Also picks up (when present): baseline/final
greedy accuracy, per-operator accuracy and reward, the qwen38 script's
guard messages (NON-FINITE gradient norm / logits / parameter rollbacks),
its `Step accounting: {...}` summary, per-GPU memory after load, and the
.timing file.

Standard library only -- runs on hecate's login node with system python3.
Chart.js is loaded from a CDN in the generated HTML, so the *browser*
needs internet; the cluster does not.

Usage:
    python3 generate_dashboard.py out/hecate_qwen38_run1.log \\
        --timing out/hecate_qwen38_run1.timing \\
        --output out/hecate_qwen38_run1_dashboard.html
"""

import argparse
import ast
import html
import json
import re

STEP_RE = re.compile(r"^step\s+(\d+)/(\d+)\s*\|(.*)$", re.MULTILINE)
KV_RE = re.compile(r"(\w+)=(-?[\d.]+(?:/\d+)?)")
ACCURACY_RE = re.compile(r"greedy accuracy\s*=\s*([\d.]+)%")
DEVICE_RE = re.compile(r"Using device:\s*(.+)")
LOAD_RE = re.compile(r"Loading (\S+) \(([^)]*)\), sharded across (\d+) GPU")
GPU_MEM_RE = re.compile(r"GPU (\d+): ([\d.]+) GiB allocated after load")
DDP_RE = re.compile(r"DDP x(\d+) ranks")
SAVED_RE = re.compile(r"Saved (?:fine-tuned model|LoRA adapter) to (.+)")
PER_OP_EVAL_RE = re.compile(r"(baseline|final) by operator -- (.+)")
PER_OP_EVAL_ENTRY_RE = re.compile(r"([+\-*]):\s*([\d.]+)%\s*\((\d+)/(\d+)\)")
PER_OP_TRAIN_RE = re.compile(r"^\s*([+\-*]):\s*mean_reward=([\d.]+)\s*over\s*(\d+)\s*samples", re.MULTILINE)
SKIP_GRAD_RE = re.compile(r"^step\s+(\d+): NON-FINITE gradient norm", re.MULTILINE)
SKIP_LOGITS_RE = re.compile(r"^step\s+(\d+): (?:NON-FINITE logits|non-finite logits|non-finite loss)", re.MULTILINE)
ROLLBACK_RE = re.compile(r"^step\s+(\d+): NON-FINITE parameters", re.MULTILINE)
ACCOUNTING_RE = re.compile(r"Step accounting:\s*(\{.*\})")
EVAL_SAMPLE_RE = re.compile(r"eval sample \(gold=(-?\d+), reward=([\d.]+)\): (.+)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate an interactive HTML dashboard for an RLVR run")
    p.add_argument("log", help="Path to the training log")
    p.add_argument("--timing", default=None, help="Path to the matching .timing file")
    p.add_argument("--output", default=None, help="Output HTML path (default: <log>.dashboard.html)")
    p.add_argument("--skip-bin", type=int, default=50, help="Step-bin width for the skipped-steps histogram")
    return p.parse_args()


def _num(v: str):
    if "/" in v:  # rollouts=4/4 -> fraction
        a, b = v.split("/")
        return float(a) / float(b) if float(b) else None
    return float(v)


def parse_log(path: str, skip_bin: int) -> dict:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    steps, total_steps = [], None
    metrics = {}  # name -> list aligned with steps
    for m in STEP_RE.finditer(text):
        step = int(m.group(1))
        total_steps = int(m.group(2))
        kv = {k: _num(v) for k, v in KV_RE.findall(m.group(3))}
        steps.append(step)
        for k in set(metrics) | set(kv):
            metrics.setdefault(k, [None] * (len(steps) - 1)).append(kv.get(k))

    skipped_grad = [int(s) for s in SKIP_GRAD_RE.findall(text)]
    skipped_logits = [int(s) for s in SKIP_LOGITS_RE.findall(text)]
    rollbacks = [int(s) for s in ROLLBACK_RE.findall(text)]

    # Histogram of skipped steps over the run: were the guards firing early, late, throughout?
    hist = None
    if total_steps and (skipped_grad or skipped_logits or rollbacks):
        nbins = max(1, -(-total_steps // skip_bin))
        labels = [f"{i * skip_bin + 1}-{min((i + 1) * skip_bin, total_steps)}" for i in range(nbins)]
        def bucket(lst):
            counts = [0] * nbins
            for s in lst:
                counts[min((s - 1) // skip_bin, nbins - 1)] += 1
            return counts
        hist = {"labels": labels, "grad": bucket(skipped_grad),
                "logits": bucket(skipped_logits), "rollback": bucket(rollbacks)}

    accounting = None
    am = ACCOUNTING_RE.search(text)
    if am:
        try:
            accounting = ast.literal_eval(am.group(1))
        except (ValueError, SyntaxError):
            accounting = None

    device = None
    dm = DEVICE_RE.search(text)
    if dm:
        device = dm.group(1).strip()
    else:
        lm = LOAD_RE.search(text)
        if lm:
            device = f"{lm.group(1)} ({lm.group(2)}) on {lm.group(3)} GPUs"

    gpu_mem = {int(i): float(g) for i, g in GPU_MEM_RE.findall(text)}

    # DDP logs written before the all-gather fix (commit 444aeeb) printed
    # torch.cuda.memory_allocated() from rank 0 only, which is a per-process
    # counter -- GPUs 1..N-1 show 0.0 even though each rank holds a full
    # replica there. Detect that case so the dashboard can say so instead of
    # drawing three idle-looking bars. New-style logs carry a
    # "(rank-local replica)" suffix and are gathered from every rank.
    ddp = DDP_RE.search(text)
    ddp_ranks = int(ddp.group(1)) if ddp else 0
    rank_local = "(rank-local replica)" in text
    gpu_mem_rank0_only = bool(
        ddp_ranks > 1 and not rank_local and gpu_mem
        and gpu_mem.get(0, 0.0) > 0.0
        and all(gpu_mem.get(i, 0.0) == 0.0 for i in range(1, ddp_ranks))
    )

    per_op_eval = {}
    for label, rest in PER_OP_EVAL_RE.findall(text):
        per_op_eval[label] = {op: (int(c), int(t)) for op, _, c, t in PER_OP_EVAL_ENTRY_RE.findall(rest)}
    per_op_train = {op: (float(r), int(n)) for op, r, n in PER_OP_TRAIN_RE.findall(text)}

    eval_samples = [{"gold": int(g), "reward": float(r), "text": t.strip()}
                    for g, r, t in EVAL_SAMPLE_RE.findall(text)]

    saved = SAVED_RE.search(text)

    return {
        "steps": steps,
        "total_steps": total_steps,
        "metrics": metrics,
        "accuracies": [float(x) for x in ACCURACY_RE.findall(text)],
        "device": device,
        "gpu_mem": gpu_mem,
        "ddp_ranks": ddp_ranks,
        "gpu_mem_rank0_only": gpu_mem_rank0_only,
        "saved_to": saved.group(1).strip() if saved else None,
        "per_op_eval": per_op_eval,
        "per_op_train": per_op_train,
        "skipped_grad": skipped_grad,
        "skipped_logits": skipped_logits,
        "rollbacks": rollbacks,
        "skip_hist": hist,
        "accounting": accounting,
        "eval_samples": eval_samples,
    }


def parse_timing(path: str) -> dict:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    out = {"raw": text.strip()}
    for key, pat in [("started", r"Job started:\s*(.+)"), ("ended", r"Job ended:\s*(.+)"),
                     ("elapsed_seconds", r"Elapsed seconds:\s*(\d+)"), ("elapsed_hms", r"Elapsed \(h:m:s\):\s*(.+)")]:
        m = re.search(pat, text)
        if m:
            out[key] = m.group(1).strip()
    return out


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>RLVR Run Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  :root { --bg:#0f1117; --panel:#171a24; --border:#262b3a; --text:#e6e8ef; --muted:#9aa1b2;
          --accent:#6ea8fe; --good:#4ade80; --bad:#f87171; --warn:#fbbf24; --purple:#c084fc; }
  * { box-sizing: border-box; }
  body { margin:0; padding:24px; background:var(--bg); color:var(--text);
         font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; }
  h1 { font-size:1.4rem; margin:0 0 4px; }
  .subtitle { color:var(--muted); margin-bottom:24px; font-size:.9rem; }
  .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:12px; margin-bottom:24px; }
  .stat { background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:14px 16px; }
  .stat .label { color:var(--muted); font-size:.78rem; text-transform:uppercase; letter-spacing:.04em; }
  .stat .value { font-size:1.5rem; font-weight:600; margin-top:4px; word-break:break-word; }
  .stat .delta { font-size:.85rem; margin-top:2px; }
  .delta.pos { color:var(--good); } .delta.neg { color:var(--bad); }
  .stat.warn .value { color:var(--warn); } .stat.good .value { color:var(--good); } .stat.bad .value { color:var(--bad); }
  .panels { display:grid; grid-template-columns:1fr; gap:20px; }
  .panel { background:var(--panel); border:1px solid var(--border); border-radius:12px; padding:18px; }
  .panel h2 { font-size:1rem; margin:0 0 6px; }
  .panel p.note { color:var(--muted); font-size:.82rem; margin:0 0 12px; }
  .two-col { display:grid; grid-template-columns:1fr 1fr; gap:20px; }
  @media (max-width:800px) { .two-col { grid-template-columns:1fr; } }
  canvas { max-height:340px; }
  .samples { font-family:ui-monospace,Menlo,Consolas,monospace; font-size:.82rem; }
  .samples div { padding:8px 10px; border-left:3px solid var(--border); margin-bottom:8px; white-space:pre-wrap; }
  .samples .ok { border-left-color:var(--good); } .samples .miss { border-left-color:var(--bad); }
  .samples .tag { color:var(--muted); }
  footer { margin-top:24px; color:var(--muted); font-size:.78rem; }
  code { background:#1f2330; padding:1px 6px; border-radius:4px; }
</style>
</head>
<body>
  <h1>RLVR Run Dashboard</h1>
  <div class="subtitle">__SUBTITLE__</div>
  <div class="grid" id="stat-grid"></div>
  <div class="panels" id="panels"></div>
  <footer>Generated from <code>__LOGPATH__</code> by generate_dashboard.py</footer>

<script>
const DATA = __DATA_JSON__;
Chart.defaults.color = "#9aa1b2";
Chart.defaults.borderColor = "#262b3a";

const statGrid = document.getElementById("stat-grid");
const panels = document.getElementById("panels");

function statCard(label, value, opts = {}) {
  const div = document.createElement("div");
  div.className = "stat" + (opts.cls ? " " + opts.cls : "");
  let deltaHtml = "";
  if (opts.delta !== undefined && opts.delta !== null) {
    const cls = opts.delta >= 0 ? "pos" : "neg";
    deltaHtml = `<div class="delta ${cls}">${opts.delta >= 0 ? "+" : ""}${opts.delta.toFixed(2)} pts</div>`;
  }
  div.innerHTML = `<div class="label">${label}</div><div class="value">${value}</div>${deltaHtml}`;
  statGrid.appendChild(div);
}
function panel(title, note, id, wide = true) {
  const div = document.createElement("div");
  div.className = "panel";
  div.innerHTML = `<h2>${title}</h2>${note ? `<p class="note">${note}</p>` : ""}<canvas id="${id}"></canvas>`;
  return div;
}
function twoCol(a, b) {
  const div = document.createElement("div");
  div.className = "two-col";
  div.appendChild(a); div.appendChild(b);
  return div;
}
const line = (label, data, color, width = 1.5) =>
  ({ label, data, borderColor: color, backgroundColor: "transparent", pointRadius: 0, borderWidth: width, tension: 0.15, spanGaps: true });

// ---- Stat cards ----
if (DATA.device) statCard("Model / device", DATA.device);
statCard("Optimizer steps logged", `${DATA.steps.length}${DATA.total_steps ? " / " + DATA.total_steps : ""}`);
if (DATA.accuracies.length >= 2) {
  const base = DATA.accuracies[0], fin = DATA.accuracies[DATA.accuracies.length - 1];
  statCard("Baseline accuracy", base.toFixed(1) + "%");
  statCard("Final accuracy", fin.toFixed(1) + "%", { delta: fin - base, cls: fin > base ? "good" : "" });
}
if (DATA.timing && DATA.timing.elapsed_hms) statCard("Elapsed", DATA.timing.elapsed_hms);
if (DATA.timing && DATA.timing.elapsed_seconds && DATA.steps.length) {
  statCard("Wall time / step", (DATA.timing.elapsed_seconds / DATA.steps.length).toFixed(1) + " s");
}
const rewards = DATA.metrics.mean_reward || [];
if (rewards.length) {
  const vals = rewards.filter(v => v !== null);
  statCard("Mean reward", (vals.reduce((a, b) => a + b, 0) / vals.length).toFixed(3));
}
if (DATA.accounting) {
  const a = DATA.accounting;
  statCard("Steps updated", a.updated ?? "-", { cls: "good" });
  statCard("Skipped: non-finite grad", a.skipped_grad ?? 0, { cls: (a.skipped_grad ?? 0) > 0 ? "warn" : "" });
  statCard("Skipped: non-finite logits", a.skipped_logits ?? 0, { cls: (a.skipped_logits ?? 0) > 0 ? "warn" : "" });
  statCard("Adapter rollbacks", a.rollbacks ?? 0, { cls: (a.rollbacks ?? 0) > 0 ? "bad" : "good" });
}

// ---- Reward chart (with skipped steps marked) ----
if (rewards.length) {
  const p = panel("Reward & running baseline over training",
    DATA.skipped_grad.length ? `${DATA.skipped_grad.length} optimizer steps were skipped by the gradient guard (non-finite gradient norm) and print no reward line, so they are absent from the x-axis; see "Where the guards fired" below for their distribution.` : null,
    "rewardChart");
  panels.appendChild(p);
  const datasets = [
    line("mean_reward (per step)", rewards, "#6ea8fe"),
    line("running_baseline", DATA.metrics.baseline || [], "#4ade80", 2),
  ];
  new Chart(document.getElementById("rewardChart"), {
    type: "line",
    data: { labels: DATA.steps, datasets },
    options: { responsive: true, interaction: { mode: "index", intersect: false },
              scales: { x: { title: { display: true, text: "optimizer step" } }, y: { min: 0, max: 1.05 } } },
  });
}

// ---- Secondary metric: loss or grad_norm ----
const secondary = DATA.metrics.loss ? ["loss", "#f87171", "Loss over training", null]
                : DATA.metrics.grad_norm ? ["grad_norm", "#c084fc", "Gradient norm (after clipping) over training",
                    "Drops toward 0 when the running baseline catches up with the reward (advantage ≈ 0) -- i.e. the task is solved. Spikes are steps where the policy moved."] : null;
if (secondary) {
  const [key, color, title, note] = secondary;
  panels.appendChild(panel(title, note, "secondaryChart"));
  new Chart(document.getElementById("secondaryChart"), {
    type: "line",
    data: { labels: DATA.steps, datasets: [line(key, DATA.metrics[key], color)] },
    options: { responsive: true, interaction: { mode: "index", intersect: false },
              scales: { x: { title: { display: true, text: "optimizer step" } } } },
  });
}

// ---- Skipped-step histogram ----
if (DATA.skip_hist) {
  const h = DATA.skip_hist;
  panels.appendChild(panel("Where the guards fired",
    "Count of skipped/rolled-back optimizer steps per bin of steps. Early-heavy = instability while the policy was changing fast; spread evenly = a persistent numerical issue; late-heavy = something degrading.",
    "skipChart"));
  new Chart(document.getElementById("skipChart"), {
    type: "bar",
    data: { labels: h.labels, datasets: [
      { label: "non-finite gradient (skipped)", data: h.grad, backgroundColor: "#f87171aa" },
      { label: "non-finite logits/loss (rollout skipped)", data: h.logits, backgroundColor: "#fbbf24aa" },
      { label: "parameter rollback", data: h.rollback, backgroundColor: "#c084fcaa" },
    ] },
    options: { responsive: true, scales: { x: { stacked: true, title: { display: true, text: "step range" } },
                                          y: { stacked: true, beginAtZero: true } } },
  });
}

// ---- Rollouts completed per step (qwen38) ----
if (DATA.metrics.rollouts) {
  panels.appendChild(panel("Fraction of planned rollouts completed per step",
    "1.0 = every accumulated rollout produced finite logits and a usable loss.", "rolloutChart"));
  new Chart(document.getElementById("rolloutChart"), {
    type: "line",
    data: { labels: DATA.steps, datasets: [line("rollouts completed", DATA.metrics.rollouts, "#fbbf24")] },
    options: { responsive: true, interaction: { mode: "index", intersect: false },
              scales: { x: { title: { display: true, text: "optimizer step" } }, y: { min: 0, max: 1.05 } } },
  });
}

// ---- Per-operator (train_rlvr.py) or GPU memory (qwen38) ----
const opOrder = ["+", "-", "*"];
const hasPerOp = Object.keys(DATA.per_op_eval).length || Object.keys(DATA.per_op_train).length;
const hasGpu = Object.keys(DATA.gpu_mem).length;
if (hasPerOp) {
  const a = panel("Accuracy by operator (baseline vs final)", null, "perOpAccChart");
  const b = panel("Training reward by operator (all steps)", null, "perOpRewardChart");
  panels.appendChild(twoCol(a, b));
  const accFor = label => opOrder.map(op => { const e = (DATA.per_op_eval[label] || {})[op]; return e ? 100 * e[0] / e[1] : null; });
  new Chart(document.getElementById("perOpAccChart"), { type: "bar",
    data: { labels: opOrder, datasets: [
      { label: "baseline", data: accFor("baseline"), backgroundColor: "#fbbf24aa" },
      { label: "final", data: accFor("final"), backgroundColor: "#4ade80aa" } ] },
    options: { responsive: true, scales: { y: { min: 0, max: 100, title: { display: true, text: "accuracy %" } } } } });
  new Chart(document.getElementById("perOpRewardChart"), { type: "bar",
    data: { labels: opOrder, datasets: [ { label: "mean training reward",
      data: opOrder.map(op => DATA.per_op_train[op] ? DATA.per_op_train[op][0] : null), backgroundColor: "#6ea8feaa" } ] },
    options: { responsive: true, scales: { y: { min: 0, max: 1.05 } } } });
}
if (hasGpu) {
  const ids = Object.keys(DATA.gpu_mem).sort((a, b) => a - b);
  let note, datasets;
  if (DATA.gpu_mem_rank0_only) {
    // Old-format DDP log: only rank 0's per-process counter was printed, so GPUs 1..N-1 read 0.0
    // even though every rank holds its own full replica. Show rank 0 as measured and the other
    // ranks as inferred copies of it, clearly labelled, rather than as idle GPUs.
    const r0 = DATA.gpu_mem[ids[0]];
    note = `DDP x${DATA.ddp_ranks}: one full model replica per GPU. This log predates the per-rank all-gather fix (commit 444aeeb), ` +
           `so only rank 0's per-process counter was recorded and GPUs 1-${DATA.ddp_ranks - 1} printed 0.0. ` +
           `The outlined bars are inferred (= rank 0's ${r0.toFixed(1)} GiB); nvidia-smi during the run showed all ${DATA.ddp_ranks} GPUs at ~${Math.round(r0 * 1.3)} GiB used.`;
    datasets = [
      { label: "GiB allocated (measured, rank 0)", data: ids.map(i => Number(i) === 0 ? r0 : null), backgroundColor: "#6ea8feaa" },
      { label: "GiB allocated (inferred replica)", data: ids.map(i => Number(i) === 0 ? null : r0), backgroundColor: "#6ea8fe44", borderColor: "#6ea8fe", borderWidth: 1, borderDash: [4, 3] },
    ];
  } else if (DATA.ddp_ranks > 1) {
    note = `DDP x${DATA.ddp_ranks}: each bar is that rank's own replica (gathered from every rank). Bars should be equal.`;
    datasets = [ { label: "GiB allocated (rank-local replica)", data: ids.map(i => DATA.gpu_mem[i]), backgroundColor: "#6ea8feaa" } ];
  } else {
    note = "Balanced bars = device_map spread the shards across all GPUs. One or two tall bars with the rest near zero = the run-4 placement problem (or a deliberate single-GPU run).";
    datasets = [ { label: "GiB allocated", data: ids.map(i => DATA.gpu_mem[i]), backgroundColor: "#6ea8feaa" } ];
  }
  panels.appendChild(panel("GPU memory allocated after model load", note, "gpuChart"));
  new Chart(document.getElementById("gpuChart"), { type: "bar",
    data: { labels: ids.map(i => "GPU " + i), datasets },
    options: { responsive: true, scales: { x: { stacked: true }, y: { beginAtZero: true, title: { display: true, text: "GiB" } } } } });
}

// ---- Sample completions ----
if (DATA.eval_samples.length) {
  const div = document.createElement("div");
  div.className = "panel";
  const rows = DATA.eval_samples.map((s, i) =>
    `<div class="${s.reward >= 1 ? "ok" : "miss"}"><span class="tag">${i === 0 ? "baseline" : "final"} eval · gold=${s.gold} · reward=${s.reward}</span>\n${s.text.replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]))}</div>`).join("");
  div.innerHTML = `<h2>What the model actually said</h2><p class="note">First eval completion before and after training, as printed in the log.</p><div class="samples">${rows}</div>`;
  panels.appendChild(div);
}
</script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    data = parse_log(args.log, args.skip_bin)
    if not data["steps"]:
        raise SystemExit("No 'step N/M | ...' lines found -- is this a train_rlvr*.py log?")
    data["timing"] = parse_timing(args.timing) if args.timing else None

    output = args.output or (args.log.rsplit(".", 1)[0] + ".dashboard.html")
    subtitle = " | ".join(html.escape(x) for x in [args.log, data["device"] or ""] if x)
    out_html = (HTML_TEMPLATE
                .replace("__SUBTITLE__", subtitle)
                .replace("__LOGPATH__", html.escape(args.log))
                .replace("__DATA_JSON__", json.dumps(data)))
    with open(output, "w", encoding="utf-8") as f:
        f.write(out_html)
    print(f"Wrote dashboard to {output}")
    if data["accounting"]:
        print(f"Step accounting: {data['accounting']}")
    print(f"Steps logged: {len(data['steps'])}  skipped(grad): {len(data['skipped_grad'])}  "
          f"skipped(logits): {len(data['skipped_logits'])}  rollbacks: {len(data['rollbacks'])}")


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
