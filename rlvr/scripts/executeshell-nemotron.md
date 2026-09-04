# Hecate execute shell — Nemotron-3.5-Lightning-30B-A3B RLVR (LoRA, BF16)

RLVR fine-tuning of NVIDIA's
[Nemotron-3.5-Lightning-30B-A3B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4)
(30B total / 3B active; Mamba-2 + MoE + attention hybrid, architecture
`nemotron_h`) on the same arithmetic task as the other guides, using
[`train_rlvr_nemotron.py`](../train_rlvr_nemotron.py) -- an adaptation of
the Qwen3.8 script that keeps every stability guard learned there.

**Use the BF16 checkpoint for training** (`...-30B-A3B-BF16`, ~60GB). The
NVFP4 sibling is a ModelOpt-quantised *inference* artifact (vLLM/SGLang,
one H100 / DGX Spark) and cannot be LoRA-trained with peft.

What is different from the Qwen3.8 flow, and why:
- Text-only causal LM: `AutoTokenizer` + `AutoModelForCausalLM`, no
  multimodal processor.
- 60GB fits on **one** ~280GB Vera Rubin GPU, so the default is a single
  process with the whole model on one GPU -- no cross-GPU pipeline hops,
  so faster per step than Qwen3.8's `device_map="auto"` sharding.
- Because it also fits *per GPU*, `RLVR_NPROC=4` runs data-parallel via
  `torchrun` (one replica per GPU, like the 1.5B run) with the guards
  synchronised across ranks. **Untested** -- complete a single-process
  run first.
- Mamba-2 layers are recurrent: `--batch-size` stays 1 (pad tokens
  entering a recurrence is what crashed Qwen3.8 runs 1-3); the batch
  comes from `--grad-accum`.
- Reasoning model: the chat template is rendered with thinking off. The
  rendered prompt is printed once -- if it still contains a think block,
  this template toggles reasoning via the system prompt instead; set
  `RLVR_SYSTEM_PROMPT="/no_think"` (or whatever the model card specifies).

Verified locally on the identical code path with Qwen2.5-1.5B-Instruct
(3 steps x 2 rollouts: 0% -> 75%, all guards clean, adapter saved). Not
yet run against the Nemotron checkpoint on hecate.

## 1. Download the model weights

Public (OpenMDW-1.1), no token needed. Download **without `--local-dir`**
so the weights land in the Hub cache layout (`hf_cache/hub/`) that
`from_pretrained` resolves repo ids against -- the Qwen3.8 `--local-dir`
download ended up duplicated (336GB) because the training job could not
see it and re-fetched.

```bash
export LUSTRE_DIR="${LUSTRE_DIR:-/lustre/fsw/general_sa/$USER}"
source "$LUSTRE_DIR/.venv-upload/bin/activate"     # created in executeshell-qwen38.md step 1
export HF_HOME="$LUSTRE_DIR/hf_cache"
mkdir -p "$LUSTRE_DIR/out"

# BF16 training checkpoint (~60GB) -- the one train_rlvr_nemotron.py uses
hf download nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 2>&1 | tee "$LUSTRE_DIR/out/nemotron35_bf16_download.log"

# Optional: NVFP4 inference checkpoint (~15-20GB), for serving with vLLM/SGLang only
# hf download nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 2>&1 | tee "$LUSTRE_DIR/out/nemotron35_nvfp4_download.log"

deactivate
ls -1 "$LUSTRE_DIR/hf_cache/hub/" | sed 's/^models--//; s/--/\//'
du -sh "$LUSTRE_DIR"/hf_cache/hub/models--nvidia--* 2>/dev/null
```

Minutes, not hours, at this size; `tmux new -s dl` if your session drops.

## 2. Write the RLVR code to Lustre

`task.py` is reused unchanged and must already be at `$LUSTRE_DIR/task.py`
(from the multi-GPU guide's Block 1).

```bash
export LUSTRE_DIR="${LUSTRE_DIR:-/lustre/fsw/general_sa/$USER}"
mkdir -p "$LUSTRE_DIR/scripts" "$LUSTRE_DIR/out"

cat > "$LUSTRE_DIR/train_rlvr_nemotron.py" << 'PYEOF'
"""RLVR (LoRA) fine-tuning for NVIDIA Nemotron-3.5-Lightning-30B-A3B (BF16).

Adapted from train_rlvr_qwen38.py, carrying over everything the four
Qwen3.8-Flash-Next runs on hecate taught us, and simplified where this
model's shape allows:

  - Text-only causal LM (AutoTokenizer + AutoModelForCausalLM), not a
    multimodal processor. Architecture `nemotron_h`: Mamba-2 + MoE +
    select attention layers, 30B total / 3B active.
  - ~60GB in bf16, so it FITS ON ONE GPU (hecate's Vera Rubin GPUs have
    ~280GB). Default: single process, whole model on one GPU -- no
    cross-GPU pipeline hops, so faster than the Qwen3.8 device_map setup.
    `--device-map auto` is available if a smaller GPU ever needs sharding.
  - Because it also fits per GPU, optional data-parallel training via
    torchrun (WORLD_SIZE>1 auto-detected, same pattern as train_rlvr.py):
    each rank runs its own single-prompt rollouts and DDP all-reduces the
    LoRA gradients. The stability guards below are synchronised across
    ranks with an all-reduce so no rank skips a step alone (which would
    deadlock DDP). UNTESTED on multi-GPU as of writing -- start with the
    single-process default.
  - Mamba-2 layers are recurrent: left-padding pad tokens into the
    recurrence is exactly what crashed Qwen3.8 runs 1-3 (its Gated
    DeltaNet layers). --batch-size stays 1; --grad-accum gives the batch.
  - Reasoning model: the chat template is rendered with the thinking
    switch OFF (--thinking-kwarg, default `enable_thinking`; set
    --system-prompt "/no_think" if this template uses that convention
    instead -- the rendered prompt is printed once so you can check).
  - The NVFP4 sibling checkpoint is an inference artifact (ModelOpt) and
    cannot be LoRA-trained with peft; use the BF16 one (the default).

Guards (from the Qwen3.8 experience):
  - finite-logits check on the prompt BEFORE generate() (avoids the
    unrecoverable CUDA assert inside sampling);
  - gradient guard: non-finite clip norm -> grads zeroed, step skipped
    before optimizer.step();
  - parameter rollback: LoRA weights snapshotted each good step; if any
    parameter is non-finite after a step, restore + rebuild optimizer;
  - the adapter is only saved when finite; a `Step accounting` line
    summarises updated / skipped / rollback counts.

Usage:
    python train_rlvr_nemotron.py --iterations 10
    torchrun --standalone --nproc_per_node=4 train_rlvr_nemotron.py --iterations 1000

Log lines are the same shape as train_rlvr_qwen38.py's, so analyze_run.py
and generate_dashboard.py work on the output unchanged.
"""

import argparse
import os
import random

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from task import sample_batch, verify_reward

try:
    from peft import LoraConfig, get_peft_model
except ImportError as e:
    raise SystemExit("peft is required: pip install -r requirements-nemotron.txt") from e

try:
    from accelerate import init_empty_weights
except ImportError as e:
    raise SystemExit("accelerate is required: pip install -r requirements-nemotron.txt") from e


DEFAULT_MODEL = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RLVR LoRA fine-tuning for Nemotron-3.5-Lightning (BF16)")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--iterations", type=int, default=10, help="Optimizer steps.")
    p.add_argument(
        "--batch-size", type=int, default=1,
        help="Prompts per rollout. KEEP AT 1: batch>1 needs left-padding, and pad tokens "
             "entering recurrent (Mamba-2) layers is what crashed the Qwen3.8 runs. Use "
             "--grad-accum for a larger effective batch.",
    )
    p.add_argument("--grad-accum", type=int, default=4, help="Single-prompt rollouts per optimizer step.")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--enable-thinking", action="store_true", help="Leave the model's thinking mode ON.")
    p.add_argument(
        "--thinking-kwarg", default="enable_thinking",
        help="Chat-template variable that toggles thinking. Passed as {kwarg: --enable-thinking}. "
             "Unknown variables are ignored by the template, so a wrong name is harmless but "
             "ineffective -- check the printed rendered prompt for a think block.",
    )
    p.add_argument(
        "--system-prompt", default=None,
        help="Optional system message. Some Nemotron templates control reasoning via the "
             "system prompt (e.g. '/no_think' or 'detailed thinking off') rather than a kwarg.",
    )
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--temperature", type=float, default=1.0, help="Model card recommends 1.0.")
    p.add_argument("--top-p", type=float, default=0.95, help="Model card recommends 0.95.")
    p.add_argument("--top-k", type=int, default=0, help="0 = disabled (model card gives no top_k).")
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-dir", default=None, help="Dir to save the LoRA adapter (not the base model).")
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--eval-n", type=int, default=10)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj",
        help="Comma-separated module-name suffixes for LoRA. Attention projections by default. "
             "This architecture has only 'select' attention layers, so for more adapter capacity "
             "consider adding the Mamba-2 projections (typically in_proj,out_proj) -- run "
             "--list-modules to see the real names first.",
    )
    p.add_argument(
        "--device-map", choices=["single", "auto"], default="single",
        help="'single': whole model on one GPU (bf16 30B ~60GB; fits on an 80GB+ GPU). "
             "'auto': shard across GPUs with --max-memory-per-gpu. Ignored under torchrun.",
    )
    p.add_argument("--max-memory-per-gpu", default="60GiB", help="Only for --device-map auto.")
    p.add_argument("--list-modules", action="store_true",
                   help="Build the model on the meta device (seconds), print module-name patterns, exit.")
    p.add_argument("--print-completion-chars", type=int, default=200)
    return p.parse_args()


# --------------------------------------------------------------------------- helpers

def build_messages(question: str, system_prompt):
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": question})
    return msgs


def build_prompt(tokenizer, problem: dict, args) -> str:
    kwargs = {args.thinking_kwarg: args.enable_thinking} if args.thinking_kwarg else {}
    return tokenizer.apply_chat_template(
        build_messages(problem["question"], args.system_prompt),
        tokenize=False,
        add_generation_prompt=True,
        **kwargs,
    )


def _meta_model(model_name):
    config = AutoConfig.from_pretrained(model_name)
    with init_empty_weights():
        return AutoModelForCausalLM.from_config(config)


def list_modules(model_name: str) -> None:
    import re
    model = _meta_model(model_name)
    seen = {}
    for name, module in model.named_modules():
        if not name:
            continue
        pattern = re.sub(r"\.\d+(\.|$)", r".N\1", name)
        key = (type(module).__name__, pattern)
        if key not in seen:
            extra = f"  in={module.in_features} out={module.out_features}" if isinstance(module, torch.nn.Linear) else ""
            seen[key] = extra
    print(f"{len(seen)} distinct module patterns in {model_name}:\n")
    for (cls, pattern), extra in sorted(seen.items(), key=lambda kv: kv[0][1]):
        print(f"  {cls:40s} {pattern}{extra}")


def _show(label: str, text: str, limit: int) -> None:
    snippet = text if len(text) <= limit else text[:limit] + "..."
    print(f"    {label}: {snippet!r}")


@torch.no_grad()
def _logits_finite(model, enc) -> bool:
    return bool(torch.isfinite(model(**enc).logits).all())


def _snapshot(params):
    return [p.detach().clone() for p in params]


def _params_finite(params) -> bool:
    return all(bool(torch.isfinite(p).all()) for p in params)


@torch.no_grad()
def _restore(params, snapshot):
    for p, s in zip(params, snapshot):
        p.copy_(s)


def _any_rank(flag: bool, distributed: bool, device) -> bool:
    """True if `flag` is True on ANY rank. Keeps guard decisions identical across ranks."""
    if not distributed:
        return flag
    t = torch.tensor([1.0 if flag else 0.0], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return bool(t.item() > 0)


@torch.no_grad()
def _greedy_eval(gen_model, tokenizer, device, rng, args, n):
    problems = sample_batch(rng, n)
    correct = 0
    nonfinite = 0
    for i, prob in enumerate(problems):
        enc = tokenizer(build_prompt(tokenizer, prob, args), return_tensors="pt").to(device)
        if not _logits_finite(gen_model, enc):
            nonfinite += 1
            print(f"  eval problem {i}: NON-FINITE logits for prompt {prob['question']!r} -- skipping")
            continue
        out = gen_model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 use_cache=True, pad_token_id=tokenizer.pad_token_id)
        completion = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        reward = verify_reward(completion, prob["answer"])
        if i == 0:
            _show(f"eval sample (gold={prob['answer']}, reward={reward})", completion, args.print_completion_chars)
        if reward == 1.0:
            correct += 1
    if nonfinite:
        print(f"  {nonfinite}/{n} eval prompts produced non-finite logits")
    return correct / n


# --------------------------------------------------------------------------- main

def main() -> None:
    args = parse_args()
    if args.list_modules:
        list_modules(args.model)
        return

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    distributed = world_size > 1
    is_main = rank == 0

    rng = random.Random(args.seed + rank)  # ranks see different problems
    torch.manual_seed(args.seed + rank)

    if not torch.cuda.is_available():
        raise SystemExit("This script requires CUDA GPUs.")

    if distributed:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        load_kwargs = {}
    elif args.device_map == "auto":
        device = None  # decided by accelerate
        n_gpu = torch.cuda.device_count()
        load_kwargs = {"device_map": "auto", "max_memory": {i: args.max_memory_per_gpu for i in range(n_gpu)}}
    else:
        device = torch.device("cuda", 0)
        load_kwargs = {}

    if is_main:
        mode = f"DDP x{world_size} ranks" if distributed else f"single process, device_map={args.device_map}"
        print(f"Loading {args.model} in bf16 ({mode})...")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, **load_kwargs)
    if device is not None and not load_kwargs:
        model = model.to(device)
    if device is None:
        device = model.device
    model.config.use_cache = False  # training forward; generate() passes use_cache=True

    if is_main:
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.memory_allocated(i) / 2**30:.1f} GiB allocated after load")

    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha,
        target_modules=[m.strip() for m in args.lora_target_modules.split(",")],
        lora_dropout=0.0,  # sampling and training forward must see the same network
        bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    if is_main:
        model.print_trainable_parameters()
    model.eval()  # gradients still flow; this only disables dropout-like layers

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    gen_model = model  # generate()/save_pretrained live on the peft model
    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank])

    def make_optimizer():
        return torch.optim.AdamW(trainable_params, lr=args.lr)

    optimizer = make_optimizer()
    last_good = _snapshot(trainable_params)

    if is_main:
        example = sample_batch(random.Random(args.seed + 12345), 1)[0]
        print(f"\nRendered prompt for one example problem ({args.thinking_kwarg}={args.enable_thinking}, "
              f"system_prompt={args.system_prompt!r}):\n{build_prompt(tokenizer, example, args)!r}\n")
        print("Baseline accuracy before training:")
        baseline_acc = _greedy_eval(gen_model, tokenizer, device, rng, args, n=args.eval_n)
        print(f"  greedy accuracy = {baseline_acc:.2%}")
        print(f"\nTraining: {args.iterations} optimizer steps x {args.grad_accum} rollout(s) of "
              f"{args.batch_size} prompt(s) each" + (f" x {world_size} ranks" if distributed else "") + "\n")
    if distributed:
        dist.barrier()

    running_baseline = 0.0
    stats = {"updated": 0, "skipped_logits": 0, "skipped_grad": 0, "rollbacks": 0}

    for step in range(1, args.iterations + 1):
        optimizer.zero_grad()
        step_rewards, losses = [], []
        first_completion = first_problem = None

        for _ in range(args.grad_accum):
            problems = sample_batch(rng, args.batch_size)
            prompts = [build_prompt(tokenizer, p, args) for p in problems]
            enc = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
            prompt_len = enc["input_ids"].shape[1]

            if not _logits_finite(gen_model, enc):
                stats["skipped_logits"] += 1
                print(f"step {step:3d}: NON-FINITE logits for {[p['question'] for p in problems]} -- skipping this rollout.")
                continue

            with torch.no_grad():
                gen_out = gen_model.generate(
                    **enc, max_new_tokens=args.max_new_tokens, do_sample=True,
                    temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                    use_cache=True, pad_token_id=tokenizer.pad_token_id,
                )

            rewards = []
            for i, prob in enumerate(problems):
                completion = tokenizer.decode(gen_out[i][prompt_len:], skip_special_tokens=True)
                rewards.append(verify_reward(completion, prob["answer"]))
                if first_completion is None:
                    first_completion, first_problem = completion, prob
            step_rewards.extend(rewards)
            rewards_t = torch.tensor(rewards, dtype=torch.float32, device=gen_out.device)
            advantages = rewards_t - running_baseline

            attention_mask = (gen_out != tokenizer.pad_token_id).long()
            # Forward through the DDP wrapper (if any) so gradient hooks are registered.
            logits = model(input_ids=gen_out, attention_mask=attention_mask).logits
            if not torch.isfinite(logits).all():
                stats["skipped_logits"] += 1
                print(f"step {step:3d}: non-finite logits in the training forward -- skipping this rollout.")
                continue
            log_probs = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
            target_ids = gen_out[:, 1:]
            token_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
            gen_mask = torch.zeros_like(target_ids, dtype=torch.float32)
            gen_mask[:, prompt_len - 1:] = 1.0
            gen_mask *= attention_mask[:, 1:].float()
            seq_log_prob = (token_log_probs * gen_mask).sum(dim=1) / gen_mask.sum(dim=1).clamp(min=1)
            loss = -(advantages.to(seq_log_prob.device) * seq_log_prob).mean()
            if torch.isfinite(loss):
                losses.append(loss)
            else:
                stats["skipped_logits"] += 1
                print(f"step {step:3d}: non-finite loss -- skipping this rollout.")

        # One backward per optimizer step (sum of usable rollouts). Under DDP every rank
        # must call backward exactly once per step or the all-reduce deadlocks, so if ANY
        # rank has nothing usable, ALL ranks skip this step.
        if _any_rank(len(losses) == 0, distributed, device):
            optimizer.zero_grad()
            if is_main:
                print(f"step {step:3d}: no usable rollouts on at least one rank -- no update.")
            continue

        total_loss = torch.stack(losses).sum() / args.grad_accum
        total_loss.backward()

        batch_mean = sum(step_rewards) / len(step_rewards)
        running_baseline = 0.9 * running_baseline + 0.1 * batch_mean

        total_norm = torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
        if _any_rank(not bool(torch.isfinite(total_norm)), distributed, device):
            optimizer.zero_grad()
            stats["skipped_grad"] += 1
            if is_main:
                print(f"step {step:3d}: NON-FINITE gradient norm -- grads zeroed, update skipped, weights untouched.")
            continue

        optimizer.step()

        if _any_rank(not _params_finite(trainable_params), distributed, device):
            stats["rollbacks"] += 1
            _restore(trainable_params, last_good)
            optimizer = make_optimizer()
            if is_main:
                print(f"step {step:3d}: NON-FINITE parameters after optimizer.step() -- rolled back, optimizer reset.")
            continue
        last_good = _snapshot(trainable_params)
        stats["updated"] += 1

        if is_main and step % args.log_every == 0:
            print(f"step {step:3d}/{args.iterations} | mean_reward={batch_mean:.3f} | "
                  f"baseline={running_baseline:.3f} | grad_norm={total_norm.item():.3f} | "
                  f"rollouts={len(losses)}/{args.grad_accum}")
            if first_completion is not None:
                _show(f"sample (gold={first_problem['answer']}, reward={step_rewards[0]})",
                      first_completion, args.print_completion_chars)

    if distributed:
        dist.barrier()

    if is_main:
        print(f"\nStep accounting: {stats}")
        print("\nAccuracy after training:")
        final_acc = _greedy_eval(gen_model, tokenizer, device, rng, args, n=args.eval_n)
        print(f"  greedy accuracy = {final_acc:.2%} (baseline was {baseline_acc:.2%})")
        if args.save_dir:
            if _params_finite(trainable_params):
                gen_model.save_pretrained(args.save_dir)
                tokenizer.save_pretrained(args.save_dir)
                print(f"Saved LoRA adapter to {args.save_dir}")
            else:
                print("Adapter is non-finite -- NOT saving.")

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
PYEOF

cat > "$LUSTRE_DIR/requirements-nemotron.txt" << 'REQEOF'
# For train_rlvr_nemotron.py inside the gitlab-master.nvidia.com/dl/dgx/pytorch:main-py3-devel
# container. The container's torch/CUDA is reused (not reinstalled). No bitsandbytes: the
# BF16 checkpoint (~60GB) fits on one ~280GB GPU, so nothing is quantised.
# If loading fails with an unrecognised `nemotron_h` architecture, the container's
# transformers is too old -- `pip install -U transformers` first.
transformers>=4.57
accelerate>=1.0
peft>=0.13
REQEOF

cat > "$LUSTRE_DIR/scripts/launch_rlvr_nemotron.sh" << 'SHEOF'
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

COMMON_ARGS=(
  --model "${RLVR_MODEL:-nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16}"
  --iterations "${RLVR_ITERATIONS:-10}"
  --batch-size "${RLVR_BATCH_SIZE:-1}"
  --grad-accum "${RLVR_GRAD_ACCUM:-4}"
  --max-new-tokens "${RLVR_MAX_NEW_TOKENS:-64}"
  --lr "${RLVR_LR:-1e-5}"
  --eval-n "${RLVR_EVAL_N:-10}"
  --lora-r "${RLVR_LORA_R:-16}"
  --lora-target-modules "${RLVR_LORA_TARGETS:-q_proj,k_proj,v_proj,o_proj}"
  --seed "${RLVR_SEED:-0}"
  --save-dir "$LUSTRE_DIR/out/rlvr-nemotron-run1"
)
# Optional: a system prompt for templates that toggle reasoning that way (e.g. "/no_think").
if [ -n "${RLVR_SYSTEM_PROMPT:-}" ]; then
  COMMON_ARGS+=(--system-prompt "$RLVR_SYSTEM_PROMPT")
fi

NPROC="${RLVR_NPROC:-1}"
if [ "$NPROC" -gt 1 ]; then
  # Data-parallel: one full bf16 replica per GPU (~60GB each). UNTESTED path -- use
  # only after the single-process default has completed a run cleanly.
  echo "$(hostname): Launching RLVR fine-tuning for Nemotron-3.5-Lightning via torchrun x$NPROC (DDP)..."
  torchrun --standalone --nproc_per_node="$NPROC" "$LUSTRE_DIR/train_rlvr_nemotron.py" "${COMMON_ARGS[@]}"
else
  echo "$(hostname): Launching RLVR fine-tuning for Nemotron-3.5-Lightning (single process, one GPU)..."
  python "$LUSTRE_DIR/train_rlvr_nemotron.py" --device-map "${RLVR_DEVICE_MAP:-single}" "${COMMON_ARGS[@]}"
fi
SHEOF
chmod +x "$LUSTRE_DIR/scripts/launch_rlvr_nemotron.sh"

cat > "$LUSTRE_DIR/scripts/submit_hecate_nemotron.sh" << 'SHEOF'
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
SHEOF
chmod +x "$LUSTRE_DIR/scripts/submit_hecate_nemotron.sh"

echo "--- written ---"
ls -la "$LUSTRE_DIR/train_rlvr_nemotron.py" "$LUSTRE_DIR/requirements-nemotron.txt" \
       "$LUSTRE_DIR/scripts/launch_rlvr_nemotron.sh" "$LUSTRE_DIR/scripts/submit_hecate_nemotron.sh"
```

## 3. Check the architecture is loadable (seconds, no GPU)

Builds the model on the meta device from its config. Confirms the
container/venv `transformers` knows `nemotron_h`, and shows the real
module names for `--lora-target-modules` (attention `q_proj/k_proj/...`
vs. Mamba-2 `in_proj/out_proj`).

```bash
source "$LUSTRE_DIR/.venv-upload/bin/activate"
pip install -q -U transformers peft accelerate
HF_HOME="$LUSTRE_DIR/hf_cache" python3 "$LUSTRE_DIR/train_rlvr_nemotron.py" --list-modules 2>&1 | tee "$LUSTRE_DIR/out/nemotron35_modules.txt" | head -60
deactivate
```

An "unrecognized architecture" error here means the same `pip install -U
transformers` is needed inside the job -- it is already the first line
of `launch_rlvr_nemotron.sh`.

## 4. Submit a short run first

```bash
bash "$LUSTRE_DIR/scripts/submit_hecate_nemotron.sh"
tail -f "$LUSTRE_DIR/out/hecate_nemotron_run1.log"
```

Read these, in order, before spending hours:
1. `GPU 0: ~60 GiB allocated after load`, GPUs 1-3 ~0 -- expected for the
   single-process default (one GPU holds the whole model).
2. `Rendered prompt ... enable_thinking=False`: if the text contains a
   think/reasoning block, the template ignores that kwarg. Re-run with
   `RLVR_SYSTEM_PROMPT="/no_think"` (check the model card for the exact
   phrase) and confirm the block is gone.
3. `eval sample ...` -- what the model actually says; baseline accuracy
   should be well above 0%.
4. `step N/10 | ... | grad_norm=... | rollouts=4/4` lines.
5. `Step accounting: {...}` -- `updated` ~10 is healthy; non-zero
   `skipped_grad`/`rollbacks` means the guards are doing their job.
6. `Saved LoRA adapter to .../out/rlvr-nemotron-run1`.

## 5. Scale up

```bash
mv -f "$LUSTRE_DIR/out/hecate_nemotron_run1.log"    "$LUSTRE_DIR/out/hecate_nemotron_10step.log"    2>/dev/null
mv -f "$LUSTRE_DIR/out/hecate_nemotron_run1.timing" "$LUSTRE_DIR/out/hecate_nemotron_10step.timing" 2>/dev/null

# single process, 1000 steps x 4 rollouts
RLVR_ITERATIONS=1000 RLVR_GRAD_ACCUM=4 RLVR_EVAL_N=20 bash "$LUSTRE_DIR/scripts/submit_hecate_nemotron.sh"

# OR data-parallel on all 4 GPUs (untested): 4 replicas x 4 rollouts = 16 samples/step
# RLVR_NPROC=4 RLVR_ITERATIONS=1000 RLVR_GRAD_ACCUM=4 RLVR_EVAL_N=20 bash "$LUSTRE_DIR/scripts/submit_hecate_nemotron.sh"
```

A 3B-active MoE on one GPU should be several times faster per step than
Qwen3.8 was; check the 10-step timing file before assuming 1000 steps
fits in the 5h `--time`. Adapter capacity: attention layers are
"select" in this architecture, so if reward plateaus early, try
`RLVR_LORA_TARGETS="q_proj,k_proj,v_proj,o_proj,in_proj,out_proj"` (names
from step 3) to also adapt the Mamba-2 blocks.

## 6. Insights and dashboard

Same tools and commands as [`executeshell-qwen38.md`](executeshell-qwen38.md)
sections 5-6, with `hecate_nemotron_run1` in place of `hecate_qwen38_run1`:

```bash
export LUSTRE_DIR="${LUSTRE_DIR:-/lustre/fsw/general_sa/$USER}"
LOG="$LUSTRE_DIR/out/hecate_nemotron_run1.log"; TIMING="$LUSTRE_DIR/out/hecate_nemotron_run1.timing"
python3 "$LUSTRE_DIR/analyze_run.py" "$LOG" --timing "$TIMING"
python3 "$LUSTRE_DIR/generate_dashboard.py" "$LOG" --timing "$TIMING" --output "$LUSTRE_DIR/out/hecate_nemotron_run1_dashboard.html"
```

## Known risks

- **Untested against the Nemotron checkpoint.** The code path was
  verified end-to-end locally with Qwen2.5-1.5B-Instruct only.
- **Thinking toggle name.** If `enable_thinking` is not this template's
  variable, thinking stays on and a 64-token budget yields the 0%
  baseline / 0.1-reward pattern seen on Qwen3.8 run 1-3. The printed
  prompt makes this obvious; the fix is `RLVR_SYSTEM_PROMPT`.
- **Mamba-2 + padding.** Never raise `RLVR_BATCH_SIZE` above 1.
- **DDP path (`RLVR_NPROC>1`)** synchronises the guards with an
  all-reduce so ranks never diverge, but has not been exercised on
  multi-GPU hardware.
- **`nemotron_h` support** requires a recent `transformers`; the launcher
  upgrades it inside the container.
