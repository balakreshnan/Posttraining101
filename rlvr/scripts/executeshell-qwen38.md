# Hecate execute shell — Qwen3.8-Flash-Next RLVR (EXPERIMENTAL)

**Experimental -- four runs on hecate so far, each one taught us
something; see "Known risks" at the bottom for the full story.**
Qwen3.8-Flash-Next is a ~180B param multimodal MoE model (model card:
[huggingface.co/Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)) --
fundamentally different from the validated [`executeshell-multigpu.md`](executeshell-multigpu.md)
/ [`executeshell-singlegpu.md`](executeshell-singlegpu.md) flows, which
use Qwen2.5-1.5B-Instruct with full-replica DDP training. At ~360GB in
BF16 it cannot be replicated per GPU -- instead the frozen base is
sharded across all visible GPUs with `device_map="auto"` (balanced via
`max_memory` so every GPU holds a slice; hecate's Vera Rubin nodes have
~280GB per GPU, so it fits in plain bf16 -- no quantisation) and only a
small LoRA adapter on the attention projections is trained. The
512-expert MoE layers stay frozen.

Read [train_rlvr_qwen38.py](../train_rlvr_qwen38.py)'s docstring for the
rationale behind every default. The important ones: `--batch-size 1`
(left-padding into the recurrent layers caused a hard crash -- use
`--grad-accum` for a real batch), thinking mode off, gradient and
parameter guards that skip/roll back a bad step instead of poisoning the
adapter. Start with the defaults in `launch_rlvr_qwen38.sh` (10 steps x
4 accumulated rollouts) and read the printed completions before scaling.

## 1. Download the model weights

~360GB. Check disk space first, then download in the background since
it will take a while:

```bash
export ACCOUNT="${ACCOUNT:-general_sa}"
export LUSTRE_DIR="${LUSTRE_DIR:-/lustre/fsw/$ACCOUNT/$USER}"

df -h /lustre/fsw/general_sa
```

Set up a small venv for the `huggingface_hub` CLI (the login node's
system Python is externally managed, PEP 668 -- same reason we use a
venv for the Hugging Face upload step in the other guides):

```bash
python3 -m venv "$LUSTRE_DIR/.venv-upload"
ls -la "$LUSTRE_DIR/.venv-upload/bin/"   # confirm 'activate' exists before continuing

source "$LUSTRE_DIR/.venv-upload/bin/activate"
which python pip

pip install --quiet -U huggingface_hub
pip show huggingface_hub 2>&1 | head -5
which hf huggingface-cli 2>&1
```

If any of those checks come back empty/missing, stop and re-run this
block -- a failed `python3 -m venv` (e.g. because `$LUSTRE_DIR` wasn't
set yet) will silently leave you with no `activate` script and a
confusing "command not found" later.

Once confirmed, download (backgrounded, with its own log):

```bash
source "$LUSTRE_DIR/.venv-upload/bin/activate"
export HF_HOME="$LUSTRE_DIR/hf_cache"
mkdir -p "$LUSTRE_DIR/out"

nohup hf download Qwen/Qwen3.8-Flash-Next --local-dir "$LUSTRE_DIR/hf_cache/models/Qwen3.8-Flash-Next" \
  > "$LUSTRE_DIR/out/qwen3.8_download.log" 2>&1 &
disown

sleep 5
tail -20 "$LUSTRE_DIR/out/qwen3.8_download.log"
```

Monitor with:

```bash
tail -f "$LUSTRE_DIR/out/qwen3.8_download.log"
```

Wait for this to fully finish before moving on -- don't submit the
training job while the download is still in progress.

## 2. Write the RLVR code to Lustre

```bash
cat > "$LUSTRE_DIR/train_rlvr_qwen38.py" << 'PYEOF'
"""RLVR fine-tuning for Qwen/Qwen3.8-Flash-Next -- EXPERIMENTAL.

This is a fundamentally different setup from train_rlvr.py, required
because Qwen3.8-Flash-Next is:
  - ~180B params (125B dense + 51B n-gram embedding + 4B MTP, ~6B
    activated per token via MoE) -- ~360GB in BF16. It cannot be
    replicated per-GPU the way train_rlvr.py's DDP setup works. Instead
    the frozen base is sharded across all visible GPUs with
    device_map="auto" (balanced via max_memory so every GPU holds a
    slice) and only a small LoRA adapter (peft) is trained.
  - Multimodal (image-text-to-text): AutoProcessor +
    AutoModelForImageTextToText, text path only, same arithmetic task as
    train_rlvr.py (task.py reused unchanged).
  - A "thinking" model: it emits <think>...</think> by default, which
    eats a small --max-new-tokens budget before any answer appears. The
    chat template is rendered with enable_thinking=False.

Run as a single process (no torchrun/DDP):

    python train_rlvr_qwen38.py --iterations 10

What has been learned on hecate (4 runs), in the order it was learned:
  1. Runs 1-3 (batch 2, 4-bit): identical CUDA assert ("probability
     tensor contains inf/nan") inside generate() at step 3, independent
     of lr. Run 4 with --batch-size 1 got through 135 real steps ->
     LEFT-PADDING INTO THE RECURRENT GATED-DELTANET LAYERS WAS THAT
     CRASH. Batch size stays 1; use --grad-accum for a real batch.
  2. enable_thinking=False took baseline accuracy from 0% to non-zero.
  3. Run 4 then went to NaN on EVERY input after ~step 135 (886 of
     1000 steps skipped by the finite-logits guard; final eval 20/20
     non-finite). A model that works for 135 steps and is then broken
     on all inputs means THE LORA WEIGHTS WERE POISONED: one step had a
     non-finite gradient (a finite loss does not guarantee finite
     grads), clip_grad_norm_ turned an inf norm into NaN grads
     (inf * 0), optimizer.step() wrote NaN into the adapter and AdamW's
     moments, and nothing could recover. The previous guard checked
     logits and loss but never gradients or parameters.
  4. nvidia-smi showed the "4-bit" model occupying ~436GB across two
     GPUs (the other two empty): bitsandbytes only quantises nn.Linear,
     and this architecture's fused MoE expert tensors and 51B n-gram
     nn.Embedding are not that -- so the model was mostly bf16 with a
     scattering of NF4 layers. These GPUs have ~280GB each; the whole
     model fits in bf16 across four of them. Quantisation is therefore
     off by default (--quant none) and layers are spread over all GPUs.

What this version does about it:
  - Gradient guard: clip_grad_norm_ returns the total norm; if it is
    not finite the grads are zeroed and the step is skipped BEFORE
    optimizer.step(). (The clip coefficient inf*0=NaN path can no
    longer reach the weights.)
  - Parameter rollback: a copy of the trainable (LoRA-only, ~5.7M
    params, ~23MB) weights is kept; after every optimizer.step() the
    params are checked and, if any are non-finite, restored from the
    copy and the optimizer state is rebuilt. A bad step costs one
    step, not the run.
  - --grad-accum N: N single-prompt rollouts per optimizer step, losses
    accumulated. Effective batch N with zero padding.
  - --quant none|4bit (default none) and --max-memory-per-gpu to balance
    the shards across all visible GPUs.
  - Everything from the diagnostic version is kept: finite-logits check
    before generate(), printed prompt/completions, --list-modules,
    --skip-quant-modules, --seed, non-finite bookkeeping.
"""

import argparse
import os
import random

import torch
from transformers import (
    AutoConfig,
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
)

from task import sample_batch, verify_reward

try:
    from peft import LoraConfig, get_peft_model
except ImportError as e:
    raise SystemExit(
        "peft is required for this script. "
        "pip install -r requirements-qwen38.txt first."
    ) from e

try:
    from accelerate import init_empty_weights
except ImportError as e:
    raise SystemExit("accelerate is required (pip install -r requirements-qwen38.txt).") from e


def build_prompt(processor, problem: dict, enable_thinking: bool) -> str:
    messages = [{"role": "user", "content": [{"type": "text", "text": problem["question"]}]}]
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experimental RLVR fine-tuning for Qwen3.8-Flash-Next")
    p.add_argument("--model", default="Qwen/Qwen3.8-Flash-Next")
    p.add_argument("--iterations", type=int, default=10, help="Optimizer steps.")
    p.add_argument(
        "--batch-size", type=int, default=1,
        help="Prompts per rollout. KEEP AT 1: batch>1 requires left-padding, which drove the "
             "recurrent Gated DeltaNet layers to NaN and crashed generate() at step 3 on "
             "three consecutive runs. Use --grad-accum for a larger effective batch.",
    )
    p.add_argument(
        "--grad-accum", type=int, default=4,
        help="Single-prompt rollouts accumulated per optimizer step. Effective batch = "
             "--batch-size x --grad-accum, with no padding.",
    )
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument(
        "--enable-thinking", action="store_true",
        help="Render the chat template with thinking ON (model default). Off by default; "
             "turning it off is what moved baseline accuracy from 0% to non-zero.",
    )
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=20, help="Model card's recommended value.")
    p.add_argument("--max-grad-norm", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-dir", default=None, help="Optional dir to save the LoRA adapter (not the full base model)")
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--eval-n", type=int, default=6, help="Held-out problems for greedy eval -- this model is slow")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj",
        help="Comma-separated module name suffixes for LoRA. Attention projections only; "
             "the 512-expert MoE stays frozen. --list-modules shows real names.",
    )
    p.add_argument(
        "--quant", choices=["none", "4bit"], default="none",
        help="'none' loads bf16 (fits: ~360GB across four ~280GB GPUs). '4bit' uses "
             "bitsandbytes NF4 -- on this architecture it only reaches a few nn.Linear "
             "layers (fused MoE experts and the n-gram nn.Embedding are untouched), so it "
             "saves little memory and is a NaN suspect. Kept for experiments only.",
    )
    p.add_argument(
        "--max-memory-per-gpu", default="200GiB",
        help="Cap passed to device_map='auto' per GPU so the shards are spread across ALL "
             "visible GPUs instead of filling GPU 0 to 98%% and leaving others empty "
             "(what run 4 did). Leave headroom for activations.",
    )
    p.add_argument(
        "--skip-quant-modules", default="",
        help="With --quant 4bit: comma-separated substrings; matching nn.Linear modules stay "
             "in bf16. Ignored for --quant none.",
    )
    p.add_argument(
        "--list-modules", action="store_true",
        help="Build the model on the meta device (no weights, seconds), print module "
             "name patterns grouped by class, and exit.",
    )
    p.add_argument("--print-completion-chars", type=int, default=200)
    return p.parse_args()


def _meta_model(model_name):
    config = AutoConfig.from_pretrained(model_name)
    with init_empty_weights():
        return AutoModelForImageTextToText.from_config(config)


def list_modules(model_name: str) -> None:
    """Print every distinct (class, name-with-layer-index-collapsed) pattern once."""
    import re

    model = _meta_model(model_name)
    seen = {}
    for name, module in model.named_modules():
        if not name:
            continue
        pattern = re.sub(r"\.\d+(\.|$)", r".N\1", name)
        key = (type(module).__name__, pattern)
        if key not in seen:
            extra = ""
            if isinstance(module, torch.nn.Linear):
                extra = f"  in={module.in_features} out={module.out_features}"
            seen[key] = extra
    print(f"{len(seen)} distinct module patterns in {model_name}:\n")
    for (cls, pattern), extra in sorted(seen.items(), key=lambda kv: kv[0][1]):
        print(f"  {cls:40s} {pattern}{extra}")


def resolve_skip_modules(model_name: str, substrings: list[str]) -> list[str]:
    if not substrings:
        return []
    model = _meta_model(model_name)
    return [
        name for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and any(s in name for s in substrings)
    ]


def _decode_completion(processor, out_ids, prompt_len):
    return processor.tokenizer.decode(out_ids[prompt_len:], skip_special_tokens=True)


def _show(label: str, text: str, limit: int) -> None:
    snippet = text if len(text) <= limit else text[:limit] + "..."
    print(f"    {label}: {snippet!r}")


@torch.no_grad()
def _logits_finite(model, enc) -> bool:
    """Plain forward over the prompt. Non-finite here means the model itself is
    broken for this input -- generate() would hit the CUDA assert."""
    return bool(torch.isfinite(model(**enc).logits).all())


@torch.no_grad()
def _greedy_eval(model, processor, rng, n, max_new_tokens, enable_thinking, show_chars):
    problems = sample_batch(rng, n)
    correct = 0
    nonfinite = 0
    for i, prob in enumerate(problems):
        text = build_prompt(processor, prob, enable_thinking)
        inputs = processor(text=text, return_tensors="pt").to(model.device)
        if not _logits_finite(model, inputs):
            nonfinite += 1
            print(f"  eval problem {i}: NON-FINITE logits for prompt {prob['question']!r} -- skipping")
            continue
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
        completion = _decode_completion(processor, out[0], inputs["input_ids"].shape[1])
        reward = verify_reward(completion, prob["answer"])
        if i == 0:
            _show(f"eval sample (gold={prob['answer']}, reward={reward})", completion, show_chars)
        if reward == 1.0:
            correct += 1
    if nonfinite:
        print(f"  {nonfinite}/{n} eval prompts produced non-finite logits")
    return correct / n


def _snapshot(params):
    return [p.detach().clone() for p in params]


def _params_finite(params) -> bool:
    return all(bool(torch.isfinite(p).all()) for p in params)


@torch.no_grad()
def _restore(params, snapshot):
    for p, s in zip(params, snapshot):
        p.copy_(s)


def main() -> None:
    args = parse_args()

    if args.list_modules:
        list_modules(args.model)
        return

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    if not torch.cuda.is_available():
        raise SystemExit("This script requires CUDA GPUs.")
    n_gpu = torch.cuda.device_count()

    quantization_config = None
    if args.quant == "4bit":
        skip_substrings = [s.strip() for s in args.skip_quant_modules.split(",") if s.strip()]
        skip_names = resolve_skip_modules(args.model, skip_substrings)
        if skip_substrings:
            print(f"Keeping {len(skip_names)} nn.Linear module(s) in bf16 (matched {skip_substrings}).")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=False,
            llm_int8_skip_modules=["lm_head", *skip_names],
        )

    max_memory = {i: args.max_memory_per_gpu for i in range(n_gpu)}
    print(f"Loading {args.model} ({'bf16' if quantization_config is None else '4-bit NF4'}), "
          f"sharded across {n_gpu} GPU(s) at up to {args.max_memory_per_gpu} each...")
    print("~180B params -- expect this step alone to take a few minutes.")

    processor = AutoProcessor.from_pretrained(args.model)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "left"

    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        quantization_config=quantization_config,
        device_map="auto",
        max_memory=max_memory,
        dtype=torch.bfloat16,
    )
    model.config.use_cache = False  # training forward; generate() passes use_cache=True

    # Where did the shards land? (run 4 put everything on GPUs 0-1.)
    placement = {}
    for _, dev in getattr(model, "hf_device_map", {}).items():
        placement[str(dev)] = placement.get(str(dev), 0) + 1
    print(f"Device map (modules per device): {placement}")
    for i in range(n_gpu):
        used = torch.cuda.memory_allocated(i) / 2**30
        print(f"  GPU {i}: {used:.1f} GiB allocated after load")

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[m.strip() for m in args.lora_target_modules.split(",")],
        lora_dropout=0.0,  # sampling and training forward must see the same network
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.eval()  # gradients still flow; this only disables dropout-like layers

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    def make_optimizer():
        return torch.optim.AdamW(trainable_params, lr=args.lr)

    optimizer = make_optimizer()
    last_good = _snapshot(trainable_params)

    example = sample_batch(random.Random(args.seed + 12345), 1)[0]
    print("\nRendered prompt for one example problem (enable_thinking="
          f"{args.enable_thinking}):\n{build_prompt(processor, example, args.enable_thinking)!r}\n")

    print("Baseline accuracy before training:")
    baseline_acc = _greedy_eval(
        model, processor, rng, n=args.eval_n, max_new_tokens=args.max_new_tokens,
        enable_thinking=args.enable_thinking, show_chars=args.print_completion_chars,
    )
    print(f"  greedy accuracy = {baseline_acc:.2%}")
    print(f"\nTraining: {args.iterations} optimizer steps x {args.grad_accum} rollout(s) "
          f"of {args.batch_size} prompt(s) each\n")

    running_baseline = 0.0
    stats = {"updated": 0, "skipped_logits": 0, "skipped_grad": 0, "rollbacks": 0}

    for step in range(1, args.iterations + 1):
        optimizer.zero_grad()
        step_rewards = []
        first_completion = None
        first_problem = None
        micro_done = 0

        for _ in range(args.grad_accum):
            problems = sample_batch(rng, args.batch_size)
            prompts = [build_prompt(processor, p, args.enable_thinking) for p in problems]
            enc = processor(text=prompts, return_tensors="pt", padding=True).to(model.device)
            prompt_len = enc["input_ids"].shape[1]

            if not _logits_finite(model, enc):
                stats["skipped_logits"] += 1
                print(f"step {step:3d}: NON-FINITE logits from the model for "
                      f"{[p['question'] for p in problems]} -- skipping this rollout.")
                continue

            with torch.no_grad():
                gen_out = model.generate(
                    **enc,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    use_cache=True,
                    pad_token_id=processor.tokenizer.eos_token_id,
                )

            rewards = []
            for i, prob in enumerate(problems):
                completion_text = _decode_completion(processor, gen_out[i], prompt_len)
                r = verify_reward(completion_text, prob["answer"])
                rewards.append(r)
                if first_completion is None:
                    first_completion, first_problem = completion_text, prob
            step_rewards.extend(rewards)
            rewards_t = torch.tensor(rewards, dtype=torch.float32, device=gen_out.device)
            advantages = rewards_t - running_baseline

            attention_mask = (gen_out != processor.tokenizer.pad_token_id).long()
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

            loss = -(advantages.to(seq_log_prob.device) * seq_log_prob).mean() / args.grad_accum
            if not torch.isfinite(loss):
                stats["skipped_logits"] += 1
                print(f"step {step:3d}: non-finite loss -- skipping this rollout.")
                continue
            loss.backward()
            micro_done += 1

        if micro_done == 0:
            optimizer.zero_grad()
            print(f"step {step:3d}: no usable rollouts -- no update.")
            continue

        batch_mean = sum(step_rewards) / len(step_rewards)
        running_baseline = 0.9 * running_baseline + 0.1 * batch_mean

        # --- Gradient guard: this is where run 4 went wrong. ---
        total_norm = torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
        if not torch.isfinite(total_norm):
            optimizer.zero_grad()
            stats["skipped_grad"] += 1
            print(f"step {step:3d}: NON-FINITE gradient norm ({total_norm.item()}) -- "
                  f"grads zeroed, update skipped, weights untouched.")
            continue

        optimizer.step()

        # --- Parameter guard: never carry a poisoned adapter into the next step. ---
        if not _params_finite(trainable_params):
            stats["rollbacks"] += 1
            _restore(trainable_params, last_good)
            optimizer = make_optimizer()  # AdamW moments would be poisoned too
            print(f"step {step:3d}: NON-FINITE parameters after optimizer.step() -- rolled back "
                  f"to the last good adapter and reset the optimizer.")
            continue
        last_good = _snapshot(trainable_params)
        stats["updated"] += 1

        if step % args.log_every == 0:
            print(
                f"step {step:3d}/{args.iterations} | "
                f"mean_reward={batch_mean:.3f} | baseline={running_baseline:.3f} | "
                f"grad_norm={total_norm.item():.3f} | rollouts={micro_done}/{args.grad_accum}"
            )
            if first_completion is not None:
                _show(f"sample (gold={first_problem['answer']}, reward={step_rewards[0]})",
                      first_completion, args.print_completion_chars)

    print(f"\nStep accounting: {stats}")

    print("\nAccuracy after training:")
    final_acc = _greedy_eval(
        model, processor, rng, n=args.eval_n, max_new_tokens=args.max_new_tokens,
        enable_thinking=args.enable_thinking, show_chars=args.print_completion_chars,
    )
    print(f"  greedy accuracy = {final_acc:.2%} (baseline was {baseline_acc:.2%})")

    if args.save_dir:
        if _params_finite(trainable_params):
            model.save_pretrained(args.save_dir)  # LoRA adapter only
            processor.save_pretrained(args.save_dir)
            print(f"Saved LoRA adapter to {args.save_dir}")
        else:
            print("Adapter is non-finite -- NOT saving.")


if __name__ == "__main__":
    main()
PYEOF

cat > "$LUSTRE_DIR/requirements-qwen38.txt" << 'REQEOF'
# For train_rlvr_qwen38.py inside the gitlab-master.nvidia.com/dl/dgx/pytorch:main-py3-devel
# container. The container's bundled torch/CUDA is reused (not reinstalled here), but
# Qwen3.8-Flash-Next's architecture (qwen4_exp) is brand new -- if model loading fails
# with an unrecognized-architecture error, the container's transformers is too old;
# try `pip install -U transformers` (or install from git main) before anything else.
transformers>=4.57
accelerate>=1.0
peft>=0.13
bitsandbytes>=0.44
REQEOF

mkdir -p "$LUSTRE_DIR/scripts"

cat > "$LUSTRE_DIR/scripts/launch_rlvr_qwen38.sh" << 'SHEOF'
#!/bin/bash
# Runs INSIDE the pyxis/enroot container on the compute node.
# EXPERIMENTAL -- see train_rlvr_qwen38.py's docstring for the full history
# of what has been observed on hecate and why the defaults are what they are.
set -e

LUSTRE_DIR="/lustre/fsw/general_sa/bbalakreshna"

echo "$(hostname): Installing RLVR + Qwen3.8-Flash-Next dependencies..."
pip install --quiet -r "$LUSTRE_DIR/requirements-qwen38.txt"

export HF_HOME="$LUSTRE_DIR/hf_cache"
mkdir -p "$HF_HOME"

# Off by default now: the script guards logits, gradients and parameters
# itself, so the CUDA assert should no longer fire, and synchronous kernel
# launches slow every step. Set CUDA_LAUNCH_BLOCKING=1 in the environment
# if you need a precise stack trace for a new failure.
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"

echo "$(hostname): Launching RLVR fine-tuning for Qwen3.8-Flash-Next (single process, device_map=auto)..."
python "$LUSTRE_DIR/train_rlvr_qwen38.py" \
  --model "${RLVR_MODEL:-Qwen/Qwen3.8-Flash-Next}" \
  --iterations "${RLVR_ITERATIONS:-10}" \
  --batch-size "${RLVR_BATCH_SIZE:-1}" \
  --grad-accum "${RLVR_GRAD_ACCUM:-4}" \
  --max-new-tokens "${RLVR_MAX_NEW_TOKENS:-64}" \
  --lr "${RLVR_LR:-5e-6}" \
  --eval-n "${RLVR_EVAL_N:-6}" \
  --lora-r "${RLVR_LORA_R:-16}" \
  --seed "${RLVR_SEED:-0}" \
  --quant "${RLVR_QUANT:-none}" \
  --max-memory-per-gpu "${RLVR_MAX_MEMORY_PER_GPU:-200GiB}" \
  --skip-quant-modules "${RLVR_SKIP_QUANT_MODULES:-}" \
  --save-dir "$LUSTRE_DIR/out/rlvr-qwen38-run1"
SHEOF
chmod +x "$LUSTRE_DIR/scripts/launch_rlvr_qwen38.sh"

cat > "$LUSTRE_DIR/scripts/submit_hecate_qwen38.sh" << 'SHEOF'
#!/bin/bash
# Run this FROM hecate's login node home directory (~), after
# train_rlvr_qwen38.py / requirements-qwen38.txt / launch_rlvr_qwen38.sh
# have been placed on Lustre. EXPERIMENTAL -- unverified against real
# hardware, start with the tiny defaults baked into launch_rlvr_qwen38.sh.
set -e

export ACCOUNT="${ACCOUNT:-general_sa}"
export LUSTRE_DIR="${LUSTRE_DIR:-/lustre/fsw/$ACCOUNT/$USER}"
mkdir -p "$LUSTRE_DIR/out"

LOG_FILE="$LUSTRE_DIR/out/hecate_qwen38_run1.log"
TIMING_FILE="$LUSTRE_DIR/out/hecate_qwen38_run1.timing"

{
  START_TS=$(date +%s)
  echo "Job started: $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$TIMING_FILE"

  srun --account=general_sa \
       --partition=batch-xdr \
       --nodes=1 \
       --ntasks-per-node=1 \
       --time=5:00:00 \
       --job-name=general_sa-rlvr.qwen38 \
       --container-image=gitlab-master.nvidia.com/dl/dgx/pytorch:main-py3-devel \
       --container-mount-home \
       --container-mounts=/lustre:/lustre \
       --no-container-remap-root \
       --mpi=pmix \
       --export=ALL \
       "$LUSTRE_DIR/scripts/launch_rlvr_qwen38.sh" > "$LOG_FILE" 2>&1

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
chmod +x "$LUSTRE_DIR/scripts/submit_hecate_qwen38.sh"

echo "--- Files written ---"
ls -la "$LUSTRE_DIR/train_rlvr_qwen38.py" "$LUSTRE_DIR/requirements-qwen38.txt" "$LUSTRE_DIR/scripts/launch_rlvr_qwen38.sh" "$LUSTRE_DIR/scripts/submit_hecate_qwen38.sh"
```

Note: `task.py` is reused unchanged from the earlier setup -- it must
already exist at `$LUSTRE_DIR/task.py` (from the multi-GPU Block 1
steps). If it doesn't, run that block first.

## 3. Submit the job

Only after the download from step 1 has fully finished:

```bash
bash "$LUSTRE_DIR/scripts/submit_hecate_qwen38.sh"
```

## 4. Check status / view output

```bash
squeue -u $USER
sacct -u $USER --format=JobID,JobName,State,Elapsed,ExitCode -j <JOBID>
tail -f "$LUSTRE_DIR/out/hecate_qwen38_run1.log"
cat "$LUSTRE_DIR/out/hecate_qwen38_run1.timing"
```

Watch the very start of the log closely -- if `AutoModelForImageTextToText.from_pretrained(...)`
fails, nothing downstream matters until that's fixed.

## 5. Gather diagnostics from the run

Paste the output of this block when asking for help interpreting a run --
the three things that matter most are the `Rendered prompt` (does it
contain `<think>`?), the `eval sample ... : '...'` line (what the model
actually said), and any `NON-FINITE ... offending prompt:` line.

```bash
export LUSTRE_DIR="${LUSTRE_DIR:-/lustre/fsw/general_sa/$USER}"
LOG="$LUSTRE_DIR/out/hecate_qwen38_run1.log"

echo "===== JOB STATE ====="; squeue -u $USER
echo "===== LOG HEAD (prompt + first completions) ====="; head -80 "$LOG"
echo "===== KEY LINES ====="; grep -nE "greedy accuracy|NON-FINITE|non-finite|offending prompt|skipped|Traceback|Error|Assertion|^step " "$LOG" | head -60
echo "===== TIMING ====="; cat "$LUSTRE_DIR/out/hecate_qwen38_run1.timing" 2>/dev/null
echo "===== MODULE LIST (first 60) ====="; head -60 "$LUSTRE_DIR/out/qwen38_modules.txt" 2>/dev/null
```

## 6. Insights: text summary + interactive HTML dashboard

`analyze_run.py` and `generate_dashboard.py` understand both log
formats: `train_rlvr.py`'s `step N/M | ... | loss=...` and this script's
`step N/M | ... | grad_norm=... | rollouts=a/b`. For a qwen38 run the
dashboard additionally shows the `Step accounting` counters as stat
cards, a histogram of *where* the gradient guard fired, the
gradient-norm trajectory (it falls to ~0 once the baseline catches the
reward -- i.e. the task is solved), rollouts completed per step,
per-GPU memory after load (confirms the shards were balanced), and the
first baseline/final eval completions verbatim. Both tools are
stdlib-only: login-node system `python3`, no venv.

**Paste the current versions of both tools first** -- the `cat` heredocs
are in [`executeshell-multigpu.md`](executeshell-multigpu.md) under
"Generating insights from the run" and "Interactive HTML insights
dashboard". Older copies (e.g. under the retired `rlvr-posttraining101`
subfolder) require a `loss=` field and report "No step lines found" on
a qwen38 log.

```bash
export LUSTRE_DIR="${LUSTRE_DIR:-/lustre/fsw/general_sa/$USER}"
LOG="$LUSTRE_DIR/out/hecate_qwen38_run1.log"
TIMING="$LUSTRE_DIR/out/hecate_qwen38_run1.timing"

python3 "$LUSTRE_DIR/analyze_run.py" "$LOG" --timing "$TIMING"
python3 "$LUSTRE_DIR/generate_dashboard.py" "$LOG" --timing "$TIMING" \
  --output "$LUSTRE_DIR/out/hecate_qwen38_run1_dashboard.html"
```

Then pull the dashboard down from a **fresh local terminal on your own
machine** (not inside the SSH session -- `scp` copies *to* wherever it
runs, and `$HOME/Downloads` doesn't exist on the login node). It
prompts for the usual MFA device code:

```powershell
scp bbalakreshna-mfa@login-hecate.nvidia.com:/lustre/fsw/general_sa/bbalakreshna/out/hecate_qwen38_run1_dashboard.html C:\Users\bbalakreshna\Downloads\hecate_qwen38_run1_dashboard.html
```

It's a single self-contained HTML file -- double-click to open. Chart.js
loads from a CDN, so the browser needs internet; the cluster doesn't.
If the run crashed at step 3 the dashboard will only have 2 data points
-- section 5's output is what matters in that case.

## 7. Scaling up to 1000 steps

**Only after a 10-step diagnostic run has completed cleanly** (a final
`Accuracy after training` line, no crash). If it still dies at step 3,
1000 steps will die at step 3 too. No code change is needed -- the
launcher is env-var driven -- but two things in the submit script don't
fit a long run, so this block adjusts them:

```bash
export LUSTRE_DIR="${LUSTRE_DIR:-/lustre/fsw/general_sa/$USER}"

# 1. keep the 10-step diagnostic log; the submit script would overwrite it
mv -f "$LUSTRE_DIR/out/hecate_qwen38_run1.log"    "$LUSTRE_DIR/out/hecate_qwen38_diag.log"    2>/dev/null
mv -f "$LUSTRE_DIR/out/hecate_qwen38_run1.timing" "$LUSTRE_DIR/out/hecate_qwen38_diag.timing" 2>/dev/null

# 2. submit: 1000 optimizer steps x 4 accumulated rollouts, bigger eval.
#    (The submit script's --time is already the 5h batch-xdr max, and
#    CUDA_LAUNCH_BLOCKING now defaults to 0 in the launcher.)
RLVR_ITERATIONS=1000 RLVR_GRAD_ACCUM=4 RLVR_EVAL_N=20 \
  bash "$LUSTRE_DIR/scripts/submit_hecate_qwen38.sh"
```

Before submitting, check the per-step time from the diagnostic run's
timing file and the `Step accounting:` line at the end of its log --
only `updated` steps cost full time. Run 4 (batch 1, one rollout per
step) averaged well under 30 s per real step; with `--grad-accum 4`
expect roughly 4x that. If 1000 steps won't fit in 5h, drop
`RLVR_ITERATIONS` or switch to the 8h `backfill-xdr` partition:
`sed -i 's/--partition=batch-xdr/--partition=backfill-xdr/' "$LUSTRE_DIR/scripts/submit_hecate_qwen38.sh"`.

**Do not raise `RLVR_BATCH_SIZE` above 1.** Batching prompts requires
left-padding, and that is what crashed runs 1-3 at step 3 (the
recurrent Gated DeltaNet layers went NaN on padded input). More signal
per step comes from `RLVR_GRAD_ACCUM` (single-prompt rollouts, losses
accumulated, no padding) -- 8 or 16 is reasonable once a run completes.

## Known risks / what to check if it fails

**Status: four runs on hecate. Each failure was different from what it
first looked like, and each one is now understood.** Model load and
LoRA attach worked every time (`trainable params: 5,701,632 || all
params: 177,398,532,208`).

**Runs 1-3** (batch 2, 4-bit): an identical CUDA device-side assert
(`probability tensor contains either inf, nan or element < 0`) inside
`model.generate()` at step 3, unchanged by a 20x learning-rate change,
structured chat content, or `use_cache=True`. Baseline accuracy was
0.00% and every rollout scored exactly 0.1 on all three.

**Run 4** (the instrumented version: batch 1, thinking off, finite-logits
guard, 1000 steps) resolved both mysteries and exposed a third:

1. **The step-3 crash was left-padding.** With `--batch-size 1` (no
   padding) the run got through 135 real steps instead of dying at 3.
   Batching prompts pads them to equal length; this architecture's
   Gated DeltaNet layers are recurrent, and pad tokens entering the
   recurrence drove it to NaN. `--batch-size` must stay 1;
   `--grad-accum` provides the batch instead (single-prompt rollouts,
   accumulated losses, zero padding).
2. **The 0% baseline was thinking mode.** `enable_thinking=False` took
   baseline accuracy from 0% to non-zero (5% with only 20 eval
   problems). The earlier "bare string content" theory was wrong.
3. **Then the adapter got poisoned.** After ~step 135 *every* input
   produced non-finite logits (886 of 1000 steps skipped; the final
   eval was 20/20 non-finite). A model that works for 135 steps and is
   then broken on all inputs is not an input problem -- **the LoRA
   weights themselves became NaN.** Mechanism: one step produced a
   non-finite gradient (a finite loss does not guarantee finite
   gradients), `clip_grad_norm_` computed an infinite norm, so the clip
   coefficient was 0 and `inf x 0 = NaN` landed in the grads;
   `optimizer.step()` wrote NaN into the adapter and into AdamW's
   moments; every forward thereafter was NaN. The run-4 guard checked
   logits and loss but never gradients or parameters.
4. **The "4-bit" model wasn't 4-bit.** `nvidia-smi` showed ~436GB in
   use across GPUs 0-1 with GPUs 2-3 empty. bitsandbytes only
   quantises `nn.Linear`; this architecture's fused MoE expert tensors
   and its 51B-param n-gram `nn.Embedding` aren't that, so the model
   was mostly bf16 with a scattering of NF4 layers -- a mixed state
   that saves little memory and is itself a NaN suspect. And these GPUs
   have ~280GB each: the whole model fits in bf16 across four of them.

What the current script does about each: `--quant none` (bf16) by
default with `--max-memory-per-gpu` so `device_map="auto"` spreads the
shards over *all* GPUs (it prints the per-GPU allocation after load);
`--batch-size 1` + `--grad-accum` (default 4); a **gradient guard**
(`clip_grad_norm_`'s returned norm is checked -- non-finite means grads
are zeroed and the step skipped *before* `optimizer.step()`); a
**parameter guard** (a ~23MB copy of the LoRA weights is kept; if any
parameter is non-finite after a step, the adapter is rolled back and
the optimizer rebuilt, so a bad step costs one step, not the run); and
the adapter is only saved if it is finite. The `Step accounting:` line
at the end of the log reports `updated / skipped_logits / skipped_grad
/ rollbacks` -- if `skipped_grad` or `rollbacks` is non-zero the guards
are earning their keep; if `updated` is most of the steps, the run is
healthy.

Not yet re-verified against hardware. If a run still shows non-finite
logits from the *base* model on the very first eval (before any
training), that would be a genuine bf16 forward-pass problem in this
architecture, and `CUDA_LAUNCH_BLOCKING=1` plus `--seed` variations are
the tools to localise it.

Other risks not yet encountered:

- **"Unrecognized architecture" / `qwen4_exp` not found** -- did not
  occur; the container's `transformers` (after `pip install -U`)
  already supports `qwen4_exp`. Leaving this note in case a different
  container image is used later.
- **OOM during model load** -- even at 4-bit, ~180B params is roughly
  90GB+ of weights alone (before activations/KV cache). If the node's
  visible GPUs don't have enough combined memory, `device_map="auto"`
  will fail to place all layers. Reduce `--max-new-tokens` and
  `--batch-size` further.
- **Only attention layers are trainable.** If the run completes but
  accuracy doesn't move much, that's expected to some degree -- the
  512-expert MoE layers (where most of the model's capacity lives)
  are frozen by design here. This is a deliberate simplicity
  trade-off, not a bug.
