# Hecate execute shell — Qwen3.8-Flash-Next RLVR (EXPERIMENTAL)

**This is unverified against real hardware.** Qwen3.8-Flash-Next is a
~180B param multimodal MoE model (see model card:
[huggingface.co/Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)) --
fundamentally different from the validated [`executeshell-multigpu.md`](executeshell-multigpu.md)
/ [`executeshell-singlegpu.md`](executeshell-singlegpu.md) flows, which
use Qwen2.5-1.5B-Instruct with full-replica DDP training. At 180B params
(~360GB in BF16), that approach doesn't fit this model at all -- instead
this loads the frozen base in 4-bit (bitsandbytes) with
`device_map="auto"` (sharded across every visible GPU on the node) and
trains only a small LoRA adapter on top of the attention projections.
The 512-expert MoE layers stay frozen.

Read [train_rlvr_qwen38.py](../train_rlvr_qwen38.py)'s docstring for the
full rationale and for what has actually been observed on hecate so far.
Start with the tiny defaults (batch 1, 10 iterations, 64 tokens, thinking
mode off) baked into `launch_rlvr_qwen38.sh` -- this run is instrumented
to *diagnose* the recurring NaN crash (see "Known risks" at the bottom),
not to be a real training run yet.

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
    activated per token via MoE). Full BF16 weights are ~360GB -- this
    will NOT fit as 4 full replicas the way train_rlvr.py's DDP setup
    works. Instead we load the frozen base in 4-bit (bitsandbytes) with
    device_map="auto" so accelerate shards it across all visible GPUs,
    and only train a small LoRA adapter (peft) on top -- the only
    thing that needs gradients/optimizer state.
  - Multimodal (image-text-to-text): loaded via AutoProcessor +
    AutoModelForImageTextToText, not AutoTokenizer + AutoModelForCausalLM.
    We only use its text path here (no images), same arithmetic task as
    train_rlvr.py, reusing task.py unchanged.
  - A "thinking" model: by default it emits <think>...</think> before
    the answer. With a small --max-new-tokens budget it never reaches
    the answer, which scores 0.0/0.1 every time. We render the chat
    template with enable_thinking=False (documented on the model card).

Run as a single process (no torchrun/DDP -- device_map="auto" already
spreads the frozen base across every visible GPU on the node):

    python train_rlvr_qwen38.py --iterations 10 --batch-size 1

What has actually been observed on hecate so far (3 runs):
  - Model load + LoRA attach work every time
    ("trainable params: 5,701,632 || all params: 177,398,532,208").
  - Baseline greedy accuracy 0.00% and every sampled rollout scoring
    exactly 0.1 -- consistent with thinking mode eating the whole
    token budget (see above), NOT with the prompt being malformed.
  - An identical CUDA device-side assert ("probability tensor contains
    either inf, nan or element < 0") inside model.generate() at step 3
    on all three runs, with lr varying 20x between runs. LoRA's B
    matrix is zero-initialised, so two tiny steps in the model is still
    ~the base model: this is not an update-magnitude problem. Because
    both the problem RNG and torch RNG are seeded, "step 3" is really
    "one specific batch of inputs" -- the NaN is data-dependent. Once
    the assert fires the CUDA context is dead; the process can't
    recover, only avoid it.

This version is therefore instrumented to *diagnose* rather than guess:
  - --list-modules: builds the model on the meta device (seconds, no
    weights) and prints every module name pattern, so quantisation
    skip-lists and LoRA target names can be chosen from real names.
  - Prints the rendered prompt once and the first decoded completion of
    every eval / training step, so we can see what the model says.
  - Runs a plain forward pass on each prompt batch and checks the
    logits are finite BEFORE calling generate(). If they aren't, it
    prints the offending prompt and skips the step -- no multinomial
    call, so no CUDA assert, and the run keeps going.
  - --skip-quant-modules: comma-separated substrings; matching
    nn.Linear modules are kept in bf16 instead of 4-bit (resolved to
    full module names via the meta model, so it works regardless of
    transformers' name-matching rules).
  - Default --batch-size 1 removes padding as a variable (this
    architecture has recurrent Gated DeltaNet layers; left-padding
    feeding a recurrence is a plausible NaN source). Raise it once a
    run completes.
  - Set CUDA_LAUNCH_BLOCKING=1 in the environment (launch_rlvr_qwen38.sh
    does) so any remaining assert reports the *real* failing op instead
    of the async-reported one.
  - Try a different --seed: if the crash moves or disappears, that
    confirms it is input-dependent.
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
    # Structured content blocks -- this is a multimodal processor and the
    # model card's examples use [{"type": "text", "text": ...}] throughout.
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
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument(
        "--batch-size", type=int, default=1,
        help="Default 1 = no padding, removing it as a variable while diagnosing the NaN "
             "crash (recurrent layers + left-padding is a plausible cause). Raise once a "
             "run completes.",
    )
    p.add_argument(
        "--max-new-tokens", type=int, default=64,
        help="With thinking disabled, 64 is plenty for 'Final Answer: N'. If you enable "
             "thinking you'll need hundreds.",
    )
    p.add_argument(
        "--enable-thinking", action="store_true",
        help="Render the chat template with thinking ON (model default). Off by default "
             "here because the <think> block otherwise consumes the whole token budget.",
    )
    p.add_argument(
        "--lr", type=float, default=5e-6,
        help="Kept conservative. Note the NaN crash was NOT lr-dependent (identical at "
             "1e-4 and 5e-6), so don't expect lowering this further to fix it.",
    )
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=20, help="Model card's recommended value.")
    p.add_argument("--max-grad-norm", type=float, default=0.3)
    p.add_argument(
        "--seed", type=int, default=0,
        help="Both the problem RNG and torch are seeded from this. The NaN crash has "
             "been at the same step every run -- change the seed to test whether it is "
             "tied to specific inputs.",
    )
    p.add_argument("--save-dir", default=None, help="Optional dir to save the LoRA adapter (not the full base model)")
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--eval-n", type=int, default=6, help="Held-out problems for greedy eval -- keep small, this model is slow")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj",
        help="Comma-separated module name suffixes to attach LoRA to. Attention "
             "projections only by default -- the 512-expert MoE layers stay frozen. "
             "Use --list-modules to see the real names in this architecture.",
    )
    p.add_argument(
        "--skip-quant-modules", default="",
        help="Comma-separated substrings; any nn.Linear whose full name contains one is "
             "kept in bf16 rather than quantised to 4-bit. lm_head is always skipped. "
             "Candidates once --list-modules shows real names: MoE routers/gates, the "
             "sparse-attention indexer, Gated DeltaNet's small gate projections, MTP.",
    )
    p.add_argument(
        "--list-modules", action="store_true",
        help="Build the model on the meta device (no weights, seconds), print module "
             "name patterns grouped by class, and exit. Use this to pick "
             "--skip-quant-modules / --lora-target-modules from real names.",
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
    print("\nnn.Linear modules are the ones bitsandbytes will quantise unless skipped.")


def resolve_skip_modules(model_name: str, substrings: list[str]) -> list[str]:
    """Turn substrings into the exact full names of matching nn.Linear modules."""
    if not substrings:
        return []
    model = _meta_model(model_name)
    names = [
        name for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and any(s in name for s in substrings)
    ]
    return names


def _decode_completion(processor, out_ids, prompt_len):
    return processor.tokenizer.decode(out_ids[prompt_len:], skip_special_tokens=True)


def _show(label: str, text: str, limit: int) -> None:
    snippet = text if len(text) <= limit else text[:limit] + "..."
    print(f"    {label}: {snippet!r}")


@torch.no_grad()
def _logits_finite(model, enc) -> bool:
    """One plain forward over the prompt. If this is non-finite the base model
    itself is broken for this input -- generate() would hit the CUDA assert."""
    logits = model(**enc).logits
    return bool(torch.isfinite(logits).all())


@torch.no_grad()
def _greedy_eval(model, processor, rng, n, max_new_tokens, enable_thinking, show_chars):
    problems = sample_batch(rng, n)
    correct = 0
    for i, prob in enumerate(problems):
        text = build_prompt(processor, prob, enable_thinking)
        inputs = processor(text=text, return_tensors="pt").to(model.device)
        if not _logits_finite(model, inputs):
            print(f"  eval problem {i}: NON-FINITE logits for prompt {prob['question']!r} -- skipping")
            continue
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,  # config.use_cache=False is set for the training forward;
                              # generation (esp. the recurrent DeltaNet layers) wants it on
            pad_token_id=processor.tokenizer.eos_token_id,
        )
        completion = _decode_completion(processor, out[0], inputs["input_ids"].shape[1])
        reward = verify_reward(completion, prob["answer"])
        if i == 0:
            _show(f"eval sample (gold={prob['answer']}, reward={reward})", completion, show_chars)
        if reward == 1.0:
            correct += 1
    return correct / n


def main() -> None:
    args = parse_args()

    if args.list_modules:
        list_modules(args.model)
        return

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    if not torch.cuda.is_available():
        raise SystemExit("This script requires CUDA GPUs -- Qwen3.8-Flash-Next is not viable on CPU.")
    if os.environ.get("CUDA_LAUNCH_BLOCKING") != "1":
        print("note: CUDA_LAUNCH_BLOCKING is not set -- if a CUDA assert fires, the reported "
              "stack trace will point at the wrong op. launch_rlvr_qwen38.sh sets it.")

    skip_substrings = [s.strip() for s in args.skip_quant_modules.split(",") if s.strip()]
    skip_names = resolve_skip_modules(args.model, skip_substrings)
    if skip_substrings:
        print(f"Keeping {len(skip_names)} nn.Linear module(s) in bf16 (matched {skip_substrings}).")
        for name in skip_names[:10]:
            print(f"    {name}")
        if len(skip_names) > 10:
            print(f"    ... and {len(skip_names) - 10} more")

    print(f"Loading {args.model} in 4-bit, sharded across {torch.cuda.device_count()} visible GPU(s)...")
    print("This is a ~180B param model -- expect this step alone to take a while.")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=False,
        llm_int8_skip_modules=["lm_head", *skip_names],
    )

    processor = AutoProcessor.from_pretrained(args.model)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "left"

    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16,
    )
    model.config.use_cache = False  # for the training forward pass; generate() overrides

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[m.strip() for m in args.lora_target_modules.split(",")],
        # 0.0 so the log-probs computed in the training forward match the distribution
        # that produced the sampled tokens in generate(); the model also stays in eval()
        # mode throughout for the same reason (gradients still flow in eval()).
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.eval()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    # Show exactly what the model is being fed, once.
    example = sample_batch(random.Random(args.seed + 12345), 1)[0]
    print("\nRendered prompt for one example problem (enable_thinking="
          f"{args.enable_thinking}):\n{build_prompt(processor, example, args.enable_thinking)!r}\n")

    print("Baseline accuracy before training:")
    baseline_acc = _greedy_eval(
        model, processor, rng, n=args.eval_n, max_new_tokens=args.max_new_tokens,
        enable_thinking=args.enable_thinking, show_chars=args.print_completion_chars,
    )
    print(f"  greedy accuracy = {baseline_acc:.2%}")

    running_baseline = 0.0
    skipped = 0
    for step in range(1, args.iterations + 1):
        problems = sample_batch(rng, args.batch_size)
        prompts = [build_prompt(processor, p, args.enable_thinking) for p in problems]

        enc = processor(text=prompts, return_tensors="pt", padding=True).to(model.device)
        prompt_len = enc["input_ids"].shape[1]

        # Guard: a plain forward first. If the base model already produces
        # non-finite logits for this input, generate() would hit the CUDA
        # assert and kill the process. Skip instead, and say which input.
        if not _logits_finite(model, enc):
            skipped += 1
            print(f"step {step:3d}: NON-FINITE logits from the base model for this batch -- skipping.")
            for p in problems:
                print(f"    offending prompt: {p['question']!r}")
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
        completions = []
        for i, prob in enumerate(problems):
            completion_text = _decode_completion(processor, gen_out[i], prompt_len)
            completions.append(completion_text)
            rewards.append(verify_reward(completion_text, prob["answer"]))
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=gen_out.device)

        batch_mean = rewards_t.mean().item()
        advantages = rewards_t - running_baseline
        running_baseline = 0.9 * running_baseline + 0.1 * batch_mean

        attention_mask = (gen_out != processor.tokenizer.pad_token_id).long()
        logits = model(input_ids=gen_out, attention_mask=attention_mask).logits
        if not torch.isfinite(logits).all():
            skipped += 1
            print(f"step {step:3d}: non-finite logits in the training forward pass -- skipping update.")
            continue
        log_probs = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
        target_ids = gen_out[:, 1:]
        token_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

        gen_mask = torch.zeros_like(target_ids, dtype=torch.float32)
        gen_mask[:, prompt_len - 1:] = 1.0
        gen_mask *= attention_mask[:, 1:].float()

        seq_log_prob = (token_log_probs * gen_mask).sum(dim=1) / gen_mask.sum(dim=1).clamp(min=1)
        loss = -(advantages.to(seq_log_prob.device) * seq_log_prob).mean()

        if not torch.isfinite(loss):
            skipped += 1
            print(f"step {step:3d}: non-finite loss -- skipping this step's optimizer update.")
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
        optimizer.step()

        if step % args.log_every == 0:
            print(
                f"step {step:3d}/{args.iterations} | "
                f"mean_reward={batch_mean:.3f} | baseline={running_baseline:.3f} | loss={loss.item():.4f}"
            )
            _show(f"sample (gold={problems[0]['answer']}, reward={rewards[0]})",
                  completions[0], args.print_completion_chars)

    if skipped:
        print(f"\n{skipped} step(s) skipped due to non-finite values -- see messages above.")

    print("\nAccuracy after training:")
    final_acc = _greedy_eval(
        model, processor, rng, n=args.eval_n, max_new_tokens=args.max_new_tokens,
        enable_thinking=args.enable_thinking, show_chars=args.print_completion_chars,
    )
    print(f"  greedy accuracy = {final_acc:.2%} (baseline was {baseline_acc:.2%})")

    if args.save_dir:
        model.save_pretrained(args.save_dir)  # LoRA adapter only, not the frozen base
        processor.save_pretrained(args.save_dir)
        print(f"Saved LoRA adapter to {args.save_dir}")


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
# EXPERIMENTAL -- see train_rlvr_qwen38.py's docstring for why this is a
# different (LoRA + 4-bit + single-process device_map="auto") setup than
# launch_rlvr.sh's DDP full-replica training, and for what has been
# observed on hecate so far.
set -e

LUSTRE_DIR="/lustre/fsw/general_sa/bbalakreshna"

echo "$(hostname): Installing RLVR + Qwen3.8-Flash-Next dependencies..."
pip install --quiet -r "$LUSTRE_DIR/requirements-qwen38.txt"

export HF_HOME="$LUSTRE_DIR/hf_cache"
mkdir -p "$HF_HOME"

# Debugging aid while the NaN crash is being chased: makes CUDA kernel
# launches synchronous so a device-side assert is reported at the op that
# actually failed, not at some later sync point. Slows things down; remove
# once a run completes cleanly.
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"

echo "$(hostname): Launching RLVR fine-tuning for Qwen3.8-Flash-Next (single process, device_map=auto)..."
python "$LUSTRE_DIR/train_rlvr_qwen38.py" \
  --model "${RLVR_MODEL:-Qwen/Qwen3.8-Flash-Next}" \
  --iterations "${RLVR_ITERATIONS:-10}" \
  --batch-size "${RLVR_BATCH_SIZE:-1}" \
  --max-new-tokens "${RLVR_MAX_NEW_TOKENS:-64}" \
  --lr "${RLVR_LR:-5e-6}" \
  --eval-n "${RLVR_EVAL_N:-6}" \
  --lora-r "${RLVR_LORA_R:-16}" \
  --seed "${RLVR_SEED:-0}" \
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
       --time=4:00:00 \
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

`analyze_run.py` and `generate_dashboard.py` (from
[`executeshell-multigpu.md`](executeshell-multigpu.md)) work on this
run unchanged -- `train_rlvr_qwen38.py` deliberately prints the same
`step N/M | mean_reward=... | baseline=... | loss=...` and
`greedy accuracy = X%` lines they parse. The per-operator charts will
be empty (this script doesn't log per-operator stats) and the "Device"
stat card is omitted (different load message); everything else
populates. Both tools are stdlib-only, so they run on the login node's
system `python3` with no venv.

```bash
export LUSTRE_DIR="${LUSTRE_DIR:-/lustre/fsw/general_sa/$USER}"
LOG="$LUSTRE_DIR/out/hecate_qwen38_run1.log"
TIMING="$LUSTRE_DIR/out/hecate_qwen38_run1.timing"

# the two tools were originally written under the old rlvr-posttraining101 subfolder -- copy up if needed
for f in analyze_run.py generate_dashboard.py; do
  [ -f "$LUSTRE_DIR/$f" ] || cp "$LUSTRE_DIR/rlvr-posttraining101/$f" "$LUSTRE_DIR/$f"
done

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

# 2. raise the SLURM time limit to the batch-xdr partition max (5h; it was 4h)
sed -i 's/--time=4:00:00/--time=5:00:00/' "$LUSTRE_DIR/scripts/submit_hecate_qwen38.sh"
grep -n -- '--time=' "$LUSTRE_DIR/scripts/submit_hecate_qwen38.sh"

# 3. submit: 1000 steps, sync-launch debugging OFF (it slows every kernel launch), bigger eval
RLVR_ITERATIONS=1000 CUDA_LAUNCH_BLOCKING=0 RLVR_EVAL_N=20 \
  bash "$LUSTRE_DIR/scripts/submit_hecate_qwen38.sh"
```

Before submitting, check the per-step time from the diagnostic run:
`cat "$LUSTRE_DIR/out/hecate_qwen38_diag.timing"`, then
(elapsed seconds - ~120s of model loading) / 10 steps. Over ~15 s/step
and 1000 steps won't fit in 5h -- either drop to `RLVR_ITERATIONS=500`
or switch to the 8h `backfill-xdr` partition with
`sed -i 's/--partition=batch-xdr/--partition=backfill-xdr/' "$LUSTRE_DIR/scripts/submit_hecate_qwen38.sh"`.

Batch size stays at 1 by default (left-padding into the recurrent
layers was a NaN suspect), so 1000 steps is only 1000 samples -- a weak
policy-gradient signal. If the diagnostic run showed no `NON-FINITE`
lines, add `RLVR_BATCH_SIZE=4` to the submit line for 4x the signal per
step.

## Known risks / what to check if it fails

**Status: confirmed on real hardware -- three runs, the same crash every
time.** Model load, LoRA attach, and 2 training steps ran successfully
on every run (`trainable params: 5,701,632 || all params:
177,398,532,208`), then hit an identical CUDA device-side assert
(`probability tensor contains either inf, nan or element < 0`) inside
`model.generate()`'s sampling at step 3. Runs 1 and 2 differed by a 20x
learning rate; run 3 additionally had structured chat content and
`use_cache=True` in `generate()`. **None of it changed anything**: the
baseline stayed at exactly 0.00% and every sampled rollout scored
exactly 0.1 on all three runs. Once the assert fires the CUDA context
is dead; the process cannot recover, only avoid it.

What the three runs actually tell us:

1. **The 0% / 0.1 numbers are almost certainly thinking mode, not a
   broken prompt.** This is a reasoning model that emits
   `<think>...</think>` before answering (model card: "operate in
   thinking mode by default"). With `--max-new-tokens 32` it never gets
   out of the thinking block, so it never emits `Final Answer: N`.
   Greedy eval can't score 1.0 (0%), and sampled rollouts score 0.1
   because some digit inside the thinking text gets picked up as a
   wrong "last integer". Rendering the template with
   `enable_thinking=False` (the model card's documented switch) and
   64 tokens should fix this. The round-2 "bare string content" theory
   was wrong -- changing it changed nothing.
2. **The crash is input-dependent, not lr-dependent.** LoRA's B matrix
   is zero-initialised, so after two tiny steps the model is still
   essentially the base model -- and the crash didn't move by even one
   step across a 20x lr change. Both `random.Random(seed)` and
   `torch.manual_seed(seed)` are fixed, so every run generates the
   *same* problems in the *same* order: "always step 3" means "always
   this one batch". Something about a specific input makes the 4-bit
   quantised forward pass produce non-finite logits.
3. **Nobody has seen a single completion yet.** The script never
   printed one. That's now fixed -- it prints the rendered prompt once
   and the first decoded completion of every eval / training step.

What the current script does about it (instrumented to diagnose, not
guess): a plain forward pass over each prompt with an `isfinite` check
*before* `generate()` -- if the base model already produces NaN for an
input, it prints the offending prompt and skips instead of hitting the
CUDA assert; `--batch-size 1` by default so left-padding into the
recurrent Gated DeltaNet layers is off the table as a variable;
`CUDA_LAUNCH_BLOCKING=1` set in `launch_rlvr_qwen38.sh` so any
remaining assert names the real failing op; `--list-modules` to dump
the architecture's real module names on the meta device in seconds;
`--skip-quant-modules` to keep named `nn.Linear` layers (routers,
gates, the sparse-attention indexer, MTP) in bf16 instead of 4-bit;
and `--seed` to check whether the crash follows the inputs.

Suggested order for the next run: (1) run `--list-modules` first
(fast, no GPU needed) and look for router/gate/indexer names; (2) run
the default config and read the printed completions -- if baseline
accuracy is now non-zero, thinking mode was the 0% cause; (3) if a
"NON-FINITE logits ... offending prompt" line appears, that is the
smoking gun -- try the same run with `RLVR_SKIP_QUANT_MODULES` set to
the router/gate names from step 1; (4) if the assert still fires
despite the guard, the synchronous trace (thanks to
`CUDA_LAUNCH_BLOCKING=1`) will name the actual kernel. If skipping
routers/gates doesn't help, the remaining lever is 8-bit
(`load_in_8bit=True`) instead of NF4, memory permitting -- 4-bit
quantisation of an architecture nobody has validated at 4-bit is the
leading suspect for the NaN itself.

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
