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
full rationale. Start with the tiny defaults (batch 2, 10 iterations,
32 tokens) baked into `launch_rlvr_qwen38.sh` -- this first run is
"does it load and take one step," not a real training run.

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
  - A brand-new architecture (qwen4_exp / Qwen4ExpForConditionalGeneration)
    -- requires a very recent `transformers` (see requirements-qwen38.txt).
    If loading fails with an unrecognized-architecture error, upgrade
    transformers first: pip install -U transformers

Run as a single process (no torchrun/DDP -- device_map="auto" already
spreads the frozen base across every visible GPU on the node):

    python train_rlvr_qwen38.py --iterations 10 --batch-size 2

Defaults are intentionally tiny (batch 2, 10 iterations, 32 new tokens).

Confirmed on real hardware: model load + LoRA attach works (144-file
fetch, weight load, "trainable params: 5.7M / 177.4B" as expected).

Round 1 failure (lr=1e-4, top_k=0, double-quant on): CUDA device-side
assert ("probability tensor contains inf, nan or element < 0") inside
model.generate()'s sampling, 2 steps in. Mitigated with a much lower
lr (5e-6), tighter grad clipping, double-quant off, and the model
card's recommended top_p/top_k instead of unrestricted top_k=0.

Round 2 failure: identical crash, same step, despite the 20x lower lr
-- which ruled out update magnitude as the cause and pointed at two
other bugs, both now fixed:
  1. Baseline accuracy was 0.00% (not even a parseable number) even
     before training. build_prompt() was passing a bare string as
     chat-message content; this is a multimodal processor whose
     documented examples all use structured content blocks
     ([{"type": "text", "text": ...}]). Fixed.
  2. model.config.use_cache=False was applied globally, including
     during generate(). Fine for a standard transformer, but this
     architecture's Gated DeltaNet layers are recurrent/stateful --
     generate() now passes use_cache=True explicitly to override that.
  Also switched lora_dropout 0.05 -> 0.0 and moved the model to a
  permanent eval() mode (never .train()) -- with dropout active, the
  log-probs computed in the training forward pass wouldn't have
  matched the distribution that actually produced the sampled tokens
  during generate(), biasing the REINFORCE gradient regardless of the
  crash. Still not re-verified against hardware.
"""

import argparse
import random

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from task import sample_batch, verify_reward

try:
    from peft import LoraConfig, get_peft_model
except ImportError as e:
    raise SystemExit(
        "peft is required for this script. "
        "pip install -r requirements-qwen38.txt first."
    ) from e


def build_prompt(processor, problem: dict) -> str:
    # Structured content blocks, not a bare string -- this is a multimodal
    # processor and every documented example on the model card uses
    # [{"type": "text", "text": ...}] even for text-only turns. A bare
    # string produced 0.00% baseline accuracy (not even a parseable
    # number), suggesting the template mishandles unstructured content.
    messages = [{"role": "user", "content": [{"type": "text", "text": problem["question"]}]}]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experimental RLVR fine-tuning for Qwen3.8-Flash-Next")
    p.add_argument("--model", default="Qwen/Qwen3.8-Flash-Next")
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument(
        "--lr", type=float, default=5e-6,
        help="Kept conservative (much lower than typical LoRA LRs) -- this base model is "
             "brand-new/exotic under 4-bit quantization and showed NaN logits after just "
             "one update at 1e-4. Only raise this once a run has completed cleanly.",
    )
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.8)
    p.add_argument(
        "--top-k", type=int, default=20,
        help="Qwen3.8-Flash-Next's model card recommends top_k=20 for both thinking and "
             "instruct modes -- top_k=0 (unrestricted) is more likely to sample a token "
             "from the distribution's unstable tail under 4-bit quantization.",
    )
    p.add_argument(
        "--max-grad-norm", type=float, default=0.3,
        help="Tighter than train_rlvr.py's 1.0 -- same rationale as --lr.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-dir", default=None, help="Optional dir to save the LoRA adapter (not the full base model)")
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--eval-n", type=int, default=6, help="Held-out problems for greedy eval -- keep small, this model is slow")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj",
        help="Comma-separated module name suffixes to attach LoRA to. Kept to attention "
             "projections by default -- the 512-expert MoE layers are left frozen; "
             "adapting attention alone is usually enough to shift behavior for RLVR "
             "and keeps adapter size (and load complexity) sane.",
    )
    return p.parse_args()


@torch.no_grad()
def _greedy_eval(model, processor, rng, n, max_new_tokens):
    problems = sample_batch(rng, n)
    correct = 0
    for prob in problems:
        text = build_prompt(processor, prob)
        inputs = processor(text=text, return_tensors="pt").to(model.device)
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,  # config.use_cache=False (set for training) would otherwise
                              # disable it here too -- this architecture's recurrent
                              # Gated DeltaNet layers likely need it for correct generation
            pad_token_id=processor.tokenizer.eos_token_id,
        )
        completion = processor.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        if verify_reward(completion, prob["answer"]) == 1.0:
            correct += 1
    return correct / n


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    if not torch.cuda.is_available():
        raise SystemExit("This script requires CUDA GPUs -- Qwen3.8-Flash-Next is not viable on CPU.")

    print(f"Loading {args.model} in 4-bit, sharded across {torch.cuda.device_count()} visible GPU(s)...")
    print("This is a ~180B param model -- expect this step alone to take a while.")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        # Double quantization adds a second layer of approximation on top of an
        # already-aggressive 4-bit format; disabled to reduce numerical error on
        # this untested-at-4-bit architecture (NaN logits appeared with it on).
        bnb_4bit_use_double_quant=False,
    )

    processor = AutoProcessor.from_pretrained(args.model)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "left"

    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False  # required during training

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[m.strip() for m in args.lora_target_modules.split(",")],
        # 0.0, not the usual 0.05 -- with dropout on, the log-probs computed in the
        # training forward pass below wouldn't match the distribution that actually
        # produced the sampled tokens during generate() (different dropout mask each
        # forward), biasing the REINFORCE gradient. Keeping the model in eval() mode
        # throughout (never calling .train()) makes this moot either way.
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.eval()  # see lora_dropout comment above -- gradients still flow fine in
                  # eval() mode, this only affects dropout/similar stochastic layers

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    print("Baseline accuracy before training:")
    baseline_acc = _greedy_eval(model, processor, rng, n=args.eval_n, max_new_tokens=args.max_new_tokens)
    print(f"  greedy accuracy = {baseline_acc:.2%}")

    running_baseline = 0.0
    for step in range(1, args.iterations + 1):
        problems = sample_batch(rng, args.batch_size)
        prompts = [build_prompt(processor, p) for p in problems]

        enc = processor(text=prompts, return_tensors="pt", padding=True).to(model.device)
        prompt_len = enc["input_ids"].shape[1]

        with torch.no_grad():
            gen_out = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                use_cache=True,  # model.config.use_cache=False (set above, needed for
                                  # the training forward pass immediately below) would
                                  # otherwise disable it here too -- this architecture's
                                  # recurrent Gated DeltaNet layers likely need it for
                                  # correct generation
                pad_token_id=processor.tokenizer.eos_token_id,
            )

        rewards = []
        for i, prob in enumerate(problems):
            completion_ids = gen_out[i][prompt_len:]
            completion_text = processor.tokenizer.decode(completion_ids, skip_special_tokens=True)
            rewards.append(verify_reward(completion_text, prob["answer"]))
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=gen_out.device)

        batch_mean = rewards_t.mean().item()
        advantages = rewards_t - running_baseline
        running_baseline = 0.9 * running_baseline + 0.1 * batch_mean

        attention_mask = (gen_out != processor.tokenizer.pad_token_id).long()
        logits = model(input_ids=gen_out, attention_mask=attention_mask).logits
        if not torch.isfinite(logits).all():
            print(f"step {step:3d}: non-finite logits from the base model forward pass -- "
                  f"skipping this step's update (this indicates the quantized base itself "
                  f"produced NaN/inf, independent of our loss code).")
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

    print("\nAccuracy after training:")
    final_acc = _greedy_eval(model, processor, rng, n=args.eval_n, max_new_tokens=args.max_new_tokens)
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
# launch_rlvr.sh's DDP full-replica training.
set -e

LUSTRE_DIR="/lustre/fsw/general_sa/bbalakreshna"

echo "$(hostname): Installing RLVR + Qwen3.8-Flash-Next dependencies..."
pip install --quiet -r "$LUSTRE_DIR/requirements-qwen38.txt"

export HF_HOME="$LUSTRE_DIR/hf_cache"
mkdir -p "$HF_HOME"

echo "$(hostname): Launching RLVR fine-tuning for Qwen3.8-Flash-Next (single process, device_map=auto)..."
python "$LUSTRE_DIR/train_rlvr_qwen38.py" \
  --model "${RLVR_MODEL:-Qwen/Qwen3.8-Flash-Next}" \
  --iterations "${RLVR_ITERATIONS:-10}" \
  --batch-size "${RLVR_BATCH_SIZE:-2}" \
  --max-new-tokens "${RLVR_MAX_NEW_TOKENS:-32}" \
  --lr "${RLVR_LR:-5e-6}" \
  --eval-n "${RLVR_EVAL_N:-6}" \
  --lora-r "${RLVR_LORA_R:-16}" \
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

## Known risks / what to check if it fails

**Status: confirmed on real hardware, two rounds of the same crash so
far.** Model load, LoRA attach, and 2 training steps ran successfully
both times (144-file fetch, weight load, `trainable params: 5,701,632
|| all params: 177,398,532,208`), then hit an identical CUDA
device-side assert (`probability tensor contains inf, nan or element
< 0`) inside `model.generate()`'s sampling at step 3 -- **both times**,
despite round 2 using a learning rate 20x lower than round 1 (`5e-6` vs
`1e-4`). Once that assert fires the CUDA context is corrupted -- the
job can't recover in-process, only a fresh run with different settings
can avoid it.

That the crash was identical at the same step regardless of a 20x LR
change ruled out update magnitude as the cause and pointed at two
actual bugs (both now fixed, not yet re-verified against hardware):

1. **Baseline accuracy was 0.00%** (not a single parseable number, out
   of 6 eval problems) even before any training happened. Root cause:
   `build_prompt()` was passing a bare string as chat-message content.
   This is a *multimodal* processor -- every documented example on the
   model card uses structured content blocks
   (`[{"type": "text", "text": ...}]`), even for text-only turns. Fixed
   to match.
2. **`model.config.use_cache = False` was applied globally**, including
   during `generate()`. Fine for a standard transformer (it's normally
   only needed to disable KV-cache bookkeeping during the *training*
   forward pass), but this architecture's Gated DeltaNet layers are
   recurrent/stateful -- disabling caching during generation may break
   how that state propagates. `generate()` now passes `use_cache=True`
   explicitly to override the config for that call.

Also switched `lora_dropout` `0.05 -> 0.0` and moved the model to a
permanent `eval()` mode (never `.train()`): with dropout active, the
log-probs computed in the training forward pass wouldn't have matched
the distribution that actually produced the sampled tokens during
`generate()` (different dropout mask each forward pass), biasing the
REINFORCE gradient independent of the crash.

If this specific crash recurs a third time even with all of the above,
the next things to try, in order: (a) drop `--lr` further (e.g. `1e-6`)
just to be certain magnitude truly isn't a factor, (b) switch from
4-bit to 8-bit quantization (`load_in_8bit=True` instead of
`load_in_4bit`) if GPU memory allows -- 8-bit is generally more
numerically stable than NF4 for architectures that haven't been
validated at 4-bit, (c) try `CUDA_LAUNCH_BLOCKING=1` to get a precise
(synchronous) stack trace instead of the async-reported one currently
shown, since the real failing op may not be in `_has_unfinished_sequences`
at all.

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
