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
