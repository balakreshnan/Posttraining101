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
