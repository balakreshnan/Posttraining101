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
