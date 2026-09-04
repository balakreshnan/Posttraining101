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
