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
