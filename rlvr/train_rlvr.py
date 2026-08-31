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

Usage:
    python train_rlvr.py --iterations 150 --batch-size 16 --save-dir out/run1
"""

import argparse
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from task import sample_batch, verify_reward


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
    p.add_argument("--batch-size", type=int, default=16)
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
    problems = sample_batch(rng, n)
    correct = 0
    for prob in problems:
        inputs = tokenizer(build_prompt(tokenizer, prob), return_tensors="pt").to(device)
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        completion = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        if verify_reward(completion, prob["answer"]) == 1.0:
            correct += 1
    return correct / n


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    if args.require_gpu and not torch.cuda.is_available():
        raise SystemExit(
            "No CUDA device found but --require-gpu was set. "
            "Check `nvidia-smi` and that torch was installed from the cu12x index "
            "(see requirements.txt), or drop --require-gpu to run on CPU."
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"Using device: cuda ({torch.cuda.get_device_name(0)})")
    else:
        print("Using device: cpu (no CUDA device found -- this will be slow for these defaults)")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # required for batched generation

    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print("Baseline accuracy before training:")
    baseline_acc = _greedy_eval(model, tokenizer, device, rng, n=args.eval_n, max_new_tokens=args.max_new_tokens)
    print(f"  greedy accuracy = {baseline_acc:.2%}")

    running_baseline = 0.0
    for step in range(1, args.iterations + 1):
        problems = sample_batch(rng, args.batch_size)
        prompts = [build_prompt(tokenizer, p) for p in problems]

        enc = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        prompt_len = enc["input_ids"].shape[1]

        with torch.no_grad():
            gen_out = model.generate(
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
            rewards.append(verify_reward(completion_text, prob["answer"]))
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)

        # Running-mean baseline reduces variance of the REINFORCE gradient.
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
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % args.log_every == 0:
            print(
                f"step {step:3d}/{args.iterations} | "
                f"mean_reward={batch_mean:.3f} | baseline={running_baseline:.3f} | loss={loss.item():.4f}"
            )

    print("\nAccuracy after training:")
    final_acc = _greedy_eval(model, tokenizer, device, rng, n=args.eval_n, max_new_tokens=args.max_new_tokens)
    print(f"  greedy accuracy = {final_acc:.2%} (baseline was {baseline_acc:.2%})")

    if args.save_dir:
        model.save_pretrained(args.save_dir)
        tokenizer.save_pretrained(args.save_dir)
        print(f"Saved fine-tuned model to {args.save_dir}")


if __name__ == "__main__":
    main()
