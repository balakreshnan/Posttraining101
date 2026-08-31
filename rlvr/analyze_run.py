"""Summarize an RLVR training run from its log (and optional .timing file).

Parses the step-by-step `step N/M | mean_reward=... | baseline=... | loss=...`
lines plus the baseline/final greedy-eval accuracy lines that train_rlvr.py
prints, and reports the key numbers you'd otherwise have to eyeball out of
the raw log: reward trend, loss trend, accuracy delta, and (if a .timing
file is given) wall-clock time and derived throughput.

Usage:
    python analyze_run.py out/hecate_run1.log
    python analyze_run.py out/hecate_run1.log --timing out/hecate_run1.timing
    python analyze_run.py out/hecate_run1.log --plot out/hecate_run1.png
"""

import argparse
import re
import statistics

STEP_RE = re.compile(
    r"step\s+(\d+)/(\d+)\s*\|\s*mean_reward=([\d.]+)\s*\|\s*baseline=([\d.]+)\s*\|\s*loss=(-?[\d.]+)"
)
ACCURACY_RE = re.compile(r"greedy accuracy\s*=\s*([\d.]+)%")
DEVICE_RE = re.compile(r"Using device:\s*(.+)")
SAVED_RE = re.compile(r"Saved fine-tuned model to (.+)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize an RLVR training run")
    p.add_argument("log", help="Path to the training log (e.g. out/hecate_run1.log)")
    p.add_argument("--timing", default=None, help="Path to the matching .timing file")
    p.add_argument("--plot", default=None, help="Optional path to save a reward/loss curve PNG (requires matplotlib)")
    p.add_argument("--head-frac", type=float, default=0.1, help="Fraction of steps counted as 'early' vs 'late' for trend comparison")
    return p.parse_args()


def _fmt_pct(x: float) -> str:
    return f"{x:.2f}%"


def main() -> None:
    args = parse_args()

    with open(args.log, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    steps, rewards, baselines, losses = [], [], [], []
    for m in STEP_RE.finditer(text):
        steps.append(int(m.group(1)))
        rewards.append(float(m.group(3)))
        baselines.append(float(m.group(4)))
        losses.append(float(m.group(5)))

    accuracies = [float(x) for x in ACCURACY_RE.findall(text)]
    device_match = DEVICE_RE.search(text)
    saved_match = SAVED_RE.search(text)

    if not steps:
        print("No 'step N/M | mean_reward=...' lines found -- is this a train_rlvr.py log?")
        return

    n = len(steps)
    head_n = max(1, int(n * args.head_frac))
    tail_n = max(1, int(n * args.head_frac))

    print("=" * 60)
    print(f"RLVR run summary: {args.log}")
    print("=" * 60)

    if device_match:
        print(f"Device:            {device_match.group(1)}")
    print(f"Total steps:       {n} (log claims {steps[-1]}/{STEP_RE.search(text).group(2)} at first match)")

    print()
    print("-- Reward --")
    print(f"  Mean (all steps): {statistics.mean(rewards):.3f}")
    print(f"  Early avg (first {head_n}): {statistics.mean(rewards[:head_n]):.3f}")
    print(f"  Late avg  (last {tail_n}):  {statistics.mean(rewards[-tail_n:]):.3f}")
    print(f"  Max / Min:         {max(rewards):.3f} / {min(rewards):.3f}")
    print(f"  Running baseline:  {baselines[0]:.3f} -> {baselines[-1]:.3f}")

    print()
    print("-- Loss --")
    print(f"  Early avg (first {head_n}): {statistics.mean(losses[:head_n]):.4f}")
    print(f"  Late avg  (last {tail_n}):  {statistics.mean(losses[-tail_n:]):.4f}")
    print(f"  Std dev (all steps): {statistics.pstdev(losses):.4f}")

    if len(accuracies) >= 2:
        baseline_acc, final_acc = accuracies[0], accuracies[-1]
        print()
        print("-- Greedy-eval accuracy --")
        print(f"  Baseline: {_fmt_pct(baseline_acc)}")
        print(f"  Final:    {_fmt_pct(final_acc)}")
        print(f"  Delta:    {'+' if final_acc >= baseline_acc else ''}{final_acc - baseline_acc:.2f} pts")
    elif accuracies:
        print()
        print(f"-- Greedy-eval accuracy -- only one reading found: {_fmt_pct(accuracies[0])} (run may be incomplete)")

    if saved_match:
        print()
        print(f"Checkpoint saved to: {saved_match.group(1).strip()}")

    if args.timing:
        with open(args.timing, "r", encoding="utf-8", errors="replace") as f:
            timing_text = f.read()
        print()
        print("-- Timing --")
        print(timing_text.strip())
        elapsed_m = re.search(r"Elapsed seconds:\s*(\d+)", timing_text)
        if elapsed_m:
            elapsed_s = int(elapsed_m.group(1))
            print(f"  Throughput: {elapsed_s / n:.2f} sec/step ({n} steps)")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("\n(--plot requested but matplotlib isn't installed: pip install matplotlib)")
        else:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
            ax1.plot(steps, rewards, label="mean_reward", alpha=0.6)
            ax1.plot(steps, baselines, label="running_baseline", linewidth=2)
            ax1.set_ylabel("reward")
            ax1.legend()
            ax1.set_title(f"RLVR training curve -- {args.log}")

            ax2.plot(steps, losses, color="tab:red", alpha=0.7)
            ax2.set_ylabel("loss")
            ax2.set_xlabel("step")

            fig.tight_layout()
            fig.savefig(args.plot, dpi=150)
            print(f"\nSaved plot to {args.plot}")


if __name__ == "__main__":
    main()
