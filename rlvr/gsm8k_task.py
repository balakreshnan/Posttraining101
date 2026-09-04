"""GSM8K as a verifiable-reward task for RLVR.

Drop-in alternative to task.py's synthetic arithmetic. Same interface the
training scripts use -- problems are dicts with "question" and "answer",
and a reward function scores a completion against the gold answer -- but:

  - Problems come from the real dataset (openai/gsm8k, config "main"):
    grade-school word problems whose gold answers are exact numbers.
  - TRAIN rollouts sample the 7,473-problem train split; EVAL samples the
    1,319-problem TEST split. That makes the eval a genuine held-out
    measurement, unlike task.py where eval draws from the same generator
    as training.
  - Gold answers are parsed from the "#### <number>" line of the dataset's
    solution text (commas stripped; all GSM8K golds are integers).
  - Completions are normalised (commas and $ removed) before the same
    "Final Answer: <number>"-first / last-number-fallback extraction as
    task.py, and compared numerically so "18", "18.0" and "18.00" match.
    Rewards: 1.0 correct, 0.1 for any parseable number (format credit),
    0.0 otherwise.

The dataset is fetched via `datasets` on first use and cached under
$HF_HOME (hecate: $LUSTRE_DIR/hf_cache). To pre-fetch:
    hf download openai/gsm8k --repo-type dataset
"""

import re

_GOLD_RE = re.compile(r"####\s*(-?[\d,]*\.?\d+)")
_FINAL_ANSWER_RE = re.compile(r"Final Answer:\s*(-?\d*\.?\d+)", re.IGNORECASE)
_NUMBER_RE = re.compile(r"-?\d*\.?\d+")

INSTRUCTION = (
    " Solve this step by step, then end your reply with exactly one line "
    "in the form 'Final Answer: <number>'."
)


def _to_number(s: str):
    s = s.replace(",", "").replace("$", "").strip().rstrip(".")
    if not s:
        return None
    try:
        f = float(s)
    except ValueError:
        return None
    return int(f) if f == int(f) else f


def parse_gold(solution_text: str):
    m = _GOLD_RE.search(solution_text)
    if not m:
        return None
    return _to_number(m.group(1))


def extract_answer(completion_text: str):
    text = completion_text.replace(",", "").replace("$", "")
    m = _FINAL_ANSWER_RE.search(text)
    if m:
        return _to_number(m.group(1))
    nums = _NUMBER_RE.findall(text)
    return _to_number(nums[-1]) if nums else None


def verify_reward(completion_text: str, gold_answer) -> float:
    predicted = extract_answer(completion_text)
    if predicted is None:
        return 0.0
    try:
        if abs(float(predicted) - float(gold_answer)) < 1e-6:
            return 1.0
    except (TypeError, ValueError):
        return 0.0
    return 0.1


class GSM8K:
    """Holds both splits in memory (a few MB) and samples from them."""

    def __init__(self, train_split: str = "train", eval_split: str = "test"):
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise SystemExit("`datasets` is required for --task gsm8k (pip install datasets)") from e

        ds = load_dataset("openai/gsm8k", "main")
        self.train = self._prepare(ds[train_split])
        self.eval = self._prepare(ds[eval_split])
        self.train_split, self.eval_split = train_split, eval_split

    @staticmethod
    def _prepare(split):
        problems = []
        for row in split:
            gold = parse_gold(row["answer"])
            if gold is None:
                continue  # a handful of malformed rows, if any
            problems.append({"question": row["question"].strip() + INSTRUCTION, "answer": gold})
        return problems

    def describe(self) -> str:
        return (f"GSM8K: {len(self.train)} train problems ({self.train_split} split) for rollouts, "
                f"{len(self.eval)} held-out problems ({self.eval_split} split) for eval")

    def sample_train(self, rng, n: int) -> list[dict]:
        return [rng.choice(self.train) for _ in range(n)]

    def sample_eval(self, rng, n: int) -> list[dict]:
        # Without replacement so the eval set is n distinct problems.
        return rng.sample(self.eval, min(n, len(self.eval)))
