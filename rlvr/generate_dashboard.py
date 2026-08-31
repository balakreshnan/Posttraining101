"""Generate a self-contained interactive HTML dashboard from an RLVR run.

Parses the same log lines as analyze_run.py (step/reward/loss, baseline/
final accuracy, per-operator breakdowns, .timing) and renders them as an
interactive HTML file: reward/loss curves, before/after accuracy, and
per-operator bar charts, all with hover tooltips and toggleable legends.

Only the Python standard library is used to *generate* the file (no
pip install needed -- runs fine with hecate's system python3). Chart.js
is loaded from a CDN in the generated HTML, so viewing it needs an
internet connection in the browser you open it in (not on the cluster).

Usage:
    python3 generate_dashboard.py out/hecate_run1.log \\
        --timing out/hecate_run1.timing \\
        --output out/hecate_run1_dashboard.html
"""

import argparse
import html
import json
import re

STEP_RE = re.compile(
    r"step\s+(\d+)/(\d+)\s*\|\s*mean_reward=([\d.]+)\s*\|\s*baseline=([\d.]+)\s*\|\s*loss=(-?[\d.]+)"
)
ACCURACY_RE = re.compile(r"greedy accuracy\s*=\s*([\d.]+)%")
DEVICE_RE = re.compile(r"Using device:\s*(.+)")
SAVED_RE = re.compile(r"Saved fine-tuned model to (.+)")
PER_OP_EVAL_RE = re.compile(r"(baseline|final) by operator -- (.+)")
PER_OP_EVAL_ENTRY_RE = re.compile(r"([+\-*]):\s*([\d.]+)%\s*\((\d+)/(\d+)\)")
PER_OP_TRAIN_RE = re.compile(r"^\s*([+\-*]):\s*mean_reward=([\d.]+)\s*over\s*(\d+)\s*samples", re.MULTILINE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate an interactive HTML dashboard for an RLVR run")
    p.add_argument("log", help="Path to the training log (e.g. out/hecate_run1.log)")
    p.add_argument("--timing", default=None, help="Path to the matching .timing file")
    p.add_argument("--output", default=None, help="Output HTML path (default: <log>.dashboard.html)")
    return p.parse_args()


def parse_log(log_path: str) -> dict:
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
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

    per_op_eval = {}  # label -> {op: (correct, total)}
    for label, rest in PER_OP_EVAL_RE.findall(text):
        per_op_eval[label] = {
            op: (int(c), int(t)) for op, pct, c, t in PER_OP_EVAL_ENTRY_RE.findall(rest)
        }

    per_op_train = {
        op: (float(mean_reward), int(count))
        for op, mean_reward, count in PER_OP_TRAIN_RE.findall(text)
    }

    return {
        "steps": steps,
        "rewards": rewards,
        "baselines": baselines,
        "losses": losses,
        "accuracies": accuracies,
        "device": device_match.group(1).strip() if device_match else None,
        "saved_to": saved_match.group(1).strip() if saved_match else None,
        "per_op_eval": per_op_eval,
        "per_op_train": per_op_train,
    }


def parse_timing(timing_path: str) -> dict:
    with open(timing_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    result = {"raw": text.strip()}
    for key, pattern in [
        ("started", r"Job started:\s*(.+)"),
        ("ended", r"Job ended:\s*(.+)"),
        ("elapsed_seconds", r"Elapsed seconds:\s*(\d+)"),
        ("elapsed_hms", r"Elapsed \(h:m:s\):\s*(.+)"),
    ]:
        m = re.search(pattern, text)
        if m:
            result[key] = m.group(1).strip()
    return result


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>RLVR Run Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  :root {
    --bg: #0f1117; --panel: #171a24; --border: #262b3a;
    --text: #e6e8ef; --muted: #9aa1b2;
    --accent: #6ea8fe; --good: #4ade80; --bad: #f87171; --warn: #fbbf24;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 24px; background: var(--bg); color: var(--text);
    font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
  }
  h1 { font-size: 1.4rem; margin: 0 0 4px; }
  .subtitle { color: var(--muted); margin-bottom: 24px; font-size: 0.9rem; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin-bottom: 24px; }
  .stat { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; }
  .stat .label { color: var(--muted); font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em; }
  .stat .value { font-size: 1.6rem; font-weight: 600; margin-top: 4px; }
  .stat .delta { font-size: 0.85rem; margin-top: 2px; }
  .delta.pos { color: var(--good); }
  .delta.neg { color: var(--bad); }
  .panels { display: grid; grid-template-columns: 1fr; gap: 20px; }
  .panel { background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 18px; }
  .panel h2 { font-size: 1rem; margin: 0 0 12px; color: var(--text); }
  .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  @media (max-width: 800px) { .two-col { grid-template-columns: 1fr; } }
  canvas { max-height: 340px; }
  footer { margin-top: 24px; color: var(--muted); font-size: 0.78rem; }
  code { background: #1f2330; padding: 1px 6px; border-radius: 4px; }
</style>
</head>
<body>
  <h1>RLVR Run Dashboard</h1>
  <div class="subtitle">__SUBTITLE__</div>

  <div class="grid" id="stat-grid"></div>

  <div class="panels">
    <div class="panel">
      <h2>Reward &amp; running baseline over training</h2>
      <canvas id="rewardChart"></canvas>
    </div>
    <div class="panel">
      <h2>Loss over training</h2>
      <canvas id="lossChart"></canvas>
    </div>
    <div class="two-col">
      <div class="panel">
        <h2>Accuracy by operator (baseline vs final)</h2>
        <canvas id="perOpAccChart"></canvas>
      </div>
      <div class="panel">
        <h2>Training reward by operator (all steps)</h2>
        <canvas id="perOpRewardChart"></canvas>
      </div>
    </div>
  </div>

  <footer>Generated from <code>__LOGPATH__</code> by generate_dashboard.py</footer>

<script>
const DATA = __DATA_JSON__;

Chart.defaults.color = "#9aa1b2";
Chart.defaults.borderColor = "#262b3a";

// -- Stat cards --
const statGrid = document.getElementById("stat-grid");
function statCard(label, value, delta) {
  const div = document.createElement("div");
  div.className = "stat";
  let deltaHtml = "";
  if (delta !== undefined && delta !== null) {
    const cls = delta >= 0 ? "pos" : "neg";
    const sign = delta >= 0 ? "+" : "";
    deltaHtml = `<div class="delta ${cls}">${sign}${delta.toFixed(2)} pts</div>`;
  }
  div.innerHTML = `<div class="label">${label}</div><div class="value">${value}</div>${deltaHtml}`;
  statGrid.appendChild(div);
}

if (DATA.device) statCard("Device", DATA.device);
statCard("Total steps", DATA.steps.length);
if (DATA.accuracies.length >= 2) {
  const base = DATA.accuracies[0], fin = DATA.accuracies[DATA.accuracies.length - 1];
  statCard("Baseline accuracy", base.toFixed(1) + "%");
  statCard("Final accuracy", fin.toFixed(1) + "%", fin - base);
}
if (DATA.timing && DATA.timing.elapsed_hms) {
  statCard("Elapsed", DATA.timing.elapsed_hms);
}
if (DATA.timing && DATA.timing.elapsed_seconds && DATA.steps.length) {
  const perStep = DATA.timing.elapsed_seconds / DATA.steps.length;
  statCard("Throughput", perStep.toFixed(2) + " sec/step");
}
if (DATA.rewards.length) {
  const mean = DATA.rewards.reduce((a, b) => a + b, 0) / DATA.rewards.length;
  statCard("Mean reward", mean.toFixed(3));
}

// -- Reward chart --
new Chart(document.getElementById("rewardChart"), {
  type: "line",
  data: {
    labels: DATA.steps,
    datasets: [
      { label: "mean_reward (per step)", data: DATA.rewards, borderColor: "#6ea8fe", backgroundColor: "transparent", pointRadius: 0, borderWidth: 1.5, tension: 0.15 },
      { label: "running_baseline", data: DATA.baselines, borderColor: "#4ade80", backgroundColor: "transparent", pointRadius: 0, borderWidth: 2, tension: 0.15 },
    ],
  },
  options: {
    responsive: true,
    interaction: { mode: "index", intersect: false },
    scales: { x: { title: { display: true, text: "step" } }, y: { min: 0, max: 1.05 } },
  },
});

// -- Loss chart --
new Chart(document.getElementById("lossChart"), {
  type: "line",
  data: {
    labels: DATA.steps,
    datasets: [
      { label: "loss", data: DATA.losses, borderColor: "#f87171", backgroundColor: "transparent", pointRadius: 0, borderWidth: 1.5, tension: 0.15 },
    ],
  },
  options: {
    responsive: true,
    interaction: { mode: "index", intersect: false },
    scales: { x: { title: { display: true, text: "step" } } },
  },
});

// -- Per-operator accuracy --
const opOrder = ["+", "-", "*"];
const perOpEval = DATA.per_op_eval || {};
function accFor(label) {
  return opOrder.map((op) => {
    const entry = (perOpEval[label] || {})[op];
    return entry ? (100 * entry[0] / entry[1]) : null;
  });
}
new Chart(document.getElementById("perOpAccChart"), {
  type: "bar",
  data: {
    labels: opOrder,
    datasets: [
      { label: "baseline", data: accFor("baseline"), backgroundColor: "#fbbf24aa" },
      { label: "final", data: accFor("final"), backgroundColor: "#4ade80aa" },
    ],
  },
  options: {
    responsive: true,
    scales: { y: { min: 0, max: 100, title: { display: true, text: "accuracy %" } } },
  },
});

// -- Per-operator training reward --
const perOpTrain = DATA.per_op_train || {};
new Chart(document.getElementById("perOpRewardChart"), {
  type: "bar",
  data: {
    labels: opOrder,
    datasets: [
      {
        label: "mean training reward",
        data: opOrder.map((op) => (perOpTrain[op] ? perOpTrain[op][0] : null)),
        backgroundColor: "#6ea8feaa",
      },
    ],
  },
  options: {
    responsive: true,
    plugins: {
      tooltip: {
        callbacks: {
          afterLabel: (ctx) => {
            const op = opOrder[ctx.dataIndex];
            const entry = perOpTrain[op];
            return entry ? `n = ${entry[1]} samples` : "";
          },
        },
      },
    },
    scales: { y: { min: 0, max: 1.05, title: { display: true, text: "mean reward" } } },
  },
});
</script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    parsed = parse_log(args.log)

    if not parsed["steps"]:
        raise SystemExit("No 'step N/M | mean_reward=...' lines found -- is this a train_rlvr.py log?")

    data = dict(parsed)
    data["timing"] = parse_timing(args.timing) if args.timing else None

    output_path = args.output or (args.log.rsplit(".", 1)[0] + ".dashboard.html")

    subtitle_bits = [html.escape(args.log)]
    if data["device"]:
        subtitle_bits.append(html.escape(data["device"]))
    subtitle = " | ".join(subtitle_bits)

    out_html = (
        HTML_TEMPLATE
        .replace("__SUBTITLE__", subtitle)
        .replace("__LOGPATH__", html.escape(args.log))
        .replace("__DATA_JSON__", json.dumps(data))
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(out_html)

    print(f"Wrote dashboard to {output_path}")


if __name__ == "__main__":
    main()
