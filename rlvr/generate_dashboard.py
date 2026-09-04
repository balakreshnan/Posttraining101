"""Generate a self-contained interactive HTML dashboard from an RLVR run log.

Works for both training scripts:
  - train_rlvr.py        : step N/M | mean_reward=.. | baseline=.. | loss=..
  - train_rlvr_qwen38.py : step N/M | mean_reward=.. | baseline=.. | grad_norm=.. | rollouts=a/b
Step lines are parsed generically as `key=value` pairs, so any numeric
metric present gets a chart. Also picks up (when present): baseline/final
greedy accuracy, per-operator accuracy and reward, the qwen38 script's
guard messages (NON-FINITE gradient norm / logits / parameter rollbacks),
its `Step accounting: {...}` summary, per-GPU memory after load, and the
.timing file.

Standard library only -- runs on hecate's login node with system python3.
Chart.js is loaded from a CDN in the generated HTML, so the *browser*
needs internet; the cluster does not.

Usage:
    python3 generate_dashboard.py out/hecate_qwen38_run1.log \\
        --timing out/hecate_qwen38_run1.timing \\
        --output out/hecate_qwen38_run1_dashboard.html
"""

import argparse
import ast
import html
import json
import re

STEP_RE = re.compile(r"^step\s+(\d+)/(\d+)\s*\|(.*)$", re.MULTILINE)
KV_RE = re.compile(r"(\w+)=(-?[\d.]+(?:/\d+)?)")
ACCURACY_RE = re.compile(r"greedy accuracy\s*=\s*([\d.]+)%")
DEVICE_RE = re.compile(r"Using device:\s*(.+)")
LOAD_RE = re.compile(r"Loading (\S+) \(([^)]*)\), sharded across (\d+) GPU")
GPU_MEM_RE = re.compile(r"GPU (\d+): ([\d.]+) GiB allocated after load")
SAVED_RE = re.compile(r"Saved (?:fine-tuned model|LoRA adapter) to (.+)")
PER_OP_EVAL_RE = re.compile(r"(baseline|final) by operator -- (.+)")
PER_OP_EVAL_ENTRY_RE = re.compile(r"([+\-*]):\s*([\d.]+)%\s*\((\d+)/(\d+)\)")
PER_OP_TRAIN_RE = re.compile(r"^\s*([+\-*]):\s*mean_reward=([\d.]+)\s*over\s*(\d+)\s*samples", re.MULTILINE)
SKIP_GRAD_RE = re.compile(r"^step\s+(\d+): NON-FINITE gradient norm", re.MULTILINE)
SKIP_LOGITS_RE = re.compile(r"^step\s+(\d+): (?:NON-FINITE logits|non-finite logits|non-finite loss)", re.MULTILINE)
ROLLBACK_RE = re.compile(r"^step\s+(\d+): NON-FINITE parameters", re.MULTILINE)
ACCOUNTING_RE = re.compile(r"Step accounting:\s*(\{.*\})")
EVAL_SAMPLE_RE = re.compile(r"eval sample \(gold=(-?\d+), reward=([\d.]+)\): (.+)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate an interactive HTML dashboard for an RLVR run")
    p.add_argument("log", help="Path to the training log")
    p.add_argument("--timing", default=None, help="Path to the matching .timing file")
    p.add_argument("--output", default=None, help="Output HTML path (default: <log>.dashboard.html)")
    p.add_argument("--skip-bin", type=int, default=50, help="Step-bin width for the skipped-steps histogram")
    return p.parse_args()


def _num(v: str):
    if "/" in v:  # rollouts=4/4 -> fraction
        a, b = v.split("/")
        return float(a) / float(b) if float(b) else None
    return float(v)


def parse_log(path: str, skip_bin: int) -> dict:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    steps, total_steps = [], None
    metrics = {}  # name -> list aligned with steps
    for m in STEP_RE.finditer(text):
        step = int(m.group(1))
        total_steps = int(m.group(2))
        kv = {k: _num(v) for k, v in KV_RE.findall(m.group(3))}
        steps.append(step)
        for k in set(metrics) | set(kv):
            metrics.setdefault(k, [None] * (len(steps) - 1)).append(kv.get(k))

    skipped_grad = [int(s) for s in SKIP_GRAD_RE.findall(text)]
    skipped_logits = [int(s) for s in SKIP_LOGITS_RE.findall(text)]
    rollbacks = [int(s) for s in ROLLBACK_RE.findall(text)]

    # Histogram of skipped steps over the run: were the guards firing early, late, throughout?
    hist = None
    if total_steps and (skipped_grad or skipped_logits or rollbacks):
        nbins = max(1, -(-total_steps // skip_bin))
        labels = [f"{i * skip_bin + 1}-{min((i + 1) * skip_bin, total_steps)}" for i in range(nbins)]
        def bucket(lst):
            counts = [0] * nbins
            for s in lst:
                counts[min((s - 1) // skip_bin, nbins - 1)] += 1
            return counts
        hist = {"labels": labels, "grad": bucket(skipped_grad),
                "logits": bucket(skipped_logits), "rollback": bucket(rollbacks)}

    accounting = None
    am = ACCOUNTING_RE.search(text)
    if am:
        try:
            accounting = ast.literal_eval(am.group(1))
        except (ValueError, SyntaxError):
            accounting = None

    device = None
    dm = DEVICE_RE.search(text)
    if dm:
        device = dm.group(1).strip()
    else:
        lm = LOAD_RE.search(text)
        if lm:
            device = f"{lm.group(1)} ({lm.group(2)}) on {lm.group(3)} GPUs"

    gpu_mem = {int(i): float(g) for i, g in GPU_MEM_RE.findall(text)}

    per_op_eval = {}
    for label, rest in PER_OP_EVAL_RE.findall(text):
        per_op_eval[label] = {op: (int(c), int(t)) for op, _, c, t in PER_OP_EVAL_ENTRY_RE.findall(rest)}
    per_op_train = {op: (float(r), int(n)) for op, r, n in PER_OP_TRAIN_RE.findall(text)}

    eval_samples = [{"gold": int(g), "reward": float(r), "text": t.strip()}
                    for g, r, t in EVAL_SAMPLE_RE.findall(text)]

    saved = SAVED_RE.search(text)

    return {
        "steps": steps,
        "total_steps": total_steps,
        "metrics": metrics,
        "accuracies": [float(x) for x in ACCURACY_RE.findall(text)],
        "device": device,
        "gpu_mem": gpu_mem,
        "saved_to": saved.group(1).strip() if saved else None,
        "per_op_eval": per_op_eval,
        "per_op_train": per_op_train,
        "skipped_grad": skipped_grad,
        "skipped_logits": skipped_logits,
        "rollbacks": rollbacks,
        "skip_hist": hist,
        "accounting": accounting,
        "eval_samples": eval_samples,
    }


def parse_timing(path: str) -> dict:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    out = {"raw": text.strip()}
    for key, pat in [("started", r"Job started:\s*(.+)"), ("ended", r"Job ended:\s*(.+)"),
                     ("elapsed_seconds", r"Elapsed seconds:\s*(\d+)"), ("elapsed_hms", r"Elapsed \(h:m:s\):\s*(.+)")]:
        m = re.search(pat, text)
        if m:
            out[key] = m.group(1).strip()
    return out


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>RLVR Run Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  :root { --bg:#0f1117; --panel:#171a24; --border:#262b3a; --text:#e6e8ef; --muted:#9aa1b2;
          --accent:#6ea8fe; --good:#4ade80; --bad:#f87171; --warn:#fbbf24; --purple:#c084fc; }
  * { box-sizing: border-box; }
  body { margin:0; padding:24px; background:var(--bg); color:var(--text);
         font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; }
  h1 { font-size:1.4rem; margin:0 0 4px; }
  .subtitle { color:var(--muted); margin-bottom:24px; font-size:.9rem; }
  .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:12px; margin-bottom:24px; }
  .stat { background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:14px 16px; }
  .stat .label { color:var(--muted); font-size:.78rem; text-transform:uppercase; letter-spacing:.04em; }
  .stat .value { font-size:1.5rem; font-weight:600; margin-top:4px; word-break:break-word; }
  .stat .delta { font-size:.85rem; margin-top:2px; }
  .delta.pos { color:var(--good); } .delta.neg { color:var(--bad); }
  .stat.warn .value { color:var(--warn); } .stat.good .value { color:var(--good); } .stat.bad .value { color:var(--bad); }
  .panels { display:grid; grid-template-columns:1fr; gap:20px; }
  .panel { background:var(--panel); border:1px solid var(--border); border-radius:12px; padding:18px; }
  .panel h2 { font-size:1rem; margin:0 0 6px; }
  .panel p.note { color:var(--muted); font-size:.82rem; margin:0 0 12px; }
  .two-col { display:grid; grid-template-columns:1fr 1fr; gap:20px; }
  @media (max-width:800px) { .two-col { grid-template-columns:1fr; } }
  canvas { max-height:340px; }
  .samples { font-family:ui-monospace,Menlo,Consolas,monospace; font-size:.82rem; }
  .samples div { padding:8px 10px; border-left:3px solid var(--border); margin-bottom:8px; white-space:pre-wrap; }
  .samples .ok { border-left-color:var(--good); } .samples .miss { border-left-color:var(--bad); }
  .samples .tag { color:var(--muted); }
  footer { margin-top:24px; color:var(--muted); font-size:.78rem; }
  code { background:#1f2330; padding:1px 6px; border-radius:4px; }
</style>
</head>
<body>
  <h1>RLVR Run Dashboard</h1>
  <div class="subtitle">__SUBTITLE__</div>
  <div class="grid" id="stat-grid"></div>
  <div class="panels" id="panels"></div>
  <footer>Generated from <code>__LOGPATH__</code> by generate_dashboard.py</footer>

<script>
const DATA = __DATA_JSON__;
Chart.defaults.color = "#9aa1b2";
Chart.defaults.borderColor = "#262b3a";

const statGrid = document.getElementById("stat-grid");
const panels = document.getElementById("panels");

function statCard(label, value, opts = {}) {
  const div = document.createElement("div");
  div.className = "stat" + (opts.cls ? " " + opts.cls : "");
  let deltaHtml = "";
  if (opts.delta !== undefined && opts.delta !== null) {
    const cls = opts.delta >= 0 ? "pos" : "neg";
    deltaHtml = `<div class="delta ${cls}">${opts.delta >= 0 ? "+" : ""}${opts.delta.toFixed(2)} pts</div>`;
  }
  div.innerHTML = `<div class="label">${label}</div><div class="value">${value}</div>${deltaHtml}`;
  statGrid.appendChild(div);
}
function panel(title, note, id, wide = true) {
  const div = document.createElement("div");
  div.className = "panel";
  div.innerHTML = `<h2>${title}</h2>${note ? `<p class="note">${note}</p>` : ""}<canvas id="${id}"></canvas>`;
  return div;
}
function twoCol(a, b) {
  const div = document.createElement("div");
  div.className = "two-col";
  div.appendChild(a); div.appendChild(b);
  return div;
}
const line = (label, data, color, width = 1.5) =>
  ({ label, data, borderColor: color, backgroundColor: "transparent", pointRadius: 0, borderWidth: width, tension: 0.15, spanGaps: true });

// ---- Stat cards ----
if (DATA.device) statCard("Model / device", DATA.device);
statCard("Optimizer steps logged", `${DATA.steps.length}${DATA.total_steps ? " / " + DATA.total_steps : ""}`);
if (DATA.accuracies.length >= 2) {
  const base = DATA.accuracies[0], fin = DATA.accuracies[DATA.accuracies.length - 1];
  statCard("Baseline accuracy", base.toFixed(1) + "%");
  statCard("Final accuracy", fin.toFixed(1) + "%", { delta: fin - base, cls: fin > base ? "good" : "" });
}
if (DATA.timing && DATA.timing.elapsed_hms) statCard("Elapsed", DATA.timing.elapsed_hms);
if (DATA.timing && DATA.timing.elapsed_seconds && DATA.steps.length) {
  statCard("Wall time / step", (DATA.timing.elapsed_seconds / DATA.steps.length).toFixed(1) + " s");
}
const rewards = DATA.metrics.mean_reward || [];
if (rewards.length) {
  const vals = rewards.filter(v => v !== null);
  statCard("Mean reward", (vals.reduce((a, b) => a + b, 0) / vals.length).toFixed(3));
}
if (DATA.accounting) {
  const a = DATA.accounting;
  statCard("Steps updated", a.updated ?? "-", { cls: "good" });
  statCard("Skipped: non-finite grad", a.skipped_grad ?? 0, { cls: (a.skipped_grad ?? 0) > 0 ? "warn" : "" });
  statCard("Skipped: non-finite logits", a.skipped_logits ?? 0, { cls: (a.skipped_logits ?? 0) > 0 ? "warn" : "" });
  statCard("Adapter rollbacks", a.rollbacks ?? 0, { cls: (a.rollbacks ?? 0) > 0 ? "bad" : "good" });
}

// ---- Reward chart (with skipped steps marked) ----
if (rewards.length) {
  const p = panel("Reward & running baseline over training",
    DATA.skipped_grad.length ? `${DATA.skipped_grad.length} optimizer steps were skipped by the gradient guard (non-finite gradient norm) and print no reward line, so they are absent from the x-axis; see "Where the guards fired" below for their distribution.` : null,
    "rewardChart");
  panels.appendChild(p);
  const datasets = [
    line("mean_reward (per step)", rewards, "#6ea8fe"),
    line("running_baseline", DATA.metrics.baseline || [], "#4ade80", 2),
  ];
  new Chart(document.getElementById("rewardChart"), {
    type: "line",
    data: { labels: DATA.steps, datasets },
    options: { responsive: true, interaction: { mode: "index", intersect: false },
              scales: { x: { title: { display: true, text: "optimizer step" } }, y: { min: 0, max: 1.05 } } },
  });
}

// ---- Secondary metric: loss or grad_norm ----
const secondary = DATA.metrics.loss ? ["loss", "#f87171", "Loss over training", null]
                : DATA.metrics.grad_norm ? ["grad_norm", "#c084fc", "Gradient norm (after clipping) over training",
                    "Drops toward 0 when the running baseline catches up with the reward (advantage ≈ 0) -- i.e. the task is solved. Spikes are steps where the policy moved."] : null;
if (secondary) {
  const [key, color, title, note] = secondary;
  panels.appendChild(panel(title, note, "secondaryChart"));
  new Chart(document.getElementById("secondaryChart"), {
    type: "line",
    data: { labels: DATA.steps, datasets: [line(key, DATA.metrics[key], color)] },
    options: { responsive: true, interaction: { mode: "index", intersect: false },
              scales: { x: { title: { display: true, text: "optimizer step" } } } },
  });
}

// ---- Skipped-step histogram ----
if (DATA.skip_hist) {
  const h = DATA.skip_hist;
  panels.appendChild(panel("Where the guards fired",
    "Count of skipped/rolled-back optimizer steps per bin of steps. Early-heavy = instability while the policy was changing fast; spread evenly = a persistent numerical issue; late-heavy = something degrading.",
    "skipChart"));
  new Chart(document.getElementById("skipChart"), {
    type: "bar",
    data: { labels: h.labels, datasets: [
      { label: "non-finite gradient (skipped)", data: h.grad, backgroundColor: "#f87171aa" },
      { label: "non-finite logits/loss (rollout skipped)", data: h.logits, backgroundColor: "#fbbf24aa" },
      { label: "parameter rollback", data: h.rollback, backgroundColor: "#c084fcaa" },
    ] },
    options: { responsive: true, scales: { x: { stacked: true, title: { display: true, text: "step range" } },
                                          y: { stacked: true, beginAtZero: true } } },
  });
}

// ---- Rollouts completed per step (qwen38) ----
if (DATA.metrics.rollouts) {
  panels.appendChild(panel("Fraction of planned rollouts completed per step",
    "1.0 = every accumulated rollout produced finite logits and a usable loss.", "rolloutChart"));
  new Chart(document.getElementById("rolloutChart"), {
    type: "line",
    data: { labels: DATA.steps, datasets: [line("rollouts completed", DATA.metrics.rollouts, "#fbbf24")] },
    options: { responsive: true, interaction: { mode: "index", intersect: false },
              scales: { x: { title: { display: true, text: "optimizer step" } }, y: { min: 0, max: 1.05 } } },
  });
}

// ---- Per-operator (train_rlvr.py) or GPU memory (qwen38) ----
const opOrder = ["+", "-", "*"];
const hasPerOp = Object.keys(DATA.per_op_eval).length || Object.keys(DATA.per_op_train).length;
const hasGpu = Object.keys(DATA.gpu_mem).length;
if (hasPerOp) {
  const a = panel("Accuracy by operator (baseline vs final)", null, "perOpAccChart");
  const b = panel("Training reward by operator (all steps)", null, "perOpRewardChart");
  panels.appendChild(twoCol(a, b));
  const accFor = label => opOrder.map(op => { const e = (DATA.per_op_eval[label] || {})[op]; return e ? 100 * e[0] / e[1] : null; });
  new Chart(document.getElementById("perOpAccChart"), { type: "bar",
    data: { labels: opOrder, datasets: [
      { label: "baseline", data: accFor("baseline"), backgroundColor: "#fbbf24aa" },
      { label: "final", data: accFor("final"), backgroundColor: "#4ade80aa" } ] },
    options: { responsive: true, scales: { y: { min: 0, max: 100, title: { display: true, text: "accuracy %" } } } } });
  new Chart(document.getElementById("perOpRewardChart"), { type: "bar",
    data: { labels: opOrder, datasets: [ { label: "mean training reward",
      data: opOrder.map(op => DATA.per_op_train[op] ? DATA.per_op_train[op][0] : null), backgroundColor: "#6ea8feaa" } ] },
    options: { responsive: true, scales: { y: { min: 0, max: 1.05 } } } });
}
if (hasGpu) {
  panels.appendChild(panel("GPU memory allocated after model load",
    "Balanced bars = device_map spread the shards across all GPUs. One or two tall bars with the rest near zero = the run-4 placement problem.", "gpuChart"));
  const ids = Object.keys(DATA.gpu_mem).sort((a, b) => a - b);
  new Chart(document.getElementById("gpuChart"), { type: "bar",
    data: { labels: ids.map(i => "GPU " + i), datasets: [ { label: "GiB allocated", data: ids.map(i => DATA.gpu_mem[i]), backgroundColor: "#6ea8feaa" } ] },
    options: { responsive: true, scales: { y: { beginAtZero: true, title: { display: true, text: "GiB" } } } } });
}

// ---- Sample completions ----
if (DATA.eval_samples.length) {
  const div = document.createElement("div");
  div.className = "panel";
  const rows = DATA.eval_samples.map((s, i) =>
    `<div class="${s.reward >= 1 ? "ok" : "miss"}"><span class="tag">${i === 0 ? "baseline" : "final"} eval · gold=${s.gold} · reward=${s.reward}</span>\n${s.text.replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]))}</div>`).join("");
  div.innerHTML = `<h2>What the model actually said</h2><p class="note">First eval completion before and after training, as printed in the log.</p><div class="samples">${rows}</div>`;
  panels.appendChild(div);
}
</script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    data = parse_log(args.log, args.skip_bin)
    if not data["steps"]:
        raise SystemExit("No 'step N/M | ...' lines found -- is this a train_rlvr*.py log?")
    data["timing"] = parse_timing(args.timing) if args.timing else None

    output = args.output or (args.log.rsplit(".", 1)[0] + ".dashboard.html")
    subtitle = " | ".join(html.escape(x) for x in [args.log, data["device"] or ""] if x)
    out_html = (HTML_TEMPLATE
                .replace("__SUBTITLE__", subtitle)
                .replace("__LOGPATH__", html.escape(args.log))
                .replace("__DATA_JSON__", json.dumps(data)))
    with open(output, "w", encoding="utf-8") as f:
        f.write(out_html)
    print(f"Wrote dashboard to {output}")
    if data["accounting"]:
        print(f"Step accounting: {data['accounting']}")
    print(f"Steps logged: {len(data['steps'])}  skipped(grad): {len(data['skipped_grad'])}  "
          f"skipped(logits): {len(data['skipped_logits'])}  rollbacks: {len(data['rollbacks'])}")


if __name__ == "__main__":
    main()
