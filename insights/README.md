# Run insights

Interactive HTML dashboards generated from training logs with
[`rlvr/generate_dashboard.py`](../rlvr/generate_dashboard.py). Each file is
self-contained (the run's data is embedded as JSON); Chart.js is loaded from
a CDN, so open the file in a browser with internet access. GitHub does not
render HTML in-repo -- download the file (or use the "Raw" view via an HTML
previewer) to see the charts.

| File | Run | Key numbers |
|---|---|---|
| [`hecate_qwen38_run1_dashboard.html`](hecate_qwen38_run1_dashboard.html) | Qwen3.8-Flash-Next, LoRA on attention projections, bf16 base sharded across a 4-GPU Vera Rubin node, 1000 optimizer steps x 4 accumulated single-prompt rollouts (`train_rlvr_qwen38.py`, hecate) | Greedy accuracy **20% -> 100%**. Step accounting: 689 updated, **311 skipped for non-finite gradient**, 0 skipped for non-finite logits, **0 adapter rollbacks**. Wall time 05:17:08 incl. queue wait. |

## Reading the Qwen3.8 dashboard

- **Skipped: non-finite grad = 311** is the headline. This is the exact
  mechanism that silently destroyed run 4 (a NaN gradient -> `inf x 0`
  in gradient clipping -> NaN written into the adapter). The gradient
  guard caught it 311 times and **never once** let it reach the weights
  (`rollbacks = 0`). The "Where the guards fired" histogram shows whether
  those steps cluster early (policy changing fast) or are spread evenly
  (a persistent bf16 numerical quirk of this architecture that the guard
  simply absorbs).
- **Gradient norm -> 0** late in the run is the signature of a *solved*
  task: reward is 1.0 every step, so the running baseline reaches 1.0,
  the advantage is zero, and there is no gradient left to take.
- **Skipped: non-finite logits = 0** -- the bf16 base model never
  produced NaN on any input, confirming the earlier NaN-on-everything
  behaviour was entirely the poisoned adapter.
- **GPU memory after load: `{0: 194 GiB, 1: 136 GiB, 2: 0, 3: 0}`.** The
  intent was to spread shards across all four GPUs, but the whole bf16
  model (~330 GiB) fit within the `--max-memory-per-gpu 200GiB` cap on
  the first two, so `device_map="auto"` never reached GPUs 2-3. To
  actually use all four, lower the cap (e.g. `RLVR_MAX_MEMORY_PER_GPU=100GiB`)
  so accelerate is forced to spill onto every GPU. Compute utilisation
  is sequential-pipeline regardless (one GPU active at a time), so this
  matters for headroom and larger `--grad-accum`, not for speed.
- Caveat shared with the 1.5B runs: eval problems are drawn from the same
  generator as training (small multiplication space), so 100% is partly
  in-distribution mastery rather than a strict held-out result.
