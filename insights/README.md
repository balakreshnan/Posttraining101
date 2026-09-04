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

## Nemotron-3.5-Lightning-30B-A3B (BF16) -- pre-run artifacts

- [`nemotron35_modules.txt`](nemotron35_modules.txt): `--list-modules`
  output (meta-device build, no weights) captured before the first hecate
  job (533496) was submitted. What it tells us:
  - The installed `transformers` recognises `nemotron_h` -- no
    architecture-support risk for this model.
  - Every block is one of three mixers: `NemotronHMamba2Mixer`
    (`in_proj` 2688->10304, `conv1d`, `out_proj` 4096->2688),
    `NemotronHMoE` (router `gate` = `NemotronHTopkRouter`; experts are a
    single fused `NemotronHExperts` module -- **not** `nn.Linear`, so
    neither LoRA-by-name nor bitsandbytes would ever touch them; plus a
    dense `shared_experts` MLP `up_proj`/`down_proj`), or
    `NemotronHAttention` (`q_proj` 2688->4096, `k_proj`/`v_proj`
    2688->256 = heavy GQA, `o_proj` 4096->2688).
  - The default LoRA targets `q_proj,k_proj,v_proj,o_proj` therefore
    match only the attention mixers ("select" layers). For more adapter
    capacity the real names to add are the Mamba-2 `in_proj` and/or
    `shared_experts.up_proj,shared_experts.down_proj` (peft matches by
    suffix, so `up_proj,down_proj` also works and hits only the shared
    experts). **Not `out_proj` or `conv1d`**: peft refuses them on
    Mamba-based models (`ValueError: Module 'out_proj' is incompatible
    with Mamba-based models (model_type='nemotron_h')`) because they sit
    on the SSM state path -- job 533662 failed at `get_peft_model` on
    exactly that; the launcher default is now
    `q_proj,k_proj,v_proj,o_proj,in_proj`.
  - `k_proj`/`v_proj` are tiny (out=256); most attention LoRA capacity
    will sit in `q_proj`/`o_proj`.
- [`hecate_nemotron_10step.log`](hecate_nemotron_10step.log) and
  [`hecate_nemotron_10step_dashboard.html`](hecate_nemotron_10step_dashboard.html):
  job 533496, the first Nemotron run on hecate (10 optimizer steps x 4
  single-prompt rollouts, LoRA on attention projections, whole bf16 model
  on one GPU). Clean on every check:
  - `nemotron_h` loaded (401 shards in 32 s); **58.9 GiB on GPU 0**, GPUs
    1-3 empty -- the single-GPU design working as intended.
  - Rendered prompt ends in an **empty `<think></think>`** block: this
    template honours `enable_thinking=False` (that is how "thinking off"
    is rendered), so no system-prompt workaround is needed.
  - `trainable params: 1,867,776 / 31.6B (0.0059%)` -- small because only
    the "select" attention mixers carry LoRA and `k_proj`/`v_proj` are
    tiny (out=256).
  - `Step accounting: updated 10, skipped_logits 0, skipped_grad 0,
    rollbacks 0`; grad norm 0.74 -> 0.05-0.09 as the baseline rose.
  - **Baseline greedy accuracy 100%, final 100%.** The model already
    solves the arithmetic task before any training. The sampled-rollout
    rewards of 0.775 are verbose answers truncated at `--max-new-tokens
    64` (e.g. the step-2 sample is cut off mid-explanation), not wrong
    arithmetic. On this task the only thing RLVR can teach this model is
    brevity -- a formatting signal, not a capability gain. A meaningful
    1000-step run needs a harder verifiable task first (larger operands
    / multi-step expressions in `task.py`, or a real dataset such as
    GSM8K).
