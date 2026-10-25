# The roster: what we benchmark, and how the numbers stay comparable

2026-10-24 · Apple M4 (10-core, 4P+6E), 16 GB unified, macOS 27.0.1 ·
`mlx-lm` 0.32.2, `mlx` 0.32.4 · text-only, in-process and `mlx_lm.server`

`registry.py` is the machine-readable version of this file. If the two disagree,
`registry.py` is what ran and this file is a bug.

## Why a roster

One model evaluated deeply answers "is this file good on this machine". It
cannot answer "which of the things that fit in 16 GB should I run", which is the
question anyone with this laptop actually has. So the Nemotron 3.5 Bolt record
stays as the baseline — same prompts, same harness, same provenance — and three
rivals are run through the identical pipeline, one model at a time, to completion.

## Roster and verified metadata

Every size below was read from the Hub on 2026-10-24 (`api/models/<repo>?blobs=true`
for file bytes and parameter counts, `<repo>/resolve/main/config.json` for
architecture), not from a model card or a blog post. `bpw` is bytes × 8 ÷
parameters over the tensors the engine actually loads.

| Model | Params (total / active) | Tier | Checkpoint | Weights | bpw | Publisher | Engine arm |
|---|---|---|---|---|---|---|---|
| **Nemotron 3.5 Bolt E4B** | 7.46B / 4B label | 4-bit | `rapid-mlx/…-MLX-4bit` | 4.20 GB | 4.50 | third-party | mlx_lm |
| | | OptiQ | `mlx-community/…-OptiQ-4bit` | 6.53 GB (7.49 dl) | 7.00 | mlx-community | mlx_lm + optiq |
| | | 8-bit | `rapid-mlx/…-MLX-8bit` | 7.96 GB | 8.53 | third-party | mlx_lm |
| **MiniCPM5-2B** | 2.52B / 2.52B | 4-bit | `openbmb/MiniCPM5-2B-MLX` | 1.43 GB | 4.55 | **vendor** | mlx_lm |
| | | OptiQ | `mlx-community/…-OptiQ-4bit` | 1.94 GB | 6.17 | mlx-community | mlx_lm + optiq |
| | | 8-bit | `mlx-community/MiniCPM5-2B-8bit` | 2.68 GB | 8.52 | mlx-community | mlx_lm |
| **Spark-X2.5-4B** | 4.11B / 4.11B | 4-bit | `abenzerps/Spark-X2.5-4B-MLX-4bit` | 2.32 GB | 4.51 | **third-party** | mlx_lm |
| | | OptiQ | `mlx-community/…-OptiQ-4bit` | 3.05 GB | 5.93 | mlx-community | mlx_lm + optiq |
| | | 8-bit | `abenzerps/Spark-X2.5-4B-MLX-8bit` | 4.38 GB | 8.52 | **third-party** | mlx_lm |
| **LFM2.5-8B-A1B** | 8.47B / ~1.3B label | 4-bit | `LiquidAI/LFM2.5-8B-A1B-MLX-4bit` | 4.85 GB | 4.58 | **vendor** | mlx_lm |
| | | OptiQ | `mlx-community/…-OptiQ-4bit` | 5.47 GB | 5.17 | mlx-community | mlx_lm + optiq |
| | | 8-bit | `LiquidAI/LFM2.5-8B-A1B-MLX-8bit` | 9.02 GB | 8.52 | **vendor** | mlx_lm |

### Why these three tiers, and not others

Every model is run at 4-bit, OptiQ mixed 4/8-bit, and 8-bit. That set is chosen because it is
the largest matrix that **all four models actually publish**, which is the only way the rows
stay comparable.

- **8-bit is a reference tier, not a recommendation.** It answers "how much is quantization
  costing this model" — the one question a 4-bit-only roster cannot answer. The baseline is
  re-run at 8-bit for this reason, even though its original study did not.
- **The baseline's 8-bit omission is a finding about the original study**, not a gap in the
  model: `rapid-mlx` publishes the file, it is 7.96 GB, and it fits.
- **`openbmb` publishes its own MLX 4-bit** for MiniCPM5, so the plain tier uses the vendor
  conversion the way LFM does, rather than a community quant. `mlx-community`'s 4-bit of the
  same weights is a free second opinion if the two ever disagree.
- **Excluded on purpose:** 5/6-bit tiers (published for LFM and several community repos, but
  not for every model, so they would break the even matrix); `DSpark` variants (published for
  MiniCPM5 and LFM, but not for the baseline — and a speculative-decoding checkpoint changes
  decode semantics, not just quality); GGUF (different engine, explicit non-goal); bf16
  (30 GB+ for LFM, does not fit, and answers nothing the 8-bit tier does not).

Architecture, in the terms that change what the numbers mean:

| Model | Architecture | Why it might win | What to watch |
|---|---|---|---|
| Nemotron 3.5 Bolt | dense GQA, 42 layers, Gemma-4 derived | the baseline; agentic post-training | 7.46B params at 4.5 bpw is still the largest dense read here |
| MiniCPM5-2B | dense Llama, 42 layers, 16 heads / **2 KV heads**, 131k vocab | intelligence per GB; tiny KV cache | 2 KV heads means long context is cheap — check 8k prefill against the 7.5× larger model |
| Spark-X2.5-4B | dense, **3:1 sliding : full attention**, head_dim 256, gated attn out, 1M trained ctx | coding / tool use at 4B | sliding attention should bend prefill tps — if it doesn't, that is the finding |
| LFM2.5-8B-A1B | **hybrid conv + attention MoE**, 18 of 24 layers conv, 32 experts top-4 | decode speed: 8.5B of weights, ~1.3B active | conv layers change prefill scaling entirely; 4.85 GB is the biggest download here |

Each model has a plain 4-bit checkpoint and an OptiQ mixed 4/8-bit checkpoint,
which is exactly the pairing the Nemotron report already measures. So the same
two questions — engine cost, and checkpoint cost — get asked of every model, and
the report gains a third: which checkpoint does this model wear best.

## Provenance, and the numbers that circulated

The three candidates were proposed with sizes attached. Two of those sizes were
for files this repo cannot use, and the baseline's parameter count was off by a
factor of ~1.9. Nothing here is a correction for its own sake — a wrong size in
the roster becomes a wrong recommendation in the table.

- **Spark-X2.5-4B ~2.6 GB GGUF** — that is a GGUF. GGUF is an explicit non-goal
  (DESIGN.md): it needs llama.cpp, a second engine and a second set of controls.
  In MLX this model is 2.32 GB plain / 3.05 GB OptiQ.
- **LFM2.5-8B-A1B ~5.3 GB at 4k context** — the weights are 4.85 GB, so 5.3 GB
  was a runtime figure, not a file size. Runtime memory is measured here, not
  quoted; the P2 peak numbers replace it.
- **Nemotron 3.5 Bolt "E4B"** — 7,463,013,376 parameters, not 4B. "E4B" is the
  effective/active label, the same convention as Gemma 3n. The 4.20 GB file is
  4.50 bpw against 7.46B, which is coherent; against 4.0B it would be 8.4 bpw,
  which is impossible for a 4-bit file. Every "efficiency" ratio that divides by
  parameter count has to use 7.46B.
- **OptiQ 7.49 GB** — 0.96 GB of that is `optiq_vision.safetensors`, which
  `mlx_lm` never loads. Live weights are 6.53 GB, which is what the P0/P2 peaks
  show. Download size and loaded size are different columns from here on.
- **Spark plain 4-bit is a third-party conversion** (`abenzerps`). No vendor or
  mlx-community plain-4-bit MLX repo exists. It is config-compatible with the
  confirmed 4.11B BF16 weights, and that is the whole of its vetting. Any table
  showing it must say so, or the OptiQ-vs-plain delta becomes a comparison of two
  different people's quantization choices.

## Template controls, per model

Every one of these was read from `chat_template.jinja` at the pinned repo. This
is the single biggest source of accidentally measuring the wrong thing.

| Model | Kwarg | Default if omitted | What it costs / breaks |
|---|---|---|---|
| Nemotron | `conversational_mode` | false | +27 prompt tokens of scaffolding; the primary table pins it true |
| Nemotron | `enable_thinking` | template false, **transformers injects true** | emits a `<\|channel>thought` trace; prompt 34 tok instead of 27 |
| MiniCPM5 | `enable_thinking` | branch guarded by `is defined` | the "undefined" path is not the same as false — pin it and compare prompt tokens |
| Spark-X2.5 | `enable_thinking` | **true** | thinking on unless passed false; changes what you measure, not just formatting |
| LFM2.5 | `preserve_thinking` | false | keeps historical `<think>` blocks in context — irrelevant single-turn, relevant the moment anyone runs a transcript |
| — | `thinking_token_budget` | n/a | a serving-API parameter, **not** a template variable (P0.md corrects AGENTS.md on this) |

LFM2.5 always emits a `<think>` block and its template has no switch to disable it. P0 therefore uses a larger token budget (`p0_max_tokens=512`) so the answer is not truncated, and its decode and accuracy rows include reasoning tokens. This is a documented deviation from "thinking off everywhere"; every LFM2.5 number must carry that caveat.

The harness never invents a kwarg: `bench.py` and `run_p3.py` read
`Model.template_kwargs` from the registry and record the exact dict in every row.
A model whose template would receive a foreign kwarg either raises or silently
drops it depending on the jinja, so "it ran" is not evidence the controls applied.

MiniCPM5 deserves one extra P0 call, because its template has **three** distinct
branches at `add_generation_prompt`: `enable_thinking` undefined renders nothing,
`false` renders an empty `<think>\n\n</think>\n\n` block, `true` opens a real
`<think>`. The registry pins `false` (consistent "thinking off" across the roster);
run P0 once with `--template-kwargs '{}'` and record both prompt-token counts, so
the scaffolding cost of that choice is a measurement instead of an assumption.

## What stays fixed, so the rows mean something

These are the controls that make a cross-model table legitimate at all. Breaking
one of them means the comparison is no longer comparing models.

1. **Same 5 speed prompts** (`short`, `medium`, `long`, `decode-short`,
   `decode-long`) and **same 30 + 8 accuracy/quality prompts**, byte-identical.
2. **Same `max_tokens`** (256 speed/accuracy, 384 quality) and **greedy** decoding.
3. **1 discarded warm-up + 5 trials, median with min–max**, never a bare median.
4. **Nonce on every HTTP request**, except the row labelled `cached`, which exists
   to measure prefix reuse.
5. **Per-model prompt-token and generation-token counts in every row.** Tokenizers
   differ, so "1,026 prompt tokens" is a fact about one model, not about the other
   three. The chart pools them and says so; the tables never do.
6. **`bpw` and publisher in every checkpoint row**, because "4-bit" is a family,
   not a number: the spread here is 4.50–7.00 bpw and the OptiQ premium ranges
   from +13% (LFM) to +56% (Nemotron) by weight.
7. **Active parameters stated for the MoE** (8.47B total, ~1.3B active by the
   vendor's label; config arithmetic lands nearer 1.6B once embed, conv and dense
   layers are counted). Decode tps is measured against active work, so the label
   goes next to the number and the arithmetic is flagged as arithmetic.
8. **Peak memory from `mx.get_peak_memory()` in-process, `ps` RSS for HTTP**, with
   the source recorded in the row (`memory` dict: method, scope, phase, unit,
   availability). The two are different bases (MLX allocator buffers vs whole-process
   resident set), are never subtracted or ranked together, and neither is GPU VRAM
   (unified memory). `weights_gb` is checkpoint file size, not runtime memory.
9. **`caffeinate`, plugged in, nothing else running**, one model per run.
10. **Blind quality: one rater, slots assigned by a balanced rotation over a
    seeded shuffle**, key not opened until scores are filed. Balance matters: with
    a plain shuffle one row can land in slot A five times out of eight, and a
    rater who anchors on slot A then manufactures a winner. `score_quality.py`
    also prints the mean per slot and warns when slots are ≥1 point apart, so
    position bias is visible instead of baked in. Single rater, 8 prompts — a
    preference signal, and labelled as one.
11. **Integrity gate before any run.** `run_bench.py --phase fetch` hashes every
    weight file and compares against the Hub's published `sha256`, writing
    `results/<model>/p0/<ckpt>-provenance.json`. A checkpoint that does not match
    its hash is not the file the report claims to have measured, so it does not
    run. This is the same check `P0.md` records for the baseline.
12. **Determinism is checked, not assumed.** P0 generates the same prompt twice
    and records `deterministic: true/false`. Greedy must reproduce exactly; MoE
    routing and hybrid attention are new here, and if a row is not reproducible
    its median is a median of something else.
13. **A non-greedy row never shares a directory with a greedy one.** `bench.py`
    appends `-t<temp>-p<top_p>` to the run id, so a card-recommended sampling
    spot-check cannot overwrite published greedy results.

## What this roster cannot tell you

- **Nothing about quantization quality in general.** Two checkpoints per model,
  each built by whoever built it. The OptiQ delta is a fact about these files.
- **Nothing about 64K context.** The KV cache does not fit in 16 GB; not tested.
- **Nothing about concurrency.** Every run is single-stream, cold, sequential.
- **Nothing about multimodal.** `mlx_lm` never loads the vision/audio towers;
  text only, all four models, regardless of what a card claims.
- **No cross-model "intelligence" ranking from 30 prompts.** 30 hand-made
  prompts with exact-match checkers will separate 50% from 100% and will not
  separate 83% from 87%. A 1-prompt difference is noise, not a ranking.

## Run order, and the gate at each step

One model at a time, all the way through, so any stopping point leaves a usable
record rather than four half-models:

```bash
# 1. fetch + integrity gate (resumable; verifies every weight file's sha256)
.venv-mlxlm/bin/python run_bench.py --model minicpm5-2b --phase fetch

# 2. GATE: does it load in mlx_lm, with this template, and answer sanely?
.venv-mlxlm/bin/python run_bench.py --model minicpm5-2b --phase p0
#    stop here if load fails, peak > ~9 GB, or the answer is garbage

# 3. speed / TTFT / memory (in-process, then server, then labelled cache row)
.venv-mlxlm/bin/python run_bench.py --model minicpm5-2b --phase p2

# 4. accuracy (30) + quality (8)
.venv-mlxlm/bin/python run_bench.py --model minicpm5-2b --phase p3

# 5. blind quality across whatever is on disk, then score by hand
.venv-mlxlm/bin/python blind.py --round r1
#    copy results/blind-r1/scores-template.json -> scores.json, score, then:
.venv-mlxlm/bin/python score_quality.py --round r1

# 6. chart everything that has results
.venv-mlxlm/bin/python visualize_p2.py
```

A model is "done" when `results/<id>/` holds `p0/`, `<ckpt>-inproc/`,
`<ckpt>-http/`, `<ckpt>-http-cached/`, `p3/` and its rows are in RESULTS or the
per-model report. Nothing is deleted to make a table look tidy: a blocked
checkpoint keeps its error rows.

## Per-model expectations to check against

Written down before the runs, so a surprising result is recognisable as
surprising rather than explained away afterwards.

| Model | Prompted by | Falsifiable expectation | If it fails |
|---|---|---|---|
| MiniCPM5-2B | "best compact intelligence per GB" | beats 60% on the 30-prompt set and stays under 2.5 GB peak | the "per GB" claim is about size, not capability — report the size win and drop the intelligence claim |
| Spark-X2.5-4B | "best 4B-class challenger, coding/tool use" | 10/10 on function-call JSON (ceiling for all current rows) and best quality on rewrite/code prompts | a 4B losing to a 2.5B on its advertised strength is the headline |
| LFM2.5-8B-A1B | "fastest on this tier" | highest decode tps of the four, despite the most weights | MoE on M4 failing to beat dense 4.5 bpw at 7.5B would be worth saying plainly |
| Nemotron (baseline) | — | reproduces its own P2/P3 numbers within its min–max range | a miss means the harness changed, and the other rows are suspect too |
