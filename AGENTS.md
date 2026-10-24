# AGENTS.md

Benchmarking 2-9 GB quantized LLMs on Apple Silicon MLX: one baseline plus a
roster of challengers, three quantization tiers each. Renamed from
`nemotron-bolt-bench` (github.com/imaddde867/mlx-model-bench). Read `DESIGN.md`
and `MODELS.md` before touching an arm. This file is the operating contract.

## Machine

Apple M4, 10-core, 16 GB unified, macOS 27.0.1. Python 3.12.10, uv 0.10.9.
119 GB free disk is enough for all four models (≈29 GB of weights).

## Pins

`mlx-lm` 0.32.2 · `mlx` 0.32.4 · `mlx-optiq` 0.7.3 (requires `mlx-lm <0.33`).
One venv per engine, so neither can silently upgrade the other. Record every
version inside each result row, not in prose.

Architecture support is a pin too: `mlx-lm` 0.32.2 registers `llama` (MiniCPM5),
`spark2_5` (Spark-X2.5) and `lfm2_moe` (LFM2.5), which is why those three are
in the roster. Check `mlx_lm.models` before adding a fourth.

## The registry

`registry.py` is the single source of truth: models, checkpoints, Hub repos,
tiers, weight sizes, bits-per-weight, publisher, engines, and **per-model chat
template controls**. No harness may hardcode a model. If a number appears in a
report, it came from a result row, and the row's metadata came from the registry.

A model is a *Model* with one *Checkpoint* per tier. The tiers are the same for
every model — `4bit`, `optiq-4bit`, `8bit` — because that is the largest set all
four publish. A missing tier is a documented absence, never a silent omission:
see the option matrix in `MODELS.md`.

`publisher` and `official` are part of a checkpoint's identity. Two models use
vendor conversions, one has no vendor MLX conversion at all, one is entirely
third-party. That distinction goes in every table the row appears in.

## Rules that make the numbers comparable

- Greedy decoding. Where a card recommends temp 1.0 / top_p 0.95 that is a
  separate, labelled row, not the default.
- Discard run 1 (Metal kernel compile, first-touch faults). Report median of 5
  **and** the range. Never a bare median.
- Fixed prompt files, fixed `max_tokens`, identical across models.
- HTTP arms: every request carries a random nonce (`bench.py`), or the server's
  prefix cache reports fake prefill speed. `--no-nonce --tag cached` measures the
  cache deliberately.
- `caffeinate` on, plugged in, nothing else running. One model per run, to
  completion, before the next starts.
- No number leaves the house without its model, checkpoint, prompt-token count
  and generation length printed next to it.
- **Prompt-token counts are per tokenizer.** Never present one model's token
  count as another's; never rank models by "prompt tokens".
- **`bpw` and publisher are part of the checkpoint's identity.** They range
  4.50–7.00 here. A third-party quant is labelled as one in every table it
  appears in.
- **Active parameters get stated for MoE**, next to decode tps, with the vendor
  label and (if computed) the config arithmetic marked as arithmetic.
- Peak memory is `mx.get_peak_memory()` in-process and `ps` RSS for HTTP; the
  source is recorded in the row and never mixed silently inside a column.
- **Integrity before measurement.** `run_bench.py --phase fetch` verifies every
  weight file against the Hub's `sha256` and writes a provenance record. No
  checksum, no row — a mismatched quant is a different file than the one named.
- **Determinism is verified at P0** (same prompt twice, must match exactly). If a
  model's greedy output is not reproducible, its medians are not comparable and
  the row is reported as such.
- **Deviation rows get their own directory.** `bench.py` tags any non-greedy run
  `-t<temp>-p<top_p>`; nothing a spot-check produces can overwrite a published
  greedy summary.
- **Blind slots are balanced, not just shuffled**, and the scorer reports the
  mean per slot. A ≥1-point spread across slots means position bias — fix the
  blind before reading the ranking.

## Chat template controls

Per model, read from `chat_template.jinja` at the pinned repo; recorded in
`registry.py`. Do not assume a kwarg exists.

- Nemotron: `conversational_mode` (primary table true; costs +27 prompt tokens)
  and `enable_thinking` (pass explicitly — the jinja defaults false but
  transformers injects `True`).
- MiniCPM5: `enable_thinking`, guarded by `is defined` — an explicit false is not
  the same as omitting it. Verify by prompt-token count in P0.
- Spark-X2.5: `enable_thinking`, `| default(true)` — thinking is ON unless passed
  false. Omitting it changes what is measured.
- LFM2.5: `preserve_thinking` only (default false). No `enable_thinking`.
- `thinking_token_budget` is a **serving-API parameter, not a template
  variable** — it appears in no `chat_template.jinja`. (This file said "three
  controls" until P0.md corrected it.)

## Status

- [x] P0 — load tests, baseline. Arms 1–3 load; arm 4 blocked by an optiq prefill buffer. See `P0.md`.
- [x] P1 — `bench.py` — verified arms 1, 2, 3 produce valid JSON with all metrics.
- [x] P2 — speed/TTFT/memory runs, arms 1–3, valid after cache-bust fix. See `REPORT.md` §1.1.
- [x] P3 — accuracy (30) + quality (8) prompt sets, blind evaluation, report in `REPORT.md` §1.2–1.5.
- [x] R0 — harness generalized: registry-driven `bench.py`/`run_p3.py`/`run_bench.py`, multi-model chart, cross-model `blind.py` + `score_quality.py`, results moved to `results/<model>/`.
- [x] R0.5 — tier matrix: every model at `4bit` / `optiq-4bit` / `8bit` (reference), three arms each (`inproc`, `http`, `optiq` engine attempted), `run_bench.py --tiers` to select. Fairness ledger in `MODELS.md`.
- [ ] R1 — MiniCPM5-2B: fetch (sha256 gate), P0 (determinism gate), P2, P2-optiq, P3.
- [ ] R2 — Spark-X2.5-4B: same. Remember: its 4-bit and 8-bit tiers are community quants, no vendor conversion exists.
- [ ] R3 — LFM2.5-8B-A1B: same. The 9.02 GB 8-bit tier may fail the P0 memory gate on 16 GB — record that, do not drop the row.
- [ ] R3b — baseline 8-bit reference row: `--model nemotron-3.5-bolt --tiers 8bit`. Retrofit parity; the original study omitted it.
- [ ] R4 — cross-model report: one table, each row carrying tier, bpw, publisher, prompt tokens, peak memory; blind quality round r1 across all four.

## Known traps

- NVIDIA's **DFlash** drafter is GGUF + NVIDIA GPUs only, and it barely moves
  TTFT. Do not plan around it.
- Paper evals are Q4_K_M GGUF, not these MLX quants. Their scores are an upper
  reference for a different file, not a target for ours.
- 64K context will not fit in 16 GB. Do not test it.
- The mixed-bit OptiQ checkpoint may not load in plain `mlx_lm`. **Resolved for
  the baseline** (it loads; see P0.md) — but re-run the P0 gate per model, since
  `kv_config.json` coverage is a per-repo accident (Nemotron's covers 24 of its
  42 layers).
- GGUF is a non-goal. A "2.6 GB GGUF" quote for a roster model is not a size you
  can use here; the MLX figure is a different file.
- A model's `E#B` / `A#B` name is an effective/active label, not a parameter
  count. Nemotron E4B is 7.46B total; LFM2.5 A1B is 8.47B total. Dividing by the
  label silently inflates every efficiency ratio by ~2×.
