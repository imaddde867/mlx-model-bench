# Nemotron 3.5 Bolt E4B 4GB on MLX — benchmark design

2026-10-24 · Apple M4 (10-core, 4P+6E), 16 GB unified, macOS 27.0.1

## Intent

Pick a runtime for this model on this machine, and know what each candidate
costs in speed, TTFT, accuracy, quality, and memory. Not a leaderboard, not a
comparison against other models or other hardware.

## The four questions

1. How much of the speed gap is the **engine**, and how much is the **checkpoint**?
2. Is the OptiQ mixed 4/8-bit checkpoint's quality distinguishable from plain 4-bit?
3. What does optiq's application layer cost in TTFT, measured like-for-like?
4. Does the assistant drafter earn its keep?

## Arms

| # | Engine | Checkpoint | Transport |
|---|--------|-----------|-----------|
| 1 | `mlx_lm` 0.32.2 | `rapid-mlx/…-MLX-4bit` (4.2 GB, affine 4-bit) | in-process |
| 2 | `mlx_lm` 0.32.2 | same | HTTP (`mlx_lm.server`) |
| 3 | `mlx_lm` 0.32.2 | `mlx-community/…OptiQ-4bit` (7.5 GB, mixed 4/8) | in-process |
| 4 | `mlx-optiq` 0.7.3 | `mlx-community/…OptiQ-4bit` | HTTP (`optiq serve`) |
| 5 | `mlx-optiq` 0.7.3 | OptiQ + Gemma-4 assistant drafter | HTTP |

Arm 5 uses Gemma-4's assistant drafter (or mlx-vlm MTP), **not** NVIDIA's
DFlash. DFlash ships as one GGUF and conditions on five hidden layers of the
target, so it needs an engine that exposes mid-layer activations — `mlx_lm`
does not. See "What the technical report changes".

Arm 2 exists so arm 4's HTTP overhead is not mistaken for engine cost.
Arm 3 exists only if `mlx_lm` can load the mixed-bit file at all — P0 says it can
(4.6 s load, 6.65 GB peak), so arm 3 stays. Arm 4 does not: `optiq serve` 0.7.3
dies on every completion with a 10.5 GiB Metal allocation on 16 GB. See `P0.md`;
until that is fixed, arm 5 has no transport and arm 4 reports as unmeasurable.

## Metrics

| Metric | Definition | Why |
|---|---|---|
| TTFT | request → first token, streaming | the number the model is sold on |
| Prefill tps | prompt tokens ÷ prefill time | inverse of TTFT; explains it |
| Decode tps | generated tokens ÷ decode time, prefill excluded | weight-read bound |
| Peak memory | `mx.get_peak_memory()` | 16 GB is the real constraint |
| Accuracy | % correct on a fixed, auto-graded set | objective |
| Quality | blind human rubric on 8 fixed prompts | the only honest judge here |

TTFT is measured at three prompt lengths (short ≈ 64, medium ≈ 1k, long ≈ 8k
tokens) because prefill cost is not linear in a useful way and the model
claims a voice-assistant use case where short-prompt TTFT dominates.

## Controls

- Greedy decoding everywhere, so runs are reproducible. Model card recommends
  temp 1.0 / top_p 0.95; noted as a deviation, spot-checked once.
- 1 discarded warm-up run (Metal kernel compile + first-touch page faults).
- 3 trials, **median** reported. Range shown alongside, never median alone.
- Fixed `max_tokens`, fixed prompt files, identical across arms.
- `caffeinate` for the duration; nothing else running; plugged in.
- Each arm in its own `uv` venv, versions pinned and recorded in the output.
- Thinking off for the primary table; thinking on reported as a separate row,
  because it changes decode volume, not TTFT.
- The chat template has three controls, not one: `enable_thinking`,
  `thinking_token_budget`, `conversational_mode`. Primary table runs
  `conversational_mode=True` (it is the intended voice-assistant mode) and
  `thinking_token_budget` fixed, so "thinking on" cannot run unbounded.

## Prompt sets

- **Accuracy (30)**: 12 arithmetic/word problems with checkable answers,
  10 function-call requests graded on valid JSON + correct function + correct
  arguments, 8 instruction-following with binary constraints ("answer in
  exactly two sentences, no lists").
- **Quality (8)**: explanation, summarization, rewrite, persona reply, refusal
  case, code explanation, correction, short-story opening. Scored 1–5 on
  accuracy / coherence / instruction-fit, by you, blind to arm.

## Non-goals

Other models; other hardware; llama.cpp or GGUF; 64K-context behavior (KV
cache would not fit 16 GB); quantization tuning; anything needing the vision
tower; serving under concurrency; **DFlash** (GGUF + NVIDIA GPUs only).

## Phases

- **P0 · load test** — does each checkpoint load in each engine. ~10 min. Gates
  arm 3.
- **P1 · harness** — `bench.py`, one file, writes per-run JSON.
- **P2 · runs** — arms × trials. ~1 h of wall clock, mostly unattended.
- **P3 · report** — `RESULTS.md` + raw JSON + 8 outputs for blind scoring.

## What the technical report changes

Read from `~/Downloads/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB-Technical-Report.pdf`.

- **DFlash is out of reach and irrelevant to TTFT.** It ships as a single GGUF
  (737 MB) and every speedup number is vLLM/NIM on RTX PRO 6000, DGX Spark,
  RTX 5070. The paper states TTFT "changes by less than 100 ms in all
  settings" — it shortens decode only. The 2.74× is 124.7 → 341.2 tok/s on an
  RTX PRO 6000. The closest analogue to a 16 GB Mac is DGX Spark at 20.1 →
  71.6 tok/s.
- **The 1.21× is the number that matters for adoption**, not the 2.74×. The
  2.74× is against *no* speculative decoding; Gemma-4-E4B-it ships MTP, and
  against that Bolt is 1.21× on RTX 5070 and **1.05× on RTX Spark** — i.e.
  nearly nothing. Bolt's edge over its parent is quality and post-training,
  not speed. Worth testing whether plain Gemma-4-E4B-it + MTP in mlx-vlm is
  the better deal.
- **Published evals are Q4_K_M GGUF via llama.cpp**, not these MLX
  quantizations. Their accuracy numbers are an upper reference for the
  original file, never an expected value for our arms.
- **Modality conflict**: the model card says Text → text; the NIM section says
  "Text, image, and audio inputs are accepted". P0 checks which is true.

## Open

- Accuracy/quality prompt authorship: mine, unless you want your own domain.
- Arm 5's drafter is a second download (bf16, few GB).
