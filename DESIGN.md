# MLX model bench on M4 / 16 GB — design

2026-10-24 · Apple M4 (10-core, 4P+6E), 16 GB unified, macOS 27.0.1

Started as `nemotron-bolt-bench`: one model, four arms, five metrics. It keeps that
record as the baseline and extends the same rigor — fixed prompts, bloat-free
controls, provenance in every row — to a roster of models that fit in 16 GB.

## Intent

Pick the runtime *and* the model for this machine, and know what each candidate
costs in speed, TTFT, accuracy, quality and memory. Not a leaderboard: no
cross-hardware extrapolation, no single-number score, no model that does not run
here.

## The four questions (baseline)

1. How much of the speed gap is the **engine**, and how much is the **checkpoint**?
2. Is the OptiQ mixed 4/8-bit checkpoint's quality distinguishable from plain 4-bit?
3. What does optiq's application layer cost in TTFT, measured like-for-like?
4. Does the assistant drafter earn its keep?

## The three questions (roster)

5. For each model, **which checkpoint does it wear best** — is the OptiQ premium in
   bits bought back in accuracy or quality, and at what cost in decode tps?
6. **Does a smaller model beat a larger one per GB?** MiniCPM5-2B at 1.43 GB
   against the 4.11–7.46B dense rows is the specific test; the answer is per task
   domain, not a single number.
7. **Does architecture show up in the numbers?** Spark's 3:1 sliding attention and
   LFM's 18-of-24 conv layers both predict a *different prefill curve* from a dense
   transformer. Same prompts, same harness: if the curves are identical, that is
   itself the finding.

Questions 1–4 are asked of every model in the roster, in the same order, so the
baseline is not a special case but the first row.

## Arms

Per model and per **tier** (4-bit, OptiQ mixed 4/8-bit, 8-bit reference), the same
three arms — so a difference between rows is a difference between models, not
between harnesses:

| Arm | Engine | Checkpoint | Transport |
|---|---|---|---|
| *n*.1 | `mlx_lm` 0.32.2 | the tier's checkpoint (vendor / mlx-community / third-party, see `MODELS.md`) | in-process |
| *n*.2 | `mlx_lm` 0.32.2 | same | HTTP (`mlx_lm.server`) |
| *n*.3 | `mlx-optiq` 0.7.3 | OptiQ tier only | HTTP (`optiq serve`) |

Arm *n*.2 exists so HTTP overhead is never mistaken for engine cost. Arm *n*.3
is attempted for every OptiQ checkpoint and recorded as unmeasurable when it
fails — for the baseline it does (`P0.md`), and "the engine cannot serve this" is
a result the comparison needs, not a gap to hide.

The three tiers exist because all four models publish all three. 8-bit is a
reference: it prices the quantization, and it is the reason the baseline is
re-run at 8-bit even though the original study stopped at 4-bit vs OptiQ.

### The baseline's historical arms

These are kept by name because `REPORT.md` and `P0.md` refer to them; the
harness now addresses the same things as `--model nemotron-3.5-bolt
--checkpoint mlx-4bit|optiq-4bit --transport inproc|http`.

| # | Engine | Checkpoint | Transport |
|---|--------|-----------|-----------|
| 1 | `mlx_lm` 0.32.2 | `rapid-mlx/…-MLX-4bit` (4.20 GB, affine 4-bit) | in-process |
| 2 | `mlx_lm` 0.32.2 | same | HTTP (`mlx_lm.server`) |
| 3 | `mlx_lm` 0.32.2 | `mlx-community/…OptiQ-4bit` (6.53 GB loaded / 7.49 GB download, mixed 4/8) | in-process |
| 4 | `mlx-optiq` 0.7.3 | `mlx-community/…OptiQ-4bit` | HTTP (`optiq serve`) |
| 5 | `mlx-optiq` 0.7.3 | OptiQ + Gemma-4 assistant drafter | HTTP (documented; not measured) |

Arm 5 is the external assistant-drafter path for the Nemotron target, distinct
from native MTP and from DFlash. The pinned OptiQ 0.7.3 source has a Gemma-4
assistant adapter that consumes target embeddings and hidden states, and the
OptiQ model card documents this exact pairing. Neither compatibility on this
checkpoint nor performance or memory was demonstrated by this study. DFlash is
a separate block-diffusion method, with NVIDIA inference-stack integrations;
this project has no verified DFlash integration in its pinned MLX stack. See
"What the technical report changes".

Arm 2 exists so arm 4's HTTP overhead is not mistaken for engine cost.
Arm 3 exists only if `mlx_lm` can load the mixed-bit file at all — P0 says it can
(4.6 s load, 6.65 GB peak), so arm 3 stays. Arm 4's no-drafter request crashes
in the recorded P0 run with a 10.5 GiB Metal allocation on this 16 GB machine.
Arm 5 is a separate `--drafter` path that was not attempted; do not infer its
compatibility or memory use from arm 4. The pinned OptiQ 0.7.3 adapter supports
Gemma-4 assistant checkpoints and the model card documents this target pairing,
but this is not a local execution result. See `P0.md` and `REPORT.md` section 1.4.

## Metrics

| Metric | Definition | Why |
|---|---|---|
| TTFT | request → first token, streaming | the number the model is sold on |
| Prefill tps | prompt tokens ÷ prefill time | inverse of TTFT; explains it |
| Decode tps | generated tokens ÷ decode time, prefill excluded | weight-read bound |
| Peak memory | `mx.get_peak_memory()` | 16 GB is the real constraint |
| Accuracy | % correct on a fixed, auto-graded set | objective |
| Quality | blind human rubric on 8 fixed prompts | the only honest judge here |

Timing boundaries are defined per field in the comment block above
`decode_window()` in `bench.py`. Historical `ttft_s` scope differs by transport
(each row's `ttft_scope` says which); `request_ttft_s` is the cross-transport
TTFT, `wall_s` the end-to-end user latency, and `decode_window_tps` the revised
decode rate, left None (not ranked) whenever `decode_flags` is non-empty.

TTFT is measured at three prompt lengths (short ≈ 64, medium ≈ 1k, long ≈ 8k
tokens) because prefill cost is not linear in a useful way and the model
claims a voice-assistant use case where short-prompt TTFT dominates.

## Controls

- Greedy decoding everywhere, so runs are reproducible. Model card recommends
  temp 1.0 / top_p 0.95; noted as a deviation, spot-checked once.
- 1 discarded warm-up run (Metal kernel compile + first-touch page faults).
- 5 trials, **median** reported. Range shown alongside, never median alone.
- Fixed `max_tokens`, fixed prompt files, identical across arms.
- `caffeinate` for the duration; plugged in. Other apps may be open, but the run says so
  (`--condition`) and each trial records a `sysinfo.py` snapshot.
- Each arm in its own `uv` venv, versions pinned and recorded in the output.
- Thinking off for the primary table; thinking on reported as a separate row,
  because it changes decode volume, not TTFT.
- Chat template controls are **per model** and pinned in `registry.py`: the
  template has two variables for the baseline (`enable_thinking`,
  `conversational_mode`), one for MiniCPM5 (`enable_thinking`), one for
  Spark-X2.5 (`enable_thinking`, default true), one for LFM2.5
  (`preserve_thinking`). `thinking_token_budget` is a serving-API parameter, not
  a template variable (P0.md). Primary table runs `conversational_mode=True` for
  the baseline, thinking off everywhere.
- Every row records its own prompt-token count. Tokenizers differ across the
  roster, so prompt length is a property of the row, never of the prompt file.

## Prompt sets

- **Accuracy (30)**: 12 arithmetic/word problems with checkable answers,
  10 function-call requests graded on valid JSON + correct function + correct
  arguments, 8 instruction-following with binary constraints ("answer in
  exactly two sentences, no lists").
- **Quality (8)**: explanation, summarization, rewrite, persona reply, refusal
  case, code explanation, correction, short-story opening. Scored 1–5 on
  accuracy / coherence / instruction-fit, by you, blind to arm.

## Non-goals

Other hardware; llama.cpp or GGUF; 64K-context behavior (KV cache would not fit
16 GB); quantization tuning (we compare published checkpoints, we do not produce
them); anything needing the vision tower; serving under concurrency; **DFlash**
(no verified integration in this repo's pinned MLX stack). Models enter the roster
only if they load in `mlx_lm` 0.32.2 at 4-bit and fit 16 GB with room for a KV
cache.

## Phases

Baseline (complete):

- **P0 · load test** — does each checkpoint load in each engine. ~10 min. Gates
  arm 3.
- **P1 · harness** — `bench.py`, one file, writes per-run JSON.
- **P2 · runs** — arms × trials. ~1 h of wall clock, mostly unattended.
- **P3 · report** — `REPORT.md` section 1 + raw JSON + 8 outputs for blind scoring.

Roster (repeated per model, in this order):

- **R0 · generalize** — registry, model-agnostic harness, per-model results
  directories, cross-model blind scoring. Done.
- **R*n* · one model end to end** — fetch → P0 gate → P2 → P3. Stop at the gate if
  it does not load, does not fit, or answers nonsense; a blocked row keeps its
  error record.
- **R4 · cross-model report** — 12 checkpoint rows, each with bpw, publisher,
  tokenizer-specific prompt tokens, generation length and peak memory; exploratory,
  single-rater blind quality round r1 for the four plain 4-bit checkpoints. LFM's
  larger P3 token budget is disclosed.

## What the technical report changes

Read from `~/Downloads/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB-Technical-Report.pdf`.

- **The earlier DFlash assessment is withdrawn.** The current NVIDIA integration
  source describes DFlash as a block-diffusion drafter integrated through
  Speculators/vLLM and SGLang, using target hidden states; it does not establish
  the old GGUF-only, sub-100-ms-TTFT, or edge-hardware claims. DFlash remains out
  of scope because this repo has no verified path in its pinned MLX stack, not
  because those older claims were demonstrated here.
- **The Nemotron OptiQ card documents an external Gemma-4 assistant-drafter
  pairing**, whereas native MTP is absent from the target. The pinned OptiQ 0.7.3
  adapter implements Gemma-4 assistant drafting using target hidden states, but
  this study has not established that this exact target/drafter combination
  executes correctly. Neither path has been measured here on the 16 GB M4; keep
  the two mechanisms and evidence separate. The card's 24 GB M4 numbers are
  upstream context only. The published
  drafter checkpoint is 78.8M parameters; its safetensors file is 159,138,208
  bytes (~0.159 GB), not the unsupported 3.5 GB estimate. Its end-to-end peak
  memory and speed on this host remain unmeasured.
- **Published evals are Q4_K_M GGUF via llama.cpp**, not these MLX
  quantizations. Their accuracy numbers are an upper reference for the
  original file, never an expected value for our arms.
- **Modality conflict**: the model card says Text → text; the NIM section says
  "Text, image, and audio inputs are accepted". P0 checks which is true.

## Open

- Accuracy/quality prompt authorship: mine, unless you want your own domain.
- Arm 5's separate Gemma-4 E4B assistant checkpoint is about 0.159 GB of
  safetensors (upstream Hub metadata); its total live runtime memory and
  compatibility with this Nemotron checkpoint are unmeasured here.
