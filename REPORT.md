# Four local models on a 16 GB M4 — speed, memory, accuracy, quality

Apple M4 (10-core, 4P+6E), 16 GB unified memory, macOS 27.0.1, `mlx` 0.32.4, `mlx-lm` 0.32.2,
Python 3.12.10. Greedy decoding and per-model template controls; the intended protocol uses a
plugged-in Mac with `caffeinate`. The historical P2 matrix rows do not record run conditions.
LFM2.5 emits reasoning tokens by design; see sections 4–5.

This report covers four models at three quantization tiers. Sections 2–4 present the challengers;
section 5 compares all 12 in-process checkpoint rows. The P2 matrix rows predate condition
snapshots and are labelled `not recorded`. The run operator recalls that many other apps were open during those runs; this may have slowed
the measurements, but it is not a per-trial system measurement. The records do not support a
slowdown estimate or clean-vs-loaded comparison. Focused repeats remain separate.
The r1 blind scores cover the four plain 4-bit checkpoints only and are exploratory, single-rater
results.

| Section | Model | Weights (4-bit / OptiQ / 8-bit) | Status |
|---|---|---|---|
| 1 | NVIDIA Nemotron 3.5 Bolt E4B | 4.20 / 6.53 / 7.96 GB | baseline complete; historical 4-bit/OptiQ and 8-bit matrix rows |
| 2 | OpenBMB MiniCPM5-2B | 1.43 / 1.94 / 2.68 GB | complete |
| 3 | XHToken Spark-X2.5-4B | 2.32 / 3.05 / 4.38 GB | complete; plain tiers are third-party quants |
| 4 | Liquid AI LFM2.5-8B-A1B | 4.85 / 5.47 / 9.02 GB | complete; thinking tokens are included in generation |
| 5 | Cross-model comparison | — | complete; 12 checkpoint rows and blind r1 |

Section 1's measurements predate the checkpoint, P0, timing, memory and grader gates (PRs #13-#18) and
are historical, not re-validated by them. Checkpoints: Arm 1 `rapid-mlx/NVIDIA-Nemotron-3.5-Bolt-E4B-MLX-4bit`
@ `601cd937`, Arm 3 `mlx-community/NVIDIA-Nemotron-3.5-Bolt-E4B-OptiQ-4bit` @ `0c204a16` (sha256-verified in
`results/nemotron-3.5-bolt/p0/*-provenance.json`).

Method, controls and comparability rules: [`MODELS.md`](MODELS.md). Machine-readable
model/checkpoint metadata: [`registry.py`](registry.py). Every row in this report is produced
by one of the commands in the README, and the raw JSON sits beside it under `results/`.

![P2 chart](p2_visual.png)

---

## 1. NVIDIA Nemotron 3.5 Bolt E4B — complete

`nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB` on Apple Silicon MLX. Arms below are the historical
names; the harness addresses them as `--model nemotron-3.5-bolt --checkpoint <tag>`.

Arms:
- **Arm 1** (= `--model nemotron-3.5-bolt --checkpoint mlx-4bit`): `rapid-mlx/NVIDIA-Nemotron-3.5-Bolt-E4B-MLX-4bit` (4.20 GB, 4.50 bpw, affine 4-bit), `mlx_lm` in-process.
- **Arm 2**: Same checkpoint served via `mlx_lm.server` (HTTP).
- **Arm 3** (= `--checkpoint optiq-4bit`): `mlx-community/NVIDIA-Nemotron-3.5-Bolt-E4B-OptiQ-4bit` (6.53 GB loaded / 7.49 GB download, 7.00 bpw, mixed 4/8-bit), `mlx_lm` in-process.
- **Arm 4 / 5**: `mlx-optiq` 0.7.3 (`optiq serve`) — blocked on 16 GB machines due to a hardcoded 10.5 GiB Metal buffer allocation (see `P0.md`).

This model is the **baseline row** of the roster (`MODELS.md`). Its parameter count is 7.46B, not 4B: "E4B" is the effective/active label. Raw results live in `results/nemotron-3.5-bolt/`.

---

### 1.1 Speed, TTFT, and Memory (P2)

Speed runs conducted with 1 discarded warm-up + 5 trials; reported as **median [min–max]**. Prompt cache busted with random nonces, except for `Arm 2 cached` which measures warm prefix reuse. Source rows: `results/nemotron-3.5-bolt/{mlx-4bit-inproc,mlx-4bit-http,mlx-4bit-http-cached,optiq-4bit-inproc}/summary.json` (`mlx` 0.32.4, `mlx-lm` 0.32.2).

**Metric scope (historical rows).** These rows predate `request_ttft_s`, `decode_window_tps`, `ttft_scope` and the `memory` provenance record, so none are in the recorded rows. Figures below are the original `ttft_s` / `decode_tps` / `peak_gb`, and their scopes differ by transport (see the comment block in `bench.py`):
- In-process TTFT (Arms 1, 3) includes chat-template rendering and tokenization; HTTP TTFT (Arm 2) starts at request send and ends at the first visible SSE delta, excluding client JSON serialization. The in-process vs HTTP TTFT columns are therefore not a like-for-like engine comparison; `request_ttft_s` is the cross-transport definition and was not recorded here.
- `Arm 2 cached` is a deliberate warm-prefix row (`--no-nonce`); it measures prefix-cache reuse, not prefill speed, and is not comparable to the cold columns.
- Decode tps is the original `decode_tps`, not `decode_window_tps`, and was not flagged for short or cache-hit windows.

### TTFT (s)

| Prompt tokens | Arm 1 (in-proc) | Arm 2 (HTTP) | Arm 3 (OptiQ in-proc) | Arm 2 cached |
|---|---|---|---|---|
| 68 | 0.195 [0.192–0.200] | 0.161 [0.159–0.167] | 0.217 [0.204–0.234] | 0.068 [0.065–0.076] |
| 1,026 | 1.62 [1.62–1.63] | 1.52 [1.52–1.61] | 1.68 [1.68–1.72] | 0.070 [0.067–0.074] |
| 8,341 | 14.3 [13.7–14.3] | 14.4 [14.1–14.6] | 14.6 [14.4–15.0] | 0.094 [0.091–0.104] |

Prefill throughput (historical `prefill_tps` medians, so it inherits the `ttft_s` scope) is not flat: 68 tokens 314–422 tok/s (Arm 3 / Arm 2), 1,026 tokens 609–674, 8,341 tokens 571–584, across Arms 1–3 (cold rows).

### Decode (tok/s, prefill excluded)

| Prompt tokens → generated | Arm 1 | Arm 2 | Arm 3 |
|---|---|---|---|
| 85 → ~100–160 (`decode-short`) | 34.4 [33.8–34.8] | 32.5 [32.0–33.6] | 25.7 [24.1–26.4] |
| 8,359 → 256 (`decode-long`) | 29.5 [28.2–31.1] | 27.9 [26.7–28.9] | 23.8 [23.7–24.5] |

### Memory (`mx.get_peak_memory`)

Basis: MLX allocator peak, in-process, GB decimal (source inferred from transport; these summaries predate the `memory` provenance record). It is not comparable with server RSS (HTTP rows), and neither is GPU VRAM on unified memory. The load peaks match the checkpoint file sizes; they are not total process memory.

| Metric | Arm 1 | Arm 3 | Delta (OptiQ overhead) |
|---|---|---|---|
| Load peak | 4.20 GB | 6.53 GB | +2.33 GB (+55.5%) |
| Peak, 68-token prompt | 4.38 GB | 6.71 GB | +2.33 GB |
| Peak, 8.3k prompt + 256 gen | 5.09 GB | 7.42 GB | +2.33 GB |

---

### 1.2 Objective Accuracy (P3 — 30 Prompts)

Evaluated on 30 fixed prompts in the **Machine Learning & Data Engineering** domain:
- **12 Arithmetic & Quantitative ML Problems**: Convolution spatial dims, parameter counts, memory bytes, linear decay schedules, precision/recall, Kafka partition throughput, gradient accumulation steps, and quantization storage. Checked against exact ground truth.
- **10 Function-Call Requests**: Tool calls requiring valid JSON schema matching `{"name": "...", "arguments": {...}}` with correct parameters and typed values.
- **8 Strict Instruction-Following Tasks**: Strict binary constraints (sentence counts, no prohibited letters like 'e', exact comma-separated single lines, strict word count ranges 25-35, 5 hyphen lines, no prohibited domain words, and all-caps constraints).

### Accuracy Summary

| Category | Total Prompts | Arm 1 (rapid-mlx 4-bit) | Arm 3 (OptiQ mixed 4/8-bit) | Delta |
|---|---|---|---|---|
| **Quantitative ML & Arithmetic** | 12 | 6 / 12 (50.0%) | **7 / 12 (58.3%)** | +8.3% |
| **Function-Call Schema & Args** | 10 | **10 / 10 (100.0%)** | **10 / 10 (100.0%)** | Parity |
| **Instruction-Following Constraints** | 8 | **8 / 8 (100.0%)** | **8 / 8 (100.0%)** | Parity |
| **Total Accuracy** | **30** | **24 / 30 (80.0%)** | **25 / 30 (83.3%)** | **+3.3%** |

Counts are unchanged by the #4 grader hardening: `p3-grader-v2` (`GRADER_VERSION` in `p3_suite.py`) reproduces every recorded verdict from the raw answers in `results/nemotron-3.5-bolt/p3/{mlx-4bit,optiq-4bit}-results.json` (24 and 25 passes); the check is `tests/test_graders.py::test_recorded_legacy_results_do_not_drift`, which re-grades those files. The original run predates the grader version tag. The 25 vs 24 difference is one net prompt: Arm 3 passes `math-03` and `math-09` where Arm 1 fails, and Arm 1 passes `math-01` where Arm 3 fails. With n=30 and one greedy run per prompt this is not evidence of superior reasoning. The "Instruction-Following" category is format-compliance only (sentence/word counts, letter and word bans, casing), not instruction-following ability in general.

### Key Accuracy Findings
1. **Tool use and format compliance at ceiling**: Both checkpoints scored 10/10 on JSON function calling and 8/8 on the format constraints. With the suite at ceiling for both, it cannot separate them here.
2. **Numerical calculations (observation, not a conclusion)**: Arm 3 computed large multi-factor memory calculations (`math-03`: `32000 * 256 * 4 = 32768000`) and dataset token counts (`math-09`: `64000000`), whereas Arm 1 returned the uncomputed expression (`'32000 * 256 * 4'`) or dropped a trailing zero (`'6400000'`). Arm 3 failed `math-01`, which Arm 1 passed.

---

### 1.3 Qualitative Evaluation (P3 — 8 Prompts Blind Scored)

Evaluated on 8 core NLP/ML engineering tasks. **Exploratory:** one rater, 8 prompts, one greedy output per arm per prompt; a preference signal, not a statistically demonstrated ranking. Scoring predates the balanced-slot rounds (`blind.py`, `score_quality.py`), so slot balance for this round was not audited. Raw scores: `results/nemotron-3.5-bolt/p3/quality_scoring.json`. Outputs were randomized blind into **Model A** and **Model B** per prompt, scored independently across 3 rubrics (1–5 scale each, 15 max per prompt):
1. **Accuracy / Technical Fidelity** (1–5)
2. **Coherence & Tone** (1–5)
3. **Instruction-Fit & Completeness** (1–5)

### Blind Quality Scoring Table

| Task ID | Task Description | Arm 1 (Score / 15) | Arm 3 (Score / 15) | Winner | Notes |
|---|---|---|---|---|---|
| `qual-01` | Vanishing Gradient & ReLU Failure | 12.0 / 15 (4.00) | 12.0 / 15 (4.00) | **Tie** | Both provide concise, accurate explanations of saturation and dying ReLU. |
| `qual-02` | Vector DB & ANN Summarization | 14.5 / 15 (4.83) | **15.0 / 15 (5.00)** | **Arm 3** | Arm 3 captures exact numbers (1.5x-2x RAM) and disk-backed architecture. |
| `qual-03` | Incident Post-Mortem Rewrite | 13.5 / 15 (4.50) | **15.0 / 15 (5.00)** | **Arm 3** | Arm 3 properly formats with the requested section header and post-mortem flow. |
| `qual-04` | Principal MLOps Persona Reply | 9.5 / 15 (3.17) | **11.5 / 15 (3.83)** | **Arm 3** | Arm 1 is a bare one-liner; Arm 3 provides constructive critique on testability and artifact storage. |
| `qual-05` | Auth Bypass & Exfiltration Refusal | 14.5 / 15 (4.83) | **15.0 / 15 (5.00)** | **Arm 3** | Arm 3 is completely non-preachy with an immediate pivot to authorized data governance. |
| `qual-06` | PyTorch Custom Autograd Analysis | **14.5 / 15 (4.83)** | 13.5 / 15 (4.50) | **Arm 1** | Arm 1 explicitly flags numerical instability with large scale/diff exponential overflow. |
| `qual-07` | Tabular CNN vs GBDT Correction | 12.5 / 15 (4.17) | **15.0 / 15 (5.00)** | **Arm 3** | Arm 3 explains why tabular data lacks spatial structure exploited by CNNs. |
| `qual-08` | Orbital Cluster Sci-Fi Opening | 12.0 / 15 (4.00) | **15.0 / 15 (5.00)** | **Arm 3** | Arm 3 delivers rich technical atmosphere (ECC checks, telemetry, 100 tok vs 42 tok). |
| **Total** | **Mean Score (out of 5.0)** | **4.29 / 5.0** (103/120) | **4.67 / 5.0** (112/120) | **Arm 3 (6-1-1)** | OptiQ +0.38 mean score (single rater, exploratory) |

### Qualitative Side-by-Side Highlights

#### `qual-03`: Post-Mortem Rewrite
- **Arm 1**: `"Upstream schema change (user_id: int64 to string) caused Spark to fail silently/OOM due to unhandled type mismatch during JSON parsing. No Dead Letter Queue prevented Kafka offset advancement, leading to 4M record lag. Remediation: Implemented schema validation and dedicated DLT topic."`
- **Arm 3**: `"**Root Cause and Pipeline Remediation:** An upstream schema change (user_id from int64 to string) caused the Spark streaming job to fail due to silent nulls or OOM errors during JSON parsing. Lack of a Dead Letter Queue (DLT) halted Kafka offset advancement, leading to 4M record lag. Remediation involved patching schema validation, implementing a dedicated DLT topic, and restarting the job."`

#### `qual-07`: False-Premise Correction
- **Arm 1**: `"No. CNNs don't inherently use spatial self-attention for tabular data superiority; GBDT often performs better on raw tabular data."`
- **Arm 3**: `"CNNs don't consistently outperform GBDTs on tabular data due to spatial self-attention; tabular data lacks inherent spatial structure that CNNs exploit. GBDTs are generally better suited for tabular data."`

---

### 1.4 Synthesis: Answering the Four Benchmark Questions

#### 1. How much of the speed gap is the engine, and how much is the checkpoint?
- **Engine**: ~0.195 s in-process vs 0.161 s HTTP TTFT and 34.4 vs 32.5 tok/s decode (~5%). The TTFT scopes differ (in-process includes template and tokenize; HTTP does not), so this is not a clean transport cost; a like-for-like `request_ttft_s` comparison has not been run.
- **Checkpoint**: OptiQ mixed 4/8-bit decodes ~25% slower (34.4 → 25.7 tok/s, same in-process transport) and loads **+2.33 GB** more (4.20 GB → 6.53 GB, MLX allocator basis; unified memory, not VRAM).

#### 2. Is the OptiQ mixed 4/8-bit checkpoint's quality distinguishable from plain 4-bit?
- **Not established.**
  - **Objective Accuracy**: 25/30 vs 24/30, one net prompt (see 1.2). Not a demonstrated difference.
  - **Subjective Quality** (exploratory, single rater, 8 prompts): 4.67 vs 4.29 out of 5.0, Arm 3 ahead on 6 of 8. Arm 3's outputs were more detailed; Arm 1's were sometimes terse one-liners. Direction is suggestive; more raters and prompts are needed for a claim.

#### 3. What does OptiQ's application layer cost in TTFT, measured like-for-like?
- `optiq serve` crashes out-of-the-box due to eager evaluation of the unquantized dummy `embed_tokens_per_layer` weight (`vocab 262,144 × layers 42 × dim 512 × 2 bytes = 11,274,289,152 bytes = 10.5 GiB`) in `optiq/mlx_lm_patches/_chunked_quantize.py#L72`, exceeding Metal's 8.88 GiB max allocation limit.
- When `_chunked_quantize`'s eager eval is bypassed:
  - `OptiqEngine` short-prompt TTFT of 0.218 s and decode of 21.3 tok/s (~17% below `mlx_lm` in-process, 25.7) are **unverified**: no `results/` row exists for the patched engine run, so they cannot be tied to a measurement and are withdrawn as findings until re-run.
- Both `mlx_lm` in-process arms share a scope: OptiQ weights add +0.022 s short-prompt TTFT (0.217 s vs 0.195 s, ranges 0.204–0.234 and 0.192–0.200, so close to the run-to-run spread) with similar prefill throughput (~570 tok/s).

#### 4. Does speculative decoding earn its keep on this architecture?
- **An external Gemma 4 assistant is a documented OptiQ pairing, but its benefit and memory cost are not measured in this study.**
  - **Native MTP**: Nemotron 3.5 Bolt has no native MTP head. An `OptiqEngine` run reported in an earlier session was autoregressive (`mode='AR', depth=0`), but it has no row in `results/` and is unverified; this says nothing about the separate assistant-drafter path.
  - **Gemma 4 assistant drafting**: This is a separate four-layer assistant, not another 3.5 GB target model. The pinned local `mlx-optiq` 0.7.3 package exposes `serve --drafter` and contains a Gemma assistant adapter that consumes target embeddings and hidden states. The [Nemotron OptiQ card](https://huggingface.co/mlx-community/NVIDIA-Nemotron-3.5-Bolt-E4B-OptiQ-4bit/tree/0c204a16b6b186d1434e70f5fc2ca2e30984a006) documents this exact target/drafter pairing and reports decode results on a 24 GB M4. These sources establish an implemented and documented path, not that this exact pair has been executed or verified in this study. The MLX Hub repo reports 78,779,908 parameters, a 159,138,208-byte `model.safetensors` file (~0.159 GB), and 191,307,648 bytes total storage. Its checked revision is [`844e008e`](https://huggingface.co/mlx-community/gemma-4-E4B-it-assistant-bf16/commit/844e008e06ef5562bdb89428d851d0634d119dcd). That upstream card reports 1.70–2.38x decode throughput on its four listed prompts, with one prose output differing at a near-tie token; those figures are not this study's measurements. Drafter file size alone does not establish total live memory, peak usage, or a safe 16 GB configuration.
  - **DFlash**: This is a distinct block-diffusion drafter, not Gemma assistant MTP or a target's native MTP head. NVIDIA's [current integration guidance](https://developer.nvidia.com/blog/boost-inference-performance-up-to-15x-on-nvidia-blackwell-using-dflash-speculative-decoding/) describes Speculators/vLLM and SGLang integrations using target hidden states on NVIDIA GPU stacks; this repo has no verified DFlash path in its pinned MLX stack. The prior claims that DFlash requires GGUF and provides <100 ms TTFT improvement are withdrawn; the cited current official source does not support either claim.
  - **Verdict**: The blanket memory-based rejection was unsupported. An implementation and exact pairing are documented, and the drafter checkpoint is small; however, compatibility and performance on this 16 GB host remain unverified and unmeasured. Do not claim an accelerated Nemotron result, a safe memory margin, or DFlash compatibility without applicable verification.

---

### 1.5 Deployment Recommendations

| Target Workload | Recommended Arm | Rationale |
|---|---|---|
| **Voice Assistant / Low Latency UI** | **Arm 1 / Arm 2 (rapid-mlx 4-bit)** | 34.4 tok/s decode, 0.195 s (in-process) / 0.161 s (HTTP) short-prompt TTFT on differing scopes, 4.20 GB load peak (MLX allocator). Fastest decode measured here. |
| **Technical Copilot / Complex RAG** | **Arm 3 (OptiQ mixed 4/8-bit)** | Tentative: 25/30 vs 24/30 and 4.67 vs 4.29 are not demonstrated differences (1.2, 1.3). Costs ~25% decode (25.7 tok/s) and 7.42 GB peak (8.3k prompt, MLX allocator). Choose it for the larger bit budget, not proven quality. |
| **Multi-turn Conversational Service** | **Arm 2 (`mlx_lm.server`)** | Warm-prefix row (`--no-nonce`): 0.068–0.094 s TTFT vs ~14 s cold at 8k (~150x, arithmetic on historical `ttft_s`). Applies only when the prefix repeats; server RSS peak 4.02–4.03 GB (`mlx-4bit-http-cached/summary.json`; process RSS, a different basis from the in-process MLX allocator peaks above). |

These recommendations are **within this model only**. Whether the 4-bit Nemotron is the right
pick for any of those workloads is now assessed in section 5, against the 12 completed checkpoint rows.

---

## 2. OpenBMB MiniCPM5-2B

A 2.517B dense Llama (42 layers, 2 KV heads) provides the smallest weights and runtime peak in
the matrix. Its long-prompt TTFT is not correspondingly small: the 8.35k-token prompt takes
14.606 s, close to the larger Nemotron and LFM rows. At 4-bit its short-prompt decode median
is 69.0 tok/s and 54.6 tok/s on the 8.37k-token decode prompt, with a 2.465 GB in-process
allocator peak.

| Tier | Weights GB | bpw | Publisher | Decode long tok/s [range] | TTFT long s [range] | Prompt tok | Gen tok | Peak GB | Math strict /12 | Tool /10 | Format /8 | Total /30 |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4-bit (`mlx-4bit`) | 1.43 | 4.55 | vendor (`openbmb`) | 54.6 [54.5–55.5] | 14.606 [13.979–14.782] | 8,354 | 256 | 2.465 | 3 | 10 | 6 | 19 |
| OptiQ 4/8 (`optiq-4bit`) | 1.94 | 6.17 | mlx-community | 44.4 [41.3–44.9] | 14.042 [13.906–14.334] | 8,354 | 256 | 2.975 | 1 | 10 | 6 | 17 |
| 8-bit (reference) | 2.68 | 8.52 | mlx-community | 32.6 [23.7–33.5] | 14.768 [14.743–14.869] | 8,354 | 256 | 3.723 | 2 | 10 | 6 | 18 |

On this 30-prompt set, all three tiers pass all ten tool-call cases, while mathematical correctness
and strict-format compliance are much lower. The numbers do not establish broad capability or an
intelligence-per-GB ranking. The format column is only the eight objective output constraints.

#### Focused 4-bit decode repeat (not a replacement for the matrix row)

The full P2 `decode-short` row reports **69.0 tok/s [54.7–71.7]** over five kept trials
(55 prompt tokens, 256 generated tokens). A separately tagged five-trial repeat measured
**71.6 tok/s [71.4–72.1]**, with TTFT **0.1133 s [0.1107–0.1180]** and peak MLX allocator memory
**1.652 GB**. The repeat's decode range is narrow and overlaps the original range; the original
range is wide, so this supports the original estimate but is not evidence of a higher speed.

The repeat was labelled `loaded`; its snapshots recorded Spotlight indexing activity up to 17%
CPU and 0.611 GB swap in use (not proof that the model was paging). The original run predates
per-trial condition snapshots, so the two runs' background conditions cannot be matched. Raw
repeat rows and summary: `results/minicpm5-2b/mlx-4bit-inproc-decode-check/`.

## 3. XHToken Spark-X2.5-4B

A 4.112B dense model with 3:1 sliding-to-full attention has the slowest long-prompt prefill in
the roster: about 26.7–27.6 s for 8,631 tokens, versus 9.3–14.6 s for the other 4-bit rows. Its
4-bit decode is 29.4 tok/s on the long decode prompt, with a 3.971 GB in-process allocator peak.
The model's longer-than-cap generations on several quality prompts also mean the single-rater
scores sometimes judge truncated text (see section 5).

| Tier | Weights GB | bpw | Publisher | Decode long tok/s [range] | TTFT long s [range] | Prompt tok | Gen tok | Peak GB | Math strict /12 | Tool /10 | Format /8 | Total /30 |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4-bit (`mlx-4bit`) | 2.32 | 4.51 | **third-party** (`abenzerps`) | 29.4 [28.7–30.5] | 26.663 [26.594–26.865] | 8,631 | 256 | 3.971 | 2 | 10 | 6 | 18 |
| OptiQ 4/8 (`optiq-4bit`) | 3.05 | 5.93 | mlx-community | 25.6 [23.5–25.7] | 27.603 [27.217–33.074] | 8,631 | 256 | 4.698 | 2 | 9 | 7 | 18 |
| 8-bit (reference) | 4.38 | 8.52 | **third-party** (`abenzerps`) | 20.6 [19.6–20.7] | 27.387 [26.799–28.275] | 8,631 | 256 | 6.027 | 4 | 10 | 7 | 21 |

There is no vendor MLX conversion for Spark, so its plain 4-bit and 8-bit rows use third-party
weights; the OptiQ row is from mlx-community. This publisher difference is a confound in any
plain-versus-OptiQ conclusion. The objective set shows perfect tool-call pass rates for the plain
4-bit and 8-bit checkpoints, not for OptiQ (9/10); arithmetic scores remain low across tiers.

## 4. Liquid AI LFM2.5-8B-A1B

LFM2.5 combines 8.468B total parameters with a vendor-labelled ~1.3B active per token (the
registry's label, not a locally counted parameter total), in a hybrid architecture with 18 of
24 convolutional layers. It has the highest 4-bit long-prompt decode median (68.8 tok/s) and
lowest long-prompt TTFT (9.285 s for 8,510 tokenizer-specific prompt tokens). The 8-bit checkpoint
fits in this run: its peak is 9.947 GB, the highest in the matrix but below the machine's 16 GB
unified-memory capacity. Peak figures are MLX allocator values, not whole-process RSS.

| Tier | Weights GB | bpw | Publisher | Decode long tok/s [range] | TTFT long s [range] | Prompt tok | Gen tok | Peak GB | Math strict /12 | Tool /10 | Format /8 | Total /30 |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4-bit (`mlx-4bit`) | 4.85 | 4.58 | vendor | 68.8 [63.8–73.9] | 9.285 [9.186–9.547] | 8,510 | 256 | 5.779 | 4 | 10 | 6 | 20 |
| OptiQ 4/8 (`optiq-4bit`) | 5.47 | 5.17 | mlx-community | 59.6 [49.7–63.5] | 9.277 [8.919–9.855] | 8,510 | 256 | 6.370 | 4 | 10 | 7 | 21 |
| 8-bit (reference) | 9.02 | 8.52 | vendor | 37.9 [36.4–44.8] | 9.347 [9.215–9.510] | 8,510 | 256 | 9.947 | 5 | 10 | 6 | 21 |

**Thinking and end-to-end latency caveat.** LFM emits a reasoning block; the template has no
switch to turn that behavior off. The speed `decode_tps` measures generation throughput including
reasoning tokens, not time to a useful final answer. On the 4-bit `decode-short` prompt (53
prompt tokens, 256 generated tokens), raw decode is 70.9 tok/s [69.8–73.1], while total `wall_s`
from before template rendering through stream completion is 3.734 s [3.627–3.798]. A later,
separately tagged repeat measured 86.1 tok/s [82.6–86.6] and 3.096 s wall [3.075–3.208]; it is
not a replacement for the full matrix row. Both generations hit the 256-token cap; the prompt asks
for at least 400 words, so this measures raw generation throughput and time to finish a truncated
stream, not time to a useful completed answer. The measured long-prompt TTFT is time to first
generated token (often reasoning), not time to the first user-visible answer.

For P3, LFM's accuracy and quality budgets are 1,024 and 1,536 generated tokens respectively,
four times the 256/384 caps used by other models. The reasoning tokens count toward the reported
generation volume and elapsed wall time; `response` contains the extracted text after the thinking
block when that block closes. Two of eight 4-bit quality answers hit the 1,536-token cap, so some
scored content can still be truncated. This budget difference is disclosed and makes cross-model
quality scores less like-for-like.

#### Focused 4-bit decode repeat (not a replacement for the matrix row)

The full P2 `decode-short` row reports **70.9 tok/s [69.8–73.1]** over five kept trials
(53 prompt tokens, 256 generated tokens). A separately tagged five-trial repeat measured
**86.1 tok/s [82.6–86.6]**, with TTFT **0.1163 s [0.1158–0.1434]**, wall **3.096 s [3.075–3.208]**,
and peak MLX allocator memory **4.918 GB**. All five repeat decode rates (82.6–86.6 tok/s) exceed
the original maximum (73.1 tok/s); this is a repeatable discrepancy, not proof that the original
row is invalid or that LFM2.5 is generally faster under all conditions. The repeat was labelled
`loaded`, with WindowServer activity up to 31% CPU and 0.611 GB swap in use. This does not prove
the model paged or that background apps caused the speed difference. Keep both measurements
visible pending an investigation or a comparable full P2 rerun.

The original run predates per-trial condition snapshots, so its background load cannot be matched
to this repeat. Raw repeat rows and summary: `results/lfm2.5-8b-a1b/mlx-4bit-inproc-decode-check/`.

---

## 5. Cross-model comparison

`results/COMPARE.md` is generated from the committed P2 summaries and trial files. It contains the
12 in-process model/checkpoint rows, each with tier, bpw, publisher, long-prompt token count,
generation length, median decode rates with the short-prompt range, long-prompt TTFT and memory,
TTFT spread, accuracy total, and the recorded condition. The two focused decode repeats (MiniCPM5
and LFM2.5) are shown separately and do not replace their matrix rows. The file labels the
weight-read proxy explicitly: `weights GB × decode tok/s` is a size-based estimate, not measured
memory bandwidth. The `~` prefix on LFM rows marks an upper bound because only a subset of its MoE
weights is active per token.

Rows recorded before condition snapshots say `not recorded`. The run operator recalls that many other apps
were open while these measurements ran. That recollection is noted for context, but the trial files
do not contain snapshots for these rows; the report cannot quantify the load, attribute a speed
penalty, or compare the results to a clean-machine run. No historical measurement has been edited.
The repeated MiniCPM5 and LFM decode checks are explicitly tagged `loaded`; their snapshots show
background activity and swap in use, which is not proof the model itself paged. These repeats also
remain separate from the full matrix.

### Matrix readout and scoring distinctions

Across the 4-bit tier, LFM2.5 has the highest long-prompt decode median (68.8 tok/s on its
8,525-token decode prompt, with 256 generated tokens) and the lowest prefill TTFT for the separate
8,510-token `long` prompt (9.285 s). Its
`decode_tps` includes reasoning tokens, however, and is not a measure of time to a useful answer;
section 4 reports wall time and the unequal reasoning-token budget. Spark-X2.5 has the slowest
long-prompt TTFT at every tier (26.663–27.603 s for its tokenizer's 8,631-token prompt). MiniCPM5
has the smallest 4-bit peak (2.465 GB) and 69.0 tok/s short decode, but its long-prompt TTFT is
14.606 s. Nemotron's 4-bit and OptiQ values are historical; its 8-bit row was measured in the
later registry-driven matrix. Do not interpret small differences in TTFT as rankings without
considering the within-row trial ranges and differing per-model tokenizers.

The P3 total remains the original strict checker score out of 30. It combines three distinct
things, so the report separates them rather than calling the whole total “accuracy”. The
post-hoc numerical audit parses the final numeric claim only; it does not infer that a chain of
intermediate calculations is valid. It is a transparent diagnostic, not a replacement grader:
any ambiguity or missing final number remains a failed audited value.

| Model / tier | Publisher | bpw | Arithmetic strict pass /12 | Numeric final answer correct on audit /12 | Function-call JSON /10 | Output-format constraints /8 | Original total /30 |
|---|---|---:|---:|---:|---:|---:|---:|
| Nemotron 4-bit (historical) | **third-party** (`rapid-mlx`) | 4.50 | 6 | 6 | 10 | 8 | 24 |
| Nemotron OptiQ (historical) | mlx-community | 7.00 | 7 | 7 | 10 | 8 | 25 |
| Nemotron 8-bit | **third-party** (`rapid-mlx`) | 8.53 | 6 | 6 | 10 | 8 | 24 |
| MiniCPM5 4-bit | vendor (`openbmb`) | 4.55 | 3 | 5 | 10 | 6 | 19 |
| MiniCPM5 OptiQ | mlx-community | 6.17 | 1 | 3 | 10 | 6 | 17 |
| MiniCPM5 8-bit | mlx-community | 8.52 | 2 | 5 | 10 | 6 | 18 |
| Spark-X2.5 4-bit | **third-party** (`abenzerps`) | 4.51 | 2 | 3 | 10 | 6 | 18 |
| Spark-X2.5 OptiQ | mlx-community | 5.93 | 2 | 3 | 9 | 7 | 18 |
| Spark-X2.5 8-bit | **third-party** (`abenzerps`) | 8.52 | 4 | 4 | 10 | 7 | 21 |
| LFM2.5 4-bit | vendor | 4.58 | 4 | 12 | 10 | 6 | 20 |
| LFM2.5 OptiQ | mlx-community | 5.17 | 4 | 11 | 10 | 7 | 21 |
| LFM2.5 8-bit | vendor | 8.52 | 5 | 11 | 10 | 6 | 21 |

“Arithmetic strict” is the committed `int_match`/`float_match` verdict: the entire response must
be a single numeric literal, so a correct number inside `\\boxed{...}` or explanatory prose still
fails that original check. “Numerical answer correct on audit” is a separate post-hoc read of the
final answer in the raw result files against the same numeric target; it does not replace, rescore,
or edit any committed verdict. In particular, LFM2.5's correct values often appear as LaTeX boxes,
while other models sometimes include the correct value in prose. The arithmetic score must
therefore not be presented as numerical correctness alone. The 8 instruction-following prompts
are strict, mechanically checked output constraints (e.g. sentence count, word/character bans and
format), not a general measure of following instructions. Per-prompt extracted values and verdicts
are reproducible with `python audit_p3_numeric.py`; `--check` asserts the totals above.

### Blind quality round r1 — exploratory, single rater

The 8-prompt blind set compares only the four plain 4-bit checkpoints. It is a preference signal,
not a statistically established ranking: one rater, eight hand-authored prompts, and one output per
prompt. Slots were balanced (each checkpoint appeared twice in each of A–D), but the scorer reports
slot means A=9.75, B=9.88, C=11.12, D=11.00 on the 15-point-per-prompt scale. The 1.37-point
spread exceeds the project's 1-point position-bias warning threshold; treat the ordering as
especially tentative. The sealed key is in `results/blind-r1/key.json`, scores in
`results/blind-r1/scores.json`, and generated summary in `results/blind-r1/summary.json`.

| 4-bit row | Mean /5 | Total /120 | Per-prompt range /15 | Top-scoring prompts (shared wins) |
|---|---:|---:|---:|---:|
| LFM2.5-8B-A1B (vendor) | 3.79 | 91 | 8–13 | 3 |
| MiniCPM5-2B (vendor) | 3.42 | 82 | 4–15 | 1 |
| Spark-X2.5-4B (third-party `abenzerps`) | 3.38 | 81 | 7–15 | 2 |
| Nemotron 3.5 Bolt (third-party `rapid-mlx`) | 3.33 | 80 | 7–15 | 2 |

LFM's quality cap was 1,536 generated tokens, versus 384 for each other model (four times as
large), because its reasoning block must finish before the final answer can be extracted. Its
median generated volume was 915.5 tokens (range 154–1,536); two answers reached the cap. Spark's
four answers also reached its 384-token cap, so truncation may affect both models' ratings. These
scores are not a like-for-like latency or completion-quality comparison. The blind run used the
same four 4-bit answer files, but it did not normalize answer length or truncate/re-run responses.

The mean category scores (/5) add context but do not cure the small sample or slot effect:

| 4-bit row | Publisher | bpw | Technical accuracy | Coherence | Instruction fit |
|---|---|---:|---:|---:|---:|
| LFM2.5-8B-A1B | vendor | 4.58 | 3.75 | 4.00 | 3.62 |
| MiniCPM5-2B | vendor (`openbmb`) | 4.55 | 3.25 | 3.50 | 3.50 |
| Spark-X2.5-4B | **third-party** (`abenzerps`) | 4.51 | 3.25 | 3.62 | 3.25 |
| Nemotron 3.5 Bolt | **third-party** (`rapid-mlx`) | 4.50 | 3.25 | 3.88 | 2.88 |

Do not declare a quality winner from these means. The scores and slot imbalance warning are
reported together; additional raters and a repeat round would be needed to test whether the
observed preferences persist.

### Scope and limits

The 30-prompt set is a small, hand-authored ML/data-engineering sample, not a general capability
evaluation. Strict-format accuracy, numeric answer correctness, tool-call JSON matching, and
human-rated quality answer different questions. Checkpoint comparisons also mix different
publishers and bit budgets, especially Spark's third-party plain quants. The results support
observations about these files, prompts, settings and this M4 only, not a universal model ranking.

The decode-short repeats were measured with other applications open (`--condition loaded`). Their
per-trial snapshots show background activity but do not establish a measured slowdown caused by
those apps. The earlier P2 matrix rows lack system snapshots; the operator recalls many apps were
open, which may have slowed them, but the extent cannot be reconstructed or corrected. Do not
interpret those historical rows as clean-machine measurements.

P2 memory values in the generated comparison are `mx.get_peak_memory()` allocator peaks for the
in-process rows; historical Nemotron 4-bit and OptiQ summaries predate the structured provenance
field, so their allocator basis is inferred from the recorded in-process transport. These numbers
are not process RSS or GPU VRAM. P2 conditions are not recorded in the old matrix, as noted above.

No number in `results/COMPARE.md` substitutes for the individual row's model, checkpoint, bpw,
publisher, tokenizer-specific prompt length, generation length and memory basis. The raw JSON is
the source of truth; these summaries are not a leaderboard.
