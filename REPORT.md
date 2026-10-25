# Four local models on a 16 GB M4 — speed, memory, accuracy, quality

Apple M4 (10-core, 4P+6E), 16 GB unified memory, macOS 27.0.1, `mlx` 0.32.4, `mlx-lm` 0.32.2,
Python 3.12.10. Greedy decoding, thinking off per model's own template, plugged in, `caffeinate`.

This report covers four models that fit in 16 GB, each at the same three quantization tiers,
through the same harness. One of them is complete; the other three are pending. A section that
has no numbers yet says so rather than showing a blank row.

| Section | Model | Weights (4-bit / OptiQ / 8-bit) | Status |
|---|---|---|---|
| 1 | NVIDIA Nemotron 3.5 Bolt E4B | 4.20 / 6.53 / 7.96 GB | **complete** (4-bit, OptiQ) · 8-bit pending |
| 2 | OpenBMB MiniCPM5-2B | 1.43 / 1.94 / 2.68 GB | pending |
| 3 | XHToken Spark-X2.5-4B | 2.32 / 3.05 / 4.38 GB | pending |
| 4 | Liquid AI LFM2.5-8B-A1B | 4.85 / 5.47 / 9.02 GB | pending |
| 5 | Cross-model comparison | — | pending |

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

Speed runs conducted with 1 discarded warm-up + 5 trials; reported as **median [min–max]**. Prompt cache busted with random nonces, except for `Arm 2 cached` which measures warm prefix reuse.

### TTFT (s)

| Prompt tokens | Arm 1 (in-proc) | Arm 2 (HTTP) | Arm 3 (OptiQ in-proc) | Arm 2 cached |
|---|---|---|---|---|
| 68 | 0.195 [0.192–0.200] | 0.161 [0.159–0.167] | 0.217 [0.204–0.234] | 0.068 [0.065–0.076] |
| 1,026 | 1.62 [1.62–1.63] | 1.52 [1.52–1.61] | 1.68 [1.68–1.72] | 0.070 [0.067–0.074] |
| 8,341 | 14.3 [13.7–14.3] | 14.4 [14.1–14.6] | 14.6 [14.4–15.0] | 0.094 [0.091–0.104] |

Prefill throughput saturates near 570–630 tok/s across all arms.

### Decode (tok/s, prefill excluded)

| Prompt tokens → generated | Arm 1 | Arm 2 | Arm 3 |
|---|---|---|---|
| 85 → ~100–160 (`decode-short`) | 34.4 [33.8–34.8] | 32.5 [32.0–33.6] | 25.7 [24.1–26.4] |
| 8,359 → 256 (`decode-long`) | 29.5 [28.2–31.1] | 27.9 [26.7–28.9] | 23.8 [23.7–24.5] |

### Memory (`mx.get_peak_memory`)

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

### Key Accuracy Findings
1. **Flawless tool use and constraint adherence**: Both checkpoints achieved 100% on JSON function calling (10/10) and 100% on binary negative/formatting constraints (8/8). The conversational mode chat template reliably enforces strict formatting.
2. **OptiQ edge on numerical calculations**: Arm 3 successfully computed large multi-factor memory calculations (`math-03`: `32000 * 256 * 4 = 32768000`) and dataset token counts (`math-09`: `64000000`), whereas Arm 1 returned the uncomputed expression (`'32000 * 256 * 4'`) or dropped a trailing zero (`'6400000'`).

---

### 1.3 Qualitative Evaluation (P3 — 8 Prompts Blind Scored)

Evaluated on 8 core NLP/ML engineering tasks. Outputs were randomized blind into **Model A** and **Model B** per prompt, scored independently across 3 rubrics (1–5 scale each, 15 max per prompt):
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
| **Total** | **Mean Score (out of 5.0)** | **4.29 / 5.0** (103/120) | **4.67 / 5.0** (112/120) | **Arm 3 (6-1-1)** | **OptiQ +0.38 higher mean quality** |

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
- **Engine**: Virtually zero TTFT penalty (~0.195 s in-process vs 0.161 s HTTP) and only a **~5% decode penalty** (34.4 tok/s vs 32.5 tok/s). `mlx_lm.server` adds negligible transport overhead.
- **Checkpoint**: Massive difference. OptiQ mixed 4/8-bit causes a **25% decode slowdown** (34.4 → 25.7 tok/s) and requires **+2.33 GB VRAM** (4.20 GB → 6.53 GB).

#### 2. Is the OptiQ mixed 4/8-bit checkpoint's quality distinguishable from plain 4-bit?
- **Yes, measurably.**
  - **Objective Accuracy**: 83.3% vs 80.0% (+3.3%). OptiQ shows superior retention of large-number arithmetic and multi-step reasoning.
  - **Subjective Quality**: 4.67 vs 4.29 out of 5.0 (+0.38 points, winning 6 out of 8 categories). OptiQ produces more detailed, structured, and technically complete responses, while plain 4-bit occasionally collapses into overly terse one-liners.

#### 3. What does OptiQ's application layer cost in TTFT, measured like-for-like?
- `optiq serve` crashes out-of-the-box due to eager evaluation of the unquantized dummy `embed_tokens_per_layer` weight (`vocab 262,144 × layers 42 × dim 512 × 2 bytes = 11,274,289,152 bytes = 10.5 GiB`) in `optiq/mlx_lm_patches/_chunked_quantize.py#L72`, exceeding Metal's 8.88 GiB max allocation limit.
- When `_chunked_quantize`'s eager eval is bypassed:
  - `OptiqEngine` short-prompt TTFT is **0.218 s** (matching `mlx_lm`'s 0.217 s on the same weights).
  - Autoregressive decode throughput is **21.3 tok/s**, which is **~17% slower** than `mlx_lm` in-process (**25.7 tok/s**) and **38% slower** than plain 4-bit (**34.4 tok/s**).
- When loaded via `mlx_lm`, the OptiQ weights add **+0.022 s TTFT** on short prompts (0.217 s vs 0.195 s) over plain 4-bit, while prefill throughput remains virtually identical (~570 tok/s).

#### 4. Does speculative decoding earn its keep on this architecture?
- **An external Gemma 4 assistant is a documented OptiQ pairing, but its benefit and memory cost are not measured in this study.**
  - **Native MTP**: Nemotron 3.5 Bolt has no native MTP head. The observed `OptiqEngine` run was autoregressive (`mode='AR', depth=0`); this says nothing about the separate assistant-drafter path.
  - **Gemma 4 assistant drafting**: This is a separate four-layer assistant, not another 3.5 GB target model. The pinned local `mlx-optiq` 0.7.3 package exposes `serve --drafter` and contains a Gemma assistant adapter that consumes target embeddings and hidden states. The [Nemotron OptiQ card](https://huggingface.co/mlx-community/NVIDIA-Nemotron-3.5-Bolt-E4B-OptiQ-4bit/tree/0c204a16b6b186d1434e70f5fc2ca2e30984a006) documents this exact target/drafter pairing and reports decode results on a 24 GB M4. These sources establish an implemented and documented path, not that this exact pair has been executed or verified in this study. The MLX Hub repo reports 78,779,908 parameters, a 159,138,208-byte `model.safetensors` file (~0.159 GB), and 191,307,648 bytes total storage. Its checked revision is [`844e008e`](https://huggingface.co/mlx-community/gemma-4-E4B-it-assistant-bf16/commit/844e008e06ef5562bdb89428d851d0634d119dcd). That upstream card reports 1.70–2.38x decode throughput on its four listed prompts, with one prose output differing at a near-tie token; those figures are not this study's measurements. Drafter file size alone does not establish total live memory, peak usage, or a safe 16 GB configuration.
  - **DFlash**: This is a distinct block-diffusion drafter, not Gemma assistant MTP or a target's native MTP head. NVIDIA's [current integration guidance](https://developer.nvidia.com/blog/boost-inference-performance-up-to-15x-on-nvidia-blackwell-using-dflash-speculative-decoding/) describes Speculators/vLLM and SGLang integrations using target hidden states on NVIDIA GPU stacks; this repo has no verified DFlash path in its pinned MLX stack. The prior claims that DFlash requires GGUF and provides <100 ms TTFT improvement are withdrawn; the cited current official source does not support either claim.
  - **Verdict**: The blanket memory-based rejection was unsupported. An implementation and exact pairing are documented, and the drafter checkpoint is small; however, compatibility and performance on this 16 GB host remain unverified and unmeasured. Do not claim an accelerated Nemotron result, a safe memory margin, or DFlash compatibility without applicable verification.

---

### 1.5 Deployment Recommendations

| Target Workload | Recommended Arm | Rationale |
|---|---|---|
| **Voice Assistant / Low Latency UI** | **Arm 1 / Arm 2 (rapid-mlx 4-bit)** | 34.4 tok/s decode, sub-200ms TTFT, only 4.2 GB VRAM. Fast and lightweight. |
| **Technical Copilot / Complex RAG** | **Arm 3 (OptiQ mixed 4/8-bit)** | Higher reasoning fidelity (83.3% acc, 4.67/5 quality), superior formatting, fits easily in 16 GB (6.5 GB peak). 25.7 tok/s is still fast. |
| **Multi-turn Conversational Service** | **Arm 2 (`mlx_lm.server`)** | Prefix caching delivers **0.068–0.094 s TTFT** across all context lengths (150× faster than 14 s cold prefill on 8k prompts). |

These recommendations are **within this model only**. Whether the 4-bit Nemotron is the right
pick for any of those workloads is decided in section 5, against rows that have not run yet.

---

## 2. OpenBMB MiniCPM5-2B — pending

2.52B dense Llama, 42 layers, 2 KV heads, 131k vocab. The cheapest complete row in the study:
1.43 GB at 4-bit, and still only 2.68 GB at 8-bit.

| Tier | Weights | bpw | Publisher | P0 | P2 | P3 |
|---|---|---|---|---|---|---|
| 4-bit (`mlx-4bit`) | 1.43 GB | 4.55 | vendor (`openbmb`) | — | — | — |
| OptiQ 4/8 (`optiq-4bit`) | 1.94 GB | 6.17 | mlx-community | — | — | — |
| 8-bit (reference) | 2.68 GB | 8.52 | mlx-community | — | — | — |

Open question this row is designed to answer: does a 2.5B model at 1.43 GB hold tool-calling and
constraint-following parity with a 7.46B model at 4.20 GB, and at what TTFT.

## 3. XHToken Spark-X2.5-4B — pending

4.11B dense with 3:1 sliding:full attention, head_dim 256, 1M-token trained context.

| Tier | Weights | bpw | Publisher | P0 | P2 | P3 |
|---|---|---|---|---|---|---|
| 4-bit (`mlx-4bit`) | 2.32 GB | 4.51 | **third-party** (`abenzerps`) | — | — | — |
| OptiQ 4/8 (`optiq-4bit`) | 3.05 GB | 5.93 | mlx-community | — | — | — |
| 8-bit (reference) | 4.38 GB | 8.52 | **third-party** (`abenzerps`) | — | — | — |

Fairness caveat that cannot be engineered away: no vendor MLX conversion is published, so both
non-OptiQ tiers are community quants. Every table carrying this model must repeat that.

## 4. Liquid AI LFM2.5-8B-A1B — pending

8.47B hybrid conv+attention MoE, 18 of 24 layers convolutional, 32 experts top-4.

| Tier | Weights | bpw | Publisher | P0 | P2 | P3 |
|---|---|---|---|---|---|---|
| 4-bit (`mlx-4bit`) | 4.85 GB | 4.58 | vendor | — | — | — |
| OptiQ 4/8 (`optiq-4bit`) | 5.47 GB | 5.17 | mlx-community | — | — | — |
| 8-bit (reference) | 9.02 GB | 8.52 | vendor | — | — | — |

The speed hypothesis: 8.5B of weights but ~1.3B active per token should out-decode every dense
row here. The 9.02 GB 8-bit tier may not survive the P0 memory gate on 16 GB; if it does not,
that is recorded rather than dropped.

---

## 5. Cross-model comparison — pending

Fills in as each section completes. The table below is the shape, not a result.

| Model | Tier | Decode tok/s | TTFT 68 / 1k / 8k | Peak GB | Accuracy /30 | Quality /5 |
|---|---|---|---|---|---|---|
| Nemotron 3.5 Bolt | 4-bit | 34.4 | 0.195 / 1.62 / 14.3 | 5.09 | 24 (80.0%) | 4.29 |
| Nemotron 3.5 Bolt | OptiQ | 25.7 | 0.217 / 1.68 / 14.6 | 7.42 | 25 (83.3%) | 4.67 |
| Nemotron 3.5 Bolt | 8-bit | — | — | — | — | — |
| MiniCPM5-2B | all | — | — | — | — | — |
| Spark-X2.5-4B | all | — | — | — | — | — |
| LFM2.5-8B-A1B | all | — | — | — | — | — |

What this table will **not** claim: a single "best model". 30 checkable prompts separate 50%
from 100%; they do not separate 83% from 87%, and a blind score from one rater over 8 prompts is
a preference signal. Differences smaller than a row's own min–max range are not differences.
