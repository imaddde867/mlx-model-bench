# LFM3 versus MiniCPM5 on the M4 — separate experiment

Status: preparation complete only after tests/review; operator confirmation is
required before model loading or timed inference. No performance winner claimed.
This experiment does not expand or overwrite the four-model study in issue #34.

## Question and candidates

Does vendor LFM3-1.7B 4-bit offer a useful compact alternative to vendor
MiniCPM5-2B 4-bit on this Apple M4, 10-core CPU / 16 GiB laptop?
Both use the same isolated `.venv-lfm3`, with MLX 0.32.4 and mlx-lm commit
`9027ac5b062175fba890c612c3905d5d8d5b5294` (upstream PR #2016, still unmerged
when checked 2026-10-29). Main `.venv-mlxlm` stays unchanged. This is explicitly
an experimental-runtime comparison, not a released-support claim.

Metadata and chat controls come from registry.py. The LFM3 checkpoint is pinned
to `d0ca73dcfa70ce18733754a492cdc1fb0123262d`, verified against the vendor Hub
SHA-256. Its template supports `enable_thinking` and `preserve_thinking`;
both are explicitly false in the primary comparison. MiniCPM retains its
registered thinking-off settings. Generation is greedy; vendor sampling and
repetition-penalty recommendations are not the primary control.

A vendor 8-bit LFM3 conversion exists (research metadata saved alongside this
protocol); it is not selected for this 4-bit comparison. OptiQ is not selected
or claimed compatible. No GGUF, vision/audio, 64K or advertised-maximum context
tests are included.

## Gates and immutable evidence

Reuse the existing fetch SHA-256 / P0 sanity and same-prompt-twice determinism
gates for both models under the experimental interpreter. Review raw/final
answers and verify thinking-off template rendering before timed performance.
Existing M4 full-precision parity evidence is compatibility evidence only.

Freeze a clean source commit, runtime and exact PR revision, model weight and
configuration/tokenizer/template hashes, prompt/task hashes, caps and model
order in an exclusive plan. Rehash before each block. Every session gets a new
process, UUID, condition observations, log and hashed manifest. Failed evidence
is retained; the study refuses in-place reruns. A failed planned session means
investigate before declaring completeness, not omit the row.

## Performance

Five paired blocks, ten fresh model processes. Seed 37029 picks the initial
order; order reverses each block. With five blocks, one model starts three
blocks and the other two: the residual imbalance is disclosed. Cool down for
120 seconds before each block, require AC and low-power mode off, wait for a
quiet CPU snapshot with bounded retries, and reject observed thermal throttling.
Unknown telemetry and any later interference remain caveats, never proof of
quiet conditions. User confirmation that applications are closed precedes launch.

Each process runs existing bench.py with identical five `prompts/speed` files
(short, medium, long, decode-short, decode-long), cap 256, no nonce, one discarded
warmup and five kept trials per regime. A length stop is a capped generation
observation, not completed-task success. Retain actual output lengths/text,
TTFT, full response wall time, prefill/decode metrics, per-trial MLX allocator
peak and separately sampled host-memory estimates. Compare tokenizer-specific
counts, not token counts as an achievement.

Reduce five trials to one median per session/regime, retain min–max, and compare
paired block differences. Any bootstrap intervals are exploratory with five
blocks and cannot establish reliable population coverage. Do not combine these
sessions with older runtime or workload records, or rank models on unequal
output lengths without explaining what was measured.

## Capability and tooling

Run P3's fixed 30 objective and eight quality prompts once per model with the
same existing graders and common caps (256 accuracy / 384 quality). Preserve
finish reason, raw/final text, token counts and pre-generation conditions.
Quality outputs remain unranked until independently rated; this is a small,
known convenience suite, not a held-out population quality estimate.

Run the existing 20-task P4 pilot once per model, common 256-token per-turn cap
and seven-turn limit. Report exact completion/file checks, tool errors, raw
steps, finish reasons and instrumented task wall time separately from summed generation time.
The task timer includes condition collection, template work and tool orchestration.
This is in-memory JSON-as-text tool simulation, not native API tool execution,
real device safety or proven agent competence. Capability timings are
exploratory single task runs, not independently replicated latency estimates.

HTTP and native-tool support require their own compatibility smoke checks after
the in-process gate; do not infer success from a chat template containing tool
tags. They must remain unverified until actual response/tool-call behavior is
recorded. The first batch makes no native-tool reliability claim.

## Readout

Publish one compact LFM3/MiniCPM comparison with exact checkpoint, publisher,
bpw, runtime, prompt/output lengths, ranges, session counts and observed flags.
Report P3 categories separately and label P4 exploratory. Retain failed or
truncated answers. Update the use-case guide only where the new measurements
support it; no general SOTA or statistically proven agent winner is implied.
