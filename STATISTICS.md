# Statistical design: local-model decisions on a 16 GB M4

**Question:** Which checkpoint/runtime is worth deploying for a *specified* workload on this machine, with defensible uncertainty about quality, latency and memory?

This project is a small, hardware-specific observational study until the replicated experiments below are performed. `REPORT.md` preserves the original experiment. `study.py` generates `results/STUDY.md` from the raw committed records; it does **not** rewrite historical observations or estimate unsupported confidence intervals.

## 1. Unit of analysis, outcomes, and constraints

A candidate is `(model identity, exact Hub revision, weight SHA-256, quantization/bpw, engine, transport, template/decoding settings)`. Changing any member of this tuple changes the candidate. The unit for a **reproducibility comparison** is a separately initialized, logged *session*, not an individual token or one of five sequential same-process requests. The unit for a **quality comparison** is a task (ideally a held-out sample from a declared workload distribution).

Measure each candidate using outcomes with separately defined meanings:

| Outcome | Direction | What it does **not** mean |
|---|---|---|
| Weight files, GB decimal | minimize | GPU VRAM or peak memory |
| Peak **total** app/system memory and swapping | minimize | MLX allocator peak alone |
| Request TTFT and time to first visible content | minimize | time to a correct final answer |
| Total task wall time, including model/tool turns | minimize | single-turn decode throughput |
| Decode tokens/s after prefill | maximize | tokens/s of correct useful output; thinking tokens count |
| Task completion, tool correctness, safety constraints | maximize | a P3 JSON-formatted tool example |
| Failure/truncation/timeout rates | minimize | zero if an experiment is missing |

Both `mlx_lm` and `mlx_lm.server` can supply meaningful measurements, but timing boundaries and memory bases differ. Compare matched transport/engine conditions only. See [NVIDIA NIM LLM metrics](https://docs.nvidia.com/nim/benchmarking/llm/metrics) and [MLPerf Client](https://mlcommons.org/benchmarks/client/) for TTFT, output throughput and end-to-end latency definitions.

## 2. What can be calculated **now**

- Recalculate actual trial medians from committed raw JSON and refuse a changed/missing median (automated by `study.py`).
- Describe session medians, observed min/max, per-checkpoint weight and actual bpw, P3 per-category scores, and the Pareto frontier for *specified* measured dimensions.
- Calculate within-model paired P3 item flips (`fail→pass`, `pass→fail`), not just differences between totals. The optional exact McNemar/binomial calculation is exploratory: the current 30 prompts were hand-authored and do not represent a random task population.
- Show quantization costs as `100 × (new / 4-bit − 1)%` for weight size and speed. An OptiQ-vs-4-bit contrast is also a checkpoint-publisher/bpw contrast, not a pure causal quantizer effect.
- Show the unvalidated 20-task P4 pilot as a *pilot*, not as established agent ability or a 0% population success estimate.

**Do not** bootstrap the five sequential P2 trials and claim a 95% hardware-performance confidence interval; that would confuse within-session jitter with between-session variability. **Do not** use a single weighted `accuracy/30` as a general intelligence measure: the categories mix numerical correctness, exact JSON arguments and format constraints. Qualitative scores are eight prompts from one rater with asymmetric output budgets.

## 3. Controlled follow-up experiment

### Phase A: reproducibility and shortlisting

1. Freeze the repo commit, OS, runtime/driver versions, checkpoints, tokenizer/templates, generation settings, prompt text hashes, power configuration and expected output conditions. Keep working changes on a separate branch and append new result directories.
2. Run **all four plain 4-bit candidates** in **at least five independently initialized sessions per candidate** (20 sessions total, exploratory minimum). Restart the inference process and clear relevant state between sessions; document whether the model/cache remained resident. Randomize/counterbalance candidate order across sessions so time, heat and load do not always favor the same model.
3. Within each session, run identical user-visible prompts across workload strata (short chat, approximately 1k and approximately 8k context, 128–256 generated tokens). Discard a documented warmup and retain at least five kept trials per stratum. For cached HTTP, use an explicitly separate arm.
4. Record session ID, run order, wall-clock date/time, battery/AC, thermal state, CPU interference, preexisting swap, memory pressure, exact input/output tokens, response stop reason, and start/end clock boundaries. Distinguish intended `clean`/`loaded` from observed conditions; do not declare a session clean from a label alone.
5. **Primary estimands:** median completed-task end-to-end latency per task class; held-out task pass rate by class; median/upper-tail memory; and request TTFT. **Secondary:** decode throughput, first-content latency and peak allocator memory. Store failed runs and missing measurements; never silently filter them.
6. If a contrast looks decision-relevant, add independent sessions until uncertainty is narrow enough to discriminate a practical difference. Five sessions are a *starting point*, not a guarantee of enough power or valid p95 estimates.

### Phase B: quantization and robustness

On the most relevant two or three models, repeat the same controlled conditions for plain 4-bit, OptiQ and 8-bit, while retaining actual bpw and publisher. Record 4-bit vs 8-bit paired task outcomes and end-to-end latency costs. To isolate quantization as a *causal* factor, use known identical base weights plus a controlled quantizer/producer; published third-party checkpoints can otherwise differ in unobserved ways.

### Phase C: tool-using workloads and deployed systems

Validate the agent sandbox and protocol with an obvious deterministic fixture and at least one deliberately well-behaved known-capable baseline. Then compare the custom JSON-text protocol with supported native tool calling; use the same tasks and tooling semantics, with a locked holdout. Real outcomes are tool execution, final file/content equality, correct tool sequence, time-to-correct-completion, refusals and failure rates. Report agent usefulness only after this validation. For a physical assistant include speech/audio I/O latency, interruptibility and actual end-to-end control reliability; a text-only MLX benchmark cannot certify those.

## 4. Statistical model *after* collecting replicated data

For each workload stratum, use a matched hierarchical/blocked analysis rather than pretending requests are independent:

\[
\log T_{m,q,s,p,r} = \mu + \alpha_m + \beta_{q(m)} + \gamma_p + u_s + \varepsilon_{m,q,s,p,r}
\]

`m` = model, `q(m)` = checkpoint/quantization nested in model (including its producer), `s` = session/block, `p` = fixed prompt or workload stratum, `r` = repeated trial within session. Session-level effects account for warm/cold model state, machine background conditions and timing. Fit a hierarchical model only if the required repeated and crossed cells exist; do **not** fit this to the current matrix as though five kept trials were five sessions.

For matched configurations A and B, first aggregate repeated trials **within** each session/task, calculate paired differences across independent sessions/blocks, and report robust effect sizes and an interval from the session-level/cluster bootstrap or another justified method. The inferential unit and sampling assumptions must be stated. If tasks are sampled from a real workload distribution, estimate quality uncertainty over *tasks*, keeping correlated examples or prompts grouped. If task sets remain fixed and curated, treat the accuracy comparison as descriptive and report item-level disagreements.

## 5. Multidimensional decision mathematics

Represent each candidate by a measured vector such as

\[
x_i = (\text{task success}_i,\,\text{latency}_i,\,\text{decode}_i,\,\text{memory}_i,\,\text{failure rate}_i).
\]

First apply **hard deployment constraints** for the actual application (for example, runtime memory must fit a 16 GB shared-memory machine and agent success must exceed a declared threshold). **Missing metrics fail eligibility**, rather than being converted into zero, average, or an imputed guess.

Then compute the Pareto frontier: candidate A dominates B if it is no worse on every eligible objective, and strictly better on at least one. The current `study.py` only publishes a **two-dimensional descriptive** frontier for stored weights and short decode throughput. That frontier cannot decide which model makes a better hardware assistant.

Finally, if a single selection is necessary, specify *before viewing results* a workload utility function over task-level outcomes, for example

\[
U_i(w) = w_Q Q_i - w_L L_i - w_M M_i - w_E E_i,
\quad w_j\ge 0,\quad \sum_jw_j=1.
\]

Use consistently interpreted, calibrated/normalized dimensions; do not add raw seconds to raw GB. Report a **weight sensitivity analysis** (which option wins across reasonable preference weights), uncertainty intervals, and where the recommendation flips. Arbitrary equal weights are not scientific truth. Prefer hard constraints and a transparent Pareto front to a single opaque leaderboard number. See [recent multi-criteria decision research](https://doi.org/10.1287/mnsc.2025.00510) on robustness to preference-weight uncertainty.

## 6. Release and reporting gates

A recommendation for real deployment requires: (a) passing provenance/P0, (b) logged and independently replicated timing, (c) task-representative held-out quality, (d) validated real tool execution for agent use, (e) memory/latency evidence at the intended context length and application load, and (f) a published workload-specific choice rule with sensitivity. Until then, publish **observations, uncertainties and next experiments**, not a universal model winner.

`study.py` stays standard-library-only and generates `results/STUDY.md`. It is a first usable analytical layer, not the final statistical model. Historical P2/P3/P4 rows remain immutable; new sessions and protocols get distinct run IDs and output directories.

## 7. Replicated-session infrastructure (issue #34)

`replicated.py` implements the Phase A follow-up protocol described
above. [SESSION_PROTOCOL.md](SESSION_PROTOCOL.md) defines the locked five-block
four-model plain-4-bit design, immutable per-session schema, Mac-only preflight,
measurement and report commands. It reuses `bench.py` for inference, rather than
creating a competing model runner. `tests/test_replicated.py` exercises the
session-design and evidence/analysis gates entirely offline.

Statistical unit: **one newly initialized model session per block**. A session's
five kept trials per fixed prompt are summarized *before* cross-session analysis.
The new report uses exploratory session-level and paired whole-block percentile
bootstraps only after all 20 planned sessions validate; it does not claim that
five sessions confer nominal 95% frequentist coverage or enough statistical
power. The full-system `vm_stat` estimate is distinct from MLX allocator peak,
and response-end wall time is distinct from independently graded time to a usable
answer. In-process first-content latency remains unmeasured. Compromised
observed conditions are flagged separately from operator labels. The source
matrix and `results/STUDY.md` remain historical until actual M4 measurements
are collected and validated; **no CI/fixture data are experimental results**.

## 8. First completed replication — 2026-10-29

[Full report](results/replicated/m4-4bit-20261029/REPLICATED_STUDY.md)
and immutable session evidence record five complete blocks / 20 independent
sessions at source `38c5c9c04d84d50041483401c9e3489f6c49576d`.
Independent revalidation passed all 20 sessions; all 300 kept answers matched
the duration target. The completion pilot revised the common prompts and raised
the cap to 512 before the plan was frozen; see SESSION_PROTOCOL.md and the
preserved failed pilots. These prompts are pilot-informed retrieval tasks,
not a held-out quality suite or a sustained decode workload.

The report estimates session-level variability and paired whole-block contrasts.
MiniCPM has the smallest allocator footprint and shortest short/medium response
latency in this workload; LFM has the shortest long-context response latency,
including its internal reasoning. Tokenizer counts and completion lengths differ:
consult the report's complete checkpoint-labelled rows rather than extrapolating
to model intelligence or a universal ranking.

All 20 sessions have observed CPU-interference flags, preexisting swap and
unavailable CPU thermal-speed telemetry. Kept-trial swap snapshots range from
2.313 to 4.077 GB. Operator intent remains `clean`; the recorded warnings are
not removed or relabelled. Cooldowns and quiet pre-block checks do not prove
quiet conditions throughout a session. Therefore this satisfies replication
and evidence completeness, but not an independently verified interference-free
measurement. More controlled replication and representative quality tasks are
needed before deployment recommendations. Historical P2/P3 measurements remain
unchanged and are not pooled with this new workload.
