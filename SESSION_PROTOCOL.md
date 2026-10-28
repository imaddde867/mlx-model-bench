# Replicated plain-4-bit performance protocol (issue #34)

**Status:** infrastructure only. No measurements are claimed by this change. The
historical P2/P3 records and `results/STUDY.md` are unchanged. Inference must run
on the **16 GiB Apple M4 Mac**, not GitHub Actions, another Mac, or a simulator.

## Experimental design

- **Candidates:** `registry.ORDER`, exactly the four existing plain `4bit` checkpoints
  (including their original HF publisher/bpw). Checkpoint selection is registry-driven.
- **Independent unit:** one **fresh subprocess** executing the existing `bench.py`
  `inproc`/MLX measurement, one model and one checkpoint per subprocess. Each
  subprocess loads the model anew and exits after measuring three regimes. A
  single process' five sequential trials **do not** provide five replicates.
- **Blocks:** five consecutive rounds, each containing four sessions (20 total).
  A predeclared integer seed randomizes model assignment to a four-row balanced
  Williams square. Every candidate takes each position once in the first four
  blocks; the fifth adds one position. Keep the planned order. Run one block at a
  time to allow cool-down and operator checks; do **not** re-sort by convenience.
  A nonblocking OS lock prohibits simultaneous blocks in the same experiment.
  Validation independently rejects overlapping UTC session intervals, reordered
  sessions, or reused session UUIDs, even if the individual records look valid.
- **Within session:** fixed `prompts/replicated/{short,medium,long}.txt` files,
  the same lighthouse question and requested two-word answer at three context
  lengths. The medium and long passages reuse historical text without changing
  those files; short uses their first paragraph. The ~1k/~8k labels are
  approximate, not a guarantee for each tokenizer. Record actual counts per
  model. One discarded warmup plus **five kept trials per regime**; greedy,
  template kwargs from `registry.py`, no explicit in-process prompt cache and
  **no nonce**. The output target is two *words*, not two generated tokens:
  tokenizers and internal reasoning differ, and model compliance is not assumed.
  The common `max_tokens=256` is a cap, not a fixed generated-token length.
  Record actual prompt/completion tokens and stop reasons; no performance metric
  here establishes that the answer is correct. Failure, empty output,
  reasoning-only output and truncation are preserved and prevent a complete
  report, not discarded or converted into successful observations.
- **Conditions:** physically plugged in and awake under `caffeinate`; same
  macOS/pinned MLX environment, screen/desktop workload held as consistent as
  practical. `--condition clean|loaded` records the **operator's stated intent**.
  Each `bench.py` trial includes its own `sysinfo.snapshot()` before the timed
  window; this layer stores before/after snapshots and derives observed warnings
  from **all** trial snapshots (including warmups) and before/after state.
  Active low-power mode is recorded and flagged; an unavailable power, low-power,
  thermal, swap, pressure or process observation is flagged as unknown, not quiet.
  A `clean` label with `mediaanalysisd` at 225% CPU stays `clean` **with an
  observed busy warning**. No historical pilot label is rewritten. Snapshots are
  intermittent and cannot establish that no background work occurred during a trial.

- **Memory:** the existing `mx.get_peak_memory()` is the *per-trial MLX allocator*
  peak, not app/system memory. Separately, `vm_stat` is polled every 0.5 s from
  an external thread for a **host-wide physical-used estimate** based on
  `total − free − inactive − speculative` pages. This is an approximate host
  value affected by other processes, not process RSS, GPU VRAM, or an exact
  workload-only peak. The raw sampled memory series is retained to permit a
  check of the stated peak. Swap, memory pressure, power and thermal CPU speed-limit
  are recorded in `sysinfo` snapshots. Direct temperature in degrees and a true
  application's total high-water mark are not available here: do not invent them.
- **Latency:** primary observed end-to-end metric is `wall_s`, i.e. time to
  response end. It is **not** time to a verified *correct or useful* answer.
  `request_ttft_s` is time to the first generated token. In-process
  `first_content_s` is unavailable and must remain missing, rather than being
  fabricated from TTFT. Later, a held-out task/quality experiment is required
  before claiming time to useful/agentic completion.

**Pilot required before the study:** these new prompts and the common cap have
not been tested on the four checkpoints. A separate, explicitly non-benchmark
text-only pilot must inspect answer text and stop reasons, including LFM's
reasoning, without publishing timing or memory as performance evidence. If any
model cannot finish, revise the common cap or prompts *before* freezing a new
plan; do not change the rules after observing study sessions. A pilot is not a
quality evaluation and does not validate usefulness. Keep its files, if any,
separate from the 20-session study and historical P2 evidence.

## On the actual Mac

Read `AGENTS.md`, `MODELS.md`, `DESIGN.md` first. Fetch all four plain 4-bit
checkpoints (one at a time), verify their complete weight SHA-256 provenance and
passing P0 using `run_bench.py --phase fetch` and `--phase p0` with `--tiers
4bit`. Check available disk capacity, power source and `python sysinfo.py`.
`plan` refuses missing provenance, P0 failures, mismatched weight SHA-256, a
dirty tracked working tree, the wrong machine or incorrect pinned MLX versions.
Before each block it additionally requires observed AC power and low-power mode
off; an unavailable reading does not count as confirmation.
It also freezes the hashes of all local checkpoint configuration/tokenizer/
chat-template files, and excludes unexpected unverified safetensors shards.
The frozen file set and hashes are verified before each block; only Hugging
Face's `.cache` and incidental macOS Finder metadata are excluded.
**Merge the infrastructure PR first**: the plan freezes the Git commit currently
checked out, along with exact machine/runtime versions, Hub revisions, individual
weight hashes, template controls, all three prompt SHA-256s and decode settings.

Run with the actual `.venv-mlxlm` interpreter (Python 3.12.10 and pinned packages):

```bash
# Choose the seed before any observations; directory must not exist.
.venv-mlxlm/bin/python replicated.py plan \
  --seed 34027 --out results/replicated/m4-4bit-2026-11

# At the Mac, plugged in; read and act on sysinfo warnings before each block.
.venv-mlxlm/bin/python sysinfo.py
caffeinate -i .venv-mlxlm/bin/python replicated.py run \
  results/replicated/m4-4bit-2026-11 --block 1 --condition clean

# Repeat --block 2, 3, 4 and 5 on the SAME frozen checkout and Mac.
# Only if prior slots already validated and the block was interrupted:
# add --resume to SKIP validated slots; it never overwrites a failed slot.

# The following are pure offline checks; no model is loaded.
python3 replicated.py validate results/replicated/m4-4bit-2026-11
python3 replicated.py report results/replicated/m4-4bit-2026-11 \
  --out results/REPLICATED_STUDY.md
```

Only run `report` **after validation** of all 20 complete independent
sessions. The report refuses incomplete, truncated, failed, drifted, or tampered
observations. It never regenerates or overwrites historical `results/STUDY.md`.
Review the report and then integrate new, validated evidence and its comparison
into the main study deliberately, in a separate PR. The auto-generated results
folder is ignored by default; explicitly curate and force-add vetted artifacts
once reviewed, not the entire unattended experiment directory.

## Immutable evidence layout and schema

```
results/replicated/<experiment-name>/
  plan.json                               # schema=2, content-addressed plan_id
  .experiment.lock                        # persistent filename; OS-released flock
  sessions/
    b01-p01-<model-id>/
      started.json                       # UUID + intended condition + pre-state
      bench.log                          # exact CLI/process output, including failures
      raw/<model-id>/<checkpoint>-inproc/
        summary.json
        short-warmup.json                # plus short-trial1..5
        medium-warmup.json               # plus medium-trial1..5
        long-warmup.json                 # plus long-trial1..5
      session_payload.json               # SHA256-protected exact manifest payload
      session.json                       # schema=2, UUID, block/position,
                                         # observed pre/post, condition caveats,
                                         # raw host-memory samples and per-file SHA256
    b01-p02-<model-id>/...
    ...
```

Schema 2 uses the replicated prompt set and expanded condition observations;
older schema-1 plans are rejected rather than silently reinterpreted.
`plan.json` carries a SHA-256 digest of its entire canonical payload (excluding
its own `plan_id`). **Every session directory and JSON file is created
exclusively** (`mkdir(exist_ok=False)`, `open('x')`); a second process cannot
silently overwrite the same slot. The experiment-wide lock also prevents
different blocks from executing concurrently; the OS releases it after a crash.
`session_payload.json` redundantly stores and hashes every non-checksum field
from the final manifest, including timestamps, observed conditions, and host
memory. Validation recomputes environmental warnings from snapshots and the
host-memory peak from raw samples, rather than trusting editable summaries.
Logs, failed measurements and incomplete
sessions remain for diagnosis. `validate` compares evidence file sets and their
SHA-256 digests and cross-checks identity, runtime, 18 raw trials, warmups,
counts, summary medians/ranges, conditions and exact generation settings. This is
write-once/integrity checking, **not a cryptographic signature or an external
archival guarantee**: back up complete directories to independent storage.

Before launching another session, the runner validates existing evidence and the
schedule prefix; it stops immediately after a completed subprocess with invalid
or truncated evidence. This prevents continuing a block that cannot be reported.

A crashed/failed slot is deliberately *not* rerun in place. Preserve it and
investigate, then plan a new clean experiment (or a separately documented
extension). Do not delete the failed evidence and pretend that slot never ran.
Never use GitHub CI, cached HTTP, OptiQ or 8-bit values to fill missing slots.

## Statistical interpretation

The report reduces each 5-trial regime to **one session median**, then reports
median and 3,000-resample percentile bootstrap interval over five independently
initialized sessions. For model comparisons, it computes differences between
*same-block session medians* and resamples complete blocks to preserve pairing.
These are exploratory **session-level**, not trial- or token-level, intervals.
Five sessions are a minimum, not a guarantee of precision; the percentile
bootstrap may have poor coverage at this n. Heavy tails, clock trends, swap,
thermal differences, output-length differences and flagged conditions limit
interpretation. Report all flags and missingness; do not silently discard a
busy session, relabel it as `loaded`, or publish a universal winner. If effects
are decision-relevant but intervals are wide, preregister a new replicated
extension with more blocks, preserving the first study unchanged.

The report prints tokenizer-specific prompt-token counts, actual completion
lengths and their ranges next to the corresponding performance measurements.
The main table separates bootstrap intervals across independent sessions from
the descriptive span of all kept trials; a per-session appendix gives the
median and min–max across **each session's five kept trials**.

Run offline regressions with `python3 -m unittest discover -s tests -v`.
