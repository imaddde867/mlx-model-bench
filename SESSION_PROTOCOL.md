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
- **Within session:** fixed `prompts/speed/{short,medium,long}.txt` files (short,
  ~1k and ~8k as tokenized by each model), one discarded warmup plus **five kept
  trials per regime**. Greedy, template kwargs from `registry.py`, no explicit
  in-process prompt cache and **no nonce** (there is no in-process prefix cache).
  The common `max_tokens=256` is a cap, **not** an assertion that every model
  produced 256 tokens. Actual prompt/completion tokens and stop reasons are in
  each trial. If a model truncates or fails, preserve the evidence and do not
  claim a complete comparable run.
- **Conditions:** physically plugged in and awake under `caffeinate`; same
  macOS/pinned MLX environment, screen/desktop workload held as consistent as
  practical. `--condition clean|loaded` records the **operator's stated intent**.
  Each `bench.py` trial includes its own `sysinfo.snapshot()` before the timed
  window; this layer stores before/after snapshots and their observed warnings.
  A `clean` label with `mediaanalysisd` at 225% CPU stays `clean` **with an
  observed busy warning**. No historical pilot label is rewritten.
- **Memory:** the existing `mx.get_peak_memory()` is the *per-trial MLX allocator*
  peak, not app/system memory. Separately, `vm_stat` is polled every 0.5 s from
  an external thread for a **host-wide physical-used estimate** based on
  `total − free − inactive − speculative` pages. This is an approximate host
  value affected by other processes, not process RSS, GPU VRAM, or an exact
  workload-only peak. Swap, memory pressure, power and thermal CPU speed-limit
  are recorded in `sysinfo` snapshots. Direct temperature in degrees and a true
  application's total high-water mark are not available here: do not invent them.
- **Latency:** primary observed end-to-end metric is `wall_s`, i.e. time to
  response end. It is **not** time to a verified *correct or useful* answer.
  `request_ttft_s` is time to the first generated token. In-process
  `first_content_s` is unavailable and must remain missing, rather than being
  fabricated from TTFT. Later, a held-out task/quality experiment is required
  before claiming time to useful/agentic completion.

## On the actual Mac

Read `AGENTS.md`, `MODELS.md`, `DESIGN.md` first. Fetch all four plain 4-bit
checkpoints (one at a time), verify their complete weight SHA-256 provenance and
passing P0 using `run_bench.py --phase fetch` and `--phase p0` with `--tiers
4bit`. Check available disk capacity, power source and `python sysinfo.py`.
`plan` refuses missing provenance, P0 failures, mismatched weight SHA-256, a
dirty tracked working tree, the wrong machine or incorrect pinned MLX versions.
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
  plan.json                               # schema=1, content-addressed plan_id
  sessions/
    b01-p01-<model-id>/
      started.json                       # UUID + intended condition + pre-state
      bench.log                          # exact CLI/process output, including failures
      raw/<model-id>/<checkpoint>-inproc/
        summary.json
        short-warmup.json                # plus short-trial1..5
        medium-warmup.json               # plus medium-trial1..5
        long-warmup.json                 # plus long-trial1..5
      session.json                       # schema=1, UUID, block/position,
                                         # observed pre/post, condition caveats,
                                         # host-memory sampling and per-file SHA256
    b01-p02-<model-id>/...
    ...
```

`plan.json` carries a SHA-256 digest of its entire canonical payload (excluding
its own `plan_id`). **Every session directory and JSON file is created
exclusively** (`mkdir(exist_ok=False)`, `open('x')`); a second process cannot
silently overwrite the same slot. Logs, failed measurements and incomplete
sessions remain for diagnosis. `validate` compares evidence file sets and their
SHA-256 digests and cross-checks identity, runtime, 18 raw trials, warmups,
counts, summary medians, conditions and exact generation settings. This is
write-once/integrity checking, **not a cryptographic signature or an external
archival guarantee**: back up complete directories to independent storage.

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

Run offline regressions with `python3 -m unittest discover -s tests -v`.