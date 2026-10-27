# Local model bench — 16 GB Apple Silicon

Speed, TTFT, memory, accuracy and blind quality for the models that actually fit on a
16 GB Apple Silicon laptop, measured under one harness, one prompt set and one set of
controls. Apple M4 (10-core), macOS 27.0.1, MLX, `mlx-lm` 0.32.2.

Not a leaderboard. Every row carries its model, checkpoint, bits-per-weight, publisher,
prompt-token count and generation length, because none of those numbers mean anything
without them. What the study can and cannot conclude is written down in
[`MODELS.md`](MODELS.md#what-this-roster-cannot-tell-you).

## Roster

Four models, each at the **same three quantization tiers**, so a difference between rows is
a difference between models rather than between quantization budgets.

| Model | Params | 4-bit | OptiQ 4/8-bit | 8-bit (reference) | Status |
|---|---|---|---|---|---|
| NVIDIA Nemotron 3.5 Bolt E4B | 7.46B | 4.20 GB | 6.53 GB | 7.96 GB | 3 tiers reported · historical baseline caveat · blind r1 scored |
| OpenBMB MiniCPM5-2B | 2.52B | 1.43 GB | 1.94 GB | 2.68 GB | 3 tiers reported · blind r1 scored |
| XHToken Spark-X2.5-4B | 4.11B | 2.32 GB | 3.05 GB | 4.38 GB | 3 tiers reported · third-party plain quants · blind r1 scored |
| Liquid AI LFM2.5-8B-A1B | 8.47B | 4.85 GB | 5.47 GB | 9.02 GB | 3 tiers reported · reasoning tokens included · blind r1 scored |

The cross-model report covers the 12 in-process model/tier rows. Matrix run conditions are
`not recorded` for the historical runs that predate per-trial snapshots. The run operator recalls
many other apps were open; this may have slowed the measurements, but without system snapshots the
effect cannot be quantified or corrected.
The blind r1 comparison is exploratory and single-rater, and LFM receives a 4× quality token cap
because its reasoning block precedes its final answer. See [`REPORT.md`](REPORT.md) for the
results, caveats, strict-format versus numerical scoring distinction, and blind scores.

Per checkpoint there are three arms: `mlx_lm` in-process, `mlx_lm.server` over HTTP, and an
`mlx-optiq` engine arm attempted for every OptiQ checkpoint (it is unmeasurable for the
baseline — see [`P0.md`](P0.md) — and is recorded as unmeasurable rather than omitted when
it fails for a new model).

Three fairness notes, because they are the difference between a comparison and a collage:

- **Publisher provenance is part of the row.** Two models have vendor MLX conversions
  (MiniCPM5, LFM2.5); Spark has none at all, so two of its three tiers are community quants;
  Nemotron's are third-party. That is stated in every table those rows appear in.
- **The 8-bit tier exists because the baseline omitted it.** Nemotron publishes an 8-bit
  MLX file that the original study never ran. Comparing four models at 4-bit while testing
  whether only one of them gains from a wider quant would not be a fair test of the
  quantizer, so the baseline gets an 8-bit row too.
- **The tiers are not the same number of bits.** Bits-per-weight runs 4.50 → 8.53; OptiQ's
  premium over plain 4-bit ranges from +13% (LFM) to +56% (Nemotron). Any "OptiQ is better"
  claim has to survive that.

Full metadata, template-control traps and the comparability protocol:
[`MODELS.md`](MODELS.md). The registry that everything reads: [`registry.py`](registry.py).

## Files

| File | What |
|---|---|
| `REPORT.md` | the report: methodology, per-model results, cross-model comparison |
| `MODELS.md` | roster, verified metadata, option matrix, fairness protocol, limits |
| `DESIGN.md` | questions, arms, controls |
| `P0.md` | load tests; why `optiq serve` cannot run the baseline on 16 GB |
| `registry.py` | single source of truth: models, checkpoints, sizes, template controls |
| `run_bench.py` | one model end to end: fetch → p0 gate → p2 → p2-optiq → p3 |
| `fetch_all.py` | the whole roster at once, in parallel; verification stays in `run_bench.py` |
| `bench.py` | speed/TTFT/memory harness, one JSON per run |
| `tests/test_runner.py` | offline fixture-driven regression suite (Python standard library only) |
| `p0_load.py` | load gate: does this checkpoint run here, deterministically, at what cost |
| `p3_suite.py`, `run_p3.py` | the fixed 30 accuracy + 8 quality prompts and their runner |
| `blind.py`, `score_quality.py` | cross-model blind set (balanced slots) and scoring summary |
| `readiness.py` | offline go/no-go before the matrix (suite + preconditions + checklist) |
| `sysinfo.py` | what else the machine is doing: swap, memory pressure, power, thermal, busiest processes. `python sysinfo.py` is the preflight; `bench.py` records the same snapshot per trial |
| `compare.py` | cross-model table (REPORT section 5) generated from committed rows, incl. the labelled weight-read proxy and TTFT spread |
| `visualize_p2.py` | the chart: every model on the same axes |
| `results/` | raw per-run JSON, one directory per model |

## Reproduce

Run the hardware-independent regression suite without models, MLX, network
access, or third-party test dependencies (also run by GitHub Actions on pushes
and pull requests):

```bash
python3 -m unittest discover -s tests -v
```

Needs a venv with `mlx-lm==0.32.2`, `mlx==0.32.4`, `matplotlib` (and `mlx-optiq==0.7.3`
in a separate venv to attempt the engine arm). No model is vendored: `registry.py` names
every Hub repo, and `--phase fetch` downloads and verifies them.

### Readiness check before the matrix

```bash
.venv-mlxlm/bin/python readiness.py [--models ...] [--tiers ...] [--out results] [--round r1]
```

One offline go/no-go: runs the unit suite above, then checks the planned run against
`--out` read-only (registry and downloaded checkpoints, `gates.problems` provenance and
P0 evidence, `--trials` >= 5, one mlx / mlx-lm / host across a model's rows, prompt files,
blind-round inputs, and that no planned output path already exists). It loads no model,
runs no benchmark and uses no network. Prints `READY` (exit 0) or `NOT READY` with every
reason (exit 1), then the reproducibility checklist (host, runtimes, Hub revisions, prompt
sha256, generation settings, trials, output paths). Existing results, including the
historical `results/nemotron-3.5-bolt/*` baseline, count as collisions: narrow
`--models/--tiers` or use another `--out`. `p3/accuracy_summary.json` is merged per tier by `run_p3.py`, so it collides only for tiers
that already have an entry there; other tiers (e.g. the baseline 8-bit row, R3b) are not blocked by it.
`READY` is not
the independently reviewed clean run, which is a separate future action; on this
repo today it reports `NOT READY` because the existing matrix collides with planned output
paths (see `verification/` on `docs/offline-results-verification`); use `--out` elsewhere.

**To fetch fast, pull several checkpoints at once — not "with a faster transfer path".**
On the pinned `huggingface_hub` 2.3.0, `HF_HUB_ENABLE_HF_TRANSFER=1` does nothing but emit a
`FutureWarning` that `hf_transfer` is no longer used, and `HF_XET_HIGH_PERFORMANCE` does
nothing for these repos either: they are LFS, not Xet. Concurrency is the lever that does
exist. One stream tops out near 4.5 MB/s here, while six concurrent `snapshot_download`s
sustain 35–45 MB/s aggregate, which is the difference between ~15 minutes and hours for the
54 GB download roster (51 GiB on disk). Downloads resume, so a killed fetch costs only the
files in flight — and a half-written shard leaves an `.incomplete` marker that
`Checkpoint.available` reads as *not downloaded*, so nothing gets measured from a partial
file.

```bash
.venv-mlxlm/bin/python fetch_all.py --jobs 6          # every checkpoint, in parallel
.venv-mlxlm/bin/python run_bench.py --model minicpm5-2b --phase fetch   # sha256 gate, per model

# inspect the matrix: tiers, sizes, bpw, publisher, what is already on disk
.venv-mlxlm/bin/python registry.py
.venv-mlxlm/bin/python bench.py --list

# one model, in order: fetch+sha256 gate, P0 gate, speed, engine arm, accuracy/quality
caffeinate -i .venv-mlxlm/bin/python run_bench.py --model minicpm5-2b --phase fetch
caffeinate -i .venv-mlxlm/bin/python run_bench.py --model minicpm5-2b --phase p0
caffeinate -i .venv-mlxlm/bin/python run_bench.py --model minicpm5-2b --phase p2
caffeinate -i .venv-mlxlm/bin/python run_bench.py --model minicpm5-2b --phase p2-optiq
caffeinate -i .venv-mlxlm/bin/python run_bench.py --model minicpm5-2b --phase p3

# or everything, stopping at the first gate that fails
caffeinate -i .venv-mlxlm/bin/python run_bench.py --model minicpm5-2b --phase all

# restrict tiers: --tiers 4bit optiq-4bit   (8bit is the reference tier)
# chart, and the cross-model blind quality round
.venv-mlxlm/bin/python visualize_p2.py --include-http
.venv-mlxlm/bin/python blind.py --round r1
# fill results/blind-r1/scores.json from the template, score blind, then:
.venv-mlxlm/bin/python score_quality.py --round r1
```

Blind quality round r1 (four plain 4-bit rows; exploratory, single rater): see `results/blind-r1/`.
The scored summary warns of a 1.37-point spread across blind slots, above the 1-point position-bias
threshold. The results are a preference signal, not a ranking.

The repository was renamed from `nemotron-bolt-bench` when the study grew from one model to
four; its original single-model record is now section 1 of the report, unchanged.

## Run conditions

Future runs should record whether other applications are open, rather than assume a clean
machine. The historical matrix rows have no per-trial snapshots; the operator recalls many apps
were open, which may have slowed them, but the impact cannot be reconstructed:

```bash
python sysinfo.py                                   # preflight: swap, memory pressure, power, thermal, busiest processes
.venv-mlxlm/bin/python bench.py ... --condition loaded     # or: clean
.venv-mlxlm/bin/python bench.py ... --max-spread 1.5       # optional: wobbly trials make the row non-publishable
python compare.py                                   # cross-model table; TTFT spread column flags unsteady rows
```

Each trial file records `started_at` and a `system` snapshot taken just before its timed
window; each summary records `condition`, `condition_caveats` and per-prompt `spread_ratio`.
Rows recorded before this existed show `condition: not recorded`; `compare.py` recomputes
their spread from the per-trial files. Do not interpret historical rows as clean runs or apply an
estimated slowdown; no such estimate can be supported by the available records.

## Pins

`mlx-lm` 0.32.2 · `mlx` 0.32.4 · `mlx-optiq` 0.7.3 (requires `mlx-lm <0.33`) · Python 3.12.10.
One venv per engine so neither can silently upgrade the other. Versions are recorded inside
each result row, never only in prose.
