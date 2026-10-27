# Offline result verification

Commands: `.venv-mlxlm/bin/python readiness.py` and `.venv-mlxlm/bin/python -m unittest discover -s tests -v`. Full output: [readiness](readiness.txt), [unit tests](unit-tests.txt). The stored readiness and test transcripts are the PR #26 offline baseline, before the separately tagged quick-repeat measurements and comparison-table addition.

The unit suite passes (151 tests). Readiness exits nonzero with 59 existing-output collisions plus two missing modern P0 records for the historical Nemotron 4-bit and OptiQ checkpoints. These are pre-run blockers, not evidence that completed result rows failed. The independent summary audit before the quick checks found 37 speed summaries: 33 explicitly `completed` and `publishable: true`, four historical Nemotron summaries with those fields unrecorded, and no explicitly failed or non-publishable summary. Added reduced checks are kept in separately tagged directories; two-trial checks are marked non-publishable, and five-trial decode-short repeats are completed/publishable. Existing full-result rows were not overwritten or retroactively validated.

During the earlier offline verification, initial sandboxed test attempts could not bind localhost fixture sockets; the approved reruns passed using synthetic HTTP fixtures. The later quick checks did run local in-process inference, used no model server, and downloaded no model data.

## Reconciliation with PR #27

PR #26 was rebased onto current `main` after PR #27 merged (`ef30f0f`, containing `e64c836`). The independent rerun on that tree passed **150 tests**, including PR #27's conditions, spread, and comparison coverage. After adding a focused test for the repeat comparison section, **151 tests pass**. `python compare.py` generates `results/COMPARE.md` from the original full-matrix rows plus a separate section for the repeat checks.

PR #27 adds condition labels and per-trial system snapshots for future runs. The existing raw trial files predate those fields: none contains `started_at` or `system`, and their summaries do not record a condition. Therefore PR #27 cannot establish retrospectively whether Safari or WindowServer overlapped any particular historical measurement. The comparison table's recomputed spreads remain useful, but condition must stay `not recorded`; no historical measurement was edited to invent this evidence.

## Summary inventory

Values below are read directly from each summary; `not recorded` distinguishes an absent field from false or an empty list.

| Summary | outcome | publishable | non_publishable_reasons |
|---|---|---|---|
| [results/lfm2.5-8b-a1b/mlx-4bit-http/summary.json](../results/lfm2.5-8b-a1b/mlx-4bit-http/summary.json) | "completed" | true | [] |
| [results/lfm2.5-8b-a1b/mlx-4bit-http-cached/summary.json](../results/lfm2.5-8b-a1b/mlx-4bit-http-cached/summary.json) | "completed" | true | [] |
| [results/lfm2.5-8b-a1b/mlx-4bit-inproc/summary.json](../results/lfm2.5-8b-a1b/mlx-4bit-inproc/summary.json) | "completed" | true | [] |
| [results/lfm2.5-8b-a1b/mlx-8bit-http/summary.json](../results/lfm2.5-8b-a1b/mlx-8bit-http/summary.json) | "completed" | true | [] |
| [results/lfm2.5-8b-a1b/mlx-8bit-http-cached/summary.json](../results/lfm2.5-8b-a1b/mlx-8bit-http-cached/summary.json) | "completed" | true | [] |
| [results/lfm2.5-8b-a1b/mlx-8bit-inproc/summary.json](../results/lfm2.5-8b-a1b/mlx-8bit-inproc/summary.json) | "completed" | true | [] |
| [results/lfm2.5-8b-a1b/optiq-4bit-http/summary.json](../results/lfm2.5-8b-a1b/optiq-4bit-http/summary.json) | "completed" | true | [] |
| [results/lfm2.5-8b-a1b/optiq-4bit-http-cached/summary.json](../results/lfm2.5-8b-a1b/optiq-4bit-http-cached/summary.json) | "completed" | true | [] |
| [results/lfm2.5-8b-a1b/optiq-4bit-http-optiq-engine/summary.json](../results/lfm2.5-8b-a1b/optiq-4bit-http-optiq-engine/summary.json) | "completed" | true | [] |
| [results/lfm2.5-8b-a1b/optiq-4bit-inproc/summary.json](../results/lfm2.5-8b-a1b/optiq-4bit-inproc/summary.json) | "completed" | true | [] |
| [results/minicpm5-2b/mlx-4bit-http/summary.json](../results/minicpm5-2b/mlx-4bit-http/summary.json) | "completed" | true | [] |
| [results/minicpm5-2b/mlx-4bit-http-cached/summary.json](../results/minicpm5-2b/mlx-4bit-http-cached/summary.json) | "completed" | true | [] |
| [results/minicpm5-2b/mlx-4bit-inproc/summary.json](../results/minicpm5-2b/mlx-4bit-inproc/summary.json) | "completed" | true | [] |
| [results/minicpm5-2b/mlx-8bit-http/summary.json](../results/minicpm5-2b/mlx-8bit-http/summary.json) | "completed" | true | [] |
| [results/minicpm5-2b/mlx-8bit-http-cached/summary.json](../results/minicpm5-2b/mlx-8bit-http-cached/summary.json) | "completed" | true | [] |
| [results/minicpm5-2b/mlx-8bit-inproc/summary.json](../results/minicpm5-2b/mlx-8bit-inproc/summary.json) | "completed" | true | [] |
| [results/minicpm5-2b/optiq-4bit-http/summary.json](../results/minicpm5-2b/optiq-4bit-http/summary.json) | "completed" | true | [] |
| [results/minicpm5-2b/optiq-4bit-http-cached/summary.json](../results/minicpm5-2b/optiq-4bit-http-cached/summary.json) | "completed" | true | [] |
| [results/minicpm5-2b/optiq-4bit-http-optiq-engine/summary.json](../results/minicpm5-2b/optiq-4bit-http-optiq-engine/summary.json) | "completed" | true | [] |
| [results/minicpm5-2b/optiq-4bit-inproc/summary.json](../results/minicpm5-2b/optiq-4bit-inproc/summary.json) | "completed" | true | [] |
| [results/nemotron-3.5-bolt/mlx-4bit-http/summary.json](../results/nemotron-3.5-bolt/mlx-4bit-http/summary.json) | not recorded | not recorded | not recorded |
| [results/nemotron-3.5-bolt/mlx-4bit-http-cached/summary.json](../results/nemotron-3.5-bolt/mlx-4bit-http-cached/summary.json) | not recorded | not recorded | not recorded |
| [results/nemotron-3.5-bolt/mlx-4bit-inproc/summary.json](../results/nemotron-3.5-bolt/mlx-4bit-inproc/summary.json) | not recorded | not recorded | not recorded |
| [results/nemotron-3.5-bolt/mlx-8bit-http/summary.json](../results/nemotron-3.5-bolt/mlx-8bit-http/summary.json) | "completed" | true | [] |
| [results/nemotron-3.5-bolt/mlx-8bit-http-cached/summary.json](../results/nemotron-3.5-bolt/mlx-8bit-http-cached/summary.json) | "completed" | true | [] |
| [results/nemotron-3.5-bolt/mlx-8bit-inproc/summary.json](../results/nemotron-3.5-bolt/mlx-8bit-inproc/summary.json) | "completed" | true | [] |
| [results/nemotron-3.5-bolt/optiq-4bit-inproc/summary.json](../results/nemotron-3.5-bolt/optiq-4bit-inproc/summary.json) | not recorded | not recorded | not recorded |
| [results/spark-x2.5-4b/mlx-4bit-http/summary.json](../results/spark-x2.5-4b/mlx-4bit-http/summary.json) | "completed" | true | [] |
| [results/spark-x2.5-4b/mlx-4bit-http-cached/summary.json](../results/spark-x2.5-4b/mlx-4bit-http-cached/summary.json) | "completed" | true | [] |
| [results/spark-x2.5-4b/mlx-4bit-inproc/summary.json](../results/spark-x2.5-4b/mlx-4bit-inproc/summary.json) | "completed" | true | [] |
| [results/spark-x2.5-4b/mlx-8bit-http/summary.json](../results/spark-x2.5-4b/mlx-8bit-http/summary.json) | "completed" | true | [] |
| [results/spark-x2.5-4b/mlx-8bit-http-cached/summary.json](../results/spark-x2.5-4b/mlx-8bit-http-cached/summary.json) | "completed" | true | [] |
| [results/spark-x2.5-4b/mlx-8bit-inproc/summary.json](../results/spark-x2.5-4b/mlx-8bit-inproc/summary.json) | "completed" | true | [] |
| [results/spark-x2.5-4b/optiq-4bit-http/summary.json](../results/spark-x2.5-4b/optiq-4bit-http/summary.json) | "completed" | true | [] |
| [results/spark-x2.5-4b/optiq-4bit-http-cached/summary.json](../results/spark-x2.5-4b/optiq-4bit-http-cached/summary.json) | "completed" | true | [] |
| [results/spark-x2.5-4b/optiq-4bit-http-optiq-engine/summary.json](../results/spark-x2.5-4b/optiq-4bit-http-optiq-engine/summary.json) | "completed" | true | [] |
| [results/spark-x2.5-4b/optiq-4bit-inproc/summary.json](../results/spark-x2.5-4b/optiq-4bit-inproc/summary.json) | "completed" | true | [] |

## Exceptions and stopping point

The Nemotron `mlx-4bit-inproc`, `mlx-4bit-http`, `mlx-4bit-http-cached` and `optiq-4bit-inproc` summaries are the old pre-gate baseline: outcome, publishable and non-publishable reasons are unrecorded. Every original full-matrix summary with recorded status is completed and publishable. The appended quick-rerun summaries are separate checks, not full-matrix replacements; two-trial summaries are explicitly non-publishable by design.

The available result logs contain no Safari or WindowServer evidence establishing which LFM2.5 or Nemotron reference-tier speed rows overlapped those processes. A row-specific caveat cannot be reconstructed from these logs.

The reduced reruns after the offline verification pass are preserved under the `*-inproc-quick-rerun/` and `*-inproc-decode-check/` directories; the latter contain five-trial MiniCPM5 and LFM2.5 repeats and are surfaced separately in `results/COMPARE.md`. The short two-trial runs remain explicitly non-publishable smoke checks. Full report/chart/blind-round completion remains stopped where evidence is absent; no blind key was opened.

PR #25 was merged after its offline CI checks succeeded. PR #27 is now also in current `main`; its merge commit has the requested author but GitHub's committer identity, contrary to the sole-committer contract. This was documented rather than rewriting shared history. The PR #26 verification commit uses the required author and committer identity.
