# Offline result verification

Commands: `.venv-mlxlm/bin/python readiness.py` and `.venv-mlxlm/bin/python -m unittest discover -s tests -v`. Full output: [readiness](readiness.txt), [unit tests](unit-tests.txt).

The unit suite passes (150 tests). Readiness exits nonzero with 59 existing-output collisions plus two missing modern P0 records for the historical Nemotron 4-bit and OptiQ checkpoints. These are pre-run blockers, not evidence that completed result rows failed. The independent summary audit found 37 speed summaries: 33 explicitly `completed` and `publishable: true`, four historical Nemotron summaries with those fields unrecorded, and no explicitly failed or non-publishable summary. No historical row was changed or retroactively validated.

The initial sandboxed test attempts could not bind localhost fixture sockets. The approved reruns passed the suite using synthetic HTTP fixtures; no model server, inference or model download was run.

## Reconciliation with PR #27

PR #26 was rebased onto current `main` after PR #27 merged (`ef30f0f`, containing `e64c836`). The independent rerun on that tree passes all **150 tests**, including PR #27's conditions, spread, and comparison coverage. `python compare.py` reproduces the committed `results/COMPARE.md` byte-for-byte.

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

The Nemotron `mlx-4bit-inproc`, `mlx-4bit-http`, `mlx-4bit-http-cached` and `optiq-4bit-inproc` summaries are the old pre-gate baseline: outcome, publishable and non-publishable reasons are unrecorded. Every summary with recorded status is completed and publishable.

The available result logs contain no Safari or WindowServer evidence establishing which LFM2.5 or Nemotron reference-tier speed rows overlapped those processes. A row-specific caveat cannot be reconstructed from these logs.

The report, chart, blind round and status-doc tasks are stopped under the request to report instead of guessing when evidence is missing or a gate fails. No blind key was opened.

PR #25 was merged after its offline CI checks succeeded. PR #27 is now also in current `main`; its merge commit has the requested author but GitHub's committer identity, contrary to the sole-committer contract. This was documented rather than rewriting shared history. The PR #26 verification commit uses the required author and committer identity.
