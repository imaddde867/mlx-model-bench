# Offline result verification

Commands: `.venv-mlxlm/bin/python readiness.py` and `.venv-mlxlm/bin/python -m unittest discover -s tests -v`. Full output: [readiness](readiness.txt), [unit tests](unit-tests.txt).

The unit suite passes. Readiness exits nonzero: the existing matrix collides with planned output paths, and the historical Nemotron baseline lacks modern P0 records. These are pre-run blockers, not evidence that completed result rows failed. No historical row was changed or retroactively validated.

The initial sandboxed test attempts could not bind localhost fixture sockets. The approved reruns passed the suite using synthetic HTTP fixtures; no model server, inference or model download was run.

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

PR #25 was merged after its offline CI checks succeeded. The hosting service created its merge commit with the requested author but its own committer identity, contrary to the sole-committer contract; history was not rewritten. All new verification commits use the required author and committer identity.
