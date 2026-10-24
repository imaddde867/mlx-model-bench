# MLX model bench — Apple Silicon, 16 GB

Speed, TTFT, memory, accuracy and blind quality for the sub-5 GB quantized models
that actually fit on an Apple Silicon laptop — measured on Apple M4 / 16 GB, MLX,
one model at a time, with the same prompts, the same controls and the same
provenance for every row.

![P2 chart](p2_visual.png)

**Headline (baseline, complete):** for Nemotron 3.5 Bolt E4B the server adds ~no
TTFT cost; the OptiQ mixed-bit checkpoint decodes ~25% slower and uses ~2.3 GB
more than plain 4-bit; a fresh 8k-token prompt takes ~14 s to first token, a
prefix-cache hit ~0.09 s. Full tables and caveats: [RESULTS.md](RESULTS.md).

**Roster:** the baseline above plus three challengers, chosen to cover the range
that fits in 16 GB — [MODELS.md](MODELS.md) has the verified metadata, the
template-control traps per model, and the rules that keep the comparison honest.

| Model | Weights (plain / OptiQ) | Why it is here | Status |
|---|---|---|---|
| NVIDIA Nemotron 3.5 Bolt E4B | 4.20 / 6.53 GB | the baseline, 7.46B dense | **complete** — P0–P3 |
| OpenBMB MiniCPM5-2B | 1.43 / 1.94 GB | best compact intelligence per GB | queued |
| XHToken Spark-X2.5-4B | 2.32 / 3.05 GB | 4B-class coding and tool use | queued |
| Liquid AI LFM2.5-8B-A1B | 4.85 / 5.47 GB | hybrid MoE; speed at 1.3B active | queued |

Each model runs in two checkpoints — plain 4-bit and OptiQ mixed 4/8-bit — so the
two questions the baseline answers (engine cost, checkpoint cost) get asked of
every model, and a third appears: which checkpoint a model wears best.

## Files

| File | What |
|---|---|
| `registry.py` | **the source of truth**: models, checkpoints, sizes, template controls, provenance |
| `MODELS.md` | roster, verified metadata, fairness protocol, what the numbers cannot say |
| `DESIGN.md` | questions, arms, controls |
| `P0.md` | per-model load tests; why `optiq serve` can't run on 16 GB |
| `RESULTS.md` | the baseline report: speed, TTFT, memory, accuracy (30), blind quality (8) |
| `run_bench.py` | one model, end to end: fetch → p0 → p2 → p3 |
| `bench.py` | the speed/TTFT/memory harness, one JSON per run |
| `p0_load.py` | load test: does this checkpoint run here, and what does it cost |
| `p3_suite.py`, `run_p3.py` | the fixed 30 accuracy + 8 quality prompts and the runner |
| `blind.py`, `score_quality.py` | cross-model blind set and its scoring summary |
| `visualize_p2.py` | the chart, every model on the same axes |
| `results/` | raw per-run JSON, one directory per model |

## Reproduce

Needs the checkpoints under `models/` (see `registry.py` for the exact repos) and
a venv with `mlx-lm==0.32.2`, `mlx==0.32.4`, `matplotlib`:

```bash
# one model, all phases, in the order that keeps failures cheap
caffeinate -i .venv-mlxlm/bin/python run_bench.py --model minicpm5-2b --phase all

# chart everything with results
.venv-mlxlm/bin/python visualize_p2.py

# blind quality across whatever is on disk
.venv-mlxlm/bin/python blind.py --round r1     # then score scores.json by hand
.venv-mlxlm/bin/python score_quality.py --round r1
```

Status: the baseline (P0 loads, P1 harness, P2 speed/memory, P3 accuracy/quality)
is complete. The three-model roster is set up and verified on the Hub; its runs
start one model at a time, plain checkpoint first.
