# nemotron-bolt-bench

Speed, TTFT and memory benchmark of `NVIDIA-Nemotron-3.5-Bolt-E4B-4GB` on Apple Silicon (MLX), M4 / 16 GB.

![P2 chart](p2_visual.png)

**Headline:** the server adds ~no TTFT cost; the OptiQ mixed-bit checkpoint decodes ~25% slower and uses ~2.3 GB more than plain 4-bit; a fresh 8k-token prompt takes ~14 s to first token, a prefix-cache hit ~0.09 s. Full tables and caveats: [RESULTS.md](RESULTS.md).

| File | What |
|---|---|
| `RESULTS.md` | numbers, findings, caveats |
| `DESIGN.md` | questions, arms, controls |
| `P0.md` | load tests; why `optiq serve` can't run on 16 GB |
| `bench.py` | harness, one JSON per run |
| `run_p2.sh`, `visualize_p2.py` | reproduce the runs and chart |
| `results/` | raw per-run JSON and summaries |

## Reproduce

Needs the two checkpoints under `models/` (`rapid-mlx-4bit`, `optiq-4bit`) and a venv with `mlx-lm==0.32.2`, `mlx==0.32.4`, `matplotlib`:

```
python -m venv .venv-mlxlm && .venv-mlxlm/bin/pip install mlx-lm==0.32.2 matplotlib
caffeinate -i ./run_p2.sh && .venv-mlxlm/bin/python visualize_p2.py
```

Status: speed done (arms 1-3); accuracy/quality sets not yet run.
