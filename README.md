# nemotron-bolt-bench

Speed, TTFT and memory benchmark of `NVIDIA-Nemotron-3.5-Bolt-E4B-4GB` on Apple Silicon (MLX), M4 / 16 GB.

![P2 chart](p2_visual.png)

**Headline:** the server adds ~no TTFT cost; the OptiQ mixed-bit checkpoint decodes ~25% slower and uses ~2.3 GB more than plain 4-bit; a fresh 8k-token prompt takes ~14 s to first token, a prefix-cache hit ~0.09 s. Full tables and caveats: [RESULTS.md](RESULTS.md).

| File | What |
|---|---|
| `RESULTS.md` | speed, TTFT, memory, accuracy (30), and blind quality (8) results |
| `DESIGN.md` | questions, arms, controls |
| `P0.md` | load tests; why `optiq serve` can't run on 16 GB |
| `bench.py` | harness, one JSON per run |
| `p3_suite.py`, `run_p3.py` | P3 accuracy and quality test harness |
| `run_p2.sh`, `visualize_p2.py` | reproduce the runs and chart |
| `results/` | raw per-run JSON and summaries |

## Reproduce

Needs the two checkpoints under `models/` (`rapid-mlx-4bit`, `optiq-4bit`) and a venv with `mlx-lm==0.32.2`, `mlx==0.32.4`, `matplotlib`:

```bash
# Speed and TTFT benchmark (P2)
caffeinate -i ./run_p2.sh && .venv-mlxlm/bin/python visualize_p2.py

# Accuracy and Quality benchmark (P3)
caffeinate -i .venv-mlxlm/bin/python run_p3.py
```

Status: Complete across P0 (loads), P1 (harness), P2 (speed/memory), and P3 (accuracy/quality).
