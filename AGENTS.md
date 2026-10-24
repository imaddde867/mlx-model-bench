# nemotron-bolt-bench

Benchmarking `nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB` on Apple Silicon MLX.
Read `DESIGN.md` before touching the arms. This file is the operating contract.

## Machine

Apple M4, 10-core, 16 GB unified, macOS 27.0.1. Python 3.12.10, uv 0.10.9.

## Pins

`mlx-lm` 0.32.2 · `mlx` 0.32.4 · `mlx-optiq` 0.7.3 (requires `mlx-lm <0.33`).
One venv per engine, so neither can silently upgrade the other. Record every
version inside each result row, not in prose.

## Rules that make the numbers comparable

- Greedy decoding. The card recommends temp 1.0 / top_p 0.95; that is a
  separate, labelled row, not the default.
- Discard run 1 (Metal kernel compile, first-touch faults). Report median of
  5 **and** the range. Never a bare median.
- Fixed prompt files, fixed `max_tokens`, identical across arms.
- HTTP arms: every request carries a random nonce (`bench.py`), or the server's prefix cache
  reports fake prefill speed. `--no-nonce --tag cached` measures the cache deliberately.
- `caffeinate` on, plugged in, nothing else running.
- No number leaves the house without its arm, prompt length and generation
  length printed next to it.

## Chat template controls

`enable_thinking` and `conversational_mode` are template variables;
`thinking_token_budget` is a serving-API parameter, not a template variable. Pass
`enable_thinking` explicitly — transformers injects `True` when it is omitted.
Primary table is `conversational_mode=True`, `enable_thinking=False`.

## Status

- [x] P0 — load tests. Arms 1–3 load; arm 4 blocked by an optiq prefill buffer. See `P0.md`.
- [x] P1 — `bench.py` — verified arms 1, 2, 3 produce valid JSON with all metrics.
- [x] P2 — speed/TTFT/memory runs, arms 1–3, valid after cache-bust fix. See `RESULTS.md`.
- [ ] P3 — accuracy (30) + quality (8) prompt sets, then final report

## Known traps

- NVIDIA's **DFlash** drafter is GGUF + NVIDIA GPUs only, and it barely moves
  TTFT. Do not plan around it.
- Paper evals are Q4_K_M GGUF, not these MLX quants. Their scores are an upper
  reference for a different file, not a target for ours.
- 64K context will not fit in 16 GB. Do not test it.
- The mixed-bit OptiQ checkpoint may not load in plain `mlx_lm`. P0 decides
  that before any harness code is written.
