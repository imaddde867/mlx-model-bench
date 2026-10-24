# P2 results — speed, TTFT, memory

`nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB` on MLX. Apple M4 (10-core), 16 GB, macOS 27.0.1,
`mlx` 0.32.4, `mlx-lm` 0.32.2, python 3.12.10. Greedy, `enable_thinking=False`,
`conversational_mode=True`, `max_tokens=256`. 1 discarded warm-up + 5 trials; **median [min–max]**.
Plugged in, `caffeinate`. Raw rows: `results/arm*/`. Reproduce: `run_p2.sh`, then `visualize_p2.py`.

![P2 chart](p2_visual.png)

Arms: **1** = plain 4-bit checkpoint, `mlx_lm` in-process · **2** = same checkpoint behind `mlx_lm.server` ·
**3** = OptiQ mixed 4/8-bit checkpoint, `mlx_lm` in-process. Arms 4/5 are unmeasurable (see below).

Each request starts with a random nonce so the server's prefix cache cannot hit. Arm 2 "cached" re-sends
identical prompts to show what that cache is worth.

## TTFT (s)

| Prompt tokens | Arm 1 | Arm 2 | Arm 3 | Arm 2 cached |
|---|---|---|---|---|
| 68 | 0.195 [0.192–0.200] | 0.161 [0.159–0.167] | 0.217 [0.204–0.234] | 0.068 [0.065–0.076] |
| 1,026 | 1.62 [1.62–1.63] | 1.52 [1.52–1.61] | 1.68 [1.68–1.72] | 0.070 [0.067–0.074] |
| 8,341 | 14.3 [13.7–14.3] | 14.4 [14.1–14.6] | 14.6 [14.4–15.0] | 0.094 [0.091–0.104] |

(The cached rows used 60 / 1,018 / 8,333 prompt tokens: no nonce.) Prefill saturates near 570–630 tok/s.

## Decode (tok/s, prefill excluded)

| Prompt tokens → generated | Arm 1 | Arm 2 | Arm 3 |
|---|---|---|---|
| 85 → ~100–160 (`decode-short`) | 34.4 [33.8–34.8] | 32.5 [32.0–33.6] | 25.7 [24.1–26.4] |
| 8,359 → 256 (`decode-long`) | 29.5 [28.2–31.1] | 27.9 [26.7–28.9] | 23.8 [23.7–24.5] |

Generated tokens on `decode-short`: arm 1 131, arm 2 96, arm 3 162 (greedy output diverges across
engine/checkpoint). Only the `decode-*` rows have a long enough generation for decode tok/s; the
other prompts generate 4–31 tokens.

## Memory (in-process, `mx.get_peak_memory`)

| | Arm 1 | Arm 3 |
|---|---|---|
| Load peak | 4.20 GB | 6.53 GB |
| Peak, 68-token prompt | 4.38 GB | 6.71 GB |
| Peak, 8.3k prompt + 256 gen | 5.09 GB | 7.42 GB |

Arm 2 reports server RSS (about 1 GB), which excludes Metal allocations; not comparable.

## Findings

1. **HTTP costs almost nothing.** Arm 2 matches arm 1 on TTFT within noise and decodes about 5% slower.
2. **OptiQ costs speed and memory for no prefill gain.** Arm 3 decodes about 25% slower than arm 1 and
   needs about 2.3 GB more. Whether it is better in quality is untested (P3).
3. **Prefix cache is the big lever for chat.** A repeated prompt returns in 0.07–0.09 s at any length,
   versus 14 s for a fresh 8k prompt. It does nothing for new prompts.
4. **Long fresh prompts are the weak spot:** about 14 s TTFT at 8k tokens.

## Caveats and invalidated data

- An earlier run sent identical prompts and measured Arm 2 at 0.09 s TTFT / 96,000 prefill tok/s on 8k
  tokens: a prefix-cache hit, not engine speed. That data was discarded.
- Earlier 4–6-token generations made decode tok/s meaningless (Arm 2 read 45–48 tok/s). Ignore any figure
  not from a `decode-*` row.
- Arm 1 vs Arm 2 short-prompt TTFT differs by 0.034 s; not explained, possibly timer placement.
- Arm 3 generated only 26–31 tokens on the medium/long prompts.
- Single machine, single session; no thermal logging. Median of 5 with range is the only dispersion shown.

## Not measured

- **Arm 4/5:** `optiq serve` 0.7.3 crashes on every completion with a 10.5 GiB Metal allocation
  (limit 8.88 GiB on this 16 GB GPU), sized for 32,768 positions at model build regardless of prompt.
  No CLI flag reaches it. See `P0.md`.
- **Accuracy (30 prompts) and quality (8 prompts):** P3, not run.
- Temp 1.0 / top_p 0.95 row, thinking-on row.
