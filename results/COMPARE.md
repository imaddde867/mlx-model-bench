Transport: inproc. Medians of 5 kept trials; spread = max/min of kept TTFT, 'not recorded' = the row predates condition recording.
Weight-read proxy = checkpoint weights GB × short-prompt decode tok/s; it is a size-based estimate, not measured memory bandwidth. `~` marks the MoE upper bound because only a subset of experts is active per token.

| Model | Tier | bpw | Publisher | Params | Weights GB | Decode short tok/s [min-max] | Decode long tok/s | TTFT long s | Prompt tok (long) | Gen tok (decode-long) | Peak GB (long) | Weight-read proxy GB/s | TTFT spread | Condition | Acc /30 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LFM2.5-8B-A1B | 4bit | 4.58 | vendor | 8.468B/1.3B active | 4.85 | 70.9 [69.8-73.1] | 68.8 | 9.285 | 8510 | 256 | 5.779 | ~343.9 | 1.06 | not recorded | 20/30 |
| LFM2.5-8B-A1B | optiq-4bit | 5.17 | mlx-community | 8.468B/1.3B active | 5.47 | 58.7 [57.4-62.4] | 59.6 | 9.2765 | 8510 | 256 | 6.37 | ~321.1 | 1.26 | not recorded | 21/30 |
| LFM2.5-8B-A1B | 8bit | 8.52 | vendor | 8.468B/1.3B active | 9.02 | 47.3 [45.7-47.5] | 37.9 | 9.3465 | 8510 | 256 | 9.947 | ~426.6 | 1.52 | not recorded | 21/30 |
| MiniCPM5-2B | 4bit | 4.55 | vendor | 2.517B/2.517B active | 1.43 | 69.0 [54.7-71.7] | 54.6 | 14.606 | 8354 | 256 | 2.465 | 98.7 | 1.34 | not recorded | 19/30 |
| MiniCPM5-2B | optiq-4bit | 6.17 | mlx-community | 2.517B/2.517B active | 1.94 | 56.4 [55.3-56.8] | 44.4 | 14.0424 | 8354 | 256 | 2.975 | 109.4 | 3.73 | not recorded | 17/30 |
| MiniCPM5-2B | 8bit | 8.52 | mlx-community | 2.517B/2.517B active | 2.68 | 37.9 [36.7-38.3] | 32.6 | 14.7678 | 8354 | 256 | 3.723 | 101.6 | 1.21 | not recorded | 18/30 |
| NVIDIA Nemotron 3.5 Bolt E4B | 4bit | 4.5 | third-party:rapid-mlx | 7.463B/4.0B active | 4.2 | 34.4 [33.8-34.8] | 29.5 | 14.2773 | 8341 | 256 | 5.089 | ~144.5 | 1.07 | not recorded | 24/30 |
| NVIDIA Nemotron 3.5 Bolt E4B | optiq-4bit | 7.0 | mlx-community | 7.463B/4.0B active | 6.53 | 25.7 [24.1-26.4] | 23.8 | 14.6099 | 8341 | 256 | 7.42 | ~167.8 | 1.15 | not recorded | 25/30 |
| NVIDIA Nemotron 3.5 Bolt E4B | 8bit | 8.53 | third-party:rapid-mlx | 7.463B/4.0B active | 7.96 | 16.5 [15.1-18.4] | 14.9 | 15.141 | 8341 | 256 | 8.82 | ~131.3 | 1.49 | not recorded | 24/30 |
| Spark-X2.5-4B | 4bit | 4.51 | third-party:abenzerps | 4.112B/4.112B active | 2.32 | 39.8 [33.4-40.4] | 29.4 | 26.6625 | 8631 | 256 | 3.971 | 92.3 | 1.09 | not recorded | 18/30 |
| Spark-X2.5-4B | optiq-4bit | 5.93 | mlx-community | 4.112B/4.112B active | 3.05 | 31.3 [29.5-31.4] | 25.6 | 27.6026 | 8631 | 256 | 4.698 | 95.5 | 1.22 | not recorded | 18/30 |
| Spark-X2.5-4B | 8bit | 8.52 | third-party:abenzerps | 4.112B/4.112B active | 4.38 | 18.6 [18.4-20.7] | 20.6 | 27.3866 | 8631 | 256 | 6.027 | 81.5 | 1.26 | not recorded | 21/30 |

Repeat decode-short checks (five kept trials; separate from full-matrix rows).

| Model | Checkpoint | bpw | Publisher | Prompt tok | Gen tok | Original decode tok/s [min-max] | Repeat decode tok/s [min-max] | Change | Original / repeat TTFT s | Rerun summary | Recorded caveats |
|---|---|---:|---|---:|---:|---|---|---:|---|---|---|
| LFM2.5-8B-A1B | mlx-4bit | 4.58 | vendor | 53 | 256 | 70.9 [69.8-73.1] | 86.1 [82.6-86.6] | +21.4% | 0.1341 / 0.1163 | [summary.json](lfm2.5-8b-a1b/mlx-4bit-inproc-decode-check/summary.json) | busy: WindowServer 31%; swap in use: 0.611 GB (the model may be paging) |
| MiniCPM5-2B | mlx-4bit | 4.55 | vendor | 55 | 256 | 69.0 [54.7-71.7] | 71.6 [71.4-72.1] | +3.8% | 0.1143 / 0.1133 | [summary.json](minicpm5-2b/mlx-4bit-inproc-decode-check/summary.json) | busy: spotlightknowledged.updater 17%; swap in use: 0.611 GB (the model may be paging) |
