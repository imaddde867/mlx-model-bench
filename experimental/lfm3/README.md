# LFM3-1.7B parity (experimental, issue #37)

**Status:** compatibility evidence only. LFM3 is not in the roster and this directory adds no
benchmark row. `registry.py`, `.venv-mlxlm`, `results/` and the replicated study are untouched.
Chat template, tool use, server compatibility and M4 performance are out of scope here.

The pinned `mlx-lm` 0.32.2 has no `lfm3` architecture. Its model code is the open PR
[ml-explore/mlx-lm#2016](https://github.com/ml-explore/mlx-lm/pull/2016). This directory checks
that PR against LiquidAI's own reference implementation (`modeling_lfm3.py`, run through
transformers), so that whenever the code lands in a release, "it loads" can be told apart from
"it computes the same model".

## Pins

| What | Pin |
|---|---|
| `mlx-lm` | PR #2016 head `9027ac5b062175fba890c612c3905d5d8d5b5294`, installed from git |
| `mlx` | `0.32.4` (same as the benchmark venv) |
| `transformers` | `5.21.0`: the remote `modeling_lfm3.py` imports `use_kernel_func_from_hub_with_fallback`, which the `5.12.1` named in `config.json` lacks |
| `torch` | `2.14.1`, CPU only |
| Reference checkpoint | `LiquidAI/LFM3-1.7B` @ `40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd` |
| 4-bit checkpoint | `LiquidAI/LFM3-1.7B-MLX-4bit` @ `d0ca73dcfa70ce18733754a492cdc1fb0123262d` |
| Fixture | `fixture.json`, sha256 of its content `cc37cfca8e66a12c91bfa397ff1e5a318aae906fa93a2bb625788b777f692922` |

Weight files (the fixture records every fetched file; `check_parity.py` refuses a mismatch):

| Checkpoint | `model.safetensors` sha256 |
|---|---|
| `LiquidAI/LFM3-1.7B` | `135922caef64e81e52af88ede8b7c3ddaa9d6da220a9737fb3065275dbfb4330` |
| `LiquidAI/LFM3-1.7B-MLX-4bit` | `1be5bf672013cd91aa62148f77c5eea8d2784a95845d5fe826e08c7b84823815` |

Only `*.safetensors`, `*.json`, `*.py` and `*.jinja` are fetched (~3.4 GB + ~1 GB); the repos'
README and image assets change nothing measured.

## What is compared

Input: the first 640 tokens of `prompts/replicated/long.txt`, as LFM3's tokenizer encodes it,
stored as ids. 640 is 2.5x the 256-token sliding window, so the sliding-attention layers run well
past the point where the window starts dropping keys. Then:

| Row | What | Gate |
|---|---|---|
| `fp32` | reference weights cast to fp32 in MLX vs HF fp32 on CPU: logits on the HF top-10 at every position, MLX full-vocabulary argmax, and a 64-token greedy decode of a fixed chat prompt | max abs diff ≤ 1e-3, top-1 agreement 1.0, greedy identical to the end |
| `cache_fp32` | prefill 128 tokens, then one at a time to 640 through `make_prompt_cache()`, vs the same model's uncached forward, full vocabulary | max abs diff ≤ 1e-3, top-1 agreement 1.0 |
| `4bit`, `cache_4bit` | the same for the 4-bit checkpoint | recorded, not gated |

The fixture holds the HF top-10 logits per position, not all 128k (that would be ~330 MB), so the
logit difference is measured where a next-token decision is made. The greedy reference was
decoded with `repetition_penalty=1.0`: the checkpoint's `generation_config.json` sets 1.1, which
HF `generate` applies even to greedy decoding and `mlx_lm`'s sampler does not.

## Result (2026-10-28, Apple M4 Pro, macOS 26.4)

From [`runs/2026-10-28-m4pro.json`](runs/2026-10-28-m4pro.json):

| vs HF reference (fp32, CPU) | max abs diff (top-10) | mean abs diff | top-1 agreement | top-1 after token 256 | greedy tokens identical |
|---|---|---|---|---|---|
| `fp32` | 0.0001 | 0.00001 | 1.0000 | 1.0000 | 64/64 |
| `4bit` | 7.0342 | 0.70679 | 0.8672 | 0.8958 | 14/64 |

| cached vs uncached | max abs diff | top-1 agreement |
|---|---|---|
| `cache_fp32` | 0.00021 | 1.0000 |
| `cache_4bit` | 0.54688 | 0.9750 |

Both gates pass: the PR's model code computes the reference model. The 4-bit checkpoint picks the
reference's next token at 87% of these 640 positions. That is a property of the quantized file, not
of the code, and it matters for any 4-bit LFM3 row placed next to the roster. The 4-bit cached
decode also differs from its own uncached forward at 2.5% of positions; that is consistent with
quantized matmuls taking different kernels for one token and for many, but it was not investigated
further here.

These numbers are for this input on this machine. The issue #37 comment measured the same setup
on a different 640-token input (top-1 0.795 for the 4-bit file), which is why it is not repeated
here.

## Reproduce

```bash
uv venv .venv-lfm3 --python 3.12
uv pip install --python .venv-lfm3/bin/python -r experimental/lfm3/requirements.txt

# The check. Does not run the HF model: the reference is fixture.json.
# Loads one model at a time, fp32 peak about 7 GB; ~2 minutes after the download.
caffeinate -i .venv-lfm3/bin/python experimental/lfm3/check_parity.py \
  --out experimental/lfm3/runs/<date>-<machine>.json

# Offline, any Python 3.12, no models:
python3 -m unittest tests.test_lfm3_parity -v
```

`check_parity.py` exits nonzero if a gate fails, refuses an `mlx-lm` that was not installed from
the pinned commit, refuses a checkpoint file whose sha256 differs from the fixture, and never
overwrites a run record. When a released `mlx-lm` includes LFM3, install it in a new venv and
rerun with `--allow-unpinned`: the record keeps the version it ran, and the gates still apply.

Regenerating the reference is only needed to move a pin, and then all pins move together:

```bash
.venv-lfm3/bin/python experimental/lfm3/make_fixture.py --out /tmp/fixture.json   # ~8 GB peak, CPU
```

## Files

| File | What |
|---|---|
| `lfm3_parity.py` | pins, fixture schema, comparison and gate arithmetic; standard library only |
| `make_fixture.py` | HF reference → `fixture.json` (transformers, torch) |
| `check_parity.py` | PR #2016 vs `fixture.json` → `runs/*.json` (MLX) |
| `requirements.txt` | the experimental venv |
| `fixture.json` | input ids, HF top-10 logits per position, greedy reference, checkpoint hashes |
| `runs/` | one write-once record per check |
| `../../tests/test_lfm3_parity.py` | offline tests, run by CI with the rest of `tests/` |
