#!/usr/bin/env python3
"""Check the LFM3 port in ml-explore/mlx-lm#2016 against the recorded HF reference.

Needs only MLX and the pinned PR, not transformers or torch: the reference is
fixture.json. Before loading a checkpoint, every file of it is hashed and must
match the sha256 the fixture recorded at the pinned revision.

Rows:
  fp32        reference weights in fp32 vs the HF logits     (gated)
  cache_fp32  prefill 128 + one token at a time through make_prompt_cache(),
              vs the same model's uncached forward              (gated)
  4bit        LFM3-1.7B-MLX-4bit vs the HF logits             (recorded, not gated)
  cache_4bit  the 4-bit model's cached vs uncached forward      (recorded, not gated)

    .venv-lfm3/bin/python experimental/lfm3/check_parity.py \\
        --out experimental/lfm3/runs/<date>-<machine>.json

Loads one model at a time; fp32 peak is about 7 GB. Exit 0 only if the gates pass.
"""

import argparse
import gc
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import lfm3_parity as lp


def pr_commit():
    """The git commit mlx-lm was installed from, if pip/uv recorded one."""
    try:
        info = json.loads(metadata.distribution("mlx-lm").read_text("direct_url.json") or "{}")
    except (metadata.PackageNotFoundError, json.JSONDecodeError):
        return None
    return info.get("vcs_info", {}).get("commit_id")


def chip():
    try:
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return platform.processor()


def verified_snapshot(repo, revision, expected):
    from huggingface_hub import snapshot_download

    path = Path(snapshot_download(repo, revision=revision, allow_patterns=lp.CHECKPOINT_FILES))
    for name, sha in expected.items():
        f = path / name
        if not f.is_file() or lp.sha_file(f) != sha:
            sys.exit(f"{repo}@{revision[:7]}/{name}: missing or sha256 differs from the fixture")
    return path


def measure(path, fx, fp32):
    import mlx.core as mx
    from mlx_lm.generate import generate_step
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.utils import load_model

    ref = fx["reference"]
    # Weights only: the fixture holds token ids, so no tokenizer is loaded, and
    # transformers never stops to ask whether to run the repo's remote code.
    model, _ = load_model(path)
    if fp32:
        model.set_dtype(mx.float32)
    x = mx.array([fx["input"]["ids"]])

    full = model(x)[0].astype(mx.float32)
    gathered = mx.take_along_axis(full, mx.array(ref["topk_ids"]), axis=-1)
    vs_ref = lp.compare(ref, gathered.tolist(), mx.argmax(full, axis=-1).tolist(), [])

    gen = [int(t) for t, _ in generate_step(mx.array(ref["chat_ids"]), model,
                                            max_tokens=len(ref["greedy_ids"]),
                                            sampler=make_sampler(temp=0.0))]
    vs_ref.update(greedy_identical=lp.first_divergence(ref["greedy_ids"], gen), greedy_ids=gen)

    cache = make_prompt_cache(model)
    steps = [model(x[:, :lp.CACHE_PREFILL], cache=cache)[0]]
    for i in range(lp.CACHE_PREFILL, x.shape[1]):
        steps.append(model(x[:, i:i + 1], cache=cache)[0])
        mx.eval(steps[-1])
    cached = mx.concatenate(steps).astype(mx.float32)
    vs_self = {
        "positions": x.shape[1],
        "prefill": lp.CACHE_PREFILL,
        "max_abs_diff": mx.abs(cached - full).max().item(),
        "top1": mx.mean(mx.argmax(cached, -1) == mx.argmax(full, -1)).item(),
        "cache_types": sorted({type(c).__name__ for c in cache}),
    }
    del model, cache
    gc.collect()
    mx.clear_cache()
    return vs_ref, vs_self


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="new run record; refuses to overwrite")
    ap.add_argument("--fixture", default=str(lp.FIXTURE))
    ap.add_argument("--allow-unpinned", action="store_true",
                    help="run against an mlx-lm that is not the pinned PR commit (e.g. a release that "
                         "merged it); the record says so and the gates still apply")
    args = ap.parse_args()

    if Path(args.out).exists():
        sys.exit(f"{args.out} exists: every run gets a new record")
    fx = lp.load(args.fixture)
    problems = lp.fixture_problems(fx)
    if problems:
        sys.exit("fixture unusable:\n  " + "\n  ".join(problems))

    import mlx.core as mx
    import mlx_lm

    versions = {"mlx": mx.__version__, "mlx_lm": mlx_lm.__version__, "mlx_lm_commit": pr_commit()}
    if versions["mlx_lm_commit"] != lp.PINS["mlx_lm_commit"] and not args.allow_unpinned:
        sys.exit(f"mlx-lm commit {versions['mlx_lm_commit']}, pinned {lp.PINS['mlx_lm_commit']}: "
                 "install experimental/lfm3/requirements.txt, or pass --allow-unpinned")

    weights = fx["weights"]
    ref_path = verified_snapshot(lp.PINS["reference_repo"], lp.PINS["reference_revision"],
                                 weights[lp.PINS["reference_repo"]])
    quant_path = verified_snapshot(lp.PINS["quant_repo"], lp.PINS["quant_revision"],
                                   weights[lp.PINS["quant_repo"]])

    rows = {}
    rows["fp32"], rows["cache_fp32"] = measure(ref_path, fx, fp32=True)
    rows["4bit"], rows["cache_4bit"] = measure(quant_path, fx, fp32=False)

    run = {
        "schema": lp.SCHEMA,
        "fixture_sha256": fx["fixture_sha256"],
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "machine": {"chip": chip(), "macos": platform.mac_ver()[0], "python": platform.python_version()},
        "versions": versions,
        "gates": lp.GATES,
        "rows": rows,
    }
    run["problems"] = lp.run_problems(run, fx)
    run["ok"] = not run["problems"]
    lp.write_once(args.out, run)

    print(f"{lp.N_TOKENS} tokens of {lp.TEXT_FILE} (sliding window {lp.WINDOW}), "
          f"{len(fx['reference']['greedy_ids'])} greedy tokens, logits compared on the HF top-{lp.TOP_K}\n")
    print("| vs HF reference (fp32, CPU) | max abs diff | mean abs diff | top-1 agreement "
          "| top-1 after token 256 | greedy tokens identical |")
    print("|---|---|---|---|---|---|")
    for name in ("fp32", "4bit"):
        r = rows[name]
        print(f"| {name} | {r['max_abs_diff_topk']:.4f} | {r['mean_abs_diff_topk']:.5f} | {r['top1']:.4f} "
              f"| {r['top1_after_window']:.4f} | {r['greedy_identical']}/{r['greedy_len']} |")
    print("\n| cached vs uncached | max abs diff | top-1 agreement | cache types |\n|---|---|---|---|")
    for name in ("cache_fp32", "cache_4bit"):
        r = rows[name]
        print(f"| {name} | {r['max_abs_diff']:.5f} | {r['top1']:.4f} | {', '.join(r['cache_types'])} |")
    print(f"\n{'PASS' if run['ok'] else 'FAIL'}: {args.out}")
    for p in run["problems"]:
        print(f"  {p}")
    sys.exit(0 if run["ok"] else 1)


if __name__ == "__main__":
    main()
