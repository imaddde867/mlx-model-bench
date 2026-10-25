#!/usr/bin/env python3
"""Fetch every checkpoint in the registry, in parallel.

`run_bench.py --phase fetch` walks one model at a time. On this box a single
Hub stream runs at ~20-25 MB/s, so the roster fetch is bandwidth-bound rather
than CPU-bound: pulling several checkpoints at once is the only thing that
makes it fast. huggingface_hub 2.3 no longer uses `hf_transfer`, so there is no
env flag to set -- concurrency is the lever.

Downloads only. Verification stays where it belongs:

    .venv-mlxlm/bin/python run_bench.py --model <id> --phase fetch

which skips files already on disk and writes the sha256 provenance record.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import snapshot_download

import registry

ROOT = Path(__file__).resolve().parent
# Weights plus everything a checkpoint needs to be reproducible: configs, tokenizer,
# chat template, and the repo's own artifacts (README, LICENSE, any trust_remote_code
# .py). The Hub's full file list is what the audit compares against, so a pattern gap
# here shows up as a missing file, not as a silently thinner checkpoint.
PATTERNS = ["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model",
            "*.py", "*.md", "*LICENSE*", "*.png", ".gitattributes"]


def gb(path):
    return sum(f.stat().st_size for f in Path(path).glob("*")
               if f.is_file() and not f.name.startswith(".")) / 1e9


def fetch(ckpt):
    try:
        snapshot_download(ckpt.hf_repo, local_dir=str(ckpt.local_path),
                          allow_patterns=PATTERNS, max_workers=4)
        return ckpt, None
    except Exception as e:                       # noqa: BLE001 - report, never hide
        return ckpt, f"{type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6,
                    help="checkpoints downloaded at once (default 6)")
    args = ap.parse_args()

    # Every checkpoint that names a repo is top-up fetched, not just the unloadable
    # ones: snapshot_download is idempotent, so an existing checkpoint costs a HEAD
    # per file and this picks up anything a narrower pattern missed (a repo's own
    # README, LICENSE or trust_remote_code .py).
    pending = [c for m in registry.list_models() for c in m.checkpoints.values()
               if c.hf_repo]
    done = [c for c in pending if c.available]

    print(f"{len(done)} of {len(pending)} checkpoint(s) loadable, topping up all "
          f"{len(pending)} against the Hub")
    for c in done:
        print(f"  have {c.tag:11s} {gb(c.local_path):5.2f} GB  {c.local_dir}")

    total = sum(c.download_gb or c.weights_gb for c in pending)
    print(f"\nup to {total:.1f} GB total, {args.jobs} checkpoints at a time\n")

    failed = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(fetch, c): c for c in pending}
        for fut in as_completed(futures):
            ckpt, err = fut.result()
            if err:
                failed.append((ckpt, err))
                print(f"  FAIL {ckpt.tag:11s} {ckpt.hf_repo}: {err}")
            else:
                print(f"  ok   {ckpt.tag:11s} {gb(ckpt.local_path):5.2f} GB  {ckpt.local_dir}")

    if failed:
        print(f"\n{len(failed)} failed:")
        for ckpt, err in failed:
            print(f"  {ckpt.hf_repo}: {err}")
        return 1
    print("\nall checkpoints present — now verify: run_bench.py --model <id> --phase fetch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
