#!/usr/bin/env python3
"""P0: can this engine load this checkpoint on 16 GB, and at what cost.

Run with the engine's own interpreter. Prints one JSON row; never raises past main.
"""

import argparse
import json
import platform
import sys
import time
import traceback
import importlib.metadata as md

import mlx.core as mx

PROMPT = "Reply with exactly one short sentence: what is 17 multiplied by 23?"


def version(pkg):
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--template-kwargs", default="{}", help='chat template controls, e.g. {"enable_thinking": false}')
    ap.add_argument("--no-determinism", action="store_true",
                    help="skip the generate-twice reproducibility check")
    args = ap.parse_args()

    row = {
        "model": args.model,
        "engine": "mlx_lm",
        "versions": {p: version(p) for p in ("mlx", "mlx-lm", "mlx-optiq", "transformers")},
        "host": f"{platform.machine()} {platform.mac_ver()[0]}",
        "max_tokens": args.max_tokens,
    }
    t0 = time.perf_counter()
    try:
        from mlx_lm import generate, load

        # trust_remote_code is passed as an explicit False, not left unset: Spark's
        # repos declare an auto_map, and an unset value makes transformers ask
        # interactively whether to execute the repo's own code -- which blocks a
        # non-interactive run. mlx_lm has its own spark2_5 implementation and never
        # needs the .py files.
        model, tokenizer = load(args.model, tokenizer_config={"trust_remote_code": False})
        row["load_s"] = round(time.perf_counter() - t0, 2)
        row["load_peak_gb"] = round(mx.get_peak_memory() / 1e9, 2)

        msgs = [{"role": "user", "content": PROMPT}]
        prompt = tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False,
            **json.loads(args.template_kwargs),
        )
        row["prompt_tokens"] = len(tokenizer.encode(prompt))

        t1 = time.perf_counter()
        out = generate(model, tokenizer, prompt=prompt, max_tokens=args.max_tokens, verbose=False)
        row["gen_s"] = round(time.perf_counter() - t1, 2)
        row["completion_tokens"] = len(tokenizer.encode(out))
        row["peak_gb"] = round(mx.get_peak_memory() / 1e9, 2)
        row["sample"] = out.strip()[:200]

        # Greedy must be reproducible, or every later median is a median of
        # something else. MoE routing and hybrid attention are new here, so this
        # is checked rather than assumed.
        if not args.no_determinism:
            again = generate(model, tokenizer, prompt=prompt, max_tokens=args.max_tokens, verbose=False)
            row["deterministic"] = again == out
            if again != out:
                row["repeat_sample"] = again.strip()[:200]
        row["ok"] = True
    except Exception as e:  # a failed arm is a P0 result, not a crash
        row["load_s"] = round(time.perf_counter() - t0, 2)
        row["ok"] = False
        row["error"] = f"{type(e).__name__}: {e}"
        row["traceback_tail"] = traceback.format_exc().strip().splitlines()[-3:]
    print(json.dumps(row, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
