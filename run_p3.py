#!/usr/bin/env python3
"""P3: 30-prompt accuracy + 8-prompt quality, one model at a time, in-process.

Model-agnostic. The prompt sets and checkers live in p3_suite.py and are shared
by every model in the roster, so 80% for one model means the same 30 questions
as 80% for another.

Controls: mlx_lm in-process, greedy, this model's own chat template controls
(from registry.py), no nonces (accuracy runs want the clean prompt).

  .venv-mlxlm/bin/python run_p3.py --model minicpm5-2b
  .venv-mlxlm/bin/python run_p3.py --model nemotron-3.5-bolt --checkpoints mlx-4bit optiq-4bit

Writes results/<model-id>/p3/<checkpoint>-results.json plus accuracy_summary.json.
Blind cross-model scoring is built separately by blind.py — this script only
records what each checkpoint said.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import gates
import registry
from p3_suite import CHECKERS, INSTR_PROMPTS, MATH_PROMPTS, QUALITY_PROMPTS, TOOL_PROMPTS

ROOT = Path(__file__).resolve().parent
ACCURACY_MAX_TOKENS = 256
QUALITY_MAX_TOKENS = 384


def run_single(model, tokenizer, prompt_text, template, max_tokens):
    import mlx.core as mx
    from mlx_lm import stream_generate

    formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        add_generation_prompt=True,
        tokenize=False,
        **template,
    )
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    first = last = None
    parts = []
    resp = None
    for resp in stream_generate(model, tokenizer, prompt=formatted, max_tokens=max_tokens):
        now = time.perf_counter()
        if first is None:
            first = now
        last = now
        parts.append(resp.text)
    wall = time.perf_counter() - t0
    gen = resp.generation_tokens if resp else 0
    prompt_tokens = resp.prompt_tokens if resp else 0
    return {
        "text": "".join(parts).strip(),
        "prompt_tokens": prompt_tokens,
        "generation_tokens": gen,
        "wall_s": round(wall, 4),
        "ttft_s": round(first - t0, 4) if first else None,
        "decode_tps": round((gen - 1) / (last - first), 1) if gen > 1 and last > first else None,
        "peak_gb": round(mx.get_peak_memory() / 1e9, 3),
    }


def evaluate_checkpoint(cfg, ckpt, out_dir):
    import mlx.core as mx
    from mlx_lm import load

    template = cfg.template_kwargs
    print(f"\n== {cfg.id} / {ckpt.tag}  ({ckpt.weights_gb} GB, {ckpt.bits_per_weight} bpw)")
    print(f"   template {json.dumps(template)}")

    t_load = time.perf_counter()
    model, tokenizer = load(str(ckpt.local_path),
                            tokenizer_config={"trust_remote_code": False})
    load_time = round(time.perf_counter() - t_load, 2)
    load_peak = round(mx.get_peak_memory() / 1e9, 3)
    print(f"   loaded in {load_time}s (peak {load_peak} GB)")

    results = {
        "model_id": cfg.id,
        "model_name": cfg.name,
        "checkpoint": ckpt.tag,
        "hf_repo": ckpt.hf_repo,
        "publisher": ckpt.publisher,
        "weights_gb": ckpt.weights_gb,
        "bits_per_weight": ckpt.bits_per_weight,
        "params_total": cfg.params_total,
        "params_active": cfg.params_active,
        "template_kwargs": template,
        "model_path": ckpt.local_dir,
        "load_time_s": load_time,
        "load_peak_gb": load_peak,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "accuracy_runs": [],
        "quality_runs": [],
    }

    all_acc = MATH_PROMPTS + TOOL_PROMPTS + INSTR_PROMPTS
    print(f"   {len(all_acc)} accuracy prompts")
    for idx, item in enumerate(all_acc, 1):
        gen = run_single(model, tokenizer, item["prompt"], template, ACCURACY_MAX_TOKENS)
        checker = CHECKERS[item["checker"]]
        target = item.get("target")
        try:
            passed = checker(gen["text"], target) if target is not None else checker(gen["text"])
        except Exception as e:
            passed = False
            gen["checker_error"] = str(e)

        results["accuracy_runs"].append({
            "id": item["id"],
            "category": item["category"],
            "prompt": item["prompt"],
            "target": target,
            "checker": item["checker"],
            "response": gen["text"],
            "passed": passed,
            "tokens": {"prompt": gen["prompt_tokens"], "generation": gen["generation_tokens"]},
            "metrics": {"wall_s": gen["wall_s"], "ttft_s": gen["ttft_s"], "decode_tps": gen["decode_tps"]},
        })
        print(f"   [{idx:02d}/{len(all_acc)}] {item['id']:10s} {'✓' if passed else '✗'} {gen['text'][:56]!r}")

    print(f"   {len(QUALITY_PROMPTS)} quality prompts")
    for idx, item in enumerate(QUALITY_PROMPTS, 1):
        gen = run_single(model, tokenizer, item["prompt"], template, QUALITY_MAX_TOKENS)
        results["quality_runs"].append({
            "id": item["id"],
            "task": item["task"],
            "prompt": item["prompt"],
            "rubric": item["rubric"],
            "response": gen["text"],
            "tokens": {"prompt": gen["prompt_tokens"], "generation": gen["generation_tokens"]},
            "metrics": {"wall_s": gen["wall_s"], "ttft_s": gen["ttft_s"], "decode_tps": gen["decode_tps"]},
        })
        print(f"   [{idx:02d}/{len(QUALITY_PROMPTS)}] {item['id']:22s} {gen['generation_tokens']} tok")

    (out_dir / f"{ckpt.tag}-results.json").write_text(json.dumps(results, indent=2))

    del model, tokenizer
    mx.clear_cache()
    time.sleep(2)
    return results


def summarize_acc(runs):
    total = len(runs)
    passed = sum(1 for r in runs if r["passed"])
    by_cat = {}
    for r in runs:
        c = by_cat.setdefault(r["category"], {"passed": 0, "total": 0})
        c["total"] += 1
        c["passed"] += bool(r["passed"])
    return {
        "total_passed": passed,
        "total_prompts": total,
        "accuracy_pct": round(100.0 * passed / total, 1) if total else None,
        "by_category": {
            k: {**v, "accuracy_pct": round(100.0 * v["passed"] / v["total"], 1)}
            for k, v in by_cat.items()
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="registry id, HF repo or local path")
    ap.add_argument("--checkpoints", nargs="+", help="checkpoint tags (default: all downloaded)")
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()

    cfg = registry.get_model(args.model)
    tags = args.checkpoints or [c.tag for c in cfg.available_checkpoints]
    if not tags:
        print(f"no checkpoint of {cfg.id} is downloaded under models/", file=sys.stderr)
        return 1

    if gates.refuse(cfg, [cfg.checkpoint(t) for t in tags], args.out):
        print("refusing to measure: provenance/P0 gate failed", file=sys.stderr)
        return 1

    out_dir = Path(args.out) / cfg.id / "p3"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "model_id": cfg.id,
        "model_name": cfg.name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "checkpoints": {},
    }
    for tag in tags:
        ckpt = cfg.checkpoint(tag)
        if not ckpt.available:
            print(f"skip {tag}: {ckpt.local_dir} not downloaded", file=sys.stderr)
            continue
        res = evaluate_checkpoint(cfg, ckpt, out_dir)
        summary["checkpoints"][tag] = {
            "name": f"{cfg.name} ({tag})",
            "hf_repo": ckpt.hf_repo,
            "weights_gb": ckpt.weights_gb,
            "bits_per_weight": ckpt.bits_per_weight,
            "load_time_s": res["load_time_s"],
            "load_peak_gb": res["load_peak_gb"],
            "accuracy": summarize_acc(res["accuracy_runs"]),
        }

    if not summary["checkpoints"]:
        print("nothing ran", file=sys.stderr)
        return 1

    (out_dir / "accuracy_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n" + json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
