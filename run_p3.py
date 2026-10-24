#!/usr/bin/env python3
"""Execute P3 benchmark: 30 accuracy prompts + 8 quality prompts for Arm 1 and Arm 3.

Controls:
- mlx_lm in-process
- greedy decoding
- conversational_mode=True, enable_thinking=False
- No nonces (clean accuracy evaluation)
"""

import json
import os
import random
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load, stream_generate

from p3_suite import (
    CHECKERS,
    INSTR_PROMPTS,
    MATH_PROMPTS,
    QUALITY_PROMPTS,
    TOOL_PROMPTS,
)

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results" / "p3"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TEMPLATE = {"enable_thinking": False, "conversational_mode": True}

ARMS = {
    "1": {"name": "Arm 1 (rapid-mlx-4bit)", "model_path": ROOT / "models/rapid-mlx-4bit"},
    "3": {"name": "Arm 3 (OptiQ-4bit)", "model_path": ROOT / "models/optiq-4bit"},
}


def run_single(model, tokenizer, prompt_text: str, max_tokens: int = 256):
    formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        add_generation_prompt=True,
        tokenize=False,
        **TEMPLATE,
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
    full_text = "".join(parts).strip()
    return {
        "text": full_text,
        "prompt_tokens": prompt_tokens,
        "generation_tokens": gen,
        "wall_s": round(wall, 4),
        "ttft_s": round(first - t0, 4) if first else None,
        "decode_tps": round((gen - 1) / (last - first), 1) if gen > 1 and last > first else None,
        "peak_gb": round(mx.get_peak_memory() / 1e9, 3),
    }


def evaluate_arm(arm_id: str):
    arm_info = ARMS[arm_id]
    model_path = str(arm_info["model_path"])
    print(f"\n=======================================================")
    print(f"Loading {arm_info['name']} from {model_path}...")
    print(f"=======================================================")
    t_load = time.perf_counter()
    model, tokenizer = load(model_path)
    load_time = round(time.perf_counter() - t_load, 2)
    load_peak = round(mx.get_peak_memory() / 1e9, 3)
    print(f"Loaded in {load_time}s (peak memory: {load_peak} GB)")

    results = {
        "arm": arm_id,
        "name": arm_info["name"],
        "model_path": model_path,
        "load_time_s": load_time,
        "load_peak_gb": load_peak,
        "accuracy_runs": [],
        "quality_runs": [],
    }

    all_acc = MATH_PROMPTS + TOOL_PROMPTS + INSTR_PROMPTS
    print(f"\nRunning {len(all_acc)} Accuracy Prompts on Arm {arm_id}...")
    for idx, item in enumerate(all_acc, 1):
        p_id = item["id"]
        cat = item["category"]
        prompt = item["prompt"]
        gen_res = run_single(model, tokenizer, prompt, max_tokens=256)
        checker_fn = CHECKERS[item["checker"]]
        target = item.get("target")

        try:
            if target is not None:
                passed = checker_fn(gen_res["text"], target)
            else:
                passed = checker_fn(gen_res["text"])
        except Exception as e:
            passed = False
            gen_res["checker_error"] = str(e)

        record = {
            "id": p_id,
            "category": cat,
            "prompt": prompt,
            "target": target,
            "checker": item["checker"],
            "response": gen_res["text"],
            "passed": passed,
            "tokens": {
                "prompt": gen_res["prompt_tokens"],
                "generation": gen_res["generation_tokens"],
            },
            "metrics": {
                "wall_s": gen_res["wall_s"],
                "ttft_s": gen_res["ttft_s"],
                "decode_tps": gen_res["decode_tps"],
            },
        }
        status_sym = "✓" if passed else "✗"
        print(f"  [{idx:02d}/30] {p_id} ({cat}): {status_sym} | resp: {gen_res['text'][:60]!r}")
        results["accuracy_runs"].append(record)

    print(f"\nRunning {len(QUALITY_PROMPTS)} Quality Prompts on Arm {arm_id}...")
    for idx, item in enumerate(QUALITY_PROMPTS, 1):
        p_id = item["id"]
        task = item["task"]
        prompt = item["prompt"]
        # Allow 384 tokens for richer quality outputs
        gen_res = run_single(model, tokenizer, prompt, max_tokens=384)
        record = {
            "id": p_id,
            "task": task,
            "prompt": prompt,
            "rubric": item["rubric"],
            "response": gen_res["text"],
            "tokens": {
                "prompt": gen_res["prompt_tokens"],
                "generation": gen_res["generation_tokens"],
            },
            "metrics": {
                "wall_s": gen_res["wall_s"],
                "ttft_s": gen_res["ttft_s"],
                "decode_tps": gen_res["decode_tps"],
            },
        }
        print(f"  [{idx:02d}/08] {p_id} ({task}): {gen_res['generation_tokens']} tokens")
        results["quality_runs"].append(record)

    # Clean up model
    del model
    del tokenizer
    mx.clear_cache()
    time.sleep(2)
    return results


def main():
    print("Starting P3 Benchmark Execution...")
    results_arm1 = evaluate_arm("1")
    (RESULTS_DIR / "arm1_results.json").write_text(json.dumps(results_arm1, indent=2))

    results_arm3 = evaluate_arm("3")
    (RESULTS_DIR / "arm3_results.json").write_text(json.dumps(results_arm3, indent=2))

    # Calculate accuracy stats
    def summarize_acc(runs):
        total = len(runs)
        passed = sum(1 for r in runs if r["passed"])
        by_cat = {}
        for r in runs:
            c = r["category"]
            if c not in by_cat:
                by_cat[c] = {"passed": 0, "total": 0}
            by_cat[c]["total"] += 1
            if r["passed"]:
                by_cat[c]["passed"] += 1
        return {
            "total_passed": passed,
            "total_prompts": total,
            "accuracy_pct": round(100.0 * passed / total, 1),
            "by_category": {
                k: {
                    "passed": v["passed"],
                    "total": v["total"],
                    "accuracy_pct": round(100.0 * v["passed"] / v["total"], 1),
                }
                for k, v in by_cat.items()
            },
        }

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "arm1": {
            "name": results_arm1["name"],
            "load_time_s": results_arm1["load_time_s"],
            "load_peak_gb": results_arm1["load_peak_gb"],
            "accuracy": summarize_acc(results_arm1["accuracy_runs"]),
        },
        "arm3": {
            "name": results_arm3["name"],
            "load_time_s": results_arm3["load_time_s"],
            "load_peak_gb": results_arm3["load_peak_gb"],
            "accuracy": summarize_acc(results_arm3["accuracy_runs"]),
        },
    }
    (RESULTS_DIR / "accuracy_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=======================================================")
    print("Accuracy Summary:")
    print(json.dumps(summary, indent=2))
    print("=======================================================")

    # Prepare Blind Quality Dataset
    # We assign Arm 1 vs Arm 3 randomly per prompt using a fixed deterministic seed
    random.seed(1337)
    blind_pairs = []
    blind_key = {}
    for q1, q3 in zip(results_arm1["quality_runs"], results_arm3["quality_runs"]):
        assert q1["id"] == q3["id"]
        flip = random.random() < 0.5
        model_a_arm = "arm1" if flip else "arm3"
        model_b_arm = "arm3" if flip else "arm1"
        model_a_text = q1["response"] if flip else q3["response"]
        model_b_text = q3["response"] if flip else q1["response"]
        blind_key[q1["id"]] = {
            "Model A": model_a_arm,
            "Model B": model_b_arm,
        }
        blind_pairs.append({
            "id": q1["id"],
            "task": q1["task"],
            "prompt": q1["prompt"],
            "rubric": q1["rubric"],
            "Model A": model_a_text,
            "Model B": model_b_text,
        })

    (RESULTS_DIR / "blind_quality_prompts.json").write_text(json.dumps(blind_pairs, indent=2))
    (RESULTS_DIR / "blind_key.json").write_text(json.dumps(blind_key, indent=2))
    print(f"\nGenerated blind quality set at {RESULTS_DIR / 'blind_quality_prompts.json'}")
    print(f"Blind key stored at {RESULTS_DIR / 'blind_key.json'}")
    print("P3 Run Completed successfully.")


if __name__ == "__main__":
    main()
