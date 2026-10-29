#!/usr/bin/env python3
"""Output-completion pilot for the replicated session study (issue #34).

SESSION_PROTOCOL.md requires a separate, explicitly non-benchmark text-only
pilot before any study session: do the fixed prompts and the common
`max_tokens=512` cap let every plain-4-bit checkpoint produce an answer, and
what are its answer texts and stop behaviors (empty output, reasoning-only
output, truncation, model failure)? No timing and no memory are captured or
published here; nothing in this pilot is performance evidence and none of its
data feeds `plan`/`run`/benchmarks.

    .venv-mlxlm/bin/python pilot.py --condition clean

Run dir: results/replicated/pilot/<utc>-pilot-completion/. Write-once; the
runner refuses an existing directory. Files stay separate from the 20-session
study and from historical P2 evidence by construction.
"""

import argparse
import hashlib
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import gates
import registry
import sysinfo
from replicated import MAX_TOKENS, PROMPT_DIR, REGIMES, checkpoint, write_once
from run_p3 import final_answer

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "replicated" / "pilot"


def now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def run_id():
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-pilot-completion"


def sha_file(path, chunk=8 << 20):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def finish_flag(resp):
    """mlx_lm 0.32.2 exposes a per-step finish_reason; None only if it does not."""
    return getattr(resp, "finish_reason", "no_finish_reason_field")


def classify(raw, gen_tokens, cap, fr, error=None):
    flags = []
    if error is not None:
        flags.append("failed")
    else:
        if not raw.strip():
            flags.append("empty_output")
        elif not final_answer(raw).strip():
            flags.append("reasoning_only")
        if fr == "length" or (fr is None and gen_tokens >= cap):
            flags.append("truncated")
        elif fr != "stop":
            flags.append(f"unexpected_finish_reason:{fr}")
        if final_answer(raw).lower().strip(" .!\n") not in {"forty years", "40 years"}:
            flags.append("wrong_answer")
    return flags


def run_model(cfg, ckpt, prompts, cap, run_dir, evidence):
    from mlx_lm import load

    prov_problems = gates.problems(cfg, ckpt, evidence, need_p0=True)
    record = {
        "schema": 1,
        "kind": "pilot_completion_text_only",
        "not_a_benchmark": "no timing or memory captured; this row is not performance evidence",
        "at": now(),
        "model_id": cfg.id,
        "model_name": cfg.name,
        "publisher": ckpt.publisher,
        "tier": ckpt.tier,
        "checkpoint_tag": ckpt.tag,
        "hf_repo": ckpt.hf_repo,
        "bpw": ckpt.bits_per_weight,
        "weights_gb": ckpt.weights_gb,
        "checkpoint_dir": ckpt.local_dir,
        "params_total": cfg.params_total,
        "params_active": cfg.params_active,
        "template_kwargs": cfg.template_kwargs,
        "max_tokens_cap": cap,
        "provenance_problems": prov_problems,
    }
    if prov_problems:
        record["status"] = "skipped_unverified"
        write_once(run_dir / f"{cfg.id}.json", record)
        return record

    try:
        model, tokenizer = load(str(ckpt.local_path),
                                tokenizer_config={"trust_remote_code": False})
    except Exception as e:
        record.update(error=f"load failed: {e}", status="load_failed",
                      regimes={}, flags=["load_failed"])
        write_once(run_dir / f"{cfg.id}.json", record)
        return record

    from mlx_lm import stream_generate

    record["regimes"] = {}
    for regime, text in prompts.items():
        row = {}
        raw = ""
        try:
            formatted = tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                add_generation_prompt=True,
                tokenize=False,
                **cfg.template_kwargs,
            )
            parts = []
            fr = None
            resp = None
            for resp in stream_generate(model, tokenizer, prompt=formatted, max_tokens=cap):
                fr = finish_flag(resp)
                parts.append(resp.text)
            raw = "".join(parts).strip()
            answer = final_answer(raw)
            gen_tokens = resp.generation_tokens if resp is not None else 0
            row.update(prompt_tokens=resp.prompt_tokens if resp is not None else 0,
                       generation_tokens=gen_tokens,
                       finish_reason=fr,
                       answer=answer,
                       raw_response=raw,
                       flags=classify(raw, gen_tokens, cap, fr))
        except Exception as e:
            row.update(error=f"{type(e).__name__}: {e}",
                       flags=classify(raw, 0, cap, None, error=e))
        record["regimes"][regime] = row
        print(f"  {cfg.id}/{regime}: {row.get('prompt_tokens')} prompt tok, "
              f"{row.get('generation_tokens')} gen tok, finish_reason={row.get('finish_reason')}"
              f"{', flags: ' + ','.join(row['flags']) if row.get('flags') else ''}")

    flags = sorted({f for r in record["regimes"].values() for f in r.get("flags", [])})
    record["flags"] = flags
    record["status"] = "ok" if not flags else "incomplete:" + "_".join(flags)
    write_once(run_dir / f"{cfg.id}.json", record)
    return record


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OUT), help="parent dir; a fresh run dir is created in it")
    ap.add_argument("--evidence", type=Path, default=ROOT / "results",
                    help="verified provenance/P0 evidence root")
    ap.add_argument("--condition", choices=("clean", "loaded"), required=True,
                    help="operator-stated intent, 'clean' or 'loaded'; recorded, nothing is inferred")
    args = ap.parse_args()

    run_dir = Path(args.out) / run_id()
    if run_dir.exists():
        sys.exit(f"{run_dir} already exists — write-once, choose a fresh run")
    run_dir.mkdir(parents=True)

    snap_before = sysinfo.snapshot()
    caffeinate_warn = []
    if (snap_before.get("power") or "") != "ac":
        caffeinate_warn.append("not on AC power at pilot start")
    print("conditions")
    for w in sysinfo.warnings(snap_before):
        print(f"  - {w}")
        caffeinate_warn.append(w)

    cap = MAX_TOKENS
    prompts = {regime: (PROMPT_DIR / f"{regime}.txt").read_text()
               for regime in REGIMES}

    manifest = {
        "schema": 1,
        "kind": "pilot_completion_text_only",
        "at": now(),
        "condition_stated": args.condition,
        "evidence_dir": str(args.evidence.resolve()),
        "prompt_dir": str(PROMPT_DIR.relative_to(ROOT)),
        "max_tokens_cap": cap,
        "temperature": "greedy (no sampler kwargs; temp/top_p never passed)",
        "prompt_files_sha256": {k: sha_file(PROMPT_DIR / f"{k}.txt")
                                for k in prompts},
        "versions": {"python": sys.version.split()[0],
                     "mlx": metadata.version("mlx"),
                     "mlx-lm": metadata.version("mlx-lm")},
        "sysinfo_before": snap_before,
        "condition_warnings_observed": caffeinate_warn,
        "models": [],
    }

    for mid in registry.ORDER:
        cfg = registry.MODELS[mid]
        ckpt = checkpoint(cfg)
        if not ckpt.available:
            manifest["models"].append({"model_id": mid, "status": "checkpoint not on disk"})
            print(f"{mid}: checkpoint not on disk — skipped")
            continue
        print(f"\n== {cfg.id} / {ckpt.tag} ({ckpt.weights_gb} GB, {ckpt.bits_per_weight} bpw)")
        rec = run_model(cfg, ckpt, prompts, cap, run_dir, args.evidence)
        manifest["models"].append({"model_id": mid, "status": rec.get("status"),
                                   "flags": rec.get("flags", [])})
    manifest["sysinfo_after"] = sysinfo.snapshot()
    manifest["warnings_after"] = sysinfo.warnings(manifest["sysinfo_after"])
    manifest_path = run_dir / "manifest.json"
    write_once(manifest_path, manifest)

    print(f"\nmanifest: {manifest_path}")
    failed = any(m.get("status") != "ok" for m in manifest["models"])
    print(("PILOT INCOMPLETE (see flags above — revise cap/prompts before freezing the plan)"
           if failed
           else "PILOT: all four checkpoints produced answers"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
