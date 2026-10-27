#!/usr/bin/env python3
"""P4 pilot: local, deterministic, text-protocol tool-use tasks.

The tool sandbox is an in-memory dictionary; no commands, network, or real files
are touched by model actions. Tool calls are JSON text, not native API tool calls.
This is a task-completion pilot, not a published cross-model ranking.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import bench
import gates
import registry
import run_p3
import sysinfo

ROOT = Path(__file__).resolve().parent
TASK_FILE = ROOT / "prompts" / "agent" / "tasks.json"
RULES = (
    "Act as a local file assistant. Answer with ONE JSON object per turn, no markdown. "
    'Call a tool with {"tool":"read_file","arguments":{"path":"..."}} or '
    '{"tool":"search_files","arguments":{"query":"..."}} or '
    '{"tool":"write_file","arguments":{"path":"...","content":"..."}}. '
    'Finish with {"final":"..."} exactly. '
    "Files exist only in a simulated workspace. You must inspect files before editing. "
    "Tool results arrive as user messages. Do not invent tool results."
)


def tasks():
    data = json.loads(TASK_FILE.read_text())
    ids = [t["id"] for t in data]
    if len(ids) != len(set(ids)) or not data:
        raise ValueError("agent tasks must have unique IDs")
    return data


def tool_call(action, files):
    """Execute one strictly typed call in the simulated workspace only."""
    if set(action) != {"tool", "arguments"} or not isinstance(action["arguments"], dict):
        raise ValueError("expected tool and arguments only")
    name, args = action["tool"], action["arguments"]
    if name == "read_file" and set(args) == {"path"} and isinstance(args["path"], str):
        if args["path"] not in files:
            raise ValueError("no such file")
        return {"content": files[args["path"]]}
    if name == "search_files" and set(args) == {"query"} and isinstance(args["query"], str):
        if not args["query"]:
            raise ValueError("search query must not be empty")
        return {"paths": sorted(p for p, v in files.items()
                                if args["query"].lower() in v.lower())}
    if name == "write_file" and set(args) == {"path", "content"} and all(
        isinstance(v, str) for v in args.values()
    ):
        if args["path"] not in files:
            raise ValueError("can only edit existing sandbox files")
        files[args["path"]] = args["content"]
        return {"ok": True}
    raise ValueError("unsupported tool or arguments")


def run_task(task, generate, max_steps=7):
    """Generate(messages) returns run_p3.run_single's result; replaceable in offline tests."""
    files = dict(task["files"])
    messages = [{"role": "user", "content": RULES + "\nTask: " + task["prompt"]}]
    steps, reads, searches, writes = [], set(), set(), set()
    answer = None
    for _ in range(max_steps):
        gen = generate(messages)
        response = gen["text"].strip()
        step = {"response": response, "raw_response": gen.get("raw_text", response),
                "wall_s": gen.get("wall_s"), "ttft_s": gen.get("ttft_s"),
                "prompt_tokens": gen.get("prompt_tokens"),
                "generation_tokens": gen.get("generation_tokens"),
                "peak_gb": gen.get("peak_gb")}
        steps.append(step)
        messages.append({"role": "assistant", "content": gen.get("raw_text", response)})
        try:
            action = json.loads(response)
            if not isinstance(action, dict):
                raise ValueError("expected a JSON object")
            if set(action) == {"final"} and isinstance(action["final"], str):
                answer = action["final"]
                break
            if (action.get("tool") == "write_file"
                    and isinstance(action.get("arguments"), dict)
                    and action["arguments"].get("path") not in reads):
                raise ValueError("read the file before editing it")
            output = tool_call(action, files)
            name, arguments = action["tool"], action["arguments"]
            if name == "read_file":
                reads.add(arguments["path"])
            elif name == "search_files":
                searches.add(arguments["query"])
            elif name == "write_file":
                writes.add(arguments["path"])
            step["tool"] = name
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            step["error"] = str(exc)
            output = {"error": str(exc)}
        messages.append({"role": "user", "content": "Tool response: " + json.dumps(output)
                         + "\nContinue using the JSON protocol."})

    expected = dict(task["files"])
    expected.update(task.get("expected_files", {}))
    ok = (answer == task["expected_final"]
          and files == expected
          and set(task.get("reads", [])) <= reads
          and set(task.get("writes", [])) <= writes
          and (not task.get("search") or task["search"] in searches))
    return {
        "id": task["id"], "category": task["category"], "passed": ok,
        "final": answer, "steps": steps, "tool_calls": sum("tool" in s for s in steps),
        "tool_errors": sum("error" in s for s in steps),
        "model_wall_s": round(sum(s.get("wall_s") or 0 for s in steps), 4),
        "time_to_first_token_s": steps[0].get("ttft_s") if steps else None,
        "peak_mlx_gb": max((s.get("peak_gb") or 0 for s in steps), default=0),
        "files_correct": files == expected,
        "system": sysinfo.snapshot(),
    }


def write_report(path, report):
    """Create the report exclusively: a repeated or concurrent run never replaces one."""
    try:
        with path.open("x") as fh:
            fh.write(json.dumps(report, indent=2) + "\n")
    except FileExistsError:
        print(f"refusing to overwrite {path}", file=sys.stderr)
        return False
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="model id in registry.py")
    ap.add_argument("--checkpoints", nargs="+", help="checkpoint tags; default downloaded")
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--evidence", default=str(ROOT / "results"),
                    help="location of existing verified checkpoint/P0 gate records")
    ap.add_argument("--condition", choices=("clean", "loaded", "unspecified"),
                    default="unspecified")
    args = ap.parse_args(argv)
    cfg = registry.get_model(args.model)
    tags = args.checkpoints or [c.tag for c in cfg.available_checkpoints]
    if not tags:
        ap.error("no downloaded checkpoints selected")
    if len(tags) != len(set(tags)):
        ap.error("duplicate checkpoint tags: " + ", ".join(tags))
    chosen = [cfg.checkpoint(tag) for tag in tags]
    if gates.refuse(cfg, chosen, args.evidence):
        return 1

    cases = tasks()
    digest = hashlib.sha256(TASK_FILE.read_bytes()).hexdigest()
    for ckpt in chosen:
        path = Path(args.out) / cfg.id / "p4" / (ckpt.tag + "-results.json")
        if path.exists():
            print(f"refusing to overwrite {path}", file=sys.stderr)
            return 1

    from mlx_lm import load
    import mlx.core as mx
    for ckpt in chosen:
        model, tokenizer = load(str(ckpt.local_path),
                                tokenizer_config={"trust_remote_code": False})
        token_cap = 256 * cfg.p3_budget_x
        def generate(messages):
            return run_p3.run_single(model, tokenizer, messages, cfg.template_kwargs, token_cap)
        records = [run_task(case, generate) for case in cases]
        path = Path(args.out) / cfg.id / "p4" / (ckpt.tag + "-results.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        report = {"model_id": cfg.id, "checkpoint": ckpt.tag, "hf_repo": ckpt.hf_repo,
                  "publisher": ckpt.publisher, "bits_per_weight": ckpt.bits_per_weight,
                  "weights_gb": ckpt.weights_gb, "fixture_sha256": digest,
                  "evidence_dir": str(args.evidence), "versions": bench.versions(),
                  "condition": args.condition, "template_kwargs": cfg.template_kwargs,
                  "temperature": 0, "per_turn_max_tokens": token_cap,
                  "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  "status": "pilot_unvalidated", "total": len(records),
                  "passed": sum(r["passed"] for r in records), "tasks": records}
        if not write_report(path, report):
            return 1
        print(f"{cfg.id}/{ckpt.tag}: {report['passed']}/{len(records)} pilot tasks -> {path}")
        del model, tokenizer
        mx.clear_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
