#!/usr/bin/env python3
"""Replicated, blocked MLX performance sessions (issue #34).

Only ``run`` launches inference; plan/validate/report are offline-compatible.
The independent observation is a newly spawned bench.py process per model/block.
Historical results are read-only inputs; all session evidence is write-once.
"""

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import gates
import registry
import sysinfo

ROOT = Path(__file__).resolve().parent
REGIMES = ("short", "medium", "long")
PROMPT_DIR = ROOT / "prompts" / "replicated"
N_BLOCKS = 5
N_TRIALS = 5
VERSION = 2
PACKAGES = ("mlx", "mlx-lm", "mlx-optiq", "transformers")
PINNED = {"mlx": "0.32.4", "mlx-lm": "0.32.2"}


def now():
    # A whole-second clock can make adjacent sessions appear to overlap.
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def canonical(doc):
    return json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def digest(doc):
    return hashlib.sha256(canonical(doc)).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_once(path, value):
    """Exclusive file creation, including when two jobs race."""
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def load_json(path):
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def packages():
    result = {}
    for name in PACKAGES:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def command_output(args):
    result = subprocess.run(args, capture_output=True, text=True, timeout=15, check=True)
    return result.stdout.strip()


def machine():
    try:
        ram_bytes = int(command_output(["sysctl", "-n", "hw.memsize"]))
    except (ValueError, OSError, subprocess.SubprocessError):
        ram_bytes = None
    try:
        chip = command_output(["sysctl", "-n", "machdep.cpu.brand_string"])
    except (OSError, subprocess.SubprocessError):
        chip = None
    return {"os": platform.system(), "macos": platform.mac_ver()[0],
            "arch": platform.machine(), "chip": chip,
            "cpu_count": os.cpu_count(), "ram_bytes": ram_bytes,
            "python": platform.python_version(), "packages": packages()}


def require_mac(m):
    if (m["os"] != "Darwin" or m["arch"] != "arm64"
            or m.get("chip") != "Apple M4" or m.get("cpu_count") != 10
            or m["ram_bytes"] is None or not 15 * 2**30 <= m["ram_bytes"] <= 17 * 2**30):
        raise ValueError("inference planning/runs require the 16 GiB Apple Silicon Mac")
    for pkg, version in PINNED.items():
        if m["packages"].get(pkg) != version:
            raise ValueError(f"{pkg} must be {version}, got {m['packages'].get(pkg)!r}")


def git_head():
    if command_output(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"]):
        raise ValueError("tracked repository edits present; commit/clean before planning or running")
    return command_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"])


def checkpoint(cfg):
    matches = [c for c in cfg.checkpoints.values() if c.tier == "4bit"]
    if len(matches) != 1:
        raise ValueError(f"{cfg.id}: expected exactly one plain 4-bit checkpoint")
    return matches[0]


def checkpoint_files(ckpt, verified_weights):
    """Freeze all local load/tokenizer/template files, not just Hub weight shards.

    The Hub's verified-weight manifest covers safetensors only. Local configuration,
    tokenizer and template files are still consumed by mlx_lm.load(). Hash their
    complete byte content and reject added/unexpected weight shards. Ignore only
    Hugging Face's local download metadata and macOS Finder metadata.
    """
    local = ckpt.local_path
    if not local.is_dir():
        raise ValueError(f"checkpoint directory absent: {local}")
    weights = set()
    settings = {}
    for path in sorted(local.rglob("*")):
        rel = path.relative_to(local)
        if ".cache" in rel.parts or "__pycache__" in rel.parts or path.name == ".DS_Store":
            continue
        if path.is_symlink():
            raise ValueError(f"symlink in checkpoint: {path}")
        if not path.is_file():
            continue
        key = rel.as_posix()
        if path.suffix == ".safetensors":
            weights.add(key)
        else:
            settings[key] = sha_file(path)
    if weights != set(verified_weights):
        raise ValueError(f"checkpoint weight file set drift: {local}")
    if "config.json" not in settings:
        raise ValueError(f"config.json missing from checkpoint: {local}")
    return settings


def verified_identity(cfg, evidence):
    ckpt = checkpoint(cfg)
    failures = gates.problems(cfg, ckpt, evidence)
    if failures:
        raise ValueError(f"{cfg.id}: " + "; ".join(failures))
    prov = load_json(Path(evidence) / cfg.id / "p0" / f"{ckpt.tag}-provenance.json")
    for entry in prov["files"]:
        if sha_file(ckpt.local_path / entry["file"]) != entry["expected_sha256"]:
            raise ValueError(f"{cfg.id}: on-disk SHA256 mismatch: {entry['file']}")
    weight_hashes = {f["file"]: f["expected_sha256"] for f in prov["files"]}
    return {"model_id": cfg.id, "checkpoint": ckpt.tag, "hf_repo": ckpt.hf_repo,
            "tier": ckpt.tier, "publisher": ckpt.publisher, "bpw": ckpt.bits_per_weight,
            "weights_gb": ckpt.weights_gb, "official": ckpt.official,
            "revision": prov["revision"], "fingerprint": gates.fingerprint(prov),
            "weight_sha256": weight_hashes,
            "checkpoint_file_sha256": checkpoint_files(ckpt, weight_hashes),
            "template_kwargs": cfg.template_kwargs}


def schedule(models, seed, n_blocks=N_BLOCKS):
    """Randomized Williams square; each model occupies each position once in four blocks."""
    if len(models) != 4 or n_blocks < 4 or len(set(models)) != 4:
        raise ValueError("balanced design requires four distinct models and >=4 blocks")
    slots = list(models)
    rng = random.Random(seed)
    rng.shuffle(slots)
    square = ((0, 1, 3, 2), (1, 2, 0, 3), (2, 3, 1, 0), (3, 0, 2, 1))
    # Complete squares remain balanced; the fifth block is an extra, randomized row.
    offset = rng.randrange(4)
    return [{"block": b + 1, "order": [slots[i] for i in square[(b + offset) % 4]]}
            for b in range(n_blocks)]


def make_plan(evidence, seed, m=None, source_commit=None):
    m = machine() if m is None else m
    require_mac(m)
    source_commit = git_head() if source_commit is None else source_commit
    models = [verified_identity(registry.MODELS[model_id], evidence)
              for model_id in registry.ORDER]
    prompts = []
    for regime in REGIMES:
        path = PROMPT_DIR / f"{regime}.txt"
        prompts.append({"regime": regime, "path": str(path.relative_to(ROOT)),
                        "sha256": sha_file(path), "bytes": path.stat().st_size})
    doc = {"schema": VERSION, "created_utc": now(), "source_commit": source_commit,
           "machine": m, "evidence_dir": str(Path(evidence).resolve()),
           "models": models, "prompts": prompts,
           "settings": {"transport": "inproc", "engine": "mlx_lm", "temperature": 0.0,
                        "nonce": False, "prompt_cache": "none (fresh process, no prompt_cache)",
                        "max_tokens": 256, "warmups_per_regime": 1,
                        "kept_trials_per_regime": N_TRIALS},
           "seed": seed, "schedule": schedule([m["model_id"] for m in models], seed)}
    doc["plan_id"] = digest(doc)
    return doc


def check_plan(plan):
    if plan.get("schema") != VERSION:
        raise ValueError("unsupported experiment schema")
    if plan.get("plan_id") != digest({k: v for k, v in plan.items() if k != "plan_id"}):
        raise ValueError("plan SHA256 mismatch")
    models = [m["model_id"] for m in plan["models"]]
    if len(models) != 4 or len(set(models)) != 4:
        raise ValueError("expected four distinct model identities")
    if plan["schedule"] != schedule(models, plan["seed"], len(plan["schedule"])):
        raise ValueError("corrupted or non-counterbalanced run order")
    if len(plan["schedule"]) < N_BLOCKS:
        raise ValueError("fewer than five independent session blocks")
    if [p["regime"] for p in plan["prompts"]] != list(REGIMES):
        raise ValueError("changed prompt regimes")
    for p in plan["prompts"]:
        expected = (PROMPT_DIR / f"{p['regime']}.txt").relative_to(ROOT).as_posix()
        if p.get("path") != expected:
            raise ValueError(f"wrong replicated prompt path for {p['regime']}")
    if plan["settings"] != {"transport": "inproc", "engine": "mlx_lm",
          "temperature": 0.0, "nonce": False,
          "prompt_cache": "none (fresh process, no prompt_cache)",
          "max_tokens": 256, "warmups_per_regime": 1,
          "kept_trials_per_regime": N_TRIALS}:
        raise ValueError("changed measurement settings")


def verify_host(plan):
    current = machine()
    require_mac(current)
    if current != plan["machine"] or git_head() != plan["source_commit"]:
        raise ValueError("machine/runtime/git revision changed since plan was frozen")
    for p in plan["prompts"]:
        if sha_file(ROOT / p["path"]) != p["sha256"]:
            raise ValueError(f"prompt drift: {p['path']}")
    for frozen in plan["models"]:
        actual = verified_identity(registry.MODELS[frozen["model_id"]], plan["evidence_dir"])
        if actual != frozen:
            raise ValueError(f"checkpoint/provenance drift: {frozen['model_id']}")
    current = sysinfo.snapshot()
    if current.get("power") != "ac":
        raise ValueError("AC power must be observed before each block")
    if current.get("low_power_mode") != 0:
        raise ValueError("low power mode must be observed off before each block")


def slot_path(folder, block, pos, model_id):
    return Path(folder) / "sessions" / f"b{block:02d}-p{pos:02d}-{model_id}"


def parse_vm_stat(output, total_bytes):
    """Est. system non-reclaimable RAM: total - free/inactive/speculative (not app RSS)."""
    import re
    size = re.search(r"page size of (\d+) bytes", output or "")
    if not size or total_bytes is None:
        return None
    fields = dict((k.lower(), int(v.replace(".", ""))) for k, v in
                  re.findall(r"^Pages (free|inactive|speculative):\s+([\d.]+)", output, re.M))
    if "free" not in fields or "inactive" not in fields:
        return None
    reclaimable = sum(fields.get(k, 0) for k in ("free", "inactive", "speculative"))
    used = total_bytes - reclaimable * int(size.group(1))
    return round(used / 1e9, 4) if 0 <= used <= total_bytes else None


class MemorySampler:
    """Read-only, low-frequency host-wide VM sampler run outside the timed process."""

    def __init__(self, ram_bytes, interval=0.5):
        self.ram_bytes = ram_bytes
        self.interval = interval
        self.stop_event = threading.Event()
        self.samples = []
        self.thread = None

    def _poll(self):
        while not self.stop_event.is_set():
            try:
                value = parse_vm_stat(command_output(["vm_stat"]), self.ram_bytes)
                if value is not None:
                    self.samples.append(value)
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
            self.stop_event.wait(self.interval)

    def start(self):
        self.thread = threading.Thread(target=self._poll, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=16)
        return {"source": "vm_stat: total - free - inactive - speculative; host-wide estimate",
                "unit": "GB decimal", "interval_s": self.interval, "samples": len(self.samples),
                "peak_gb_est": max(self.samples) if self.samples else None,
                "sampled_gb": self.samples}


def evidence_hashes(folder):
    result = {}
    for p in sorted(Path(folder).rglob("*")):
        if p.is_symlink():
            raise ValueError(f"symlink in evidence: {p}")
        if p.is_file() and p.name != "session.json":
            result[p.relative_to(folder).as_posix()] = sha_file(p)
    return result


def trial_rows(raw):
    """Read every available warmup/kept JSON (including failed partial sessions)."""
    return [load_json(p) for p in sorted(Path(raw).rglob("*.json"))
            if p.name != "summary.json"]


def observed_caveats(before, after, trials):
    """Derive warnings from preserved snapshots, never the operator's label."""
    snapshots = [before, after] + [r.get("system") for r in trials]
    warnings = {warning for snap in snapshots if isinstance(snap, dict)
                for warning in sysinfo.warnings(snap)}
    for field, label in (("power", "power source"), ("low_power_mode", "low power mode"),
                         ("cpu_speed_limit_pct", "CPU thermal speed limit"),
                         ("swap_used_gb", "swap usage"), ("mem_free_pct", "memory pressure"),
                         ("top_cpu", "CPU interference")):
        if any(not isinstance(snap, dict) or snap.get(field) is None for snap in snapshots):
            warnings.add(f"{label} unavailable in one or more snapshots")
    return sorted(warnings)


@contextmanager
def experiment_lock(folder):
    """Non-blocking OS lock for the entire experiment; automatically freed on crash.

    The lock file persists, but locking is on an open file descriptor, not the
    existence of a stale sentinel. Two different blocks cannot run together.
    """
    with (Path(folder) / ".experiment.lock").open("a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another benchmark block is running in this experiment") from exc
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def run_block(folder, block, condition, resume=False, timeout_s=3600):
    """Reserve the whole experiment for the full lifetime of this block."""
    with experiment_lock(folder):
        return _run_block_locked(folder, block, condition, resume, timeout_s)


def _run_block_locked(folder, block, condition, resume, timeout_s):
    folder = Path(folder)
    plan = load_json(folder / "plan.json")
    check_plan(plan)
    if not 1 <= block <= len(plan["schedule"]):
        raise ValueError("invalid block")
    # Never launch after incomplete, reordered, overlapping or duplicate evidence.
    # validate() also detects unexpected slots and checks global chronology.
    _, _, existing_errors = validate(folder)
    missing = []
    for item in plan["schedule"]:
        for pos, model_id in enumerate(item["order"], 1):
            path = slot_path(folder, item["block"], pos, model_id)
            if not path.is_dir():
                missing.append((item["block"], pos, path.name))
    expected_missing = {f"{name}: missing independent session" for _, _, name in missing}
    other_errors = [e for e in existing_errors if e not in expected_missing]
    if other_errors:
        raise ValueError("existing session evidence is invalid: " + "; ".join(other_errors))
    if any(b < block for b, _, _ in missing):
        raise ValueError("prior session is missing; blocks must run in order")
    existing = len(plan["schedule"]) * 4 - len(missing)
    if missing and existing != (missing[0][0] - 1) * 4 + missing[0][1] - 1:
        raise ValueError("existing sessions are not a contiguous schedule prefix")
    verify_host(plan)  # before any new directory or inference
    order = plan["schedule"][block - 1]["order"]
    root = folder / "sessions"
    root.mkdir(exist_ok=True)
    for pos, model_id in enumerate(order, 1):
        path = slot_path(folder, block, pos, model_id)
        if path.exists():
            if resume and validate_session(path, plan, block, pos, model_id) == []:
                print(f"already validated, skipping {path}")
                continue
            raise FileExistsError(f"refusing existing session evidence: {path}")
        path.mkdir(exist_ok=False)
        session_id = uuid.uuid4().hex
        start = now()
        pre = sysinfo.snapshot()
        write_once(path / "started.json", {"schema": VERSION, "session_id": session_id, "plan_id": plan["plan_id"],
                  "block": block, "position": pos, "model_id": model_id,
                  "operator_condition": condition, "started_utc": start, "pre_system": pre})
        raw = path / "raw"
        raw.mkdir()
        command = [sys.executable, str(ROOT / "bench.py"), "--model", model_id,
                   "--checkpoint", next(m["checkpoint"] for m in plan["models"]
                                           if m["model_id"] == model_id),
                   "--transport", "inproc", "--session-id", session_id,
                   "--trials", str(N_TRIALS),
                   "--max-tokens", "256", "--no-nonce", "--condition", condition,
                   "--prompts", *REGIMES, "--prompt-dir", str(PROMPT_DIR), "--out", str(raw),
                   "--evidence", plan["evidence_dir"]]
        sampler = MemorySampler(plan["machine"]["ram_bytes"])
        rc = None
        failure = None
        sampler.start()
        try:
            with (path / "bench.log").open("x", encoding="utf-8") as stream:
                try:
                    result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                            timeout=timeout_s, check=False)
                    rc = result.returncode
                except subprocess.TimeoutExpired:
                    rc, failure = 124, f"timeout after {timeout_s}s"
                except OSError as exc:
                    rc, failure = 127, str(exc)
        finally:
            host_memory = sampler.stop()
            post = sysinfo.snapshot()
            try:
                observed_trials = trial_rows(raw)
            except (OSError, ValueError) as exc:
                failure = failure or f"unreadable trial observations: {exc}"
                rc, observed_trials = 1, []
            caveats = observed_caveats(pre, post, observed_trials)
            details = {"schema": VERSION, "session_id": session_id,
                       "plan_id": plan["plan_id"], "block": block, "position": pos,
                       "model_id": model_id, "started_utc": start, "ended_utc": now(),
                       "operator_condition": condition,
                       "observed": {"before": pre, "after": post, "caveats": caveats},
                       "host_memory": host_memory, "return_code": rc, "error": failure,
                       "status": "completed" if rc == 0 else "failed"}
            # The final manifest contains checksums, so cannot hash itself. Store
            # its COMPLETE non-checksum payload as a separate hashed artifact.
            write_once(path / "session_payload.json", details)
            manifest = {**details, "artifacts_sha256": evidence_hashes(path)}
            write_once(path / "session.json", manifest)
        print(f"{model_id} block {block} position {pos}: rc={rc}; {path}")
        if rc != 0:
            raise RuntimeError(f"session failed and is preserved: {path}")
        issues = validate_session(path, plan, block, pos, model_id)
        if issues:
            raise RuntimeError(f"session evidence is invalid and preserved: {path}: {'; '.join(issues)}")


def close(a, b):
    return (isinstance(a, (int, float)) and not isinstance(a, bool)
            and math.isfinite(a) and abs(a - b) < 0.00051)


def utc_time(value):
    """Require an offset-aware UTC interval for isolation/ordering checks."""
    stamp = datetime.fromisoformat(value)
    if stamp.tzinfo is None or stamp.utcoffset() != timedelta(0):
        raise ValueError("non-UTC or naive session timestamp")
    return stamp


def validate_session(folder, plan, block, pos, model_id):
    """Strict offline verification of identity, hashes, trials, completeness and metrics."""
    folder = Path(folder)
    errors = []
    if not (folder / "session.json").is_file():
        return ["missing session.json"]
    try:
        s = load_json(folder / "session.json")
        payload = load_json(folder / "session_payload.json")
        started = load_json(folder / "started.json")
        if {k: v for k, v in s.items() if k != "artifacts_sha256"} != payload:
            errors.append("manifest changed independently of hashed session payload")
        for k, value in {"schema": VERSION, "plan_id": plan["plan_id"], "block": block,
                         "position": pos, "model_id": model_id}.items():
            if s.get(k) != value or started.get(k) != value:
                errors.append(f"{k} identity mismatch")
        if s.get("session_id") != started.get("session_id") or not s.get("session_id"):
            errors.append("session UUID mismatch")
        if s.get("operator_condition") not in ("clean", "loaded") or s.get("operator_condition") != started.get("operator_condition"):
            errors.append("operator condition provenance mismatch")
        if s.get("status") != "completed" or s.get("return_code") != 0:
            errors.append("session process incomplete or failed")
        if s.get("error") is not None:
            errors.append("session records a process error")
        if s.get("started_utc") != started.get("started_utc"):
            errors.append("session start timestamp mismatch")
        if utc_time(s["started_utc"]) >= utc_time(s["ended_utc"]):
            errors.append("session end does not follow start")
        expected = s["artifacts_sha256"]
        if expected != evidence_hashes(folder):
            errors.append("evidence file list or SHA256 mismatch")
        if not s.get("observed") or s["observed"].get("before") != started.get("pre_system"):
            errors.append("system snapshots mismatch")
        mem = s.get("host_memory") or {}
        samples = mem.get("sampled_gb")
        if (mem.get("source") != "vm_stat: total - free - inactive - speculative; host-wide estimate"
                or mem.get("interval_s") != 0.5
                or not isinstance(samples, list) or not samples
                or not all(isinstance(x, (int, float)) and not isinstance(x, bool)
                           and math.isfinite(x) and 0 <= x <= 1e3 for x in samples)
                or mem.get("samples") != len(samples)
                or mem.get("unit") != "GB decimal"
                or not close(mem.get("peak_gb_est"), max(samples))):
            errors.append("host-wide memory estimate missing")
        model = next(m for m in plan["models"] if m["model_id"] == model_id)
        raw = folder / "raw" / model_id / f"{model['checkpoint']}-inproc"
        summary = load_json(raw / "summary.json")
        expected_summary = {"model_id": model_id, "checkpoint": model["checkpoint"],
                            "tier": "4bit", "hf_repo": model["hf_repo"],
                            "publisher": model["publisher"], "transport": "inproc",
                            "engine": "mlx_lm", "trials": N_TRIALS, "max_tokens": 256,
                            "greedy": True, "template_kwargs": model["template_kwargs"],
                            "condition": s["operator_condition"],
                            "outcome": "completed", "publishable": True,
                            "session_id": s["session_id"]}
        for k, value in expected_summary.items():
            if summary.get(k) != value:
                errors.append(f"summary {k} mismatch")
        if set(summary.get("prompts", {})) != set(REGIMES):
            errors.append("regime completeness mismatch")
        if summary.get("versions") != plan["machine"]["packages"]:
            errors.append("runtime version provenance mismatch")
        all_rows = []
        kept_rows = []
        for regime in REGIMES:
            prompt = next(p for p in plan["prompts"] if p["regime"] == regime)
            if not isinstance(prompt["sha256"], str):
                errors.append(f"{regime}: missing prompt hash")
            rows = []
            for i in range(N_TRIALS + 1):
                trial_file = raw / f"{regime}-{'warmup' if i == 0 else f'trial{i}'}.json"
                row = load_json(trial_file)
                rows.append(row)
                all_rows.append(row)
                if i:
                    kept_rows.append(row)
                for k, value in {"model_id": model_id, "checkpoint": model["checkpoint"],
                                 "prompt": regime, "trial": i, "discarded": i == 0,
                                 "condition": s["operator_condition"], "ok": True,
                                 "nonce": False, "max_tokens": 256,
                                 "template_kwargs": model["template_kwargs"],
                                 "hf_repo": model["hf_repo"],
                                 "publisher": model["publisher"],
                                 "session_id": s["session_id"]}.items():
                    if row.get(k) != value:
                        errors.append(f"{regime} trial{i}: {k} mismatch")
                if not isinstance(row.get("system"), dict) or not row.get("started_at"):
                    errors.append(f"{regime} trial{i}: system/time provenance missing")
                recorded_path = Path(row.get("prompt_file", ""))
                expected_parts = Path(prompt["path"]).parts
                if (not recorded_path.is_absolute()
                        or recorded_path.parts[-len(expected_parts):] != expected_parts):
                    errors.append(f"{regime} trial{i}: prompt file identity mismatch")
                if row.get("versions") != plan["machine"]["packages"]:
                    errors.append(f"{regime} trial{i}: runtime version mismatch")
                if row.get("finish_reason") != "stop":
                    errors.append(f"{regime} trial{i}: truncated or unknown stop reason")
                text = row.get("text")
                if (not isinstance(text, str) or not text.strip()
                        or ("</think>" in text and not text.rsplit("</think>", 1)[1].strip())
                        or (text.lstrip().startswith("<think>") and "</think>" not in text)
                        or row.get("reasoning_only")):
                    errors.append(f"{regime} trial{i}: empty or reasoning-only response")
                for metric in ("wall_s", "request_ttft_s", "peak_gb", "prompt_tokens", "generation_tokens"):
                    value = row.get(metric)
                    if (not isinstance(value, (int, float)) or isinstance(value, bool)
                            or not math.isfinite(value) or value <= 0):
                        errors.append(f"{regime} trial{i}: missing/invalid {metric}")
                for metric in ("prompt_tokens", "generation_tokens"):
                    if type(row.get(metric)) is not int:
                        errors.append(f"{regime} trial{i}: {metric} must be an integer")
                if type(row.get("generation_tokens")) is int and row["generation_tokens"] > 256:
                    errors.append(f"{regime} trial{i}: generation exceeds token cap")
                wall, ttft = row.get("wall_s"), row.get("request_ttft_s")
                # wall_s is rounded to milliseconds; request_ttft_s to 0.1 ms.
                if (type(wall) in (int, float) and type(ttft) in (int, float)
                        and math.isfinite(wall) and math.isfinite(ttft) and ttft > wall + 0.00055):
                    errors.append(f"{regime} trial{i}: TTFT exceeds response-end wall time")
            if len({row.get("prompt_tokens") for row in rows}) != 1:
                errors.append(f"{regime}: inconsistent tokenizer prompt counts")
            declared = summary["prompts"].get(regime, {})
            if declared.get("prompt_tokens") != rows[1].get("prompt_tokens"):
                errors.append(f"{regime}: summary prompt token mismatch")
            if declared.get("kept_ok") != N_TRIALS or declared.get("warmup_ok") is not True:
                errors.append(f"{regime}: warmup/kept count mismatch")
            for metric in ("wall_s", "request_ttft_s", "peak_gb", "generation_tokens"):
                reported = declared.get(metric) or {}
                values = [r[metric] for r in rows[1:] if isinstance(r.get(metric), (int, float))
                          and not isinstance(r[metric], bool) and math.isfinite(r[metric])]
                if len(values) != N_TRIALS:
                    errors.append(f"{regime}: {metric} incomplete")
                elif (not close(reported.get("median"), statistics.median(values))
                      or not close(reported.get("min"), min(values))
                      or not close(reported.get("max"), max(values))
                      or reported.get("n") != N_TRIALS):
                    errors.append(f"{regime}: {metric} median/range mismatch")
        expected_warnings = observed_caveats(started["pre_system"], s["observed"]["after"], all_rows)
        if s["observed"].get("caveats") != expected_warnings:
            errors.append("observed warning derivation mismatch")
        kept_warnings = sorted({warning for row in kept_rows
                                for warning in sysinfo.warnings(row.get("system") or {})})
        if summary.get("condition_caveats") != kept_warnings:
            errors.append("summary warning derivation mismatch")
        actual_trial_files = {p.name for p in raw.glob("*.json") if p.name != "summary.json"}
        expected_trial_files = {f"{r}-{'warmup' if i == 0 else f'trial{i}'}.json"
                                for r in REGIMES for i in range(N_TRIALS + 1)}
        if actual_trial_files != expected_trial_files:
            errors.append("missing or unexpected trial files")
    except (OSError, ValueError, KeyError, TypeError, StopIteration, ZeroDivisionError) as exc:
        errors.append(f"invalid or absent evidence: {exc}")
    return errors


def validate(folder):
    folder = Path(folder)
    plan = load_json(folder / "plan.json")
    check_plan(plan)
    good = {}
    errors = []
    sessions = folder / "sessions"
    expected_paths = set()
    intervals = []
    seen_ids = set()
    for block in plan["schedule"]:
        for pos, model_id in enumerate(block["order"], 1):
            path = slot_path(folder, block["block"], pos, model_id)
            expected_paths.add(path.name)
            if not path.is_dir():
                errors.append(f"{path.name}: missing independent session")
                continue
            issues = validate_session(path, plan, block["block"], pos, model_id)
            if issues:
                errors.extend(f"{path.name}: {i}" for i in issues)
            else:
                session = load_json(path / "session.json")
                sid = session["session_id"]
                if sid in seen_ids:
                    errors.append(f"{path.name}: reused session UUID")
                seen_ids.add(sid)
                intervals.append((utc_time(session["started_utc"]), utc_time(session["ended_utc"]),
                                  block["block"], pos, path.name))
                good[(block["block"], model_id)] = session
    # Detect independently plausible but physically impossible concurrent sessions.
    # Also reject an operator accidentally executing the frozen counterbalance out of order.
    if intervals:
        for prev, current in zip(sorted(intervals), sorted(intervals)[1:]):
            if current[0] < prev[1]:
                errors.append(f"overlapping independent sessions: {prev[4]} / {current[4]}")
        executed = [(b, p) for _, _, b, p, _ in sorted(intervals)]
        if executed != sorted(executed):
            errors.append("sessions executed out of the frozen block/position order")
    if sessions.exists():
        extras = {p.name for p in sessions.iterdir()} - expected_paths
        if extras:
            errors.append(f"unplanned session entries: {sorted(extras)}")
    return plan, good, errors


def percentile(data, pct):
    seq = sorted(data)
    x = (len(seq) - 1) * pct
    lo = int(x)
    return seq[lo] + (seq[min(lo + 1, len(seq) - 1)] - seq[lo]) * (x - lo)


def bootstrap_median(values, seed, repetitions=3000):
    """Resample whole independent sessions (or whole matched blocks for deltas)."""
    if len(values) < N_BLOCKS:
        raise ValueError("fewer than five independent sessions")
    rng = random.Random(seed)
    draw = [statistics.median(rng.choices(values, k=len(values))) for _ in range(repetitions)]
    return statistics.median(values), percentile(draw, 0.025), percentile(draw, 0.975)


def report(folder):
    plan, good, errors = validate(folder)
    if errors:
        raise ValueError("cannot report incomplete/invalid experiment:\n" + "\n".join(errors))
    rows = []
    medians = {}
    kept_trials = {}
    compromised = []
    for b in plan["schedule"]:
        for model_id in b["order"]:
            s = good[(b["block"], model_id)]
            if s["observed"]["caveats"]:
                compromised.append((b["block"], model_id, s["observed"]["caveats"]))
            model = next(m for m in plan["models"] if m["model_id"] == model_id)
            raw = slot_path(folder, b["block"], b["order"].index(model_id) + 1, model_id) / "raw" / model_id / f"{model['checkpoint']}-inproc"
            summary = load_json(raw / "summary.json")
            for regime in REGIMES:
                for metric in ("wall_s", "request_ttft_s", "peak_gb"):
                    medians[(b["block"], model_id, regime, metric)] = summary["prompts"][regime][metric]["median"]
                kept_trials[(b["block"], model_id, regime)] = [
                    load_json(raw / f"{regime}-trial{i}.json") for i in range(1, N_TRIALS + 1)]
            medians[(b["block"], model_id, "all", "system_gb_est")] = s["host_memory"]["peak_gb_est"]
    heading = ["# Replicated plain-4-bit session study", "",
               f"Frozen source: `{plan['source_commit']}` · Plan SHA256: `{plan['plan_id']}`",
               f"Machine: {plan['machine']['arch']} macOS {plan['machine']['macos']}, 16 GiB unified; "
               f"mlx {plan['machine']['packages']['mlx']}, mlx-lm {plan['machine']['packages']['mlx-lm']}",
               f"Design: {len(plan['schedule'])} counterbalanced blocks × 4 models, "
               "one fresh MLX process per model/block; one discarded warmup + five kept trials "
               "per prompt length. n is independent **sessions**, not requests.", "",
               "All intervals are exploratory percentile 95% **session/block bootstrap** intervals "
               "(3,000 resamples); n=5 gives unstable coverage and little power. The fixed "
               "prompts and non-random hardware conditions do not justify population/general-purpose claims.", "",
               "In the main table each timing/memory cell shows the **median of five session medians "
               "[95% session-bootstrap interval]**, then the **observed kept-trial min–max across "
               "all 25 requests**. The latter is descriptive jitter, not an uncertainty interval. "
               "The session appendix separately reports each five-trial median and range.", "",
               "| Model / checkpoint / publisher | Regime | Prompt tokens (model tokenizer) | "
               "Completion tokens median [min–max] | Response-end wall s: sessions [95%]; trials min–max | "
               "Request TTFT s: sessions [95%]; trials min–max | "
               "MLX allocator GB: sessions [95%]; trials min–max | n sessions × kept trials |",
               "|---|---|---:|---:|---:|---:|---:|---:|"]
    def estimate(values, salt):
        v = bootstrap_median(values, plan["seed"] + salt)
        return f"{v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]"
    def five_trial_range(values, digits=3):
        return (f"{statistics.median(values):.{digits}f} "
                f"[{min(values):.{digits}f}–{max(values):.{digits}f}]")

    def context(model_id, regime):
        trials = [trial for b in plan["schedule"]
                  for trial in kept_trials[(b["block"], model_id, regime)]]
        prompts = {t["prompt_tokens"] for t in trials}
        if len(prompts) != 1:
            raise ValueError(f"prompt-token counts drift across sessions: {model_id}/{regime}")
        return next(iter(prompts)), trials

    models = [m["model_id"] for m in plan["models"]]
    for mi, model_id in enumerate(models):
        model = plan["models"][mi]
        for ri, regime in enumerate(REGIMES):
            prompt_tokens, trials = context(model_id, regime)
            gen_tokens = [t["generation_tokens"] for t in trials]
            columns = []
            for idx, metric in enumerate(("wall_s", "request_ttft_s", "peak_gb")):
                values = [medians[(b["block"], model_id, regime, metric)] for b in plan["schedule"]]
                raw_values = [t[metric] for t in trials]
                columns.append(f"{estimate(values, mi * 100 + ri * 10 + idx)}; "
                               f"{min(raw_values):.3f}–{max(raw_values):.3f}")
            rows.append(f"| {model_id} / {model['checkpoint']} / {model['publisher']} "
                        f"({model['bpw']:.2f} bpw, {model['weights_gb']:.2f} GB weights) "
                        f"| {regime} | {prompt_tokens} | {five_trial_range(gen_tokens, 0)} "
                        f"| {' | '.join(columns)} | {len(plan['schedule'])} × {N_TRIALS} |")
    heading += rows + ["", "## Host-wide memory (per session, all regimes combined)", "",
                        "`vm_stat` estimate: total physical RAM − free − inactive − speculative pages, "
                        "polled every 0.5 s. Not process RSS, not a precise app peak; "
                        "the baseline includes other applications. GB decimal.", "",
                        "| Model | Host-wide used peak GB: median [95%] | n sessions |",
                        "|---|---:|---:|"]
    for mi, model_id in enumerate(models):
        values = [medians[(b["block"], model_id, "all", "system_gb_est")] for b in plan["schedule"]]
        heading.append(f"| {model_id} | {estimate(values, mi + 400)} | {len(values)} |")
    heading += ["", "## Block-matched comparisons (paired response-end wall time)", "",
                f"Reference: `{models[0]}`; Δ = other − reference, seconds. Positive is slower. "
                "Resample *blocks* rather than trials. These are conditional exploratory contrasts.", "",
                "| Candidate − reference | Regime | Ref prompt / completion tokens | "
                "Candidate prompt / completion tokens | Median paired Δ s [95% block bootstrap] | n blocks |",
                "|---|---|---:|---:|---:|---:|"]
    def token_context(mid, regime):
        pt, trials = context(mid, regime)
        return f"{pt} / {five_trial_range([t['generation_tokens'] for t in trials], 0)}"
    for mi, model_id in enumerate(models[1:], 1):
        for ri, regime in enumerate(REGIMES):
            deltas = [medians[(b["block"], model_id, regime, "wall_s")] -
                      medians[(b["block"], models[0], regime, "wall_s")]
                      for b in plan["schedule"]]
            heading.append(f"| {model_id} − {models[0]} | {regime} | "
                           f"{token_context(models[0], regime)} | {token_context(model_id, regime)} | "
                           f"{estimate(deltas, mi * 100 + ri + 900)} | {len(deltas)} |")
    heading += ["", "## Each independent session: five kept trials and their ranges", "",
                "One row is **one** initialized model process and prompt regime. All values "
                "are measured, not bootstrapped; the displayed [min–max] is over exactly "
                "five kept trials. Prompt tokens are tokenizer-specific.", "",
                "| Block / slot | Model / checkpoint / publisher | Regime | Prompt tokens | "
                "Completion tokens median [range] | Response-end wall s median [range] | "
                "Request TTFT s median [range] | MLX allocator GB median [range] | n kept |",
                "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for b in plan["schedule"]:
        for pos, model_id in enumerate(b["order"], 1):
            model = next(m for m in plan["models"] if m["model_id"] == model_id)
            for regime in REGIMES:
                trials = kept_trials[(b["block"], model_id, regime)]
                pt = {t["prompt_tokens"] for t in trials}
                if len(pt) != 1:
                    raise ValueError(f"prompt-token counts vary within session: {model_id}/{regime}")
                heading.append(f"| {b['block']} / {pos} | {model_id} / {model['checkpoint']} / "
                               f"{model['publisher']} | {regime} | {next(iter(pt))} | "
                               f"{five_trial_range([t['generation_tokens'] for t in trials], 0)} | "
                               f"{five_trial_range([t['wall_s'] for t in trials])} | "
                               f"{five_trial_range([t['request_ttft_s'] for t in trials])} | "
                               f"{five_trial_range([t['peak_gb'] for t in trials])} | {len(trials)} |")
    heading += ["", "## Condition provenance and limitations", "",
                f"Observed caveats in {len(compromised)}/{len(good)} sessions; operator-selected "
                "condition labels are **not** observations and have not been silently relabelled.", ""]
    for b, m, caveats in compromised:
        heading.append(f"- Block {b}, `{m}`: {'; '.join(caveats)}")
    heading += ["", "### Per-session intended vs observed conditions", "",
                "| Block | Model | Operator-selected condition | Observed warnings | Samples |",
                "|---:|---|---|---|---:|"]
    for b in plan["schedule"]:
        for model_id in b["order"]:
            s = good[(b["block"], model_id)]
            notes = "; ".join(s["observed"]["caveats"]) or "none reported"
            heading.append(f"| {b['block']} | {model_id} | {s['operator_condition']} "
                           f"| {notes} | {s['host_memory']['samples']} |")
    heading += ["", "This experiment measures **response completion time**, not independently "
                "judged time to a correct/usable answer. `first_content_s` is unavailable for "
                "the in-process harness; TTFT is time to first generated token. MLX allocator "
                "peak excludes some system allocations and must not be confused with host RAM. "
                "All response lengths and stop reasons remain in immutable trial records. "
                "Conditions, model publisher, and tokenizer-specific prompt lengths must be "
                "considered before deployment. Legacy `results/STUDY.md` remains historical. "
                "No universal winner or agent-ability claim follows from this run.", ""]
    return "\n".join(heading)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    create = sub.add_parser("plan", help="freeze checkpoints/prompts, verify Mac and provenance")
    create.add_argument("--out", type=Path, required=True, help="new, non-existent experiment directory")
    create.add_argument("--evidence", type=Path, default=ROOT / "results")
    create.add_argument("--seed", type=int, required=True, help="predeclared counterbalancing seed")
    run = sub.add_parser("run", help="run one ordered block on the M4; never overwrite")
    run.add_argument("experiment", type=Path)
    run.add_argument("--block", type=int, required=True)
    run.add_argument("--condition", choices=("clean", "loaded"), required=True,
                     help="operator-selected intention, never inferred from observed snapshots")
    run.add_argument("--resume", action="store_true", help="skip already-validated slots only")
    run.add_argument("--timeout-s", type=int, default=3600)
    val = sub.add_parser("validate", help="offline check all planned session evidence")
    val.add_argument("experiment", type=Path)
    rep = sub.add_parser("report", help="offline report; fails on incomplete evidence")
    rep.add_argument("experiment", type=Path)
    rep.add_argument("--out", type=Path, help="new report path (refuses overwrite)")
    args = p.parse_args(argv)
    try:
        if args.action == "plan":
            doc = make_plan(args.evidence, args.seed)
            args.out.mkdir(parents=True, exist_ok=False)
            write_once(args.out / "plan.json", doc)
            print(f"Plan {doc['plan_id']} saved in {args.out}; run five blocks on the M4")
        elif args.action == "run":
            if args.timeout_s < 1:
                raise ValueError("timeout must be positive")
            run_block(args.experiment, args.block, args.condition, args.resume, args.timeout_s)
        elif args.action == "validate":
            doc, good, errors = validate(args.experiment)
            print(f"{len(good)}/{len(doc['schedule']) * 4} independent sessions validated")
            for err in errors:
                print("INVALID: " + err)
            return 1 if errors else 0
        else:
            output = report(args.experiment)
            if args.out:
                with args.out.open("x", encoding="utf-8") as f:
                    f.write(output)
                print(args.out)
            else:
                print(output)
        return 0
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
