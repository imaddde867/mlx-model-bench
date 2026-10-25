"""Measurement preconditions: verified provenance and a passing P0 row, per checkpoint.

Evidence lives in results/<model>/p0/ as <tag>-provenance.json (written by
`run_bench.py --phase fetch`) and <tag>.json (written by `--phase p0`). Every
entry point that measures calls `problems()` and refuses on a non-empty list.
"""

import hashlib
import json
from pathlib import Path


def fingerprint(record):
    """Identity of the verified weights; P0 stamps it so a later re-fetch makes old P0 rows stale."""
    pairs = sorted((f["file"], f.get("expected_sha256") or "") for f in record["files"])
    return hashlib.sha256(json.dumps(pairs).encode()).hexdigest()


def _load(path):
    try:
        return json.loads(path.read_text()), None
    except FileNotFoundError:
        return None, f"missing {path.name}"
    except (OSError, ValueError) as e:
        return None, f"unreadable {path.name}: {e}"


def problems(cfg, ckpt, out, need_p0=True):
    p0_dir = Path(out) / cfg.id / "p0"
    prov, err = _load(p0_dir / f"{ckpt.tag}-provenance.json")
    if err:
        return [f"{err} — run --phase fetch"]

    bad = []
    if prov.get("hf_repo") != ckpt.hf_repo:
        bad.append("provenance is for a different repo")
    if not prov.get("revision"):
        bad.append("provenance records no Hub revision — re-run --phase fetch")
    if prov.get("verified") is not True:
        bad.append("weights not verified against the Hub")
    files = prov.get("files") or []
    if not files:
        bad.append("provenance lists no weight files")
    for f in files:
        name = f.get("file") or ""
        if not f.get("expected_sha256") or f.get("local_sha256") != f["expected_sha256"]:
            bad.append(f"{name}: no matching sha256 (a size match alone is not verification)")
        local = ckpt.local_path / name
        if not local.is_file() or local.stat().st_size != f.get("local_bytes"):
            bad.append(f"{name}: on-disk file differs from the verified record")
    if bad or not need_p0:
        return bad

    row, err = _load(p0_dir / f"{ckpt.tag}.json")
    if err:
        return [f"{err} — run --phase p0"]
    if row.get("hf_repo") != ckpt.hf_repo:
        bad.append("P0 row is for a different repo")
    if row.get("weights_fingerprint") != fingerprint(prov):
        bad.append("P0 row predates the current provenance — re-run --phase p0")
    if row.get("template_kwargs") != cfg.template_kwargs:
        bad.append("P0 ran with different chat-template controls")
    if row.get("ok") is not True:
        bad.append(f"P0 failed: {row.get('error', 'no error recorded')}")
    if row.get("sanity") is not True:
        bad.append("P0 sanity answer missing or wrong")
    if row.get("deterministic") is not True:
        bad.append("P0 greedy output not shown reproducible")
    return bad


def refuse(cfg, selected, out, need_p0=True):
    """Print every problem for every selected checkpoint; True if any blocks measurement."""
    blocked = False
    for ckpt in selected:
        for p in problems(cfg, ckpt, out, need_p0):
            print(f"  gate {cfg.id}/{ckpt.tag}: {p}")
            blocked = True
    return blocked
