#!/usr/bin/env python3
"""Pre-run go/no-go: can the model matrix safely start? Offline, no model is loaded.

  .venv-mlxlm/bin/python readiness.py                          # whole roster, every tier
  .venv-mlxlm/bin/python readiness.py --models minicpm5-2b --tiers 4bit --out results

Runs the CI unit suite, then checks every precondition of the planned run against
`--out` (read-only): registry entries and downloaded checkpoints, verified
provenance and a current P0 row (gates.problems), trial count, one set of runtime
versions across a model's rows, prompt files, blind-round inputs, and that no
planned output path already exists. Exit 0 and "READY" only if nothing blocks;
exit 1 and "NOT READY" with every reason otherwise. Finishes with the
reproducibility checklist.

It starts no benchmark and does no inference or network I/O. READY means the
preconditions hold; it is not the independently reviewed clean run, and it
measures nothing.
"""

import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import bench
import blind
import gates
import registry
import run_bench
import run_p3
import score_quality
from p3_suite import INSTR_PROMPTS, MATH_PROMPTS, QUALITY_PROMPTS, TOOL_PROMPTS

ROOT = Path(__file__).resolve().parent
NESTED = "MLX_BENCH_READINESS_SUITE"  # set for the unit-suite subprocess so tests cannot recurse


def unit_suite():
    """(ok, output tail). Inside the suite it refuses to recurse and reports not-ok, never a pass."""
    if os.environ.get(NESTED):
        return False, "SKIPPED (recursion guard) - not a pass; unset " + NESTED
    r = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", str(ROOT / "tests")],
                       cwd=ROOT, env={**os.environ, NESTED: "1"}, capture_output=True, text=True)
    return r.returncode == 0, "\n".join((r.stdout + r.stderr).strip().splitlines()[-8:])


def environment():
    return {"host": bench.host_string(), "versions": bench.versions(), "optiq": run_bench.optiq_version()}


def rel(path):
    path = Path(path)
    return path.relative_to(ROOT) if path.is_relative_to(ROOT) else path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dir_digest(path):
    files = sorted(Path(path).glob("*.txt"))
    h = hashlib.sha256("".join(f"{f.name}:{sha256(f)}\n" for f in files).encode()).hexdigest()
    return len(files), h


def planned_paths(cfg, ckpt, out):
    """Everything run_bench.py would write for this checkpoint, apart from logs and gate evidence."""
    base = Path(out) / cfg.id
    paths = [base / f"{ckpt.tag}-{arm}" for arm in ("inproc", "http", "http-cached")]
    if "mlx-optiq" in ckpt.engines:
        paths.append(base / f"{ckpt.tag}-http-optiq-engine")
    return paths + [base / "p3" / f"{ckpt.tag}-results.json"]


def version_problems(cfg, out, env):
    """mlx, mlx-lm and host must be one value across this interpreter and all of the model's rows."""
    rows = {"this interpreter": (env["host"], env["versions"])}
    for p in sorted((Path(out) / cfg.id).glob("*/summary.json")):
        s, _ = gates._load(p)
        if s:
            rows[p.parent.name] = (s.get("host"), s.get("versions") or {})
    bad = []
    # None = the row legitimately lacks that package (e.g. mlx-optiq lives in another venv).
    for key, get in [("host", lambda r: r[0])] + [(k, lambda r, k=k: r[1].get(k)) for k in bench.versions()]:
        seen = {}
        for name, r in rows.items():
            if get(r) is not None:
                seen.setdefault(get(r), []).append(name)
        if len(seen) > 1:
            bad.append(f"{cfg.id}: {key} differs across rows: "
                       + "; ".join(f"{v!r} in {', '.join(n)}" for v, n in seen.items()))
    return bad


def evaluate(model_ids, out, tiers, trials, max_tokens, prompts, round_name, env, run_suite=unit_suite):
    """-> (blocking reasons, checklist lines). Reads `out`; writes nothing."""
    out = Path(out)
    bad, lines = [], []

    def flag(area, msg):
        bad.append(f"[{area}] {msg}")

    ok, detail = run_suite()
    lines.append(f"unit suite: {'ok' if ok else 'FAILED'} ({detail.splitlines()[-1] if detail else ''})")
    if not ok:
        flag("tests", f"unit suite failed:\n{detail}")
    for pkg in ("mlx", "mlx-lm"):
        if not env["versions"].get(pkg):
            flag("versions", f"{pkg} is not installed in this interpreter — run with the engine venv")
    if trials < bench.MIN_PUBLISHABLE_TRIALS:
        flag("trials", f"--trials {trials} < {bench.MIN_PUBLISHABLE_TRIALS}: rows would not be publishable")

    # Prompts and report inputs.
    speed = []
    for name in prompts:
        f = bench.PROMPTS / "speed" / f"{name}.txt"
        try:
            if f.is_file():
                speed.append((rel(f), sha256(f)))
            else:
                flag("report inputs", f"speed prompt file missing: {f}")
        except OSError as e:
            flag("report inputs", f"speed prompt file unreadable: {f} ({e})")
    acc_ids = {p["id"] for p in MATH_PROMPTS + TOOL_PROMPTS + INSTR_PROMPTS}
    try:
        acc_n, acc_sha = dir_digest(bench.PROMPTS / "accuracy")
        qual_n, qual_sha = dir_digest(bench.PROMPTS / "quality")
    except OSError as e:
        flag("report inputs", f"prompt file unreadable: {e}")
        acc_n = qual_n = 0
        acc_sha = qual_sha = "UNREADABLE"
    if acc_n != len(acc_ids) or {f.stem for f in (bench.PROMPTS / "accuracy").glob("*.txt")} != acc_ids:
        flag("report inputs", f"prompts/accuracy has {acc_n} files, p3_suite defines {len(acc_ids)}; ids differ")
    try:
        if blind.expected_ids() != sorted(p["id"] for p in QUALITY_PROMPTS):
            flag("scoring", "prompts/quality ids differ from p3_suite.QUALITY_PROMPTS")
    except SystemExit as e:
        flag("scoring", str(e))
    for doc in ("REPORT.md", "MODELS.md"):
        if not (ROOT / doc).is_file():
            flag("report inputs", f"{doc} missing")
    if len(model_ids) < 2:
        flag("scoring", "a blind round needs at least 2 models")
    if len(model_ids) > len(blind.LABELS):
        flag("scoring", f"{len(model_ids)} models but only {len(blind.LABELS)} blind slots")

    plan, revisions = [], []
    for mid in model_ids:
        cfg = registry.MODELS.get(mid)
        if cfg is None:
            flag("models", f"{mid}: not in the registry")
            continue
        selected = []
        for tier in tiers:
            ck = next((c for c in cfg.checkpoints.values() if c.tier == tier), None)
            if ck is None:
                flag("models", f"{mid}: no {tier} checkpoint in the registry")
            else:
                selected.append(ck)
        bad_before = len(bad)
        for ck in selected:
            who = f"{mid}/{ck.tag}"
            if not ck.available:
                flag("models", f"{who}: not downloaded ({ck.local_dir}) — run --phase fetch")
            if not (ck.publisher and ck.hf_repo and ck.params and ck.weights_gb):
                flag("report inputs", f"{who}: registry lacks publisher/repo/params/size for the report table")
            prov = gates.problems(cfg, ck, out, need_p0=False)
            for p in prov:
                flag("provenance", f"{who}: {p}")
            if not prov:
                for p in gates.problems(cfg, ck, out):
                    flag("load gate", f"{who}: {p}")
            rec, _ = gates._load(out / cfg.id / "p0" / f"{ck.tag}-provenance.json")
            revisions.append(f"{who}: {ck.hf_repo} @ {(rec or {}).get('revision') or 'NO REVISION RECORDED'}")

            for path in planned_paths(cfg, ck, out):
                plan.append(path)
                if not path.exists():
                    continue
                s, err = gates._load(path / "summary.json") if path.is_dir() else (None, "a file")
                state = (f"outcome={s.get('outcome', 'not recorded (legacy summary)')}, "
                         f"publishable={s.get('publishable', 'not recorded')}" if s else err)
                flag("collision", f"{rel(path)} exists ({state}); a new run would overwrite it — "
                                  "narrow --models/--tiers or use another --out")
                if path.is_dir() and (not s or "outcome" in s) and not (
                        s and s.get("outcome") == "completed" and s.get("publishable") is True):
                    flag("speed results", f"{rel(path)}: existing row is not a completed, publishable result")
                if s and (s.get("trials") or 0) < bench.MIN_PUBLISHABLE_TRIALS:
                    flag("trials", f"{rel(path)}: row recorded {s.get('trials')} trials < {bench.MIN_PUBLISHABLE_TRIALS}")
        summary = out / cfg.id / "p3" / "accuracy_summary.json"
        if selected and summary.exists():
            flag("collision", f"{rel(summary)} exists; run_p3.py would overwrite it — narrowing --tiers "
                                  "cannot clear this; use a different --out or change run_p3.py first")
        for p in version_problems(cfg, out, env):
            flag("versions", p)
        plan.append(summary)
        lines.append(f"{mid}: {len(selected)} checkpoint(s) checked, "
                     f"{'ok' if len(bad) == bad_before else 'BLOCKED'}")

    rd = out / f"blind-{round_name}"
    if rd.exists():
        flag("collision", f"{rel(rd)} exists; pick a new --round")
        if (rd / "key.json").is_file() and (rd / "scores.json").is_file():
            k, _ = gates._load(rd / "key.json")
            d, _ = gates._load(rd / "scores.json")
            if k and d:
                for p in score_quality.validate(d, k):
                    flag("scoring", f"{rd.name}: {p}")
            else:
                flag("scoring", f"{rd.name}: key.json or scores.json unreadable")
    plan.append(rd)

    lines += ["", "Reproducibility checklist",
              f"  host:      {env['host']}",
              "  runtimes:  " + ", ".join(f"{k} {v or 'NOT INSTALLED'}" for k, v in env["versions"].items())
              + f"; {env['optiq']} (separate venv)",
              "  revisions: " + ("\n             ".join(revisions) or "none"),
              "  prompts:   " + "\n             ".join(f"{n}  sha256 {h}" for n, h in speed)
              + f"\n             prompts/accuracy  {acc_n} files  sha256 {acc_sha}"
              + f"\n             prompts/quality   {qual_n} files  sha256 {qual_sha}",
              f"  generation: greedy (temperature 0), speed max_tokens {max_tokens}, "
              f"accuracy {run_p3.ACCURACY_MAX_TOKENS}, quality {run_p3.QUALITY_MAX_TOKENS}, "
              f"nonce on, decode_min_tokens {bench.MIN_DECODE_TOKENS}",
              "  template controls: " + "; ".join(
                  f"{m}={registry.MODELS[m].template_kwargs}" for m in model_ids if m in registry.MODELS),
              f"  trials:    {trials} kept per prompt + 1 discarded warm-up "
              f"(minimum {bench.MIN_PUBLISHABLE_TRIALS})",
              "  outputs:   " + "\n             ".join(str(p) for p in plan)]
    return bad, lines


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--models", nargs="+", default=registry.ORDER)
    ap.add_argument("--tiers", nargs="+", default=list(registry.TIER_ORDER), choices=registry.TIER_ORDER)
    ap.add_argument("--trials", type=int, default=bench.MIN_PUBLISHABLE_TRIALS)
    ap.add_argument("--max-tokens", type=int, default=run_bench.DEFAULT_MAX_TOKENS)
    ap.add_argument("--prompts", nargs="+", default=run_bench.DEFAULT_PROMPTS)
    ap.add_argument("--round", default="r1", help="blind round label the matrix will create")
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args(argv)

    bad, lines = evaluate(args.models, args.out, args.tiers, args.trials, args.max_tokens,
                          args.prompts, args.round, environment(), unit_suite)
    print("\n".join(lines))
    print("\nNot done here: a clean, independently reviewed run is a separate future action; "
          "no benchmark was run and no model was loaded.")
    if bad:
        print(f"\nNOT READY: {len(bad)} blocking problem(s)")
        for b in bad:
            print(f"  {b}")
        return 1
    print("\nREADY: every precondition holds (this is a pre-run check, not evidence of any result)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
