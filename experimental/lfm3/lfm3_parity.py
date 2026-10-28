"""LFM3 parity: pins, fixture schema and comparison arithmetic. Standard library only.

LFM3 is not in the benchmark roster: `mlx-lm` 0.32.2 has no `lfm3` architecture.
Its model code is the open PR ml-explore/mlx-lm#2016. This directory checks that
PR against the Hugging Face reference implementation, in its own venv, without
touching `registry.py`, `.venv-mlxlm` or anything under `results/`.

Two steps, two scripts:

    make_fixture.py   HF transformers, fp32, CPU -> fixture.json (once; committed)
    check_parity.py   the PR's mlx_lm.models.lfm3 vs fixture.json -> runs/<name>.json

Both import this module, and so does the offline test suite, so nothing here may
import numpy, mlx or torch.
"""

import hashlib
import json
import math
from fnmatch import fnmatch
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIXTURE = HERE / "fixture.json"
RUNS = HERE / "runs"
SCHEMA = 1

# Every version that changes a number below. A run against anything else is a
# different experiment, and check_parity.py refuses it unless told otherwise.
PINS = {
    "mlx_lm_pr": "https://github.com/ml-explore/mlx-lm/pull/2016",
    "mlx_lm_commit": "9027ac5b062175fba890c612c3905d5d8d5b5294",
    "mlx": "0.32.4",
    "transformers": "5.21.0",
    "torch": "2.14.1",
    "reference_repo": "LiquidAI/LFM3-1.7B",
    "reference_revision": "40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd",
    "quant_repo": "LiquidAI/LFM3-1.7B-MLX-4bit",
    "quant_revision": "d0ca73dcfa70ce18733754a492cdc1fb0123262d",
}

# The input: a fixed file already in the repo, cut to a fixed token count. The
# 640 tokens are 2.5x the 256-token sliding window, so sliding-attention layers
# are exercised past the point where the window starts dropping keys.
TEXT_FILE = "prompts/replicated/long.txt"
N_TOKENS = 640
WINDOW = 256
TOP_K = 10
N_GEN = 64
CHAT = [{"role": "user", "content": "Explain in three sentences why the sky is blue."}]
CACHE_PREFILL = 128
# What a load reads: weights, configs, tokenizer, chat template, remote code.
# The repos also carry README and image assets that change nothing measured.
CHECKPOINT_FILES = ["*.safetensors", "*.json", "*.py", "*.jinja"]

# Gates. fp32 MLX against fp32 HF must agree on every next token; the cached
# decode must reproduce the uncached forward. The 4-bit checkpoint is a different
# file: its agreement is recorded, never gated (quantization error, not code).
GATES = {
    "fp32": {"max_abs_diff": 1e-3, "top1": 1.0},
    "cache_fp32": {"max_abs_diff": 1e-3, "top1": 1.0},
}


def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def checkpoint_file(name):
    return any(fnmatch(name, pattern) for pattern in CHECKPOINT_FILES)


def canonical(doc):
    return json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()


def digest(doc, key="fixture_sha256"):
    """SHA-256 of the document without its own digest field."""
    return hashlib.sha256(canonical({k: v for k, v in doc.items() if k != key})).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dump(doc):
    """JSON with nested dicts indented and lists of scalars on one line, so a
    640-row fixture stays reviewable and diffs line by line."""

    def emit(v, depth):
        pad = "  " * (depth + 1)
        if isinstance(v, dict) and v:
            items = [f'{pad}{json.dumps(k)}: {emit(v[k], depth + 1)}' for k in v]
            return "{\n" + ",\n".join(items) + "\n" + "  " * depth + "}"
        if isinstance(v, list) and v and any(isinstance(x, (dict, list)) for x in v):
            return "[\n" + ",\n".join(pad + emit(x, depth + 1) for x in v) + "\n" + "  " * depth + "]"
        return json.dumps(v, separators=(", ", ": "))

    return emit(doc, 0) + "\n"


def write_once(path, doc):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x") as f:
        f.write(dump(doc))


def load(path=FIXTURE):
    return json.loads(Path(path).read_text())


def first_divergence(a, b):
    """Number of leading tokens two greedy decodes share."""
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def compare(ref, gathered, argmax, gen):
    """Score one MLX run against the reference fixture.

    `gathered[i][j]` is the MLX logit at `ref["topk_ids"][i][j]`; the fixture does
    not hold full-vocabulary logits (640 x 128k floats), so the logit difference is
    measured on the reference's top-k candidates at every position, which is where
    a next-token decision is made. `argmax` is MLX's own full-vocabulary argmax.
    """
    diffs = [abs(o - r) for orow, rrow in zip(gathered, ref["topk_logits"]) for o, r in zip(orow, rrow)]
    top1 = [int(a == ids[0]) for a, ids in zip(argmax, ref["topk_ids"])]
    tail = top1[ref["window"]:]
    return {
        "positions": len(top1),
        "max_abs_diff_topk": max(diffs),
        "mean_abs_diff_topk": sum(diffs) / len(diffs),
        "top1": sum(top1) / len(top1),
        "top1_after_window": sum(tail) / len(tail),
        "greedy_identical": first_divergence(ref["greedy_ids"], gen),
        "greedy_len": len(ref["greedy_ids"]),
    }


def fixture_problems(fx):
    """Everything wrong with a fixture, empty when it is usable."""
    p = []
    if fx.get("schema") != SCHEMA:
        return [f"schema {fx.get('schema')!r}, expected {SCHEMA}"]
    if fx.get("fixture_sha256") != digest(fx):
        p.append("fixture_sha256 does not match its content")
    if fx.get("pins") != PINS:
        p.append("pins differ from lfm3_parity.PINS")
    inp = fx.get("input", {})
    if inp.get("text_file") != TEXT_FILE or len(inp.get("ids", [])) != N_TOKENS:
        p.append(f"input is not the first {N_TOKENS} tokens of {TEXT_FILE}")
    if inp.get("text_file") == TEXT_FILE and inp.get("text_sha256") != sha_file(ROOT / TEXT_FILE):
        p.append(f"{TEXT_FILE} changed since the fixture was made")
    ref = fx.get("reference", {})
    if ref.get("window") != WINDOW or N_TOKENS <= 2 * WINDOW:
        p.append("sequence does not run past twice the sliding window")
    ids, logits = ref.get("topk_ids", []), ref.get("topk_logits", [])
    if len(ids) != N_TOKENS or len(logits) != N_TOKENS:
        p.append(f"top-k rows: {len(ids)} ids / {len(logits)} logits, expected {N_TOKENS}")
    elif any(len(r) != TOP_K for r in ids) or any(len(r) != TOP_K for r in logits):
        p.append(f"a top-k row is not {TOP_K} wide")
    elif any(r != sorted(r, reverse=True) for r in logits):
        p.append("top-k logits are not sorted, so column 0 is not the argmax")
    if not 0 < len(ref.get("greedy_ids", [])) <= N_GEN or not ref.get("chat_ids"):
        p.append(f"greedy reference must hold 1..{N_GEN} tokens after a chat prompt")
    files = fx.get("weights", {})
    for repo in (PINS["reference_repo"], PINS["quant_repo"]):
        if not any(f.endswith(".safetensors") for f in files.get(repo, {})):
            p.append(f"no weight sha256 recorded for {repo}")
        stray = sorted(f for f in files.get(repo, {}) if not checkpoint_file(f))
        if stray:
            p.append(f"{repo}: {stray} are outside CHECKPOINT_FILES, so a fresh fetch never has them")
    return p


def run_problems(run, fx):
    """Why a run record does not show parity, empty when it does.

    A record made with `--allow-unpinned` says so, and is judged on the gates
    alone: a released mlx-lm records no git commit and may move `mlx`.
    """
    p = []
    if run.get("fixture_sha256") != fx.get("fixture_sha256"):
        p.append("run was made against a different fixture")
    observed = run.get("versions", {})
    if run.get("allow_unpinned") is not True:
        for key in ("mlx_lm_commit", "mlx"):
            if observed.get(key) != PINS[key]:
                p.append(f"{key} {observed.get(key)!r}, pinned {PINS[key]!r}")
    rows = run.get("rows", {})
    for name, gate in GATES.items():
        row = rows.get(name)
        if row is None:
            p.append(f"no {name} row")
            continue
        # Comparisons with NaN are all false, so a NaN must fail on its own.
        diff = row.get("max_abs_diff_topk", row.get("max_abs_diff"))
        if not finite(diff) or diff > gate["max_abs_diff"]:
            p.append(f"{name}: max abs diff {diff} > {gate['max_abs_diff']}")
        top1 = row.get("top1")
        if not finite(top1) or top1 < gate["top1"]:
            p.append(f"{name}: top-1 agreement {top1} < {gate['top1']}")
    fp32 = rows.get("fp32", {})
    if fp32 and fp32.get("greedy_identical") != fp32.get("greedy_len"):
        p.append(f"fp32: greedy decode diverges at token {fp32.get('greedy_identical')}")
    return p
