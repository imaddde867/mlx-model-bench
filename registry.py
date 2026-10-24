#!/usr/bin/env python3
"""Model registry: the one place that says what this repo benchmarks.

Nothing in the harness hardcodes a model. `bench.py`, `run_p3.py` and
`run_bench.py` take a model id from here and read everything else — checkpoint
paths, template controls, provenance — off the entry.

Numbers in here are **measured from the Hub**, not copied from model cards:

    https://huggingface.co/api/models/<repo>?blobs=true   (file sizes, params)
    https://huggingface.co/<repo>/resolve/main/config.json (architecture)

fetched 2026-10-24. `weights_gb` is the sum of file bytes / 1e9, so it is the
download size; runtime peak is measured separately by `bench.py` and is always
larger. `bits_per_weight` = weights_gb * 8 / params is the only honest way to
compare checkpoints that all call themselves "4-bit" — the spread across these
eight checkpoints runs 4.51 to 6.17 bpw, and that spread is a confound the
report has to carry.

Chat-template controls are per model and must be passed explicitly. Do not
assume a kwarg exists — see MODELS.md ("Template controls"). Two ways this bites:
a template that writes `enable_thinking | default(true)` silently turns thinking
on, and an unknown kwarg either raises or is dropped depending on the jinja.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent


@dataclass
class Checkpoint:
    """One downloadable, loadable file set for a model."""

    tag: str            # short label used in paths: "4bit", "optiq-4bit", "8bit"
    hf_repo: str        # what to fetch
    local_dir: str      # where fetch puts it, relative to the repo root
    weights_gb: float   # bytes the engine actually loads, GB decimal
    params: float       # total parameters, billions
    quant: str          # human description of the quantization
    publisher: str      # "vendor" | "mlx-community" | "third-party:<user>"
    tier: str = "4bit"  # "4bit" | "optiq-4bit" | "8bit" — the same three per model
    official: bool = False    # converted and published by the model's own vendor
    engines: tuple = ("mlx_lm",)  # runtimes this checkpoint can be loaded in
    download_gb: float = 0.0  # Hub file total; > weights_gb when the repo carries dead weight
    sha_verified: bool = False
    notes: str = ""

    @property
    def local_path(self) -> Path:
        return ROOT / self.local_dir

    @property
    def present(self) -> bool:
        """Config is on disk. Not the same as complete — see `available`."""
        return (self.local_path / "config.json").exists()

    @property
    def available(self) -> bool:
        """Loadable: config, every shard the index names, and no interrupted download.

        A killed fetch leaves `config.json` plus partial shards, which used to read
        as "download complete" and then fail verification instead of resuming. The
        `.incomplete` marker in the hub's cache is the reliable tell.
        """
        if not self.present:
            return False
        if any(self.local_path.rglob("*.incomplete")):
            return False

        index = self.local_path / "model.safetensors.index.json"
        if index.exists():
            try:
                shards = set(json.loads(index.read_text()).get("weight_map", {}).values())
            except (json.JSONDecodeError, OSError):
                shards = set()
            if shards:
                return all((self.local_path / s).exists() for s in shards)
        return bool(list(self.local_path.glob("*.safetensors")))

    @property
    def bits_per_weight(self) -> float:
        """Bits per parameter actually loaded. Only meaningful when params is known."""
        if not self.params:
            return 0.0
        return round(self.weights_gb * 8 / self.params, 2)


@dataclass
class Model:
    id: str
    name: str
    vendor: str
    arch: str
    params_total: float
    params_active: Optional[float]
    context_train: int
    template_kwargs: Dict[str, object]
    template_notes: str
    checkpoints: Dict[str, Checkpoint]
    role: str = ""            # why this model is in the roster
    source: str = ""          # Hub repo or paper the metadata came from
    notes: str = ""

    def checkpoint(self, tag: Optional[str] = None) -> Checkpoint:
        if tag is None:
            tag = next(iter(self.checkpoints))
        if tag not in self.checkpoints:
            raise KeyError(
                f"{self.id} has no checkpoint {tag!r}; "
                f"known: {sorted(self.checkpoints)}"
            )
        return self.checkpoints[tag]

    @property
    def available_checkpoints(self) -> List[Checkpoint]:
        return [c for c in self.checkpoints.values() if c.available]


# -----------------------------------------------------------------------------
# Template controls, one note per model. All read from chat_template.jinja at
# the pinned repo, not from documentation.
# -----------------------------------------------------------------------------

TIER_ORDER = ("4bit", "optiq-4bit", "8bit")

MODELS: Dict[str, Model] = {
    # =========================================================================
    # Baseline (P0-P3 complete): the model this repo was built around.
    # =========================================================================
    "nemotron-3.5-bolt": Model(
        id="nemotron-3.5-bolt",
        name="NVIDIA Nemotron 3.5 Bolt E4B",
        vendor="NVIDIA",
        arch="Dense transformer (GQA), Gemma-4 derived",
        params_total=7.463,
        params_active=4.0,
        context_train=131072,
        template_kwargs={"enable_thinking": False, "conversational_mode": True},
        template_notes=(
            "Two controls. conversational_mode=true is the intended voice-assistant "
            "mode and costs +27 prompt tokens of scaffolding. enable_thinking must be "
            "passed explicitly: the jinja defaults it false but transformers injects "
            "true, so omitting it emits a <|channel>thought trace. "
            "thinking_token_budget is a serving-API parameter, not a template variable."
        ),
        checkpoints={
            "mlx-4bit": Checkpoint(
                tag="mlx-4bit",
                hf_repo="rapid-mlx/NVIDIA-Nemotron-3.5-Bolt-E4B-MLX-4bit",
                local_dir="models/rapid-mlx-4bit",
                weights_gb=4.20,
                download_gb=4.20,
                params=7.463,
                quant="affine 4-bit, group 64",
                publisher="third-party:rapid-mlx",
                tier="4bit",
                sha_verified=True,
                notes="Text-only by construction: vision/audio tensors excluded.",
            ),
            "optiq-4bit": Checkpoint(
                tag="optiq-4bit",
                hf_repo="mlx-community/NVIDIA-Nemotron-3.5-Bolt-E4B-OptiQ-4bit",
                local_dir="models/optiq-4bit",
                weights_gb=6.53,
                download_gb=7.49,
                params=7.463,
                quant="mixed 4/8-bit (OptiQ), per-module 8-bit overrides",
                publisher="mlx-community",
                tier="optiq-4bit",
                engines=("mlx_lm", "mlx-optiq"),
                sha_verified=True,
                notes=(
                    "Ships optiq_vision.safetensors (0.96 GB) that mlx_lm never loads: "
                    "7.49 GB download, 6.53 GB of live weights. "
                    "kv_config.json covers 24 of 42 layers — a release defect; the bit "
                    "allocation was transferred from gemma-4-e4b-it, not measured."
                ),
            ),
            "mlx-8bit": Checkpoint(
                tag="mlx-8bit",
                hf_repo="rapid-mlx/NVIDIA-Nemotron-3.5-Bolt-E4B-MLX-8bit",
                local_dir="models/rapid-mlx-8bit",
                weights_gb=7.96,
                download_gb=7.96,
                params=7.463,
                quant="affine 8-bit, group 64",
                publisher="third-party:rapid-mlx",
                tier="8bit",
                notes=(
                    "Reference tier, not measured in the original study. Its existence is the "
                    "reason the baseline gets re-run: comparing four models at 4-bit while only "
                    "checking whether one of them gains from a wider quant is not a fair test of "
                    "the quantizer. lmstudio-community publishes the same 8-bit file at the same "
                    "size if a second publisher is wanted."
                ),
            ),
        },
        role="Baseline. The only model with a full P0-P3 record; parity check for the roster.",
        source="https://huggingface.co/rapid-mlx/NVIDIA-Nemotron-3.5-Bolt-E4B-MLX-4bit",
        notes="See REPORT.md section 1 and P0.md. Arm 4 (optiq serve) is dead on 16 GB: P0.md.",
    ),
    # =========================================================================
    # Candidate 1: compact intelligence per GB.
    # =========================================================================
    "minicpm5-2b": Model(
        id="minicpm5-2b",
        name="MiniCPM5-2B",
        vendor="OpenBMB",
        arch="Dense Llama (GQA, 42 layers, 16 heads / 2 KV heads, head_dim 128)",
        params_total=2.517,
        params_active=2.517,
        context_train=131072,
        template_kwargs={"enable_thinking": False},
        template_notes=(
            "enable_thinking exists but is guarded by `is defined`, so the default "
            "branch differs from an explicit false — pass it explicitly and check the "
            "prompt length in P0. Template is tool-call aware (<tools>/<function> XML). "
            "No conversational_mode."
        ),
        checkpoints={
            "mlx-4bit": Checkpoint(
                tag="mlx-4bit",
                hf_repo="openbmb/MiniCPM5-2B-MLX",
                local_dir="models/minicpm5-2b-4bit",
                weights_gb=1.43,
                download_gb=1.43,
                params=2.517,
                quant="affine 4-bit, group 64",
                publisher="vendor",
                tier="4bit",
                official=True,
                notes=(
                    "OpenBMB publishes this MLX 4-bit itself, so the plain tier uses the "
                    "vendor conversion the way LFM does. mlx-community/MiniCPM5-2B-mlx-4Bit "
                    "is a second, independently produced 4-bit of the same weights — a free "
                    "consistency check if the two ever disagree."
                ),
            ),
            "optiq-4bit": Checkpoint(
                tag="optiq-4bit",
                hf_repo="mlx-community/MiniCPM5-2B-OptiQ-4bit",
                local_dir="models/minicpm5-2b-optiq",
                weights_gb=1.94,
                download_gb=1.94,
                params=2.517,
                quant="mixed 4/8-bit (OptiQ): embed + early-layer projections at 8-bit",
                publisher="mlx-community",
                tier="optiq-4bit",
                engines=("mlx_lm", "mlx-optiq"),
            ),
            "mlx-8bit": Checkpoint(
                tag="mlx-8bit",
                hf_repo="mlx-community/MiniCPM5-2B-8bit",
                local_dir="models/minicpm5-2b-8bit",
                weights_gb=2.68,
                download_gb=2.68,
                params=2.517,
                quant="affine 8-bit, group 64",
                publisher="mlx-community",
                tier="8bit",
                notes="Reference tier: at 2.68 GB this is still smaller than three of the four 4-bit rows.",
            ),
        },
        role="Best compact-intelligence candidate: 2.5B at 1.43 GB is the cheapest full tool-use row.",
        source="https://huggingface.co/openbmb/MiniCPM5-2B",
        notes=(
            "42 layers, 2 KV heads, 131k vocab — tiny KV cache, long context is cheap here. "
            "Also on the Hub: MiniCPM5-2B-DSpark (a speculative-decoding family) and a "
            "MiniCPM5-1B tier; out of scope until the 2B row exists."
        ),
    ),
    # =========================================================================
    # Candidate 2: direct 4B-class challenger.
    # =========================================================================
    "spark-x2.5-4b": Model(
        id="spark-x2.5-4b",
        name="Spark-X2.5-4B",
        vendor="XHToken",
        arch="Dense, hybrid attention 3:1 sliding:full (36 layers, head_dim 256, gated attn out)",
        params_total=4.112,
        params_active=4.112,
        context_train=1048576,
        template_kwargs={"enable_thinking": False},
        template_notes=(
            "enable_thinking is `| default(true)` — thinking is ON unless passed false. "
            "This is the same trap as Nemotron but louder: forgetting the kwarg changes "
            "what you measure, not just how it formats. Template has tool-call tags."
        ),
        checkpoints={
            "mlx-4bit": Checkpoint(
                tag="mlx-4bit",
                hf_repo="abenzerps/Spark-X2.5-4B-MLX-4bit",
                local_dir="models/spark-x2.5-4b-4bit",
                weights_gb=2.32,
                download_gb=2.32,
                params=4.112,
                quant="affine 4-bit, group 64",
                publisher="third-party:abenzerps",
                tier="4bit",
                notes=(
                    "No vendor or mlx-community plain 4-bit MLX conversion exists. This is a "
                    "third-party quant of the confirmed 4.11B BF16 weights, unvetted beyond "
                    "config parity — say so in every table it appears in."
                ),
            ),
            "optiq-4bit": Checkpoint(
                tag="optiq-4bit",
                hf_repo="mlx-community/Spark-X2.5-4B-OptiQ-4bit",
                local_dir="models/spark-x2.5-4b-optiq",
                weights_gb=3.05,
                download_gb=3.05,
                params=4.112,
                quant="mixed 4/8-bit (OptiQ), 8-bit on embed + per-layer projections",
                publisher="mlx-community",
                tier="optiq-4bit",
                engines=("mlx_lm", "mlx-optiq"),
            ),
            "mlx-8bit": Checkpoint(
                tag="mlx-8bit",
                hf_repo="abenzerps/Spark-X2.5-4B-MLX-8bit",
                local_dir="models/spark-x2.5-4b-8bit",
                weights_gb=4.38,
                download_gb=4.38,
                params=4.112,
                quant="affine 8-bit, group 64",
                publisher="third-party:abenzerps",
                tier="8bit",
                notes=(
                    "Third-party like the 4-bit; hermitdave publishes 4/6/8-bit too. Spark is the "
                    "one model here with no vendor MLX conversion at all, which is a fairness "
                    "caveat that cannot be engineered away — only stated."
                ),
            ),
        },
        role="Best direct 4B-class challenger: coding, tool use, instruction following at 2.3-3.1 GB.",
        source="https://huggingface.co/XHToken/Spark-X2.5-4B",
        notes=(
            "The ~2.6 GB figure that circulated for this model is a GGUF; GGUF is a "
            "non-goal here. The MLX numbers are 2.32 / 3.05 GB. 1M-token trained context "
            "with 3:1 sliding attention — the one model here whose long-context prefill "
            "might not look like the others; if it does, that is the finding."
        ),
    ),
    # =========================================================================
    # Candidate 3: hybrid MoE, speed.
    # =========================================================================
    "lfm2.5-8b-a1b": Model(
        id="lfm2.5-8b-a1b",
        name="LFM2.5-8B-A1B",
        vendor="Liquid AI",
        arch="Hybrid conv+attention MoE (24 layers: 18 conv, 6 full attention; 32 experts, top-4)",
        params_total=8.468,
        params_active=1.3,
        context_train=128000,
        template_kwargs={"preserve_thinking": False},
        template_notes=(
            "No enable_thinking. The only control is preserve_thinking (default false), "
            "which keeps historical <think> blocks in context. Template is tool-call aware."
        ),
        checkpoints={
            "mlx-4bit": Checkpoint(
                tag="mlx-4bit",
                hf_repo="LiquidAI/LFM2.5-8B-A1B-MLX-4bit",
                local_dir="models/lfm2.5-8b-a1b-4bit",
                weights_gb=4.85,
                download_gb=4.85,
                params=8.468,
                quant="affine 4-bit, group 64; embed_tokens at 6-bit",
                publisher="vendor",
                tier="4bit",
                official=True,
                notes="Official Liquid AI MLX conversion. Embedding is 6-bit, not 4 — unlike every other row here.",
            ),
            "optiq-4bit": Checkpoint(
                tag="optiq-4bit",
                hf_repo="mlx-community/LFM2.5-8B-A1B-OptiQ-4bit",
                local_dir="models/lfm2.5-8b-a1b-optiq",
                weights_gb=5.47,
                download_gb=5.47,
                params=8.468,
                quant="mixed 4/8-bit (OptiQ), 8-bit on embed + conv projections",
                publisher="mlx-community",
                tier="optiq-4bit",
                engines=("mlx_lm", "mlx-optiq"),
            ),
            "mlx-8bit": Checkpoint(
                tag="mlx-8bit",
                hf_repo="LiquidAI/LFM2.5-8B-A1B-MLX-8bit",
                local_dir="models/lfm2.5-8b-a1b-8bit",
                weights_gb=9.02,
                download_gb=9.02,
                params=8.468,
                quant="affine 8-bit, group 64",
                publisher="vendor",
                tier="8bit",
                official=True,
                notes=(
                    "9.02 GB is the largest row here; with KV it may brush 10-11 GB on a 16 GB "
                    "box at 8k context. The P0 gate decides whether it is measurable, not "
                    "whether it is interesting. Liquid also publishes 5- and 6-bit MLX."
                ),
            ),
        },
        role="Speed row: 8.5B of weights, ~1.3B active per token. Tests whether MoE wins decode tps on M4.",
        source="https://huggingface.co/LiquidAI/LFM2.5-8B-A1B",
        notes=(
            "params_active is the vendor's A1B label. Arithmetic from config (24 layers, "
            "32 experts, top-4, moe_intermediate 1792) lands nearer 1.6B once embed, conv "
            "and dense layers are counted — measured decode tps is the arbiter, and the "
            "report should say which number it used. 18 of 24 layers are conv, so a long "
            "prompt's prefill will not scale like a pure-attention model."
        ),
    ),
}

# Roster order: baseline first, then the three candidates. The report keeps this order.
ORDER = ["nemotron-3.5-bolt", "minicpm5-2b", "spark-x2.5-4b", "lfm2.5-8b-a1b"]


def get_model(model_id_or_path: str) -> Model:
    """Resolve a registry id (or an ad-hoc local path) to a Model."""
    if model_id_or_path in MODELS:
        return MODELS[model_id_or_path]

    for cfg in MODELS.values():
        for ck in cfg.checkpoints.values():
            if model_id_or_path in (ck.local_dir, str(ck.local_path), ck.hf_repo):
                return cfg

    p = Path(model_id_or_path)
    clean_id = p.name.lower().replace("_", "-") or "adhoc"
    size_gb = (
        round(sum(f.stat().st_size for f in p.glob("**/*") if f.is_file()) / 1e9, 2)
        if p.exists() else 0.0
    )
    return Model(
        id=clean_id,
        name=clean_id,
        vendor="unknown",
        arch="unknown / ad-hoc",
        params_total=0.0,
        params_active=None,
        context_train=0,
        template_kwargs={},
        template_notes="Ad-hoc path: template controls unknown, passed through as given.",
        checkpoints={
            "local": Checkpoint(
                tag="local",
                hf_repo="",
                local_dir=str(p),
                weights_gb=size_gb,
                params=0.0,
                quant="unknown",
                publisher="local",
            )
        },
        role="ad-hoc",
    )


def list_models() -> List[Model]:
    return [MODELS[i] for i in ORDER if i in MODELS]


def dump_catalog_json(out_path: Optional[Path] = None) -> Path:
    target = out_path or (ROOT / "models.json")
    data = {}
    for mid, m in MODELS.items():
        data[mid] = {
            **{k: v for k, v in m.__dict__.items() if k != "checkpoints"},
            "checkpoints": {t: c.__dict__ for t, c in m.checkpoints.items()},
        }
    target.write_text(json.dumps(data, indent=2) + "\n")
    return target


def _fmt_row(m: Model, c: Checkpoint) -> str:
    have = "local" if c.available else "     "
    dl = f" dl{c.download_gb:.2f}" if c.download_gb and abs(c.download_gb - c.weights_gb) > 0.01 else ""
    src = "vendor" if c.official else "third-party"
    engines = "+".join("optiq" if e == "mlx-optiq" else "mlx_lm" for e in c.engines)
    return (
        f"  {c.tag:11s} {c.weights_gb:5.2f}GB{dl:9s} {c.bits_per_weight:4.2f}bpw "
        f"{have:5s} {src:11s} {engines:12s} {c.hf_repo}"
    )


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="Inspect the model registry.")
    ap.add_argument("--json", action="store_true", help="dump models.json")
    args = ap.parse_args(argv)

    if args.json:
        print(f"wrote {dump_catalog_json()}")
        return 0

    for m in list_models():
        print(f"\n{m.id}  ·  {m.name}  ({m.vendor})")
        print(f"  {m.arch}")
        print(f"  params {m.params_total}B total"
              + (f" / {m.params_active}B active" if m.params_active else "")
              + f" · ctx {m.context_train} · template {json.dumps(m.template_kwargs)}")
        print(f"  {m.role}")
        for c in m.checkpoints.values():
            print(_fmt_row(m, c))
    print("\ntiers are identical across the roster: 4bit (core), optiq-4bit (core), "
          "8bit (reference). 'third-party' means no vendor MLX conversion is published.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
