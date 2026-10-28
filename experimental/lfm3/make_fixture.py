#!/usr/bin/env python3
"""Record the Hugging Face reference for LFM3-1.7B into fixture.json.

Runs LiquidAI's own `modeling_lfm3.py` (trust_remote_code, pinned revision) in
fp32 on CPU, so no CUDA and no MLX are involved. Records, for the first 640
tokens of prompts/replicated/long.txt, the top-k logits at every position, plus
a 64-token greedy decode of a fixed chat prompt and the sha256 of every file in
both checkpoints. The committed fixture means check_parity.py never needs
transformers or torch; regenerate only to move a pin.

    .venv-lfm3/bin/python experimental/lfm3/make_fixture.py --out /tmp/fixture.json

Peak memory is about 8 GB (fp32 weights plus activations). The 1.7B reference
repo is ~3.4 GB to download, the 4-bit checkpoint ~1 GB (hashed, not loaded).
"""

import argparse
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import lfm3_parity as lp


def snapshot_hashes(repo, revision):
    """sha256 of every file at a pinned revision; weights cross-checked against the Hub."""
    from huggingface_hub import HfApi, snapshot_download

    path = Path(snapshot_download(repo, revision=revision, allow_patterns=lp.CHECKPOINT_FILES))
    # A cache filled by an earlier full download holds more than the allow-list;
    # hash only what a fresh fetch with the same patterns would bring down.
    local = {f.name: lp.sha_file(f) for f in sorted(path.iterdir()) if f.is_file() and lp.checkpoint_file(f.name)}
    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    for s in info.siblings:
        if s.lfs and s.rfilename in local and local[s.rfilename] != s.lfs.sha256:
            sys.exit(f"{repo}/{s.rfilename}: local sha256 does not match the Hub's")
    return path, local


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(lp.FIXTURE), help="written once; refuses to overwrite")
    args = ap.parse_args()

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    found = {"transformers": transformers.__version__, "torch": torch.__version__.split("+")[0]}
    for key, version in found.items():
        if version != lp.PINS[key]:
            sys.exit(f"{key} {version}, pinned {lp.PINS[key]}: install experimental/lfm3/requirements.txt")

    ref_path, ref_files = snapshot_hashes(lp.PINS["reference_repo"], lp.PINS["reference_revision"])
    _, quant_files = snapshot_hashes(lp.PINS["quant_repo"], lp.PINS["quant_revision"])

    tok = AutoTokenizer.from_pretrained(ref_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(ref_path, trust_remote_code=True, dtype=torch.float32).eval()
    window = model.config.sliding_window
    text = (lp.ROOT / lp.TEXT_FILE).read_text()
    ids = tok.encode(text)[: lp.N_TOKENS]
    if len(ids) != lp.N_TOKENS:
        sys.exit(f"{lp.TEXT_FILE} is only {len(ids)} tokens")

    with torch.no_grad():
        logits = model(torch.tensor([ids])).logits[0].float()
        top = torch.topk(logits, lp.TOP_K, dim=-1)
        chat = tok.apply_chat_template(lp.CHAT, add_generation_prompt=True, return_tensors="pt")
        if not isinstance(chat, torch.Tensor):
            chat = chat["input_ids"]
        # generation_config.json sets repetition_penalty 1.1, which HF applies even
        # to greedy decoding and mlx_lm's sampler does not: switch it off for parity.
        out = model.generate(chat, max_new_tokens=lp.N_GEN, do_sample=False, repetition_penalty=1.0)
    greedy = out[0, chat.shape[1]:].tolist()

    fx = {
        "schema": lp.SCHEMA,
        "pins": lp.PINS,
        "made": {
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "by": "make_fixture.py",
            "reference": "transformers AutoModelForCausalLM, trust_remote_code, fp32, CPU",
            "python": platform.python_version(),
            "machine": f"{platform.system()} {platform.mac_ver()[0] or platform.release()} {platform.machine()}",
            "versions": found,
        },
        "input": {
            "text_file": lp.TEXT_FILE,
            "text_sha256": lp.sha_file(lp.ROOT / lp.TEXT_FILE),
            "tokenizer": f"{lp.PINS['reference_repo']}@{lp.PINS['reference_revision']}, encode() defaults",
            "ids": ids,
        },
        "reference": {
            "window": window,
            "top_k": lp.TOP_K,
            "topk_ids": top.indices.tolist(),
            "topk_logits": [[round(x, 5) for x in row] for row in top.values.tolist()],
            "chat": lp.CHAT,
            "chat_ids": chat[0].tolist(),
            "greedy_ids": greedy,
            "greedy_text": tok.decode(greedy),
        },
        "weights": {lp.PINS["reference_repo"]: ref_files, lp.PINS["quant_repo"]: quant_files},
    }
    fx["fixture_sha256"] = lp.digest(fx)
    problems = lp.fixture_problems(fx)
    if problems:
        sys.exit("fixture failed its own checks:\n  " + "\n  ".join(problems))
    lp.write_once(args.out, fx)
    print(f"wrote {args.out} ({fx['fixture_sha256'][:12]}), {len(ids)} tokens, {len(greedy)} greedy")
    print(f"greedy: {fx['reference']['greedy_text']!r}")


if __name__ == "__main__":
    main()
