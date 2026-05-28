"""Estimate the diagonal empirical Fisher of one expert on its task data.

This is the data-using curvature for the Fisher-merging comparison. For expert
weights w_i, the diagonal empirical Fisher is

    F_i^{(k)} = (1/N) sum_n ( d/dw^{(k)}  log p_{w_i}(text_n) )^2,

i.e. the per-parameter mean squared gradient of the causal-LM log-likelihood over
N samples of the expert's own training-domain data (Matena & Raffel 2022, Eq. 2,
empirical-Fisher form with the data's own tokens as targets). We accumulate grad^2
on CPU so the 2.6B-param Fisher (~10GB fp32) does not have to live on the GPU, and
save it as ``fisher.safetensors`` keyed exactly like the model state dict, so
``mergebench/llm_merge.py``'s ``curvature="fisher"`` path can read it via
``ShardedStateReader``.

Runs in the `merging` env (cuda torch + transformers + datasets). gemma2 needs
eager attention here (SDPA cutlassF failure on this GPU).

Usage
-----
    python -u scripts/mb_fisher_estimate.py \
        --expert mb_ckpts/MergeBench__gemma-2-2b_math \
        --tokenizer mb_ckpts/google__gemma-2-2b \
        --dataset MergeBench/math_val \
        --out mb_fisher/gemma-2-2b/math \
        --n-samples 256 --max-len 1024
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Common dataset schemas -> a single text string. Override with --text-field.
def example_to_text(ex: dict, text_field: str | None) -> str:
    """Best-effort extraction of a training text string from one example."""
    if text_field and text_field in ex:
        return str(ex[text_field])
    for f in ("text", "content"):
        if f in ex and ex[f]:
            return str(ex[f])
    if "messages" in ex and ex["messages"]:           # chat format
        return "\n".join(str(m.get("content", "")) for m in ex["messages"])
    instr = ex.get("instruction") or ex.get("prompt") or ex.get("question") or ""
    inp = ex.get("input", "")
    out = ex.get("output") or ex.get("completion") or ex.get("answer") or ex.get("response") or ""
    joined = "\n".join(s for s in (str(instr), str(inp), str(out)) if s)
    if joined:
        return joined
    # Fallback: stringify the first string-valued field.
    for v in ex.values():
        if isinstance(v, str) and v:
            return v
    raise ValueError(f"Could not extract text from example with keys {list(ex)}; "
                     f"pass --text-field explicitly.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expert", required=True, help="Local expert checkpoint dir.")
    ap.add_argument("--tokenizer", required=True,
                    help="Tokenizer dir (use the base model's; vocab is shared).")
    ap.add_argument("--dataset", required=True,
                    help="HF dataset id, e.g. MergeBench/math_val.")
    ap.add_argument("--split", default="train")
    ap.add_argument("--text-field", default=None,
                    help="Force a specific dataset column as the text.")
    ap.add_argument("--out", required=True, help="Output dir for fisher.safetensors.")
    ap.add_argument("--n-samples", type=int, default=256)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[fisher] expert={args.expert} dataset={args.dataset} "
          f"n={args.n_samples} device={device}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.expert, torch_dtype=torch.bfloat16, attn_implementation="eager")
    model.to(device)
    model.eval()                       # no dropout; we still backprop for grads
    for p in model.parameters():
        p.requires_grad_(True)

    ds = load_dataset(args.dataset, split=args.split)
    if len(ds) > args.n_samples:
        ds = ds.shuffle(seed=args.seed).select(range(args.n_samples))

    # CPU fp32 accumulator over EVERY float key in the state dict (params get
    # grad^2; float buffers stay zero). The merge reader does f.get(key) for any
    # float key whose shape agrees across models, including buffers, so all of
    # them must be present or the merge KeyErrors. Zeros are harmless: a zero
    # Fisher entry just makes whc_diag fall back to the mean for that key.
    fisher = {n: torch.zeros(t.shape, dtype=torch.float32, device="cpu")
              for n, t in model.state_dict().items()
              if t.dtype.is_floating_point}

    n_used = 0
    for i, ex in enumerate(ds):
        text = example_to_text(ex, args.text_field)
        enc = tok(text, return_tensors="pt", truncation=True, max_length=args.max_len)
        input_ids = enc["input_ids"].to(device)
        if input_ids.shape[1] < 2:
            continue
        model.zero_grad(set_to_none=True)
        out = model(input_ids=input_ids, labels=input_ids)   # causal LM CE loss
        out.loss.backward()
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.grad is not None and n in fisher:
                    fisher[n] += (p.grad.detach().float() ** 2).cpu()
        n_used += 1
        if (i + 1) % 25 == 0:
            print(f"  [{n_used}/{len(ds)}] loss={out.loss.item():.4f}", flush=True)

    if n_used == 0:
        raise RuntimeError("No usable samples; check --dataset / --text-field.")
    for n in fisher:
        fisher[n] /= n_used

    # Save as model.safetensors so ShardedStateReader treats the fisher dir like
    # a checkpoint (the merge opens it with ShardedStateReader(fisher_dir)).
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    fisher = {k: v.contiguous() for k, v in fisher.items()}
    save_file(fisher, str(out_dir / "model.safetensors"), metadata={"format": "pt"})
    print(f"[fisher] saved {len(fisher)} tensors over {n_used} samples -> "
          f"{out_dir}/model.safetensors", flush=True)


if __name__ == "__main__":
    main()
