"""Estimate per-Linear input Gram matrices of one expert on its task data.

This is the data-using statistic for the ``whc_gram`` merge (the LLM port of the
GLUE ``whc_tree`` winner) and for the RegMean / RegMean++ comparison. For each
``nn.Linear`` layer with input activations ``x`` of shape ``[tokens, in]``, the
input Gram is

    G = (1 / T) * sum_t x_t x_t^T   in R^{in x in},

the (token-averaged) second moment of the layer input over ``T`` tokens of the
expert's own training-domain data. We register forward hooks on the target
Linear modules, accumulate ``X^T X`` on CPU so the largest Gram
(``mlp.down_proj``: ``[14336, 14336]`` for Llama-3.1-8B) does not have to live on
the GPU, and save the result as ``model.safetensors`` keyed exactly like the
model state dict (``...<proj>.weight``), so ``mergebench/llm_merge.py``'s
``whc_gram`` path can read it via ``ShardedStateReader`` and key-match the weight
it is merging.

Memory note (Llama-3.1-8B). The six hidden-width projections
(q/k/v/o/gate/up_proj) each give a ``[4096, 4096]`` Gram (~67 MB fp32); 32 layers
of those total ~13 GB. Adding ``down_proj`` (``[14336, 14336]``, ~3.3 GB each)
pushes the per-expert CPU accumulator to ~105 GB. Start with
``--exclude-modules down_proj`` (fits a 96 GB node) to validate the pipeline,
then include it on a high-RAM node for the faithful all-Linear RegMean
comparison. Grams are stored fp16 by default to halve disk.

Runs in the `merging` env (cuda torch + transformers + datasets). Llama needs
eager attention here too (SDPA cutlassF failure on the V100/Volta nodes).

Usage
-----
    python -u scripts/mb_gram_estimate.py \
        --expert mb_ckpts/MergeBench__Llama-3.1-8B_math \
        --tokenizer mb_ckpts/NousResearch__Meta-Llama-3.1-8B \
        --dataset MergeBench/math_val \
        --out mb_grams/Llama-3.1-8B/math \
        --n-samples 256 --max-len 1024 --exclude-modules down_proj
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Default Linear projections to collect Grams for (Llama / most decoder LLMs).
DEFAULT_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Same best-effort text extraction as mb_fisher_estimate.py so the two
# data-using statistics see identical inputs.
def example_to_text(ex: dict, text_field: Optional[str]) -> str:
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
    for v in ex.values():
        if isinstance(v, str) and v:
            return v
    raise ValueError(f"Could not extract text from example with keys {list(ex)}; "
                     f"pass --text-field explicitly.")


def _is_target(name: str, targets: List[str], excludes: List[str]) -> bool:
    """A Linear is a target if its module path ends with one of ``targets``
    (e.g. ``...self_attn.q_proj``) and matches none of ``excludes``."""
    if any(name.endswith(t) for t in excludes):
        return False
    return any(name.endswith(t) for t in targets)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expert", required=True,
                    help="Model dir to hook. Round 0: the expert checkpoint. "
                         "Iterative round k: the merged checkpoint from round k-1.")
    ap.add_argument("--tokenizer", required=True,
                    help="Tokenizer dir (use the base model's; vocab is shared).")
    ap.add_argument("--dataset", required=True,
                    help="HF dataset id, e.g. MergeBench/math_val.")
    ap.add_argument("--split", default="train")
    ap.add_argument("--text-field", default=None,
                    help="Force a specific dataset column as the text.")
    ap.add_argument("--out", required=True, help="Output dir for model.safetensors.")
    ap.add_argument("--n-samples", type=int, default=256)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--target-modules", default=",".join(DEFAULT_TARGETS),
                    help="Comma-separated Linear suffixes to Gram-merge.")
    ap.add_argument("--exclude-modules", default="",
                    help="Comma-separated Linear suffixes to skip "
                         "(e.g. 'down_proj' to bound memory).")
    ap.add_argument("--gram-dtype", choices=["fp16", "fp32"], default="fp16",
                    help="On-disk dtype for the saved Grams (default fp16).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    targets = [t.strip() for t in args.target_modules.split(",") if t.strip()]
    excludes = [t.strip() for t in args.exclude_modules.split(",") if t.strip()]
    print(f"[gram] expert={args.expert} dataset={args.dataset} "
          f"n={args.n_samples} device={device} targets={targets} "
          f"exclude={excludes}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.expert, torch_dtype=torch.bfloat16, attn_implementation="eager")
    model.to(device)
    model.eval()

    # Map each target Linear module -> its weight key, and set up CPU fp32
    # accumulators. token_counts[key] tracks T for the (1/T) normalization.
    name_by_module: Dict[nn.Module, str] = {}
    grams: Dict[str, torch.Tensor] = {}
    token_counts: Dict[str, int] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and _is_target(name, targets, excludes):
            key = f"{name}.weight"
            name_by_module[mod] = key
            in_dim = mod.in_features
            grams[key] = torch.zeros((in_dim, in_dim),
                                     dtype=torch.float32, device="cpu")
            token_counts[key] = 0
    if not grams:
        raise RuntimeError("No target Linear layers matched; check "
                           "--target-modules / --exclude-modules.")
    print(f"[gram] hooking {len(grams)} Linear layers", flush=True)

    def make_hook(key: str):
        def hook(_mod, inputs, _output):
            x = inputs[0].detach()
            if x.dim() > 2:
                x = x.reshape(-1, x.size(-1))      # [tokens, in]
            x = x.float()
            grams[key].add_((x.t() @ x).cpu())     # X^T X accumulated on CPU
            token_counts[key] += x.size(0)
        return hook

    handles = [mod.register_forward_hook(make_hook(name_by_module[mod]))
               for mod in name_by_module]

    ds = load_dataset(args.dataset, split=args.split)
    if len(ds) > args.n_samples:
        ds = ds.shuffle(seed=args.seed).select(range(args.n_samples))

    n_used = 0
    with torch.no_grad():
        for i, ex in enumerate(ds):
            text = example_to_text(ex, args.text_field)
            enc = tok(text, return_tensors="pt", truncation=True,
                      max_length=args.max_len)
            input_ids = enc["input_ids"].to(device)
            if input_ids.shape[1] < 2:
                continue
            model(input_ids=input_ids)              # forward only; hooks fire
            n_used += 1
            if (i + 1) % 25 == 0:
                print(f"  [{n_used}/{len(ds)}] tokens/key~{token_counts[next(iter(token_counts))]}",
                      flush=True)

    for h in handles:
        h.remove()
    if n_used == 0:
        raise RuntimeError("No usable samples; check --dataset / --text-field.")

    # Normalize each Gram by its token count: G = (1/T) X^T X.
    out_dtype = torch.float16 if args.gram_dtype == "fp16" else torch.float32
    saved: Dict[str, torch.Tensor] = {}
    for key, g in grams.items():
        t = max(token_counts[key], 1)
        saved[key] = (g / t).to(out_dtype).contiguous()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(saved, str(out_dir / "model.safetensors"), metadata={"format": "pt"})
    print(f"[gram] saved {len(saved)} Grams ({args.gram_dtype}) over {n_used} "
          f"samples -> {out_dir}/model.safetensors", flush=True)


if __name__ == "__main__":
    main()
