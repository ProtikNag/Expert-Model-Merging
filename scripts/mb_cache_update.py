"""Single-pass cache of the whc_diag consensus update, for fast per-LAYER
alpha search/compose (Bet B) WITHOUT re-reading the six 8B models each time.

The per-parameter routing probe (consensus_routed) showed coding degrades via
within-layer activation CROSS-TALK, not own-param over-extrapolation -- so the
right scaling unit is the LAYER. To explore per-layer alpha cheaply we read the
six models ONCE and cache:

  - update.safetensors : u_p = w_M^HTCL(alpha=1) - w_pre, per merged key (fp16)
  - ratio.safetensors  : uncapped consensus ratio |c_p|/(|u_p|+eps), per key (fp16)
  - meta.json          : merged-key list, copied-key list, and a per-(layer,group,
                         expert) task-vector ENERGY table (sum of squares).

A composed merge is then  w = w_pre + clip(ratio, 1, alpha_layer) * u  for any
per-layer alpha vector (scripts/mb_compose_layerscale.py), and the energy table
answers the decisive diagnostic: is coding's task-vector energy CONCENTRATED in
a few layers (=> per-layer scaling can decouple it) or spread uniformly (=> it
cannot, pivot).

    python -u scripts/mb_cache_update.py --config configs/mergebench_tier2.yaml \
        --lam 1e-3 --out mb_cache/Llama-3.1-8B_l1e-3
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mergebench.io_utils import ShardedStateReader, copy_aux_files  # noqa: E402
from mergebench.llm_merge import _consensus_ratio, _is_float, _shapes_agree  # noqa: E402
from scripts.mb_download import expert_repo, local_dir_for  # noqa: E402
from src.utils import ensure_dir, load_config  # noqa: E402

_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


def _group_of(key: str) -> str:
    """Coarse module group for a parameter key (for per-group energy / scaling)."""
    for g in ("q_proj", "k_proj", "v_proj", "o_proj",
              "gate_proj", "up_proj", "down_proj",
              "input_layernorm", "post_attention_layernorm"):
        if g in key:
            return g
    if "embed_tokens" in key:
        return "embed"
    if "lm_head" in key:
        return "lm_head"
    if key.endswith("model.norm.weight"):
        return "final_norm"
    return "other"


def _layer_of(key: str) -> int:
    m = _LAYER_RE.search(key)
    return int(m.group(1)) if m else -1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--domains", default=None)
    ap.add_argument("--lam", default="1e-3", type=str)
    ap.add_argument("--out", required=True, help="Cache output directory.")
    ap.add_argument("--log-every", default=50, type=int)
    args = ap.parse_args()

    cfg = load_config(args.config)
    download_dir = Path(cfg["download_dir"])
    domains = (args.domains.split(",") if args.domains else cfg["tier_domains"])
    base_dir = str(local_dir_for(download_dir, cfg["base_model"]))
    expert_dirs = [str(local_dir_for(download_dir,
                                     expert_repo(cfg["base_name"], d)))
                   for d in domains]
    lam = float(args.lam)
    out = ensure_dir(Path(args.out))
    print(f"[cache] base={cfg['base_name']} domains={domains} lam={lam} -> {out}",
          flush=True)

    base = ShardedStateReader(base_dir)
    experts = [ShardedStateReader(d) for d in expert_dirs]
    n = len(experts)

    updates: dict = {}
    ratios: dict = {}
    merged_keys: list = []
    copied_keys: list = []
    # energy[layer][group][expert_idx] = sum of squared task-vector entries
    energy: dict = defaultdict(lambda: defaultdict(lambda: [0.0] * n))

    keys = base.keys()
    t0 = time.time()
    for idx, key in enumerate(keys):
        base_t = base.get(key)
        if not _is_float(base_t) or not _shapes_agree(key, base, experts):
            copied_keys.append(key)
        else:
            w_pre = base_t.float()
            w_experts = [e.get(key).float() for e in experts]
            w_bar = torch.zeros_like(w_pre)
            for w in w_experts:
                w_bar += w
            w_bar /= n
            num = lam * w_bar
            den = torch.full_like(w_pre, lam)
            tvs = []
            for w_i in w_experts:
                tv = w_i - w_pre
                f_i = tv * tv
                num += f_i * w_i
                den += f_i
                tvs.append(tv)
            out0 = num / (den + 1e-12)
            u = out0 - w_pre
            ratio = _consensus_ratio(u, tvs)
            updates[key] = u.half().contiguous()
            ratios[key] = ratio.half().contiguous()
            merged_keys.append(key)
            lyr, grp = _layer_of(key), _group_of(key)
            for i, tv in enumerate(tvs):
                energy[lyr][grp][i] += float((tv * tv).sum())
        if (idx + 1) % args.log_every == 0 or (idx + 1) == len(keys):
            print(f"  [cache] {idx + 1}/{len(keys)} "
                  f"(merged={len(merged_keys)}, copied={len(copied_keys)})",
                  flush=True)

    save_file(updates, str(out / "update.safetensors"), metadata={"format": "pt"})
    save_file(ratios, str(out / "ratio.safetensors"), metadata={"format": "pt"})
    copy_aux_files(base_dir, out)        # config/tokenizer for composed models
    meta = dict(base_name=cfg["base_name"], base_dir=base_dir, lam=lam,
                domains=domains, n_experts=n,
                merged_keys=merged_keys, copied_keys=copied_keys,
                energy={str(l): {g: v for g, v in gd.items()}
                        for l, gd in energy.items()})
    (out / "meta.json").write_text(json.dumps(meta))
    print(f"\n[cache] done in {time.time() - t0:.1f}s -> {out}", flush=True)

    # --- decisive diagnostic: per-layer coding-energy fraction ---------------
    cod = domains.index("coding") if "coding" in domains else -1
    print(f"\n[diag] per-layer coding(idx={cod}) task-vector energy fraction "
          f"(sum over groups):", flush=True)
    layers = sorted((l for l in energy if l >= 0))
    fracs = []
    for l in layers:
        tot = [0.0] * n
        for grp, ev in energy[l].items():
            for i in range(n):
                tot[i] += ev[i]
        s = sum(tot) + 1e-12
        frac = tot[cod] / s if cod >= 0 else 0.0
        fracs.append(frac)
        bar = "#" * int(frac * 60)
        print(f"  layer {l:2d}: coding_frac={frac:5.3f} {bar}", flush=True)
    if fracs:
        import statistics as st
        print(f"\n[diag] coding_frac across layers: min={min(fracs):.3f} "
              f"max={max(fracs):.3f} mean={st.mean(fracs):.3f} "
              f"stdev={st.pstdev(fracs):.3f}", flush=True)
        print("[diag] high stdev / a few dominant layers => per-layer scaling "
              "CAN decouple coding; near-flat => it cannot (pivot).", flush=True)


if __name__ == "__main__":
    main()
