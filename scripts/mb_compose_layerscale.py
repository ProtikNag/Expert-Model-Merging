"""Compose per-LAYER alpha merges from the cached consensus update (Bet B),
without re-reading the six 8B models. Reads scripts/mb_cache_update.py output.

For each merged key:  w = w_pre + clip(ratio, 1, alpha_layer) * u
where alpha_layer is assigned per transformer layer by a coding-energy routing
rule: layers whose coding task-vector energy fraction >= --coding-thresh get the
LOW cap (protect coding from the within-layer cross-talk that over-boosting
causes); all other layers get the HIGH cap (give instruction its large scale).
Non-layer keys (embed / final norm / lm_head) use the HIGH cap.

Many (high, low, thresh) variants are composed in ONE pass over the cache.

    python -u scripts/mb_compose_layerscale.py \
        --cache mb_cache/Llama-3.1-8B_l1e-3 \
        --variants "3,1,0.30;4,1,0.30;3,1,0.25;5,1,0.35" \
        --manifest mb_merged/Llama-3.1-8B/layerscale_r0.txt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mergebench.io_utils import ShardedStateReader, save_merged  # noqa: E402
from scripts.mb_cache_update import _group_of, _layer_of  # noqa: E402
from src.utils import ensure_dir, load_config  # noqa: E402

_EPS = 1e-12


def _coding_frac_by_layer(meta: dict) -> dict:
    """layer -> coding task-vector energy fraction (sum over module groups)."""
    domains = meta["domains"]
    cod = domains.index("coding")
    n = meta["n_experts"]
    out = {}
    for l_str, gd in meta["energy"].items():
        l = int(l_str)
        tot = [0.0] * n
        for grp, ev in gd.items():
            for i in range(n):
                tot[i] += ev[i]
        s = sum(tot) + _EPS
        out[l] = tot[cod] / s
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/mergebench_tier2.yaml")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--variants", required=True,
                    help="Semicolon-separated 'high,low,codingthresh' triples.")
    ap.add_argument("--manifest", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    merged_root = ensure_dir(Path(cfg["paths"]["merged"]) / cfg["base_name"])
    cache = Path(args.cache)
    meta = json.loads((cache / "meta.json").read_text())
    base = ShardedStateReader(meta["base_dir"])
    cfrac = _coding_frac_by_layer(meta)

    triples = []
    for spec in args.variants.split(";"):
        spec = spec.strip()
        if not spec:
            continue
        h, lo, t = (float(x) for x in spec.split(","))
        triples.append((h, lo, t))

    def _tag(h, lo, t):
        f = lambda x: (str(int(x)) if x == int(x) else ("%g" % x))
        return f"whc_lscale_h{f(h)}_l{f(lo)}_t{f(t)}"

    tags = [_tag(*tr) for tr in triples]
    save_dirs = [str(merged_root / tg) for tg in tags]
    print(f"[lscale] cache={cache} variants={len(triples)} -> {args.manifest}",
          flush=True)
    for (h, lo, t), tg in zip(triples, tags):
        n_low = sum(1 for l, fr in cfrac.items() if fr >= t)
        print(f"  - {tg}: high={h} low={lo} thresh={t} "
              f"({n_low}/{len(cfrac)} layers routed LOW)", flush=True)

    print("[lscale] loading cached update/ratio ...", flush=True)
    updates = load_file(str(cache / "update.safetensors"))
    ratios = load_file(str(cache / "ratio.safetensors"))
    merged_keys = meta["merged_keys"]
    copied_keys = meta["copied_keys"]

    outs = [dict() for _ in triples]
    for key in copied_keys:
        bt = base.get(key)
        for m in outs:
            m[key] = bt.clone()
    for ki, key in enumerate(merged_keys):
        w_pre = base.get(key).float()
        u = updates[key].float()
        ratio = ratios[key].float()
        lyr = _layer_of(key)
        cf = cfrac.get(lyr, 0.0)            # non-layer keys -> 0 -> HIGH cap
        for m, (h, lo, t) in zip(outs, triples):
            cap = lo if cf >= t else h
            a_p = ratio.clamp(min=1.0, max=cap)
            m[key] = (w_pre + a_p * u).to(base.get(key).dtype)
        if (ki + 1) % 50 == 0 or (ki + 1) == len(merged_keys):
            print(f"  [lscale] {ki + 1}/{len(merged_keys)} keys", flush=True)

    for m, sd in zip(outs, save_dirs):
        save_merged(m, sd, aux_src_dir=meta["base_dir"])
        print(f"  [lscale] saved -> {sd}", flush=True)
    with open(args.manifest, "w") as f:
        f.write("\n".join(f"{tg} {sd}" for tg, sd in zip(tags, save_dirs)) + "\n")
    print(f"[lscale] manifest -> {args.manifest}", flush=True)


if __name__ == "__main__":
    main()
