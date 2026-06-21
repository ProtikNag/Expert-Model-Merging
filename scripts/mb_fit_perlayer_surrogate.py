"""PER-LAYER (per-block) curvature-aware merge — break the scalar interference wall.

The 5-scalar per-expert champion (ta_pe_inst0.8_codi0.4, avg ~55) is PARETO-OPTIMAL
in scalar space: instruction/math/coding want HIGH own-coefficient (generative — need
their expert present), while safety/multilingual want LOW interference from every other
expert (discriminative/refusal — degraded by merging). On a single shared scalar per
expert those two families pull in opposite directions, so no scalar point beats the
champion (proven: cold-pooled, frozen-refine, and clean ml-boost candidates all lose).

This script gives each (expert, transformer-block) pair its OWN coefficient. Hand-tuning
cannot sweep ~N*B coefficients; second-order curvature can. The bet: the safety/ml
interference is concentrated in SPECIFIC blocks, so the solve can keep a generative
expert strong in the blocks IT needs while shrinking its footprint in the blocks
safety/multilingual rely on — decoupling the two families that a scalar cannot separate.

Merge:  w = w_pre + sum_i sum_b  S[i,b] * tau_i  (restricted to block b's parameters)
with S initialised by broadcasting the champion's per-expert scalar across all blocks
(--n-blocks 1 reproduces the scalar method exactly). Curvature = empirical-Fisher over
the per-(expert,block) directional derivatives d[i,b] = <grad_blockb, tau_i|blockb>;
damped Newton step, trust-region clipped, re-linearised K times. The merged checkpoint
is saved straight from the in-memory model (no scalar-only external merge).

    conda activate /work/pnag/envs/merging
    python -u scripts/mb_fit_perlayer_surrogate.py --config configs/mergebench_tier2.yaml \
        --init-from 0.8,0.4,0.4,0.4,0.4 --n-blocks 8 \
        --objective teacher_mix --attribution pooled \
        --steps 4 --ridge 0.05 --step-clip 0.2 --out-tag ta_pl_b8 --merge
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mergebench.io_utils import ShardedStateReader  # noqa: E402
from scripts.mb_download import expert_repo, local_dir_for  # noqa: E402
from scripts.mb_fisher_estimate import set_seed  # noqa: E402
from src.utils import load_config  # noqa: E402
# reuse the proven buffer/dataset helpers from the scalar script
from scripts.mb_fit_perexpert_surrogate import _parse_datasets, _load_buffer  # noqa: E402

import re

_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


def build_block_map(keys, n_blocks):
    """Map each parameter key -> block id in [0, n_blocks). Transformer layer L is
    bucketed by L*n_blocks//num_layers; embeddings -> block 0; final norm / lm_head ->
    the last block. n_blocks==1 collapses everything to block 0 (== scalar method)."""
    layer_ids = {}
    max_layer = -1
    for k in keys:
        m = _LAYER_RE.search(k)
        if m:
            L = int(m.group(1))
            layer_ids[k] = L
            max_layer = max(max_layer, L)
    num_layers = max_layer + 1 if max_layer >= 0 else 1
    block_of = {}
    for k in keys:
        if k in layer_ids:
            b = layer_ids[k] * n_blocks // num_layers
        elif "embed" in k:
            b = 0
        else:                       # model.norm, lm_head, anything trailing
            b = n_blocks - 1
        block_of[k] = min(max(b, 0), n_blocks - 1)
    return block_of, num_layers


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--domains", default=None)
    ap.add_argument("--datasets", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--text-field", default=None)
    ap.add_argument("--init-from", default=None,
                    help="Comma list of N per-expert scalars, broadcast across blocks.")
    ap.add_argument("--base", type=float, default=0.4)
    ap.add_argument("--n-blocks", type=int, default=8,
                    help="Number of contiguous transformer blocks (DOF = N*n_blocks). "
                         "1 == the scalar per-expert method.")
    ap.add_argument("--n-per-domain", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--ridge", type=float, default=5e-2)
    ap.add_argument("--step-clip", type=float, default=0.2)
    ap.add_argument("--s-min", type=float, default=0.0)
    ap.add_argument("--s-max", type=float, default=1.5)
    ap.add_argument("--kl-mix", default=None,
                    help="teacher_mix per-domain lambda. Default 0,0,0,1,1.")
    ap.add_argument("--soft-discrepancy", default="kl",
                    choices=["kl", "js", "logit_mse", "feature_mse"])
    ap.add_argument("--objective", default="teacher_mix",
                    choices=["teacher_hard", "teacher_mix", "ce"])
    ap.add_argument("--attribution", default="pooled",
                    choices=["own_domain", "pooled"])
    ap.add_argument("--freeze", default=None,
                    help="Comma domain names held at init across ALL their blocks.")
    ap.add_argument("--out-tag", default="ta_pl_surrogate")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    cfg = load_config(args.config)
    download_dir = Path(cfg["download_dir"])
    domains = (args.domains.split(",") if args.domains else cfg["tier_domains"])
    n = len(domains)
    B = args.n_blocks
    base_dir = str(local_dir_for(download_dir, cfg["base_model"]))
    expert_dirs = [str(local_dir_for(download_dir, expert_repo(cfg["base_name"], d)))
                   for d in domains]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    s_init = (np.array([float(x) for x in args.init_from.split(",")], dtype=np.float64)
              if args.init_from else np.full(n, args.base, dtype=np.float64))
    if s_init.shape[0] != n:
        raise ValueError(f"--init-from needs {n} values, got {s_init.shape[0]}")
    # S[i, b] coefficient matrix; start by broadcasting the per-expert scalar.
    S = np.repeat(s_init[:, None], B, axis=1)            # (n, B)

    frozen = np.zeros(n, dtype=bool)
    if args.freeze:
        fset = {x.strip() for x in args.freeze.split(",") if x.strip()}
        unknown = fset - set(domains)
        if unknown:
            raise ValueError(f"--freeze names not in {domains}: {unknown}")
        frozen = np.array([d in fset for d in domains], dtype=bool)

    if args.kl_mix is None:
        klm = np.array([0.0 if d in ("instruction", "math", "coding") else 1.0
                        for d in domains], dtype=np.float64)
    else:
        vals = [float(x) for x in args.kl_mix.split(",")]
        klm = (np.full(n, vals[0]) if len(vals) == 1 else np.array(vals, dtype=np.float64))
    klm = np.clip(klm, 0.0, 1.0)

    print(f"[perlayer] domains={domains} n_blocks={B} DOF={n*B} obj={args.objective} "
          f"attr={args.attribution} freeze={args.freeze or '-'}", flush=True)
    print(f"[perlayer] init per-expert s={s_init.tolist()} klm={klm.tolist()} "
          f"steps={args.steps} ridge={args.ridge} clip={args.step_clip}", flush=True)

    # --- task vectors tau_i and w_pre (CPU bf16) -----------------------------
    base = ShardedStateReader(base_dir)
    keys = base.keys()
    w_pre, taus = {}, [{} for _ in range(n)]
    for i, ed in enumerate(expert_dirs):
        er = ShardedStateReader(ed)
        cnt = 0
        for k in keys:
            bt = base.get(k)
            if not bt.dtype.is_floating_point or not er.has(k) or er.get(k).shape != bt.shape:
                continue
            if i == 0:
                w_pre[k] = bt.to(torch.bfloat16)
            taus[i][k] = (er.get(k).float() - bt.float()).to(torch.bfloat16)
            cnt += 1
        del er
        print(f"  [tau] {domains[i]}: {cnt} float keys cached", flush=True)
    proj_keys = list(w_pre.keys())
    block_of, num_layers = build_block_map(proj_keys, B)
    blk_idx = np.array([block_of[k] for k in proj_keys])
    counts = np.bincount(blk_idx, minlength=B)
    print(f"[perlayer] {num_layers} transformer layers -> {B} blocks; "
          f"keys/block={counts.tolist()}", flush=True)

    # --- model, tokenizer, buffer --------------------------------------------
    tok = AutoTokenizer.from_pretrained(base_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_dir, torch_dtype=torch.bfloat16, attn_implementation="eager").to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)
    sd = dict(model.named_parameters())

    datasets = _parse_datasets(args.datasets, domains)
    buf = _load_buffer(datasets, tok, args.n_per_domain, args.max_len,
                       args.split, args.text_field, args.seed)
    if not buf:
        raise RuntimeError("Empty buffer; check --datasets / --text-field.")

    def set_weights(S_mat):
        """p.data <- w_pre + sum_i S[i, block(k)] * tau_i[k], streamed per param."""
        with torch.no_grad():
            for k in proj_keys:
                b = block_of[k]
                acc = w_pre[k].to(device, torch.float32)
                for i in range(n):
                    c = float(S_mat[i, b])
                    if c != 0.0:
                        acc += c * taus[i][k].to(device, torch.float32)
                sd[k].data.copy_(acc.to(sd[k].dtype))
                del acc

    # --- teacher precompute (pure expert i = unit coeff everywhere) ----------
    feat = args.soft_discrepancy == "feature_mse"
    teachers = {}
    if args.objective in ("teacher_hard", "teacher_mix"):
        mix = args.objective == "teacher_mix"
        print("[perlayer] precomputing per-domain expert teachers ...", flush=True)
        for i in range(n):
            unit = np.zeros((n, B)); unit[i, :] = 1.0
            set_weights(unit)
            cnt = 0
            with torch.no_grad():
                for j, (di, ids) in enumerate(buf):
                    if di != i:
                        continue
                    if not mix:
                        lg = model(input_ids=ids.to(device)).logits
                        teachers[j] = lg.argmax(-1).detach().to("cpu")
                    else:
                        out = model(input_ids=ids.to(device), output_hidden_states=feat)
                        lg = out.logits
                        tj = {"argmax": lg.argmax(-1).detach().to("cpu")}
                        if klm[di] > 0.0:
                            tj["soft"] = (out.hidden_states[-1].detach().to("cpu", torch.float16)
                                          if feat else lg.detach().to("cpu", torch.float16))
                        teachers[j] = tj
                    cnt += 1
            print(f"  [teacher] {domains[i]}: {cnt} cached", flush=True)

    def compute_loss(di, ids):
        if args.objective == "ce":
            return model(input_ids=ids, labels=ids).loss
        if args.objective == "teacher_hard":
            logits = model(input_ids=ids).logits
            tgt = teachers[J].to(device)
            return torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), tgt.view(-1))
        # teacher_mix
        lam = float(klm[di]); tj = teachers[J]
        need_hidden = feat and lam > 0.0
        out = (model(input_ids=ids, output_hidden_states=True) if need_hidden
               else model(input_ids=ids))
        logits = out.logits.float()
        loss = logits.new_zeros(())
        if lam < 1.0:
            tgt = tj["argmax"].to(device)
            loss = loss + (1.0 - lam) * torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), tgt.view(-1))
        if lam > 0.0:
            tsoft = tj["soft"].to(device).float()
            if args.soft_discrepancy == "feature_mse":
                cos = torch.nn.functional.cosine_similarity(
                    out.hidden_states[-1].float(), tsoft, dim=-1)
                loss = loss + lam * (1.0 - cos).mean()
            elif args.soft_discrepancy == "logit_mse":
                loss = loss + lam * torch.nn.functional.mse_loss(logits, tsoft)
            elif args.soft_discrepancy == "js":
                lp_m = torch.log_softmax(logits, dim=-1); lp_t = torch.log_softmax(tsoft, dim=-1)
                p_m, p_t = lp_m.exp(), lp_t.exp()
                logm = torch.log((0.5 * (p_m + p_t)).clamp_min(1e-9))
                js = (0.5 * (p_m * (lp_m - logm)).sum(-1) + 0.5 * (p_t * (lp_t - logm)).sum(-1))
                loss = loss + lam * js.mean()
            else:
                tprob = torch.softmax(tsoft, dim=-1)
                loss = loss + lam * -(tprob * torch.log_softmax(logits, dim=-1)).sum(-1).mean()
        return loss

    # --- per-(expert,block) Gauss-Newton steps -------------------------------
    NB = n * B
    for step in range(args.steps):
        t0 = time.time()
        set_weights(S)
        grad = np.zeros((n, B), dtype=np.float64)
        if args.attribution == "pooled":
            A = np.zeros((NB, NB), dtype=np.float64)
        else:
            Adiag = np.zeros((n, B), dtype=np.float64)
        wtot = 0.0; loss_sum = 0.0
        for J, (di, ids) in enumerate(buf):
            ids = ids.to(device)
            model.zero_grad(set_to_none=True)
            loss = compute_loss(di, ids)
            loss.backward()
            need = list(range(n)) if args.attribution == "pooled" else [di]
            d_gpu = torch.zeros(n, B, device=device, dtype=torch.float32)
            with torch.no_grad():
                for k in proj_keys:
                    g = sd[k].grad
                    if g is None:
                        continue
                    b = block_of[k]
                    gf = g.reshape(-1).float()
                    for i in need:
                        ti = taus[i][k].to(device, non_blocking=True).reshape(-1).float()
                        d_gpu[i, b] += torch.dot(gf, ti)
            d = d_gpu.cpu().numpy().astype(np.float64)      # (n, B)
            if args.attribution == "own_domain":
                grad[di] += d[di]
                Adiag[di] += d[di] * d[di]
            else:
                df = d.reshape(-1)
                grad += d
                A += np.outer(df, df)
            wtot += 1.0
            loss_sum += float(loss.item())
            if (J + 1) % 10 == 0:
                print(f"    [{step}] {J + 1}/{len(buf)} loss={loss.item():.4f}", flush=True)

        grad /= wtot
        mean_loss = loss_sum / wtot
        if args.attribution == "pooled":
            A /= wtot
            delta = -np.linalg.solve(A + args.ridge * np.eye(NB), grad.reshape(-1))
            delta = delta.reshape(n, B)
        else:
            Adiag /= wtot
            delta = -grad / (Adiag + args.ridge)
        delta = np.clip(delta, -args.step_clip, args.step_clip)
        delta[frozen, :] = 0.0
        S_new = np.clip(S + delta, args.s_min, args.s_max)
        print(f"[perlayer] step {step}: mean_loss={mean_loss:.4f} "
              f"|grad|={np.linalg.norm(grad):.3e} dt={time.time() - t0:.1f}s", flush=True)
        for i in range(n):
            print(f"    {domains[i]:12s} S {np.round(S[i], 3).tolist()} -> "
                  f"{np.round(S_new[i], 3).tolist()}", flush=True)
        S = S_new

    print("\n[perlayer] DERIVED per-expert per-block S*:", flush=True)
    for i in range(n):
        print(f"    {domains[i]:12s} mean={S[i].mean():.3f} "
              f"range=[{S[i].min():.3f},{S[i].max():.3f}] {np.round(S[i], 3).tolist()}",
              flush=True)

    if args.merge:
        merged_root = Path(cfg["paths"]["merged"]) / cfg["base_name"]
        merged_root.mkdir(parents=True, exist_ok=True)
        save_dir = str(merged_root / args.out_tag)
        print(f"[perlayer] applying S* and saving in-memory model -> {save_dir}", flush=True)
        set_weights(S)
        model.save_pretrained(save_dir, safe_serialization=True)
        tok.save_pretrained(save_dir)
        np.save(str(Path(save_dir) / "perblock_S.npy"), S)
        manifest = Path(args.manifest) if args.manifest else (merged_root / "perlayer.txt")
        with open(manifest, "w") as f:
            f.write(f"{args.out_tag} {save_dir}\n")
        print(f"[perlayer] manifest -> {manifest}", flush=True)
    else:
        print("[perlayer] --merge not set; reporting S* only.", flush=True)


if __name__ == "__main__":
    main()
