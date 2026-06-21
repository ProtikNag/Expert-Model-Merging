"""DERIVE the per-expert coefficients of the champion from a curvature/Taylor
surrogate, instead of hand-sweeping them.

The champion is per-EXPERT scaled task arithmetic
    w(s) = w_pre + sum_i s_i * tau_i,     tau_i = w_i - w_pre,
and the winning vector s = (0.8, 0.4, 0.4, 0.4, 0.4) was picked by a 2-D grid
on the gate (only instruction & coding were swept; the other three sit at the
0.4 baseline). This script replaces that grid with a principled solve.

Method (the "fold into the curvature framework" plan, form (2) = data-light
surrogate). Taylor-expand the pooled multitask loss in the N-dim coefficient
space. With dw/ds_i = tau_i and per-example gradient g_n = grad_w l_n, the
DIRECTIONAL derivative of example n along expert i is

    d_{n,i} = <g_n, tau_i>.                      (one backward pass + N dots)

The gradient and the Gauss-Newton (empirical-Fisher, PSD) curvature of the loss
in s-space are then just N-vectors / NxN matrices:

    grad_s,i = mean_n d_{n,i},
    A_{ij}   = mean_n d_{n,i} d_{n,j},

and the (ridge-regularised, trust-region-clipped) Newton step is

    delta = -(A + eps I)^{-1} grad_s,   s <- clip(s + delta).

Re-linearise K times (rebuild w(s), recompute grads). K=1 is the cheap
single-pass estimate; K=2-3 refines. This derives ALL N coefficients jointly,
and the OFF-DIAGONAL A_{ij} captures the instruction<->coding cross-talk the
weight-space probes (Bets A/B) exposed -- the exact coupling the scalar grid
could not model.

Curvature note: this is the per-expert collapse of WHC's per-parameter Hessian
weighting -- WHC keeps full diag H_i but one GLOBAL scale; this keeps a free
per-EXPERT scale but collapses H to its bilinear form on the N tau directions.
Form (1) (per-expert scalar x per-param curvature) is the natural next step.

Data: a tiny per-domain validation buffer (MergeBench/<domain>_val) -> this is a
DATA-LIGHT win axis, not fully dataless. Causal-LM CE loss, base tokenizer, eager
attention -- identical to scripts/mb_fisher_estimate.py. Runs in the `merging` env.

Memory: base model lives on the GPU as trainable params (~16 GB bf16 + grads on a
48 GB L40S, as the Fisher estimator already proves at 8B). The N task vectors are
cached on CPU (bf16, ~16 GB each) and streamed per-parameter for the projection,
so only one tau slice is on the GPU at a time. node493 (257 GB RAM) holds them.

Usage
-----
    python -u scripts/mb_fit_perexpert_surrogate.py \
        --config configs/mergebench_tier2.yaml \
        --init-from 0.4,0.4,0.4,0.4,0.4 \
        --n-per-domain 32 --max-len 512 --steps 1 \
        --out-tag ta_pe_surrogate \
        --manifest mb_merged/Llama-3.1-8B/surrogate.txt --merge

Then evaluate the merged tag through the normal gate/forgetting drivers and
compare s* to the hand-swept champion (0.8, 0.4, 0.4, 0.4, 0.4).
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mergebench.io_utils import ShardedStateReader  # noqa: E402
from mergebench.llm_merge import merge_task_arith_perexpert_multi  # noqa: E402
from scripts.mb_download import expert_repo, local_dir_for  # noqa: E402
from scripts.mb_fisher_estimate import example_to_text, set_seed  # noqa: E402
from src.utils import load_config  # noqa: E402

_DEFAULT_VAL = "MergeBench/{domain}_val"   # MergeBench's <domain>_val convention


def _parse_datasets(spec: str | None, domains: list[str]) -> dict[str, str]:
    """`d1=repo1;d2=repo2` -> map; unlisted domains fall back to <domain>_val."""
    out = {d: _DEFAULT_VAL.format(domain=d) for d in domains}
    if spec:
        for part in spec.split(";"):
            if not part.strip():
                continue
            dom, repo = part.split("=", 1)
            if dom not in out:
                raise ValueError(f"--datasets domain {dom!r} not in {domains}")
            out[dom] = repo
    return out


def _load_buffer(datasets: dict[str, str], tok, n_per_domain: int, max_len: int,
                 split: str, text_field: str | None, seed: int):
    """Pooled list of (domain_idx, input_ids[1,T]) over all domains."""
    buf = []
    for di, (dom, repo) in enumerate(datasets.items()):
        ds = load_dataset(repo, split=split)
        if len(ds) > n_per_domain:
            ds = ds.shuffle(seed=seed).select(range(n_per_domain))
        kept = 0
        for ex in ds:
            enc = tok(example_to_text(ex, text_field), return_tensors="pt",
                      truncation=True, max_length=max_len)
            ids = enc["input_ids"]
            if ids.shape[1] < 2:
                continue
            buf.append((di, ids))
            kept += 1
        print(f"  [buffer] {dom}: {kept} examples from {repo}", flush=True)
    return buf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--domains", default=None,
                    help="Comma list; default cfg['tier_domains'] (expert order).")
    ap.add_argument("--datasets", default=None,
                    help="d1=repo1;d2=repo2 ; unlisted -> MergeBench/<domain>_val.")
    ap.add_argument("--split", default="train")
    ap.add_argument("--text-field", default=None)
    ap.add_argument("--init-from", default=None,
                    help="Comma list of N initial s_i (default: all --base).")
    ap.add_argument("--base", type=float, default=0.4,
                    help="Initial s for every expert if --init-from is omitted.")
    ap.add_argument("--n-per-domain", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--steps", type=int, default=1,
                    help="Re-linearisation steps (K). 1 = single-pass estimate.")
    ap.add_argument("--ridge", type=float, default=1e-2,
                    help="Tikhonov ridge added to A before the solve.")
    ap.add_argument("--step-clip", type=float, default=0.3,
                    help="Trust region: max |delta_i| per coordinate per step.")
    ap.add_argument("--s-min", type=float, default=0.0)
    ap.add_argument("--s-max", type=float, default=1.5)
    ap.add_argument("--domain-weights", default=None,
                    help="Comma list of N weights for the pooled loss (default equal).")
    ap.add_argument("--kl-mix", default=None,
                    help="teacher_mix only: per-domain KL fraction lambda_i in [0,1]. "
                         "loss_i = (1-lam_i)*CE_hard + lam_i*KL_soft. Comma list of N, "
                         "or a single scalar broadcast to all. Default '0,0,0,1,1' = "
                         "HARD (argmax-aligned, aggressive) for the generative domains "
                         "instruction/math/coding whose metrics reward token alignment, "
                         "SOFT-KL (distribution-faithful, conservative) for safety & "
                         "multilingual whose metrics (RTA / acc_norm) reward matching the "
                         "expert's full distribution -- this is what stops the hard "
                         "objective from over-scaling safety and collapsing refusal.")
    ap.add_argument("--soft-discrepancy", default="kl",
                    choices=["kl", "js", "logit_mse", "feature_mse"],
                    help="teacher_mix: which statistical discrepancy realises the SOFT "
                         "'mimic the expert' term (KL is just one member of the family). "
                         "'kl': forward KL(expert||merged) = mode-covering cross-entropy. "
                         "'js': Jensen-Shannon -- symmetric, bounded, more stable than KL "
                         "when the merged distribution is far off. 'logit_mse': L2 between "
                         "raw logit vectors -- matches the expert's confidence MAGNITUDES, "
                         "not just argmax (softmax-invariant info KL discards). "
                         "'feature_mse': cosine/L2 on the FINAL hidden state -- forces the "
                         "merged model's internal representation to mimic the expert's, the "
                         "richest behavioural signal (refusal is decided in representation "
                         "space). All but feature_mse share the cached teacher logits.")
    ap.add_argument("--objective", default="teacher_hard",
                    choices=["teacher_hard", "teacher_kl", "teacher_mix", "ce"],
                    help="'teacher_hard' (default): per domain, CE of merged onto the "
                         "domain expert's GREEDY (argmax) token. Hard labels don't "
                         "penalise distributional sharpening until a token actually "
                         "flips, so the optimum sits at the more aggressive, argmax-"
                         "ALIGNED coefficients the benchmark rewards (soft-KL's optimum "
                         "is too conservative -- metric mismatch). "
                         "'teacher_kl': soft KL to the expert's full distribution -- "
                         "optimum below 0.4, under-merges. 'ce': next-token CE on val "
                         "text -- basin is the BASE model. The last two are kept as the "
                         "ablations that exhibit exactly those failure modes.")
    ap.add_argument("--attribution", default="own_domain",
                    choices=["own_domain", "pooled"],
                    help="How each example informs the s-coordinates. 'own_domain' "
                         "(default): an example from domain i updates ONLY s_i "
                         "(block-diagonal Gauss-Newton) -- each expert solved on its "
                         "OWN turf, so s_i is pushed up to recover expert_i and is "
                         "NOT penalised for perturbing the other 4 domains. This is "
                         "the per-expert decoupling that wins. 'pooled': every "
                         "example contributes to every coordinate (true gradient of "
                         "the summed loss); with 5 experts this is interference-"
                         "dominated and shrinks all s toward 0 -- the ablation.")
    ap.add_argument("--dump-derivatives", default=None,
                    help="npz path: at step 0, save per-example d=<grad,tau_i> "
                         "(n_ex x N), domain idx, and loss. Lets the own-vs-cross "
                         "interference weighting (mu) be swept in numpy offline, "
                         "without re-running the model. Use with --attribution pooled.")
    ap.add_argument("--freeze", default=None,
                    help="Comma list of domain names whose coordinate is HELD at its "
                         "init value (delta zeroed every step). Metric-aware curvature: "
                         "the distillation surrogate is faithful only to distribution-"
                         "matching metrics (multilingual acc_norm, safety RTA); on the "
                         "argmax-generative domains (instruction/math/coding) it is anti-"
                         "faithful, so freeze those at the hand-tuned champion and let "
                         "the second-order solve refine ONLY the coords it can track.")
    ap.add_argument("--out-tag", default="ta_pe_surrogate")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--merge", action="store_true",
                    help="After the fit, write the merged ckpt at s* (else just report).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    cfg = load_config(args.config)
    download_dir = Path(cfg["download_dir"])
    domains = (args.domains.split(",") if args.domains else cfg["tier_domains"])
    n = len(domains)
    frozen = np.zeros(n, dtype=bool)
    if args.freeze:
        fset = {x.strip() for x in args.freeze.split(",") if x.strip()}
        unknown = fset - set(domains)
        if unknown:
            raise ValueError(f"--freeze names not in domains {domains}: {unknown}")
        frozen = np.array([d in fset for d in domains], dtype=bool)
    base_dir = str(local_dir_for(download_dir, cfg["base_model"]))
    expert_dirs = [str(local_dir_for(download_dir, expert_repo(cfg["base_name"], d)))
                   for d in domains]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    s = (np.array([float(x) for x in args.init_from.split(",")], dtype=np.float64)
         if args.init_from else np.full(n, args.base, dtype=np.float64))
    if s.shape[0] != n:
        raise ValueError(f"--init-from needs {n} values, got {s.shape[0]}")
    dw = (np.array([float(x) for x in args.domain_weights.split(",")], dtype=np.float64)
          if args.domain_weights else np.ones(n, dtype=np.float64))
    if dw.shape[0] != n:
        raise ValueError(f"--domain-weights needs {n} values, got {dw.shape[0]}")

    # Per-domain KL/soft mixing lambda_i for teacher_mix. Default: HARD for the
    # generative domains (instruction/math/coding -> argmax-aligned, aggressive),
    # SOFT-discrepancy for safety & multilingual (distribution-faithful,
    # conservative) -- the metric-matched split that fixes the safety collapse.
    if args.kl_mix is None:
        klm = np.array([0.0 if d in ("instruction", "math", "coding") else 1.0
                        for d in domains], dtype=np.float64)
    else:
        vals = [float(x) for x in args.kl_mix.split(",")]
        klm = (np.full(n, vals[0], dtype=np.float64) if len(vals) == 1
               else np.array(vals, dtype=np.float64))
    if klm.shape[0] != n:
        raise ValueError(f"--kl-mix needs 1 or {n} values, got {klm.shape[0]}")
    klm = np.clip(klm, 0.0, 1.0)

    print(f"[surrogate] domains={domains} objective={args.objective} "
          f"attribution={args.attribution}", flush=True)
    if args.objective == "teacher_mix":
        print(f"[surrogate] teacher_mix: soft_discrepancy={args.soft_discrepancy} "
              f"kl_mix(lambda)={klm.tolist()}  (0=pure hard CE, 1=pure soft)", flush=True)
    print(f"[surrogate] init s={s.tolist()} steps={args.steps} ridge={args.ridge} "
          f"clip={args.step_clip} box=[{args.s_min},{args.s_max}]", flush=True)

    # --- task vectors tau_i and w_pre, cached on CPU (bf16) -------------------
    base = ShardedStateReader(base_dir)
    keys = base.keys()
    w_pre = {}      # name -> base param (bf16, CPU)
    taus = [{} for _ in range(n)]   # expert i -> {name -> tau (bf16, CPU)}
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

    # --- model on GPU (trainable), tokenizer, buffer -------------------------
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

    def set_weights(s_vec):
        """p.data <- w_pre + sum_i s_i tau_i, streamed per parameter."""
        with torch.no_grad():
            for k in proj_keys:
                acc = w_pre[k].to(device, torch.float32)
                for i in range(n):
                    if s_vec[i] != 0.0:
                        acc += float(s_vec[i]) * taus[i][k].to(device, torch.float32)
                sd[k].data.copy_(acc.to(sd[k].dtype))
                del acc

    # --- teacher targets: each domain's expert logits on its own data --------
    # expert_i = w_pre + 1.0 * tau_i, i.e. set_weights(e_i). We reuse the cached
    # tau directions (NO extra checkpoint I/O) to materialise each expert in turn
    # and cache its logits (fp16, CPU) on the domain-i examples. These are the
    # distillation targets; minimising KL(expert_i || merged) on domain i has its
    # basin AT the experts, so the Newton step pushes s up toward them -- unlike
    # the CE objective whose basin is the base model.
    teachers: list = [None] * len(buf)
    feat = args.soft_discrepancy == "feature_mse"
    if args.objective in ("teacher_kl", "teacher_hard", "teacher_mix"):
        mix = args.objective == "teacher_mix"
        hard = args.objective == "teacher_hard"
        what = ("argmax labels" if hard else
                ("mix: argmax + per-domain soft target" if mix else "logits"))
        print(f"[surrogate] precomputing per-domain expert teacher {what} ...",
              flush=True)
        for i in range(n):
            unit = np.zeros(n, dtype=np.float64)
            unit[i] = 1.0
            set_weights(unit)
            cnt = 0
            with torch.no_grad():
                for j, (di, ids) in enumerate(buf):
                    if di != i:
                        continue
                    if not mix:
                        lg = model(input_ids=ids.to(device)).logits
                        # hard: greedy token ids (int, tiny); soft: fp16 logits
                        teachers[j] = (lg.argmax(-1).detach().to("cpu") if hard
                                       else lg.detach().to("cpu", torch.float16))
                    else:
                        out = model(input_ids=ids.to(device),
                                    output_hidden_states=feat)
                        lg = out.logits
                        tj = {"argmax": lg.argmax(-1).detach().to("cpu")}
                        if klm[di] > 0.0:   # only pay fp16 cache where soft is used
                            tj["soft"] = (out.hidden_states[-1].detach().to("cpu",
                                          torch.float16) if feat
                                          else lg.detach().to("cpu", torch.float16))
                        teachers[j] = tj
                    cnt += 1
            print(f"  [teacher] {domains[i]}: {cnt} examples cached", flush=True)

    # --- Gauss-Newton surrogate steps ---------------------------------------
    for step in range(args.steps):
        t0 = time.time()
        set_weights(s)
        grad_s = np.zeros(n, dtype=np.float64)
        A = np.zeros((n, n), dtype=np.float64)
        dcount = np.zeros(n, dtype=np.float64)   # own_domain: per-coordinate weight
        wsum = 0.0
        wtot = 0.0
        loss_sum = 0.0
        dump_d, dump_dom, dump_loss = [], [], []   # for --dump-derivatives (step 0)
        for j, (di, ids) in enumerate(buf):
            ids = ids.to(device)
            model.zero_grad(set_to_none=True)
            if args.objective == "ce":
                loss = model(input_ids=ids, labels=ids).loss
            elif args.objective == "teacher_hard":
                # CE of merged onto the expert's greedy tokens (argmax-aligned).
                logits = model(input_ids=ids).logits
                tgt = teachers[j].to(device)
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), tgt.view(-1))
            elif args.objective == "teacher_mix":
                # Per domain: (1-lam)*HARD argmax-CE + lam*SOFT behavioural mimicry.
                lam = float(klm[di])
                tj = teachers[j]
                need_hidden = feat and lam > 0.0
                out = (model(input_ids=ids, output_hidden_states=True)
                       if need_hidden else model(input_ids=ids))
                logits = out.logits.float()
                loss = logits.new_zeros(())
                if lam < 1.0:    # generative metrics reward argmax alignment
                    tgt = tj["argmax"].to(device)
                    loss = loss + (1.0 - lam) * torch.nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)), tgt.view(-1))
                if lam > 0.0:    # mimic the expert's full behaviour, conservatively
                    tsoft = tj["soft"].to(device).float()
                    if args.soft_discrepancy == "feature_mse":
                        cos = torch.nn.functional.cosine_similarity(
                            out.hidden_states[-1].float(), tsoft, dim=-1)
                        loss = loss + lam * (1.0 - cos).mean()
                    elif args.soft_discrepancy == "logit_mse":
                        loss = loss + lam * torch.nn.functional.mse_loss(logits, tsoft)
                    elif args.soft_discrepancy == "js":
                        lp_m = torch.log_softmax(logits, dim=-1)
                        lp_t = torch.log_softmax(tsoft, dim=-1)
                        p_m, p_t = lp_m.exp(), lp_t.exp()
                        logm = torch.log((0.5 * (p_m + p_t)).clamp_min(1e-9))
                        js = (0.5 * (p_m * (lp_m - logm)).sum(-1)
                              + 0.5 * (p_t * (lp_t - logm)).sum(-1))
                        loss = loss + lam * js.mean()
                    else:        # kl: forward KL(expert||merged) cross-entropy
                        tprob = torch.softmax(tsoft, dim=-1)
                        loss = loss + lam * -(
                            tprob * torch.log_softmax(logits, dim=-1)).sum(-1).mean()
            else:  # teacher_kl: soft cross-entropy to the domain expert's logits
                logits = model(input_ids=ids).logits.float()
                with torch.no_grad():
                    tprob = torch.softmax(teachers[j].to(device).float(), dim=-1)
                loss = -(tprob * torch.log_softmax(logits, dim=-1)).sum(-1).mean()
            loss.backward()
            # d_i = <grad, tau_i>.  Accumulate on-GPU and sync ONCE per example.
            # own_domain attribution only needs the OWN coordinate d[di], so we
            # project onto tau_di alone (5x less CPU->GPU transfer); pooled needs
            # the full d vector for the off-diagonal outer product.
            need = list(range(n)) if args.attribution == "pooled" else [di]
            d_gpu = torch.zeros(n, device=device, dtype=torch.float32)
            with torch.no_grad():
                for k in proj_keys:
                    g = sd[k].grad
                    if g is None:
                        continue
                    gf = g.reshape(-1).float()
                    for i in need:
                        ti = taus[i][k].to(device, non_blocking=True).reshape(-1).float()
                        d_gpu[i] += torch.dot(gf, ti)
            d = d_gpu.cpu().numpy().astype(np.float64)
            w = float(dw[di])
            if args.attribution == "own_domain":
                # example in domain di informs ONLY s_di (block-diagonal GN)
                grad_s[di] += w * d[di]
                A[di, di] += w * d[di] * d[di]
                dcount[di] += w
            else:
                grad_s += w * d
                A += w * np.outer(d, d)
                wsum += w
            wtot += w
            loss_sum += w * loss.item()
            if args.dump_derivatives and step == 0:
                dump_d.append(d.copy()); dump_dom.append(di)
                dump_loss.append(float(loss.item()))
            if (j + 1) % 10 == 0:
                print(f"    [{step}] {j + 1}/{len(buf)} loss={loss.item():.4f}",
                      flush=True)
        if args.attribution == "own_domain":
            for i in range(n):
                if dcount[i] > 0:
                    grad_s[i] /= dcount[i]
                    A[i, i] /= dcount[i]
        else:
            grad_s /= wsum
            A /= wsum
        mean_loss = loss_sum / wtot

        delta = -np.linalg.solve(A + args.ridge * np.eye(n), grad_s)
        delta = np.clip(delta, -args.step_clip, args.step_clip)
        delta[frozen] = 0.0   # metric-aware: hold frozen coords at their init value
        s_new = np.clip(s + delta, args.s_min, args.s_max)
        print(f"[surrogate] step {step}: mean_loss={mean_loss:.4f} "
              f"|grad_s|={np.linalg.norm(grad_s):.3e} "
              f"dt={time.time() - t0:.1f}s", flush=True)
        print(f"            grad_s={np.round(grad_s, 5).tolist()}", flush=True)
        print(f"            A diag={np.round(np.diag(A), 5).tolist()}", flush=True)
        print(f"            delta ={np.round(delta, 4).tolist()}", flush=True)
        print(f"            s: {np.round(s, 4).tolist()} -> "
              f"{np.round(s_new, 4).tolist()}", flush=True)
        if args.dump_derivatives and step == 0 and dump_d:
            np.savez(args.dump_derivatives,
                     d=np.array(dump_d), domain=np.array(dump_dom),
                     loss=np.array(dump_loss), s0=s.copy(),
                     domains=np.array(domains))
            print(f"            [dump] {len(dump_d)} per-example derivatives -> "
                  f"{args.dump_derivatives}", flush=True)
        s = s_new

    print(f"\n[surrogate] DERIVED s* = "
          f"{ {d: round(float(v), 4) for d, v in zip(domains, s)} }", flush=True)
    print(f"[surrogate] champion (hand-swept) for reference: "
          f"instruction=0.8, coding=0.4, others=0.4", flush=True)

    # free the GPU model AND the cached CPU tensors before the (CPU) merge --
    # the merge reloads base+experts (~96 GB); holding taus(~80 GB)+teachers(~21 GB)
    # too pushed the job over the 200 GB cap (OOM-killed mid-save). Drop them now.
    import gc
    del model, sd, taus, w_pre, teachers, buf
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    if args.merge:
        merged_root = Path(cfg["paths"]["merged"]) / cfg["base_name"]
        merged_root.mkdir(parents=True, exist_ok=True)
        save_dir = str(merged_root / args.out_tag)
        coeffs = [float(x) for x in s]
        print(f"[surrogate] merging s* -> {save_dir}", flush=True)
        merge_task_arith_perexpert_multi(
            base_dir=base_dir, expert_dirs=expert_dirs,
            variants=[dict(tag=args.out_tag, coeffs=coeffs)], save_dirs=[save_dir])
        manifest = Path(args.manifest) if args.manifest else (
            merged_root / "surrogate.txt")
        with open(manifest, "w") as f:
            f.write(f"{args.out_tag} {save_dir}\n")
        print(f"[surrogate] manifest -> {manifest}", flush=True)
    else:
        print("[surrogate] --merge not set; reporting s* only (no checkpoint written).",
              flush=True)


if __name__ == "__main__":
    main()
