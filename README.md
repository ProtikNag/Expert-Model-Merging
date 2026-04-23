# Expert-Model-Merging: WHC vs. Recent Model Merging Methods

Porting the consolidation primitives from HTCL / CRL (Nag et al., 2026) to
the **model-merging** problem, with a push to position the resulting method
(**WHC** — Weighted Hessian Consolidation) as a *dataless, one-shot,
curvature-aware* merger.

Reference papers in [`docs/`](docs/):

- HTCL and CRL (our derivation).
- Merging baselines: Fisher (Matena & Raffel, NeurIPS 2022), RegMean (Jin
  et al., ICLR 2023), MaTS (Tam et al., TMLR 2024), CAMEx (Nguyen et al.,
  ICLR 2025, SMoE-specific, not a direct comparison), RegMean++ (Nguyen et
  al., 2025 preprint).

## Pilot finding (RotatedMNIST, archived)

See [`results/pilot/ANALYSIS.md`](results/pilot/ANALYSIS.md) and
[`results/pilot/figures/`](results/pilot/figures/). Headline: both HTCL-A and
HTCL-B beat every baseline (Fisher, RegMean, TIES, Task Arithmetic, Simple
Averaging) by 1.5–2.3 pp average accuracy; the Fisher+εI ablation
established that ~75% of HTCL-A's gain over Fisher comes from the
ensemble-mean anchor, not from numerical stabilization.

## Main experiment (GLUE, GPU)

**Setup.** RoBERTa-base fine-tuned into 7 experts on {CoLA, SST-2, MRPC,
QQP, MNLI, QNLI, RTE}. One seed (per user request). Val/test split of each
task's standard validation set. Merged model is evaluated per-task with
its own task head.

**Methods.** Dataless tier: Simple Averaging, Task Arithmetic, TIES,
**WHC-A/B/C/D (dataless)**. Statistics-using tier: Fisher, Fisher+εI,
RegMean, RegMean++, **WHC-A (Fisher)** (ablation).

**WHC primitives.**

- **WHC-A** — Eq. 9 (CRL) closed form over all N experts.
- **WHC-B** — Eq. 9 applied recursively in a tree.
- **WHC-C** — tree using symmetric Eq. 6 (pair update) at each internal node.
- **WHC-D** — sequential Eq. 6 absorption over the N experts.

**Dataless curvature proxy.** Every WHC variant can use either the squared
task vector (default, dataless) or the empirical diagonal Fisher (ablation).

## Running the pipeline

### Smoke test (CPU, ~1–2 min)

```bash
python scripts/smoke_test.py
```

Runs the full pipeline with DistilBERT on 2 tasks with tiny subsets. Use
this to confirm your environment is wired up before submitting a GPU job.

### Full GLUE run (single GPU, ~12–16h)

Submit the provided SLURM script:

```bash
sbatch gpu_run.sh
```

Or run directly:

```bash
python scripts/lm_train_experts.py --config configs/glue_roberta.yaml --device cuda
python scripts/lm_run_merging.py   --config configs/glue_roberta.yaml --device cuda
python scripts/lm_make_figures.py  --config configs/glue_roberta.yaml
```

All scripts produce structured logs under [`results/logs/`](results/logs/):

- `*.log`          — full stdout/stderr tee of the run.
- `*.manifest.json` — host, platform, package versions, config hash, git HEAD.
- `*.metrics.jsonl` — every sweep, every epoch, every per-task metric,
  streamed as newline-delimited JSON so the analysis stage needs no re-run.

## Repository layout

```text
Expert-Model-Merging/
├── configs/
│   ├── glue_roberta.yaml      # Full GPU run
│   ├── glue_smoke.yaml        # CPU smoke test
│   └── pilot.yaml             # Archived RotatedMNIST pilot
├── src/
│   ├── glue_data.py           # GLUE loaders w/ tokenizer, val/test split
│   ├── lm_models.py           # Encoder + per-task head wrapper
│   ├── lm_train.py            # HF fine-tuning, diagonal Fisher, Grams, g
│   ├── logging_utils.py       # RunLogger (tee, manifest, metrics JSONL)
│   ├── metrics.py             # Accuracy, Matthews, F1, param-space stats
│   ├── utils.py               # set_seed, load_config, ensure_dir, ...
│   ├── data.py models.py train.py fisher.py   # pilot (RotatedMNIST)
│   └── merging/
│       ├── simple.py task_arith.py ties.py
│       ├── fisher_merge.py regmean.py regmean_plus.py
│       └── whc.py             # WHC-A/B/C/D, pair, taskvec proxy, Eq 6 + 9
├── scripts/
│   ├── lm_train_experts.py    # Stage 1
│   ├── lm_run_merging.py      # Stage 2
│   ├── lm_make_figures.py     # Stage 3
│   ├── smoke_test.py          # CPU end-to-end validation
│   └── (pilot scripts retained for archive)
├── results/
│   ├── logs/                  # Timestamped logs, manifests, metrics JSONL
│   ├── glue/                  # Full run outputs
│   │   ├── merge_results.json
│   │   └── figures/{png,svg}/
│   ├── smoke/                 # Smoke test outputs
│   └── pilot/                 # Archived pilot
├── checkpoints/
│   ├── glue/<task>/           # backbone.pt, head.pt, fisher.pt, grams.pt
│   ├── glue/pretrained_backbone.pt
│   └── smoke/
├── gpu_run.sh                 # SLURM script
├── docs/                      # Reference papers
├── NOTES.md                   # Math modifications when porting HTCL→WHC
└── README.md
```

## Metrics captured

During the run every scalar worth later analysis is either written into
`merge_results.json` or streamed to the metrics JSONL. Specifically:

- Per-task, per-method, per-hparam val and test primary metrics, plus
  auxiliary (accuracy / F1 / Matthews where applicable).
- Merge wall-clock per method and per hparam.
- Parameter-space L2 distances and cosine similarities from the merged
  model to (i) the pretrained init, (ii) the ensemble mean, (iii) each
  expert.
- Curvature statistics: per-expert mean/median/max/min/std, and the
  fraction of non-zero entries for both the task-vec proxy and the true
  Fisher.
- Gradient-at-optimum norm for each expert (sanity check).

## Figures

Produced in separate folders:

- `results/<exp>/figures/png/*.png` (300 dpi)
- `results/<exp>/figures/svg/*.svg` (vector)

Current figures:

- `fig_methods_bar`              — avg test primary per method, tier-grouped.
- `fig_methods_per_task_heatmap` — task × method heatmap.
- `fig_whc_lambda_sensitivity`   — WHC-A (dataless) λ sweep vs. baselines.
- `fig_taskvec_vs_fisher`        — dataless proxy vs. true Fisher scatter.
- `fig_param_space_positions`    — where each merged model lives.
- `fig_sweep_curves`             — per-method hparam sweep curves.

## Modifications tracking

Every non-trivial change to the HTCL / CRL math when porting to merging is
documented in [`NOTES.md`](NOTES.md), including:

- Eq. 6 and Eq. 9 interpretations in the parallel-merging setting.
- The dataless task-vector curvature proxy and its derivation.
- Option X (dataless, $g = 0$) vs. Option Y (empirical $g$) for Eq. 6.
- Bug fixes for baseline implementations (TIES global trim).

## What to look at after the GPU run finishes

1. `results/glue/merge_results.json` — single source of truth.
2. `results/glue/figures/png/fig_methods_bar.png` — headline.
3. `results/logs/lm_run_merging_*.metrics.jsonl` — streaming metrics.
4. `NOTES.md` §10 — the claims we set out to validate.
