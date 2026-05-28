# Tier 1 Results — WHC vs. Baselines (Gemma-2-2B, math + coding)

**Setup.** Two expert checkpoints from MergeBench, both fine-tuned from the same
`google/gemma-2-2b` base, are merged into a single model:

- **math expert** — fine-tuned on DartMath (math reasoning).
- **coding expert** — fine-tuned on MagiCoder (code generation).

The merged model is evaluated on **three benchmarks spanning the two domains**:

| Domain | Benchmark | What it measures | Metric |
|---|---|---|---|
| Math | `gsm8k` (CoT, 8-shot) | grade-school word problems | exact-match accuracy, full 1319 |
| Coding | `humanevalplus` | function completion from docstring | pass@1, n_samples=10 |
| Coding | `mbppplus` | basic Python tasks from NL spec | pass@1, n_samples=10 |

So there is **one math expert and one coding expert** (two experts total). The two
coding columns (`humaneval+`, `mbpp+`) are two *benchmarks* of the same coding
ability, both from the single coding expert's domain. All eval in bf16, eager
attention, base tokenizer. Single seed.

**Tiers.** *Dataless* merges use weights only. *Data* merges use the per-expert
diagonal Fisher (estimated here on 256 samples of each expert's own task data).
`whc_diag` is ours (dataless); `whc_diag_fisher` is the same anchored formula with
true Fisher instead of the dataless task-vector proxy; `fisher_merge` is plain
Fisher-weighted averaging (Matena & Raffel 2022) with no anchor.

## Results (pass@1 / accuracy, %)

<table>
<thead>
<tr>
<th rowspan="2">Category</th><th rowspan="2">Model</th>
<th>Math</th><th colspan="2">Coding</th><th rowspan="2">Avg</th>
</tr>
<tr><th>gsm8k</th><th>humaneval+</th><th>mbpp+</th></tr>
</thead>
<tbody>
<tr><td>Reference</td><td>base (no merge)</td><td>28.1</td><td>15.7</td><td>33.4</td><td>25.7</td></tr>
<tr><td rowspan="2">Specialist<br>(single expert,<br>no merge)</td><td>math expert</td><td><b>56.0</b></td><td>18.1</td><td>32.3</td><td>35.5</td></tr>
<tr><td>coding expert</td><td>31.8</td><td><b>32.1</b></td><td>39.8</td><td>34.6</td></tr>
<tr><td rowspan="7">Dataless merge<br>(weights only)</td><td>Consensus</td><td>33.1</td><td>17.9</td><td>34.9</td><td>28.6</td></tr>
<tr><td>LocalizeAndStitch</td><td>42.2</td><td>20.5</td><td>36.6</td><td>33.1</td></tr>
<tr><td>TIES</td><td>42.3</td><td>22.2</td><td>37.5</td><td>34.0</td></tr>
<tr><td>task_arith (cross-check)</td><td>42.4</td><td>23.8</td><td>38.4</td><td>34.9</td></tr>
<tr><td>DARE</td><td>43.1</td><td>24.3</td><td>38.5</td><td>35.3</td></tr>
<tr><td>TaskArithmetic (MergeBench)</td><td>42.2</td><td>24.7</td><td>39.3</td><td>35.4</td></tr>
<tr><td><b>whc_diag (ours)</b></td><td>47.2</td><td><b>26.9</b></td><td><b>40.4</b></td><td><b>38.2</b></td></tr>
<tr><td rowspan="2">Data merge<br>(uses Fisher)</td><td>fisher_merge (Matena&Raffel)</td><td>49.4</td><td>25.1</td><td>38.5</td><td>37.7</td></tr>
<tr><td>whc_diag_fisher (ours)</td><td>47.1</td><td>25.7</td><td>39.6</td><td>37.4</td></tr>
</tbody>
</table>

Bold = column max. `whc_diag` (dataless, ours) has the **best cross-domain average
of all 12 models** — above both Fisher variants and both specialists — and is the
column max on mbpp+. Each specialist wins only its own domain (math expert 56.0 on
gsm8k; coding expert 32.1 on humaneval+) and drops to ~base off-domain.

## Reading the result

**Dataless tier.** whc_diag beats every weights-only baseline on all three
benchmarks; field ordering is consistent (Consensus < L&S < TIES < TA/DARE cluster
< WHC). gsm8k margin +4.2 over best baseline (~2σ at n=1319, significant);
humaneval+ +2.2 and mbpp+ +1.1 are directional (sub-1σ).

**Dataless proxy vs true Fisher (the headline ablation).** whc_diag (taskvec,
dataless) vs whc_diag_fisher (true Fisher, same formula): 38.2 vs 37.4 avg — the
**free task-vector proxy matches/slightly beats true Fisher**, tying on gsm8k
(47.2 vs 47.1) and winning both coding benchmarks. Curvature-aware merging without
any data is as good as with it, here.

**Anchor.** whc_diag_fisher (anchor) vs fisher_merge (no anchor): 37.4 vs 37.7 —
roughly a wash; the anchor helps coding, slightly hurts math. (Our fisher_merge
includes a mean fallback for zero-Fisher keys, so it is not the pathological
no-anchor variant.)

**Data helps on math, not coding.** Plain fisher_merge is the best *merge* on
gsm8k (49.4, above dataless whc_diag's 47.2) — the Fisher's data signal helps math
— but it trails the dataless proxy on both coding benchmarks and on average.

**Cross-check.** task_arith (ours) ≈ TaskArithmetic (MergeBench) within ~1 point on
all three, validating the merge pipeline.

## Caveats

- The Fisher was estimated on **256 samples** per expert (MergeBench used 1000);
  more samples could lift the Fisher variants. The "dataless ≥ Fisher" reading is
  conditioned on this estimate.
- Single seed, single base (gemma-2-2b), two domains.
- Not yet run: RegMean/RegMean++ (the other data-tier precursor), the iterative
  `whc_tree` variant, multi-domain (N=5), larger bases. See HANDOFF.md.

## Raw values (%, from the eval readers)

```
model               gsm8k   humaneval+   mbpp+
base                28.1    15.7         33.4
math_expert         56.0    18.1         32.3
coding_expert       31.8    32.1         39.8
Consensus           33.1    17.9         34.9
LocalizeAndStitch   42.2    20.5         36.6
TIES                42.3    22.2         37.5
task_arith          42.4    23.8         38.4
DARE                43.1    24.3         38.5
TaskArithmetic      42.2    24.7         39.3
whc_diag            47.2    26.9         40.4
fisher_merge        49.4    25.1         38.5
whc_diag_fisher     47.1    25.7         39.6
```
(gsm8k/coding raw fractions for the dataless rows are in git history / the eval
JSONs; values here are the reader's reported percentages.)
