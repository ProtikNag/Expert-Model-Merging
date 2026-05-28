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
ability — both come from the single coding expert's domain, not two coding experts.
All eval in bf16, eager attention, base tokenizer. Single seed.

The **specialist** rows evaluate each individual expert alone (no merging): they
are the per-domain ceilings (the expert in its own domain) and floors (the expert
out of its domain). The **dataless merge** rows combine both experts using
weights-only methods. **whc_diag** is ours.

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
<tr><td rowspan="6">Dataless merge<br>(weights only)</td><td>Consensus</td><td>33.1</td><td>17.9</td><td>34.9</td><td>28.6</td></tr>
<tr><td>LocalizeAndStitch</td><td>42.2</td><td>20.5</td><td>36.6</td><td>33.1</td></tr>
<tr><td>TIES</td><td>42.3</td><td>22.2</td><td>37.5</td><td>34.0</td></tr>
<tr><td>task_arith (cross-check)</td><td>42.4</td><td>23.8</td><td>38.4</td><td>34.9</td></tr>
<tr><td>DARE</td><td>43.1</td><td>24.3</td><td>38.5</td><td>35.3</td></tr>
<tr><td>TaskArithmetic (MergeBench)</td><td>42.2</td><td>24.7</td><td>39.3</td><td>35.4</td></tr>
<tr><td>Ours</td><td><b>whc_diag</b></td><td><b>47.2</b></td><td><b>26.9</b></td><td><b>40.4</b></td><td><b>38.2</b></td></tr>
</tbody>
</table>

Bold = best in column. Among the *merge* methods, whc_diag is first on all three
benchmarks and has the best cross-domain average (38.2), exceeding even each single
expert's average (math 35.5, coding 34.6) — the merged model is the only one strong
across both domains.

## Reading the result

- **WHC tops every dataless baseline on all three benchmarks**, with a consistent
  field ordering (Consensus < L&S < TIES < TA/DARE cluster < WHC).
- **Significance:** gsm8k +4.2 over the best baseline (~2σ at n=1319,
  significant); humaneval+ +2.2 and mbpp+ +1.1 are directional (sub-1σ at 164 /
  378 problems). The case rests on consistency across benchmarks plus the
  significant math margin.
- **Specialist retention:** WHC keeps ~84% of the specialist ceiling on gsm8k
  (47.2/56.0) and humaneval+ (26.9/32.1), and **ties the coding specialist on
  mbpp+** (40.4 vs 39.8).
- **Cross-check:** task_arith (ours) ≈ TaskArithmetic (MergeBench) within ~1 point
  on all three, validating the merge pipeline.
- **Specialists confirm the brackets:** each expert wins its own domain (math
  expert 56.0 on gsm8k; coding expert 32.1/39.8 on coding) and drops to ~base
  off-domain (math expert 18.1 on humaneval+; coding expert 31.8 on gsm8k).

## Scope

Dataless tier only (Option B, whc_diag). Not yet compared against the data tier
(Fisher, RegMean, RegMean++) or the iterative `whc_tree` variant. Single seed,
single base, two domains. See HANDOFF.md for the parked follow-ups.

## Raw values (fractions, for reproducibility)

```
model               gsm8k     humaneval+  mbpp+
base                0.2813    0.1567      0.3339
math_expert         0.5603    0.1811      0.3233
coding_expert       0.3184    0.3207      0.3976
Consensus           0.3306    0.1793      0.3492
LocalizeAndStitch   0.4223    0.2049      0.3656
TIES                0.4230    0.2220      0.3749
task_arith          0.4238    0.2378      0.3844
DARE                0.4306    0.2433      0.3852
TaskArithmetic      0.4215    0.2470      0.3934
whc_diag            0.4723    0.2689      0.4040
```
