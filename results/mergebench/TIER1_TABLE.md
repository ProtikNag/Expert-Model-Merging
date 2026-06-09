# Tier 1 Results — WHC vs. Baselines (Gemma-2-2B, math + coding)

**Setup.** Two expert checkpoints from MergeBench, both fine-tuned from the same
`google/gemma-2-2b` base, are merged into a single model:

- **math expert** — fine-tuned on DartMath (math reasoning).
- **coding expert** — fine-tuned on MagiCoder (code generation).

The merged model is evaluated on **three benchmarks spanning the two domains**:

| Domain | Benchmark | What it measures | Metric |
|---|---|---|---|
| Math | `gsm8k` (CoT, 8-shot) | grade-school word problems | exact-match accuracy, full 1319 |
| Coding | `humanevalplus` | function completion from docstring | pass@1 |
| Coding | `mbppplus` | basic Python tasks from NL spec | pass@1 |

So there is **one math expert and one coding expert** (two experts total). The two
coding columns (`humaneval+`, `mbpp+`) are two *benchmarks* of the same coding
ability, both from the single coding expert's domain. All eval in bf16, eager
attention, base tokenizer. Single seed.


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
<tr><td>Reference</td><td>base (google/gemma-2-2b, no merge)</td><td>28.1</td><td>15.7</td><td>33.4</td><td>25.7</td></tr>
<tr><td rowspan="2">Specialist<br>(single expert,<br>no merge)</td><td>math expert</td><td><b>56.0</b></td><td>18.1</td><td>32.3</td><td>35.5</td></tr>
<tr><td>coding expert</td><td>31.8</td><td><b>32.1</b></td><td>39.8</td><td>34.6</td></tr>
<tr><td rowspan="7">Dataless merge<br>(weights only)</td><td>Consensus</td><td>33.1</td><td>17.9</td><td>34.9</td><td>28.6</td></tr>
<tr><td>LocalizeAndStitch</td><td>42.2</td><td>20.5</td><td>36.6</td><td>33.1</td></tr>
<tr><td>TIES</td><td>42.3</td><td>22.2</td><td>37.5</td><td>34.0</td></tr>
<tr><td>task_arith (cross-check)</td><td>42.4</td><td>23.8</td><td>38.4</td><td>34.9</td></tr>
<tr><td>DARE</td><td>43.1</td><td>24.3</td><td>38.5</td><td>35.3</td></tr>
<tr><td>TaskArithmetic (MergeBench)</td><td>42.2</td><td>24.7</td><td>39.3</td><td>35.4</td></tr>
<tr><td><b>HTCL</b></td><td>47.2</td><td><b>26.9</b></td><td><b>40.4</b></td><td><b>38.2</b></td></tr>
<tr><td rowspan="2">Data merge<br>(uses Fisher)</td><td>fisher_merge (Matena&Raffel)</td><td>49.4</td><td>25.1</td><td>38.5</td><td>37.7</td></tr>
<tr><td>HTCL (Fisher)</td><td>47.1</td><td>25.7</td><td>39.6</td><td>37.4</td></tr>
</tbody>
</table>

Bold = column max. **HTCL** (dataless, ours) has the **best cross-domain average
of all 12 models** — above both Fisher variants and both specialists — and is the
column max on mbpp+. Each specialist wins only its own domain (math expert 56.0 on
gsm8k; coding expert 32.1 on humaneval+) and drops to ~base off-domain.

## Tier 2: all five domains on the largest base (Llama-3.1-8B)

Merge all five domain experts (instruction, math, coding, safety, multilingual)
from `meta-llama/Llama-3.1-8B` and evaluate on MergeBench's full protocol
(Table 3). Dataless tier only; RegMean/RegMean++ deferred to a later data tier.

**Per-domain metric.** Instruction = IFEval prompt-level accuracy. Math = GSM8k
exact-match (8-shot CoT). Coding = HumanEval+ / MBPP+ pass@1. Safety = mean over
WildGuardTest / HarmBench / DoAnythingNow RTA and XSTest accuracy. Multilingual =
mean over M_MMLU / M_ARC / M_Hellaswag across {fr, es, de, ru}. All values %. Single
seed. **Safety column pending** (4th eval env not yet built); Avg is over the four
filled domains. The `HTCL` row is the default config (lam=1e-4, alpha=1); the tuned
sweep result is below the table.

<table>
<thead>
<tr>
<th rowspan="2">Category</th><th rowspan="2">Model</th>
<th rowspan="2">Instruction</th><th rowspan="2">Math</th>
<th colspan="2">Coding</th><th rowspan="2">Safety</th>
<th rowspan="2">Multilingual</th><th rowspan="2">Avg</th>
</tr>
<tr><th>humaneval+</th><th>mbpp+</th></tr>
</thead>
<tbody>
<tr><td>Reference</td><td>Llama-3.1-8B (no merge)</td><td>8.9</td><td>56.3</td><td></td><td></td><td></td><td>54.0</td><td>39.7</td></tr>
<tr><td rowspan="7">Dataless merge<br>(weights only)</td><td>Consensus</td><td>25.1</td><td>78.2</td><td><b>49.8</b></td><td>54.1</td><td></td><td>52.5</td><td><b>51.9</b></td></tr>
<tr><td>LocalizeAndStitch</td><td>12.4</td><td>76.4</td><td></td><td></td><td></td><td>54.2</td><td>47.7</td></tr>
<tr><td>TIES</td><td>16.6</td><td>77.9</td><td>44.6</td><td>55.4</td><td></td><td>54.1</td><td>49.6</td></tr>
<tr><td>DARE</td><td>25.1</td><td>74.9</td><td>43.4</td><td>52.8</td><td></td><td>51.5</td><td>49.9</td></tr>
<tr><td>TaskArithmetic</td><td>25.7</td><td><b>78.9</b></td><td>47.9</td><td>53.9</td><td></td><td>52.1</td><td><b>51.9</b></td></tr>
<tr><td>task_arith (cross-check)</td><td><b>27.0</b></td><td>78.5</td><td>44.0</td><td>53.5</td><td></td><td>52.1</td><td>51.6</td></tr>
<tr><td><b>HTCL</b> (alpha=1)</td><td>15.3</td><td>73.2</td><td>41.5</td><td><b>55.7</b></td><td></td><td><b>54.4</b></td><td>47.9</td></tr>
<tr><td rowspan="5">Specialist<br>(single expert,<br>no merge)<br><i>avg excl. coding</i></td><td>instruction expert</td><td>47.0</td><td>60.3</td><td></td><td></td><td></td><td>55.2</td><td>54.2</td></tr>
<tr><td>math expert *</td><td>13.3</td><td>36.8</td><td></td><td></td><td></td><td>48.1</td><td>32.7</td></tr>
<tr><td>coding expert</td><td>22.9</td><td>52.1</td><td></td><td></td><td></td><td>53.1</td><td>42.7</td></tr>
<tr><td>safety expert</td><td>7.4</td><td>52.5</td><td></td><td></td><td></td><td>48.7</td><td>36.2</td></tr>
<tr><td>multilingual expert</td><td>4.4</td><td>54.8</td><td></td><td></td><td></td><td>56.0</td><td>38.4</td></tr>
</tbody>
</table>

Bold = column max among the dataless merges. Cross-check holds (our `task_arith`
≈ MergeBench `TaskArithmetic` on every domain), so the pipeline is validated. The
specialist coding columns are blank (coding eval not run for experts); `*` the math
expert row is suspect (gsm8k 36.8 < base 56.3 — the base-tokenizer override breaks
its chat format; a `TOK=self` rerun is pending). Multilingual barely separates the
methods (everyone ~52-54, base included).

### Hyperparameter sweep — the alpha scale (gate: math+instr+coding, LIMIT=500)

Default HTCL (alpha=1) trails the dataless baselines at N=5 (Avg 47.9 vs ~51.9):
the closed form is a curvature-weighted *mean*, diluting each update ~1/N vs task
arithmetic's *sum* (NOTES.md §11). The update scale `alpha` (`w_M = w_pre +
alpha*(w_M^HTCL - w_pre)`) was swept over a 3λ × 4α grid:

| variant | math | instr | heval+ | mbpp+ | gate |
|---|---|---|---|---|---|
| Consensus (top baseline) | 78.2 | 25.1 | 49.8 | 54.1 | **51.8** |
| TaskArithmetic | 78.9 | 25.7 | 47.9 | 53.9 | 51.6 |
| HTCL `l1e-3_a2` (best) | 79.8 | 26.8 | 45.6 | 53.0 | 51.3 |
| HTCL `l1e-3_a3` | 76.6 | 31.4 | 44.0 | 48.1 | 50.0 |
| HTCL `l1e-3_a1` (default) | 74.0 | 14.0 | 43.0 | 55.4 | 46.6 |

**Verdict.** `alpha` raises math and instruction sharply (instr 14→27→31 as
α=1→2→3) but **trades away coding** (heval+/mbpp+ fall as α rises). No (λ, α) clears
the baseline cluster; the best variant (`l1e-3_a2`, α=2) **ties** at 51.3 vs 51.8.
So tuned HTCL is competitive with, but does not beat, the dataless tier at N=5 — a
single global α cannot satisfy instruction (wants high α) and coding (wants low α)
at once. The dataless contribution is a tie; the leverage for a stronger result is
the data-using `whc_tree` variant (vs RegMean/RegMean++) and the safety column.

