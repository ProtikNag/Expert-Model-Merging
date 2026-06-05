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
mean over M_MMLU / M_ARC / M_Hellaswag across {fr, es, de, ru}. All values %.
Cells fill from `scripts/mb_make_tier2_table.py` once the eval jobs land; see
[`TIER2_RUNBOOK.md`](../../TIER2_RUNBOOK.md).

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
<tr><td>Reference</td><td>Llama-3.1-8B (no merge)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
<tr><td rowspan="7">Dataless merge<br>(weights only)</td><td>Consensus</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
<tr><td>LocalizeAndStitch</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
<tr><td>TIES</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
<tr><td>DARE</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
<tr><td>TaskArithmetic</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
<tr><td>task_arith (cross-check)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
<tr><td><b>HTCL</b></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
<tr><td rowspan="5">Specialist<br>(single expert,<br>no merge)</td><td>instruction expert</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
<tr><td>math expert</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
<tr><td>coding expert</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
<tr><td>safety expert</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
<tr><td>multilingual expert</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
</tbody>
</table>

**Go/no-go.** Tier 2 passes if HTCL is the dataless-tier column max (or tied) on
the cross-domain Avg, as it was at Tier 1. The specialist rows give the per-domain
ceilings for normalized-performance context.

