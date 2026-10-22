# Spatial Association Rules

Finds rules like *"where there is a CD8T cell, macrophages are nearby"* — and the
opposite, *"where there is a CD8T cell, macrophages are not"* — in spatial data.

## The idea

Every cell becomes the center of a **patch**: itself plus the cells around it. Each
patch is one **transaction**:

```
{CD8T_CENTER: 1.0, Macrophage_NEIGHBOR: 0.8, Plasma_NEIGHBOR: 0.2}
```

How much a neighbor counts is the only choice that changes the math:

| `weighting` | a neighbor counts |
|---|---|
| `WEIGHTED` | `exp(-0.5 * (distance / bandwidth) ** 2)` — a far neighbor counts less |
| `BINARY` | `1.0` — near or far |

Support is **min-based** — a pattern is only as strong as its weakest member:

```
support(I) = mean over transactions of min(weight of each item in I)
```

With binary weights that is the plain fraction of transactions holding every item.

A rule always reads *center → neighbors*. The left side is the **antecedent**, the right
side the **consequent**. The center is on the left, never on the right.

## Install

```
pip install -e .
```

## Quickstart

One sample, start to finish:

```python
import numpy as np
from spatial_association_rules import Settings, Weighting, Method, mine

# Made-up tissue: 20 clumps, each of one cell type, and Paneth scattered everywhere.
rng    = np.random.default_rng(0)
clumps = np.repeat(rng.random((20, 2)) * 200, 30, axis=0) + rng.normal(0, 8, (600, 2))
coords = np.vstack([clumps, rng.random((400, 2)) * 200])              # (n_cells, 2)
labels = np.array(["CD8T"] * 300 + ["Macrophage"] * 300 + ["Paneth"] * 400)

settings = Settings(weighting=Weighting.WEIGHTED, method=Method.CN,
                    radius=25.0, min_support=0.01,
                    min_lift=1.2, max_items_per_rule=3,
                    avoidance_max_lift=0.8)

result = mine(coords, labels, settings)
rules  = result.add_p_values(n_shuffles=1000, random_seed=42, max_individual_fdr=0.05)

print(rules[["antecedents", "consequents", "kind", "lift", "regular_p_value"]])
```

Two calls, in that order: mine, then test. Complex rules get their own optional
[conditional test](#optional-conditional-tests).

Each cell type is found sitting with itself, and the two clumped types are found keeping
apart:

```
        antecedents             consequents      kind  lift  regular_p_value
     (CD8T_CENTER,)        (CD8T_NEIGHBOR,)  attracts  1.25            0.001
(Macrophage_CENTER,)  (Macrophage_NEIGHBOR,) attracts  1.43            0.001
     (CD8T_CENTER,)  (Macrophage_NEIGHBOR,)    avoids  0.76            0.001
```

## Many samples

`run_samples` does the same steps for every sample:

```python
from spatial_association_rules import run_samples

if __name__ == "__main__":                       # required when workers is set
    report = run_samples(
        samples,                                 # (sample_id, coords, labels) triples
        settings,
        n_shuffles=1000, random_seed=42,
        max_individual_fdr=0.05,
        workers=8, output_path="results/",
    )

report.rules()      # every rule, with raw p-values and a sample_id column
report.failures     # (sample_id, traceback) for samples that raised
```

`result.raw_rules` and `report.raw_rules()` return a separate
table of rules measured before the final mining thresholds and rare-label filter.
It has no p-values; earlier search pruning still applies. The library does not save
this table unless you choose to.

A sample that raises is recorded and the rest carry on. If every sample fails, that
raises.

By default, the library writes only `run_config.json`, and only if you pass `output_path`.
Optional permutation recording also writes compressed metric arrays; see below.
Results come back as DataFrames.

## What comes back

One row per rule. `mine` gives the first block, `add_p_values` the second, and
`run_samples` adds `sample_id`. The optional conditional test adds the last three columns below.

| column | what it is |
|---|---|
| `antecedents`, `consequents` | the two sides, as tuples of items like `CD8T_CENTER` |
| `rule_idx` | the rule's id within its sample, set by `mine`; `comparisons` refers to it |
| `kind` | `"attracts"` or `"avoids"` — which search found it |
| `support` | how often the whole rule appears |
| `antecedent support`, `consequent support` | the same, for each side alone |
| `confidence`, `lift`, `leverage`, `conviction` | how strong it is. `lift > 1` attracts, `< 1` avoids |
| `len_ant`, `len_con` | items on each side |
| `regular_p_value` | raw p-value from the shuffle test; pairwise rules only, `NaN` for complex rules |
| `individual_fdr` | optional corrected p-value, grouped by sample; `NaN` for complex rules and when too few shuffles can meet the chosen cutoff |
| `sample_id` | which sample, from `run_samples` only |
| `conditional_p_value` | complex rules only: the largest p-value over its simpler rules; missing if any comparison is untestable |
| `conditional_individual_fdr` | optional corrected conditional p-value within this sample and rule size; missing when the conditional p-value is missing or too few shuffles can meet `max_conditional_individual_fdr` |
| `conditional_status` | `tested`, `no_simpler`, `not_applicable`, or `untestable` |

## Pipeline

| step | file | what it does |
|---|---|---|
| 1 | `transactions.py` | group cells into patches — everything within `radius`, or the `k_neighbors` nearest |
| 2 | `transactions.py` | weigh each neighbor: distance decay, or a flat 1.0 |
| 3 | `transactions.py` | patch → transaction. Same-label neighbors add up, capped at 1.0. Patches dominated by one label are dropped |
| 4 | `attraction.py`, `avoidance.py` | two searches, one per claim — see below |
| 5 | `rules.py` | measure and judge. Rules naming a too-rare cell type are dropped |
| 6 | `validation/significance.py` | shuffle the labels, see how often a pairwise rule still passes → `regular_p_value` |
| 7 | `validation/conditional.py` | optional: shuffle with each simpler rule's types fixed, see how often a complex rule is as strong as mined → `conditional_p_value` |

Every step runs per sample. Nothing is pooled across samples: the library gives no
dataset-wide answer.

## Parameters

### Settings

Required fields must be chosen. Other defaults are listed below; an optional
threshold set to `None` is not applied.

| | required | what it is |
|---|---|---|
| `weighting` | **yes** | `WEIGHTED` or `BINARY` |
| `method` | **yes** | `CN`: everything inside the radius. `KNN_R`: the k nearest, capped by it |
| `radius` | **yes** | how far a patch reaches |
| `min_support` | **yes** | how often a pattern must appear. `0` never terminates |
| `min_lift` | **yes** | what counts as attraction. Must be `>= 1` ([why it is required](DESIGN.md#why-the-two-lift-thresholds-are-required)) |
| `max_items_per_rule` | **yes** | longest rule to build, 2 to 5 |
| `one_sided_complex_rules` | | default `True`: only one side may contain multiple items. `False` also mines mixed rules, which are never tested |
| `avoidance_max_lift` | **when avoidance is on** | what counts as avoidance. Must be `< 1` ([why it is required](DESIGN.md#why-the-two-lift-thresholds-are-required)) |
| `bandwidth` | | distance at which a neighbor counts ~0.6. Unset, it follows `radius` |
| `k_neighbors` | for `KNN_R` | how many neighbors to take |
| `min_cells_per_patch` | | skip patches smaller than this. Minimum 2 |
| `max_one_type_share` | | skip a patch this dominated by one label |
| `min_patches` | | how many patches must back a rule. Counted in weight, so exact under `BINARY` and conservative under `WEIGHTED` |
| `strong_confidence` + `min_support_when_strong` | | above that confidence, allow this lower support |
| `min_confidence`, `min_leverage`, `min_conviction` | | tighten attraction further |
| `include_avoidance_rules` | | search for cell types that keep apart too. On by default |
| `avoidance_max_leverage` | | tighten avoidance further |
| `avoidance_min_expected_meetings` | | meetings that had to be expected before a miss counts. Default 10 ([why](DESIGN.md#why-expected-meetings-not-a-support-bar)) |
| `avoidance_min_expected_support` | | minimum expected fraction `support(A) * support(B)`, in `(0, 1)`. Combined with the absolute meetings floor using `max`. Default `None` |
| `avoidance_max_confidence` | | maximum `P(B given A)`, in `[0, 1]`, inclusive. Default `None` |
| `min_label_count`, `min_label_share` | | ignore rules naming a label this rare in the sample |

### add_p_values

| | what it is |
|---|---|
| `n_shuffles` | **required.** The smallest possible p-value is `1/(n_shuffles+1)`, so 5 shuffles can never reach 0.05 |
| `random_seed` | fix it and re-runs give identical p-values |
| `labels_kept_fixed` | labels that never move. `"Name"` is exact; `"Name*"` matches anything starting with Name, so `"CD4*"` also catches `CD45` |
| `calculate_fdr` | default `True`. Set `False` to return raw `regular_p_value` only, without an `individual_fdr` column |
| `max_individual_fdr` | chosen cutoff, greater than 0 and at most 1. Checks whether enough shuffles are planned. Default `None` skips this check; `calculate_fdr` controls correction |

Only pairwise rules are shuffled here. Complex rules keep `NaN` in both columns;
the [conditional test](#optional-conditional-tests) judges them instead.

Testing many rules increases the risk of chance findings. The correction accounts
for all allowed pairwise rules, including those the search dropped. It groups rules
by sample. Each group includes both attraction and avoidance rules.

With `max_individual_fdr` set, the library warns before shuffling if even the best
possible result cannot meet the cutoff. Those rule sizes still get raw p-values,
but their `individual_fdr` is `NaN` (missing). More shuffles are needed to have any
chance of passing. See [DESIGN](DESIGN.md#testing-many-rules-at-once) for details.

For raw p-values alone, call `result.add_p_values(n_shuffles=1000,
calculate_fdr=False)`.

If `labels_kept_fixed` leaves too few cells free to shuffle, testing raises an error.

### Optional conditional tests

Is a complex rule more than its simpler rules? For `A + B -> C` and its simpler rule
`A -> C`, every A and C cell stays in place and the other labels are shuffled. If the
shuffled tissue rarely makes `A + B -> C` as strong as it was when mined, B adds
something. Each simpler rule is one comparison.

```python
plan = result.conditional_test_plan(tested)       # what will run; no shuffles
conditional, comparisons = result.add_conditional_p_values(
    tested, n_shuffles=1000, random_seed=42, max_conditional_individual_fdr=0.05,
)
```

| parameter | what it is |
|---|---|
| `n_shuffles` | **required.** Shuffles per set of fixed cell types |
| `metric` | `"lift"`, `"conviction"`, or `None` (default): lift for complex antecedents, conviction for complex consequents |
| `labels_kept_fixed` | extra labels that never move, added to those recorded by `add_p_values` |
| `calculate_fdr` | default `True`; `False` skips `conditional_individual_fdr` |
| `max_conditional_individual_fdr` | warns when too few shuffles can reach this cutoff; default `None` skips the check |

- **Simpler rules** come from `result.raw_rules`: every rule the searches measured,
  of either kind, before the final thresholds. Matching is by cell type, so simpler
  rules that fix the same types are one comparison. Only their cell types are used.
- **The count.** A shuffle counts when the rule's metric is at least as high as mined
  (`attracts`) or at least as low (`avoids`). Ties count; a shuffle missing a side
  never does. Mining thresholds play no part. The p-value is
  `(count + 1) / (n_shuffles + 1)`, and the rule keeps the largest over its comparisons.
- **FDR.** `conditional_individual_fdr` is Benjamini–Hochberg per sample and rule size, over all
  allowed candidates, as `individual_fdr` is.
- **Missing values.** Pairwise and mixed rules are not tested. No simpler rule, a
  missing cell type, all types fixed, nothing left to shuffle, or a missing mined
  metric leaves the p-value missing. `comparisons` says why.

`comparisons` has one row per comparison, linked to the rules by `rule_idx`.
`report.comparisons()` joins to `report.rules()` on `["sample_id", "rule_idx"]`.
Rules with the same fixed types share shuffles, so the work is
`plan.attrs["shuffle_batches"] × n_shuffles` tissue rebuilds.

This is a restricted shuffle test, not Webb's test, and FDR control after mining is
not proven. [Details](DESIGN.md#conditional-shuffle-tests).

### run_samples

| | what it is |
|---|---|
| `samples` | iterable of `(sample_id, coords, labels)`. Not a DataFrame, so no column names are assumed |
| `workers` | `None` runs here. An integer runs that many **processes** — mining is CPU-bound. On Windows, guard the caller with `if __name__ == "__main__"` |
| `output_path` | where to write `run_config.json`. `None` writes nothing |
| `save_permutation_metrics` | default `False`; record per-sample shuffle metrics. Requires `output_path` |
| `n_conditional_shuffles` | shuffles for the conditional test of complex rules; default `None` skips it |
| `conditional_metric` | `metric` of the conditional test; default `None` |
| `calculate_individual_fdr` | default `True`; `False` returns raw p-values without `individual_fdr`. Requires `max_individual_fdr=None` |
| `calculate_conditional_individual_fdr` | default `True`; `False` returns conditional p-values without `conditional_individual_fdr` |
| `max_individual_fdr`, `max_conditional_individual_fdr` | cutoffs for the shuffle-count check of each test. `None` skips it |

Each sample derives its own seed from `random_seed`, so a parallel run matches a serial
one.

### Optional permutation metrics

Set `save_permutation_metrics=True` in `run_samples` with an `output_path` to save
shuffle distributions. The default is `False`; recording does not change test results.
Files are written by each worker under `output_path/permutations/sample_<id>/`,
with URL-encoded sample IDs. Each sample gets `regular_p_value.npz` and one
`conditional_0001.npz`, etc., per conditional batch actually shuffled. Skipped tests
write no archive. Recording uses additional temporary disk space. Python chooses the
temporary folder automatically; on a cluster, set `TMPDIR` to local disk in the job
script if needed. Final archives still go under `output_path`.

`run_samples` rejects a non-empty permutation folder or sample IDs sharing a folder.
Use a fresh output location or clear the permutation folder before re-running.
Existing archives are never overwritten or deleted.

See [the recording design](DESIGN.md#recording-permutation-metrics) for buffering,
memory use, and cleanup.

Each `.npz` contains:

| array | meaning |
|---|---|
| `rule_idx` | rule IDs identifying metric columns, matching the returned rules |
| `lift`, `conviction` | float64 arrays with shape `(permutations, rules)` |
| `passed_mining_filters` | regular p-value archive only: whether each rule passed its mining thresholds |
| `sample_id`, `test` | string scalars identifying the sample and test |
| `fixed_types` | cell types held fixed in this batch |

Rows identify permutations in execution order. Undefined metrics are `NaN`; infinite
conviction is retained. The regular p-value archive holds pairwise rules. Each conditional
archive holds the complex rules of one batch; compare a rule's column with `observed`
in `comparisons` to rebuild its conditional p-value.

```python
with np.load("output/permutations/sample_FOV1/regular_p_value.npz") as data:
    column = np.flatnonzero(data["rule_idx"] == rule_idx)[0]
    shuffled_lifts = data["lift"][:, column]
    # The regular p-value test counts passes of mining thresholds, rather than lift exceedances.
    regular_p_value = (data["passed_mining_filters"][:, column].sum() + 1) / (len(shuffled_lifts) + 1)
```

For single-sample calls, pass `permutation_dir` (for example, `"output/permutations"`)
to `Result.add_p_values` or `Result.add_conditional_p_values`. The shared saver builds
every sample folder and batch filename under that directory. `permutation_dir` defaults
to `None`, which writes no permutation files.

## Attraction and avoidance

*Sit together* and *keep apart* are opposite claims, so they are two searches. Every
rule records which one found it in its `kind` column. `lift >= 1` and `lift < 1` are
structural, one per search, so no rule can be both.

**Attraction** — `attraction.py`. FP-growth over the transactions. A rule must:

- clear `min_support` (or `min_support_when_strong` when confidence is high) and
  `min_patches`
- then `min_lift`, plus `min_leverage` / `min_conviction` / `min_confidence` if set

**Avoidance** — `avoidance.py`. It cannot be the same search: low joint support *is* the
finding here, so there is nothing to prune on. It prunes on each half of the rule
instead. A rule must:

- have **no joint-support requirement**
- have enough patches holding the antecedent to measure a rate on (`min_patches`)
- have had enough meetings expected that seeing none is surprising
  (`avoidance_min_expected_meetings`), or the proportional floor
  (`avoidance_min_expected_support * n_patches`) if larger
- then clear `avoidance_max_lift`, plus `avoidance_max_leverage` / `avoidance_max_confidence` if set

Why avoidance is measured this way:
[expected meetings, not a support bar](DESIGN.md#why-expected-meetings-not-a-support-bar).
What keeps its search from exploding:
[what bounds the avoidance search](DESIGN.md#what-bounds-the-avoidance-search).

---

[Design notes](DESIGN.md) explain the choices behind all of this.
