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
from spatial_association_rules import Settings, Weighting, Method, mine, classify_rules

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
tested = result.add_p_values(n_shuffles=1000, random_seed=42, max_individual_fdr=0.05)
rules  = classify_rules(tested, max_individual_fdr=0.05)

print(rules[["antecedents", "consequents", "kind", "lift", "p_value"]])
```

Three calls, in that order: mine, test, filter. You can stop after any of them.

Each cell type is found sitting with itself, and the two clumped types are found keeping
apart:

```
        antecedents             consequents      kind  lift  p_value
     (CD8T_CENTER,)        (CD8T_NEIGHBOR,)  attracts  1.25    0.001
(Macrophage_CENTER,)  (Macrophage_NEIGHBOR,) attracts  1.43    0.001
     (CD8T_CENTER,)  (Macrophage_NEIGHBOR,)    avoids  0.76    0.001
```

## Many samples

`run_samples` does the same three steps for every sample:

```python
from spatial_association_rules import run_samples

if __name__ == "__main__":                       # required when workers is set
    report = run_samples(
        samples,                                 # (sample_id, coords, labels) triples
        settings,
        n_shuffles=1000, random_seed=42,
        min_lift_gain=1.1, max_individual_fdr=0.05,
        workers=8, output_path="results/",
    )

report.rules()      # every rule, with raw p-values and a sample_id column
report.failures     # (sample_id, traceback) for samples that raised
```

A sample that raises is recorded and the rest carry on. If every sample fails, that
raises.

The library writes nothing except `run_config.json`, and only if you pass `output_path`.
Results come back as DataFrames.

## What comes back

One row per rule. `mine` gives the first block, `add_p_values` the second,
`classify_rules` the third, and `run_samples` adds `sample_id`.

| column | what it is |
|---|---|
| `antecedents`, `consequents` | the two sides, as tuples of items like `CD8T_CENTER` |
| `kind` | `"attracts"` or `"avoids"` — which search found it |
| `support` | how often the whole rule appears |
| `antecedent support`, `consequent support` | the same, for each side alone |
| `confidence`, `lift`, `leverage`, `conviction` | how strong it is. `lift > 1` attracts, `< 1` avoids |
| `len_ant`, `len_con` | items on each side |
| `p_value` | raw p-value from the shuffle test |
| `individual_fdr` | p-value corrected for testing many rules, grouped by sample and rule size; `NaN` (missing) when too few shuffles can meet the chosen cutoff |
| `rule_type` | `pairwise`, `complex-antecedents`, `complex-consequents`, `complex-mixed` |
| `complex_class` | why the rule was kept or dismissed ([how](DESIGN.md#complex-rules-classification)) |
| `adds_information` | whether the rule passes the simpler-rule comparison; missing for unclassified mixed rules |
| `simpler_rules` | what it was weighed against |
| `sample_id` | which sample, from `run_samples` only |

## Pipeline

| step | file | what it does |
|---|---|---|
| 1 | `transactions.py` | group cells into patches — everything within `radius`, or the `k_neighbors` nearest |
| 2 | `transactions.py` | weigh each neighbor: distance decay, or a flat 1.0 |
| 3 | `transactions.py` | patch → transaction. Same-label neighbors add up, capped at 1.0. Patches dominated by one label are dropped |
| 4 | `attraction.py`, `avoidance.py` | two searches, one per claim — see below |
| 5 | `rules.py` | measure and judge. Rules naming a too-rare cell type are dropped |
| 6 | `validation/significance.py` | shuffle the labels, see how often the rule still passes → `p_value` |
| 7 | `rules.py` | label rules a shorter rule already said (`classify_rules`) |

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
| `one_sided_complex_rules` | | default `True`: only one side may contain multiple items. `False` also mines mixed rules, which remain unclassified |
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
| `min_label_count`, `min_label_share` | | ignore rules naming a label this rare in the sample |

### add_p_values

| | what it is |
|---|---|
| `n_shuffles` | **required.** The smallest possible p-value is `1/(n_shuffles+1)`, so 5 shuffles can never reach 0.05 |
| `random_seed` | fix it and re-runs give identical p-values |
| `labels_kept_fixed` | labels that never move. `"Name"` is exact; `"Name*"` matches anything starting with Name, so `"CD4*"` also catches `CD45` |
| `max_individual_fdr` | chosen cutoff, greater than 0 and at most 1. Checks whether enough shuffles are planned. Use the same cutoff in `classify_rules`. Default `None` skips this check but still calculates corrected p-values |

Testing many rules increases the risk of chance findings. The correction accounts
for all allowed rules, including those the search dropped. It groups rules by
sample and size: two-item rules together, three-item rules together, and so on.
Each group includes both attraction and avoidance rules. The candidate count respects
`one_sided_complex_rules`, including mixed rules only when it is `False`.

With `max_individual_fdr` set, the library warns before shuffling if even the best
possible result cannot meet the cutoff. Those rule sizes still get raw p-values,
but their `individual_fdr` is `NaN` (missing). More shuffles are needed to have any
chance of passing. See [DESIGN](DESIGN.md#testing-many-rules-at-once) for details.

If `labels_kept_fixed` leaves too few cells free to shuffle, testing raises an error.

### classify_rules

Returns every row, with a type and a simpler-rule classification. `kind` still means
`attracts` or `avoids`. `filter_rules` remains an alias for compatibility.

| parameter | what it is |
|---|---|
| `min_lift_gain` | minimum lift ratio for complex antecedents; default `1.1` |
| `min_consequent_conviction_gain` | minimum conviction ratio for complex consequents; default `1.1` |
| `max_individual_fdr` | simpler rules must pass this cutoff to block a complex rule; `None` uses effects alone |

Both gains accept `None` or `0` for any strict improvement, or a finite ratio `>= 1`.
Simpler rules can be attraction or avoidance. The **complex rule's kind** sets the
direction: attraction requires at least `simpler * gain`; avoidance requires at most
`simpler / gain`. These are ratios, not absolute differences. Ties never count as
improvement, including two infinities.
The rule must beat **every** qualifying simpler rule. Antecedents use lift;
consequents use conviction.

For example, complex attraction lift `1.4` beats simpler avoidance lift `0.8` at
gain `1.1`, because `1.4 >= 0.8 * 1.1`. A change of kind must still meet the gain.

| `complex_class` | meaning | `adds_information` |
|---|---|---|
| `no_simpler` | no simpler rule passed mining thresholds | `True` |
| `no_significant_simpler` | simpler rules exist, but none passes the FDR cutoff | `True` |
| `stronger_than_simpler` | beats every qualifying simpler rule by the chosen metric and gain | `True` |
| `redundant_by_simpler` | fails to beat at least one qualifying simpler rule | `False` |

Pairwise rules have no class and `adds_information=True`. Mixed rules have no class
and a missing `adds_information` value. To select only rows marked informative, use
`rules[rules["adds_information"].fillna(False)]`; select mixed rules separately by
`rule_type` if you want to evaluate them yourself.

For individual rows, check for a missing value before using it as a boolean:

```python
if pd.notna(row.adds_information) and row.adds_information:
    ...
```

Here `pd` is pandas. Plain `if row.adds_information:` raises for an unclassified
mixed rule because its value is missing.

With an FDR cutoff set, missing corrected values cannot qualify a simpler rule.
If the whole `individual_fdr` column is absent, comparison uses effects alone, as it
does with no cutoff. A complex rule's own FDR does not decide its class: check it
separately. `adds_information=True` means the rule survives this comparison, not that
its additional value is statistically established. There is no improvement significance
test. See [DESIGN](DESIGN.md#complex-rules-classification) for matching and references.

### run_samples

| | what it is |
|---|---|
| `samples` | iterable of `(sample_id, coords, labels)`. Not a DataFrame, so no column names are assumed |
| `workers` | `None` runs here. An integer runs that many **processes** — mining is CPU-bound. On Windows, guard the caller with `if __name__ == "__main__"` |
| `output_path` | where to write `run_config.json`. `None` writes nothing |
| `min_lift_gain` | lift gain for complex antecedents, as in `classify_rules` |
| `min_consequent_conviction_gain` | conviction gain for complex consequents, as in `classify_rules`; default `1.1` |
| `max_individual_fdr` | same cutoff for checking the shuffle count and deciding whether shorter rules can dismiss longer ones. `None` skips both checks but still calculates corrected p-values |

Each sample derives its own seed from `random_seed`, so a parallel run matches a serial
one.

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
  (`avoidance_min_expected_meetings`)
- then clear `avoidance_max_lift`, plus `avoidance_max_leverage` if set

Why avoidance is measured this way:
[expected meetings, not a support bar](DESIGN.md#why-expected-meetings-not-a-support-bar).
What keeps its search from exploding:
[what bounds the avoidance search](DESIGN.md#what-bounds-the-avoidance-search).

---

[Design notes](DESIGN.md) explain the choices behind all of this.
