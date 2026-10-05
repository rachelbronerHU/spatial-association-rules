# Design notes

Why the library works the way it does. You do not need any of this to use it —
see the [README](README.md) for that.

## Why expected meetings, not a support bar

lift is observed over expected, so it needs enough expected to divide by. If none were
seen, `e^-expected` is the best p-value the evidence could support — expect 2 and that
is about 1 in 7; expect 20 and it is 1 in 500 million.

A support bar cannot say that, and it is not even one bar. A patch holds **one** center
and **many** neighbors, so:

```
support as a NEIGHBOR  <=  k x support as a CENTER      k = mean neighbors per patch
```

One fraction is up to `k` times harsher on the left than the right. Past a point, no
rare cell type can be the center of an avoidance rule at all.

**How the two checks relate.** `con_support <= 1` always, so:

```
expected meetings = ant_support x con_support x n  <=  ant_support x n = the patch count
```

The expected-meetings check therefore covers the patch check unless
`min_patches > avoidance_min_expected_meetings`. Each still catches what the other
cannot: few patches with a very common neighbor has the meetings but no rate worth
measuring; many patches with a very rare neighbor has the rate but nothing to deplete.

**What it does not do.** The bar makes *total absence* meaningful. It cannot detect a
*partial* shortfall — `lift <= 0.8` is a 20% deficit, needing an expected count near 100
to clear Poisson noise. That is what the shuffle test and the FDR correction are for.

## What bounds the avoidance search

A rule is built from two **sides**: the antecedent and the consequent. Sides are grown
one item at a time and measured a level at a time, then paired.

```
expected meetings = ant_support x con_support x n,   and neither share exceeds 1
```

So each side alone must clear `avoidance_min_expected_meetings / n`, and an antecedent
must also clear `min_patches / n`. A side is never more common than the shorter side it
grew from, so one that fails is dropped and never grown again — the whole branch above
it goes with it.

That leaves:

- `max_items_per_rule` is capped at 5, and a side holds at most `max_items_per_rule - 1`
  items, since the other side needs one
- a center seeds a side and only neighbors extend it, so **no side ever holds two
  centers** — a patch has one center, so those could only ever measure zero
- pairing walks the consequents most-common-first and stops as soon as one is too rare
  for the antecedent in hand

## Why the two lift thresholds are required

The shuffle test asks how often a rule still passes **its own thresholds** in a shuffled
tissue. With no lift threshold there is nothing left to fail:

- avoidance reduces to `lift < 1`, which a shuffle clears about half the time, so every
  p-value lands near 0.5 and the test carries no information
- attraction still has the support policy, so it degrades less sharply, but the same way

Every other threshold is genuinely optional. Unset means no extra tightening, not no
filtering.

## Testing many rules at once

`raw_rules` holds rules after their metrics are measured, before the final
thresholds and rare-label filter. It excludes candidates pruned earlier by the
attraction support floor or avoidance search. Only `rules` are shuffled and classified.

Searching many rules makes chance findings more likely. `add_p_values()` returns
the raw `p_value` and, by default, a corrected value, `individual_fdr`. Set
`calculate_fdr=False` to request raw p-values alone. The correction uses
[Benjamini-Hochberg (BH)](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x),
separately for each sample and rule size. Size counts items on both sides of a
rule. Attraction and avoidance rules of the same size are corrected together.

All allowed rules count, even those the search dropped. For example, if the search
keeps 20 of 1,000 possible rules of one size, the correction counts all 1,000.
Only the 20 kept rules are shuffled; the other 980 count as p = 1, with no extra
rows. The allowed rules depend on the maximum rule size and minimum cell-count/share
settings, and `one_sided_complex_rules`. Counting only the kept rules would ignore the wider search that found them.
See [Hämäläinen and Webb (2019), §6](https://doi.org/10.1007/s10618-018-0590-x).

The raw p-value is `(s + 1) / (B + 1)`: `s` is how many shuffles pass the rule's
thresholds, and `B` is the number of shuffles. Adding one prevents zero p-values
([Phipson and Smyth, 2010](https://gksmyth.github.io/pubs/PermPValuesPreprint.pdf)).
The smallest possible p-value is therefore `1 / (B + 1)`.

With `max_individual_fdr=q`, the library checks whether enough shuffles are planned.
Let `m` be the number of allowed rules of one size and `R` the number kept by the
search. The best case is that all `R` rules get the smallest possible p-value.
Using the BH formula gives:

```
best possible adjusted value = min(1, m / (R * (B + 1)))
minimum shuffles that could be enough = ceil(m / (q * R)) - 1   (R > 0, 0 < q < 1)
```

Here `ceil` means round up. This shuffle limit is derived here from BH. If even
the best case exceeds `q`, the library warns and leaves that size's corrected
values as `NaN` (missing). Raw p-values are still calculated. More shuffles make
passing possible, but do not guarantee it. `R` counts rules after all search filters.
Empty groups need no check. A cutoff of 1 needs no minimum because corrected values
cannot exceed 1. `None` skips the check but still calculates corrected values.

Grouping by size is our choice; BH does not require it. The correction applies to
each group separately. Combining sizes or samples has no automatic false discovery
rate (FDR) guarantee. For background on separate groups, see
[Sun et al. (2006)](https://utstat.utoronto.ca/craiu/Papers/strat-FDR.pdf).

The shuffle test and BH also rely on statistical assumptions. BH requires independent
tests or a particular form of dependence called PRDS
([Hämäläinen and Webb, §6.2](https://doi.org/10.1007/s10618-018-0590-x)).
Rules share cell types and can be related; we have not proved they meet this condition.

## Support from bits

A cell type is either in a patch or it is not, so one bit records it. `packed()` builds
that map once, and it is read two ways.

**Binary weights** are 0 or 1, so the bits are the whole answer: count them.

**Real weights** must still be compared, but the bits say which patches to read. A patch
missing any item of a group counts 0 towards it, and adding 0 changes nothing, so the
rest are skipped. When the map is more than `MOSTLY_ABSENT` full, too few get skipped to
be worth the trouble, and every patch is read.

Both give the same answer, and a test proves it.

## Complex rules classification

`Settings.one_sided_complex_rules=True` permits multiple items on only one side.
Both searches enforce this before measuring candidate rules. Set it to `False`
to also mine mixed rules. The full candidate count used for FDR follows this setting:
for one center and `r` chosen neighbor types, there are `2**r - 1` unrestricted
splits. With one-sided complexity there are `r + 1` splits for `r >= 2`, and one
for `r = 1` (the two possible descriptions of that split coincide).

`classify_rules` returns every row with four classification columns:

- `rule_type`: `pairwise`, `complex-antecedents`, `complex-consequents`, `complex-mixed`.
  Types count items, so a center and a neighbor of the same cell type count separately.
- `complex_class`: the decision below; unset for pairwise and mixed rules.
- `adds_information`: whether the rule survives comparison; missing for mixed rules.
- `simpler_rules`: every matching mined simpler rule, including those failing FDR.
  Mixed rules have an empty list because they are not evaluated.

### The decision

Pairwise rules have `adds_information=True`. Mixed rules are left for the caller to
analyze. For either one-sided complex type:

1. Find all mined simpler rules of either `kind`, keeping the single-item side
   fixed and dropping one or more items from the complex side.
2. If none exist: `no_simpler`, `adds_information=True`.
3. Keep those with `individual_fdr <= max_individual_fdr`. If none qualify:
   `no_significant_simpler`, `adds_information=True`.
4. Compare the complex rule with every qualifying simpler rule. If all comparisons
   pass: `stronger_than_simpler`, `adds_information=True`. Otherwise:
   `redundant_by_simpler`, `adds_information=False`.

A simpler rule still counts if it was itself classified redundant. Classification
is independent of row order and does not change any p-values or FDR values.
There is no separate check for associations between consequent items.

With a cutoff set, missing FDR values fail the gate. No cutoff, or no
`individual_fdr` column at all, means effects alone are compared. A complex rule's
own FDR is separate from this classification and should still be checked.

### Comparing effects

| complex side | metric | gain parameter | default |
|---|---|---|---|
| antecedents | lift | `min_lift_gain` | `1.1` |
| consequents | conviction | `min_consequent_conviction_gain` | `1.1` |

Both gains accept `None` or `0` for any strict improvement, or a finite ratio `>= 1`.
The complex rule's `kind` sets the direction, regardless of the simpler rule's kind.
With gain `g`, complex attraction requires `complex >= simpler * g`; complex avoidance
requires `complex <= simpler / g`. These are ratios, not absolute differences.
Both also require strict improvement, so equal values never pass, even at zero or
infinity. A missing metric cannot establish improvement.

For example, complex attraction lift `1.4` beats simpler avoidance lift `0.8` at
gain `1.1`: `1.4 >= 0.8 * 1.1`. Crossing from avoidance to attraction, or vice versa,
does not bypass the gain requirement or the simpler rule's FDR gate.

With the consequent fixed, a relative lift gain equals the relative confidence gain.
Conviction compares prediction failures; for avoidance, lower is stronger. Its lower
bound depends on consequent support, so a requested reduction may be unattainable.
Neither comparison is a statistical test of the gain.

### Matching simpler rules

The existing cell-type matching is retained: remove `_CENTER` and `_NEIGHBOR` for
matching, but keep antecedent/consequent direction and repeated types. The `kind`
does not restrict matching. Thus `Paneth_CENTER +
Paneth_NEIGHBOR -> Epithelial_NEIGHBOR` is a complex antecedent rule and can match
`Paneth_CENTER -> Epithelial_NEIGHBOR`. A rule cannot match itself because a simpler
signature always contains fewer items. All matching arrangements are considered;
only those passing the FDR gate can block an improvement. `simpler_rules` retains
the full item names so the comparisons can be read back.

### Changes from the previous classifier

Consequent clustering alone no longer marks a rule redundant. Every qualifying
simpler rule of either kind is now considered, including previously redundant
arrangements that the old grouping could skip. Ties do not count as attraction
improvement at gain 1.
Both gain defaults are now 1.1. The public name is `classify_rules`; `filter_rules`
remains an alias. Mixed rules are unclassified, so their `adds_information` is missing.

### Relation to Webb's productivity test

This classifier is an effect filter against qualifying mined simpler rules.
`no_simpler` and `no_significant_simpler` mean no qualifying simpler explanation was
found; they do not establish that the complex rule adds statistically significant
information. `redundant_by_simpler` is a filtering decision, not proof of equivalence.

Webb asks whether adding antecedent conditions improves consequent occurrence within
observations satisfying the simpler antecedent. The Fisher test in Webb (2006),
section 3, uses binary transaction counts. The 2019 tutorial, section 4.2, describes
comparisons against all proper antecedent subsets, without requiring their individual
significance first. Our classifier does not implement that test: weighted supports
are fractional and spatial patches overlap. The current individual-rule shuffle
also does not test conditional improvement. The optional conditional shuffle below
uses a different null and does not change this classifier.

References:

- Geoffrey I. Webb (2006), *Discovering Significant Rules*, KDD, pp. 434-443,
  section 3. [Local paper](references/fdr%20papers/Discovering%20Significant%20Rules%20Webb.pdf).
- Wilhelmiina Hämäläinen and Geoffrey I. Webb (2019), *A Tutorial on Statistically
  Sound Pattern Discovery*, section 4.2. [Published paper](https://doi.org/10.1007/s10618-018-0590-x).

## Conditional shuffle tests

`Result.add_conditional_p_values` is an optional second step after `add_p_values`.
It returns separate `(rules, comparisons)` tables; `run_samples` exposes these as
`report.rules()` and `report.comparisons()`, both with `sample_id`.
It preserves the original p-values and FDR, and does not change classification.
`run_samples(n_conditional_shuffles=...)` runs it before classification.

### Comparisons and fixed cells

Use the same simpler-rule matching as the classifier, across both kinds. Only rows
with `individual_fdr <= max_individual_fdr` qualify as simpler rules. Their previous
classification is irrelevant. Pairwise and mixed rules are not targets.

For `A + B -> C` compared with `A -> C`, freeze every physical A and C cell in the
sample, wherever it is. Merge these with the configured fixed cells. Shuffle the
remaining labels among the remaining cell positions, preserving their counts.
If `B -> C` also qualifies, run a separate comparison with B and C fixed. Do not
freeze the union of both parents in one test: that would immobilize the full rule.
Automatic parent type names are exact; only user-supplied patterns interpret `*`.

The null is that all assignments of the remaining labels to the remaining positions
are equally likely, conditional on the fixed type locations and label counts. This
preserves the fixed types' geometry but can destroy clustering of other types and
their relationships with the fixed types. It is not Webb's conditional independence
null. Overlapping patches use the same shuffled physical cells; weighted supports
are recomputed directly, without rounding them into Fisher counts.

### Statistic and missing cases

For each comparison, let `complex` and `simpler` be their lift values for a complex
antecedent, or conviction values for a complex consequent. The gain statistic is:

- complex attraction: `complex / simpler`;
- complex avoidance: `simpler / complex`.

Equal zeros and equal infinities count as a ratio of 1. A finite positive number
divided by zero is infinite; a finite number divided by infinity is zero. A missing
side makes the metric undefined. A universal consequent makes conviction undefined.
Observed undefined metrics leave that comparison untestable. Observed gains no
greater than 1 get p = 1 without shuffling. The classification's minimum gain remains
a separate effect requirement; it is not the permutation reference value.

Rebuild transactions and reapply the crowding filter for the observed data and every
shuffle. This can change which patches survive, even when parent-type cells stay
fixed, so measure both the complex and simpler rule again. The comparison uses the
same procedure throughout. An undefined shuffled gain counts as no improvement
(score 0); no permutation is discarded or removed from the denominator.

Count gains at least as large as the observed gain, including ties (relative tolerance
1e-12 for floating-point arithmetic). The comparison p-value is `(count + 1)/(B + 1)`.
An infinite observed gain is matched only by infinite shuffled gains.

If a comparison has all relevant cell types fixed, missing cell types, or fewer than
two distinct movable labels, it is untestable. Repeated types can cause this: freezing
Paneth and Epithelial also freezes every type in
`Paneth_CENTER + Paneth_NEIGHBOR -> Epithelial_NEIGHBOR`.

### Combining and interpreting results

The separate `comparisons` table stores the parent, metric, observed gain, fixed
types, status, and p-value for every comparison. `conditional_p_value` in the rules
table is their maximum. Thus every required
comparison must have a small p-value. If any is untestable, retain the individual
results but leave the combined value missing. No qualifying parent also means missing.

For a fixed collection of valid component tests, this maximum is the
[intersection-union test](https://pmc.ncbi.nlm.nih.gov/articles/PMC2752611/);
independence between comparisons is not required. It does not provide correction
across different complex rules.

By default, `conditional_fdr` applies Benjamini–Hochberg to the combined p-values, separately
for each sample and rule size. Attraction and avoidance, complex antecedents and
complex consequents share one group. Each rule contributes one p-value, regardless
of its number of comparisons or fixed-type sets. The candidate count is the same
as for `individual_fdr`, including allowed mixed candidates when enabled. Unreturned
or untested candidates count as p=1 for correction; missing conditional p-values
and their adjusted values stay missing in the output. No extra shuffles are needed.
Set `calculate_fdr=False` for the conditional test to skip this correction while
keeping the comparison p-values. The parent selection still requires the original
`individual_fdr` column and a cutoff.

The parent cutoff `max_individual_fdr` selects simpler rules and is reused for the
resolution check that `individual_fdr` runs. If too few shuffles are planned for any
rule of a size to reach it, a warning is logged and that size's `conditional_fdr` is
missing. So is a size with more rules than candidates, which means repeated rows or
rows from another sample. Original statistics and classification are unchanged;
`individual_fdr` still applies only to the original `p_value`.

Here the rules and qualifying parents were selected using the same data. This
selection and dependence between rules still need validation before claiming FDR
control for the full procedure. BH adjustment alone does not establish that control.

The tests check permutation arithmetic and conservative rejection rates over a small,
fully enumerated null with prespecified comparisons. This does not validate FDR after
mining and parent selection, or the random-label null as a biological model.

### Work and reproducibility

`Result.conditional_test_plan` lists comparisons without building shuffled tissues.
Its `attrs["shuffle_batches"]` counts distinct usable fixed-type sets: an upper bound
because observed non-improvements and undefined metrics can remove work later.
Each batch shares rebuilt tissues among all its comparisons. Geometry is built once,
and supports are measured only for the rules used by that batch. The actual counts
are recorded in `rules.attrs["conditional_test_summary"]` and logged.

Seeds depend on the base seed and the fixed-type set, so reordering rows does not
change results. In `run_samples`, the base seed is already specific to the sample.
`add_p_values` records `labels_kept_fixed` in the returned frame's metadata. Conditional
testing inherits that configuration and adds any extra fixed-label patterns.
