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
attraction support floor or avoidance search. Only pairwise `rules` are shuffled by
`add_p_values`; `raw_rules` supply the simpler rules for the conditional test.

Searching many rules makes chance findings more likely. `add_p_values()` returns
the raw `regular_p_value` and, by default, a corrected value, `individual_fdr`. Set
`calculate_fdr=False` to request raw p-values alone. Only pairwise rules are shuffled;
complex rules keep `NaN` for both, and the conditional test below judges them. The
correction uses
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

## Complex rules

`Settings.one_sided_complex_rules=True` permits multiple items on only one side.
Both searches enforce this before measuring candidate rules. Set it to `False`
to also mine mixed rules. The full candidate count used for FDR follows this setting:
for one center and `r` chosen neighbor types, there are `2**r - 1` unrestricted
splits. With one-sided complexity there are `r + 1` splits for `r >= 2`, and one
for `r = 1` (the two possible descriptions of that split coincide).

Complex rules are not classified, and `add_p_values` does not shuffle them. Their
only test is the optional conditional shuffle below. Mixed rules are never tested.

### Relation to Webb's productivity test

Webb asks whether adding antecedent conditions improves consequent occurrence within
observations satisfying the simpler antecedent. The Fisher test in Webb (2006),
section 3, uses binary transaction counts. The 2019 tutorial, section 4.2, describes
comparisons against all proper antecedent subsets, without requiring their individual
significance first. The conditional shuffle below follows that choice of simpler
rules, but it does not implement Webb's test: weighted supports are fractional,
spatial patches overlap, and the null is a restricted random-label one.

References:

- Geoffrey I. Webb (2006), *Discovering Significant Rules*, KDD, pp. 434-443,
  section 3. [Local paper](references/fdr%20papers/Discovering%20Significant%20Rules%20Webb.pdf).
- Wilhelmiina Hämäläinen and Geoffrey I. Webb (2019), *A Tutorial on Statistically
  Sound Pattern Discovery*, section 4.2. [Published paper](https://doi.org/10.1007/s10618-018-0590-x).

## Conditional shuffle tests

`Result.add_conditional_p_values` runs after `add_p_values` and returns
`(rules, comparisons)`. `run_samples(n_conditional_shuffles=...)` runs it per sample.

### Simpler rules

A complex rule's simpler rules are the rules in `raw_rules` that keep its single-item
side and drop one or more items from its complex side. `raw_rules` holds every rule
the searches measured, of either kind, before the final thresholds, so no threshold
or FDR decides which simpler rules count.

Matching ignores `_CENTER` and `_NEIGHBOR` but keeps the sides and repeated types.
So `Paneth_CENTER + Paneth_NEIGHBOR -> Epithelial_NEIGHBOR` matches
`Paneth -> Epithelial`. Simpler rules that fix the same cell types are one comparison.

Every valid simpler rule of a mined complex rule is in `raw_rules`. For attraction,
its itemset is part of a frequent itemset. For avoidance, a shorter side is at least
as common as the longer one, so it clears the same bars. A shorter rule without the
center is not a valid rule in either search and is never matched.

### Fixed cells and the null

For `A + B -> C` compared with `A -> C`, every A and C cell in the sample stays in
place, with any configured fixed labels. The other labels are shuffled among the
other cells, keeping their counts. `B -> C` is a separate comparison with B and C
fixed; fixing both at once would fix the whole rule.

The null: every arrangement of the movable labels over the movable cells is equally
likely. This keeps the fixed types' geometry but can break clustering among the
others. It is not Webb's conditional independence null. Weighted supports are
recomputed from the shuffled cells, not rounded into Fisher counts.

### Counting

The rule is scored by `metric`: lift, conviction, or by default lift for complex
antecedents and conviction for complex consequents. The observed value is the mined
one, read from the rules table. A shuffle counts when the rule's value is at least as
high as observed (`attracts`) or at least as low (`avoids`). Ties count (relative
tolerance 1e-12), including two infinities. The p-value is `(count + 1)/(B + 1)`.

Mining thresholds play no part, and neither do the simpler rule's metrics: the simpler
rule only says which types stay fixed. Every shuffle rebuilds the patches and reapplies
the crowding filter. A missing side, or a universal consequent for conviction, makes
the value undefined; such a shuffle never counts but stays in the denominator.

A comparison is untestable when all the rule's types are fixed, a type is missing,
fewer than two movable labels remain, or the mined metric is missing. Repeated types
can cause the first: fixing Paneth and Epithelial fixes all of
`Paneth_CENTER + Paneth_NEIGHBOR -> Epithelial_NEIGHBOR`.

### Combining and correcting

`conditional_p_value` is the largest comparison p-value, so the rule must beat every
simpler rule. One untestable comparison, or no simpler rule, leaves it missing;
`comparisons` keeps each result and its status.

`conditional_individual_fdr` is Benjamini–Hochberg per sample and rule size, one p-value per rule,
both kinds together. The family is all allowed candidates, as for `individual_fdr`;
untested candidates count as p=1. With `max_conditional_individual_fdr` set, the same shuffle-count
check warns before shuffling, and a size that cannot reach the cutoff gets missing
values. So does a size with more rules than candidates (repeated rows, or rows from
another sample).

The complex rules were selected by mining the same data, and the observed value is the
one that got them mined. FDR control for the whole procedure is not proven; BH alone
does not establish it. The tests check the counting against explicit permutations,
not the FDR after mining or the null as a biological model.

### Work and reproducibility

`Result.conditional_test_plan` lists comparisons without building shuffled tissues.
Its `attrs["shuffle_batches"]` counts distinct usable fixed-type sets.
Each batch shares rebuilt tissues among all its comparisons. Geometry is built once,
and supports are measured only for the complex rules of that batch. The actual counts
are recorded in `rules.attrs["conditional_test_summary"]` and logged.

Seeds depend on the base seed and the fixed-type set, so reordering rows does not
change results. In `run_samples`, the base seed is already specific to the sample.
`add_p_values` records `labels_kept_fixed` in the returned frame's metadata. Conditional
testing inherits that configuration and adds any extra fixed-label patterns.


## Recording permutation metrics

Optional recording reuses the measurements already made by p-value and conditional
tests, so enabling it does not change their results. Each archive stores the rule IDs
and metrics for one sample and shuffle batch. The regular p-value archive holds pairwise
rules; conditional batches hold their complex rules only. See the
[README](README.md#optional-permutation-metrics) for the flag and archive contents.

Arrays use Python's public `mmap` API with temporary files, allowing the operating
system to reclaim their pages under memory pressure. Buffer space per worker is
`shuffles × rules × 17` bytes for the regular p-value test, including threshold-pass flags,
and `× 16` for the conditional test.
Python chooses the temporary directory, respecting `TMPDIR`, `TEMP`, and `TMP`, so
buffers can use local disk while final archives go to the chosen output folder.
Mapped pages still count toward resident memory, including job memory limits; this
approach does not guarantee lower peak memory usage or prevent out-of-memory failures.
A temporary folder backed by RAM does not offer the same disk-backed reclaim.

Both test types reuse two ordinary row buffers, then copy each row into the mapped
arrays. This prevents the calculation from retaining views into memory that has been
unmapped.
Every row is filled as it is calculated, avoiding an initial write over every mapped
page. The mapped arrays are valid only inside the recording context. On exit, their
references are cleared, mappings are closed, and temporary files are removed.

The sample folder is created only when the batch is ready to save. Early checks reject
existing archives, non-empty run folders, and sample IDs mapping to the same folder
when compared ignoring case. Exclusive archive creation also prevents concurrent
writers from overwriting each other's files. If compression fails, the newly created
archive is closed and removed; previously saved archives remain untouched.
