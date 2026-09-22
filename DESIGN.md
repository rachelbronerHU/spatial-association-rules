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

Searching many rules makes chance findings more likely. `add_p_values()` returns
the raw `p_value` and a corrected value, `individual_fdr`. The correction uses
[Benjamini-Hochberg (BH)](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x),
separately for each sample and rule size. Size counts items on both sides of a
rule. Attraction and avoidance rules of the same size are corrected together.

All allowed rules count, even those the search dropped. For example, if the search
keeps 20 of 1,000 possible rules of one size, the correction counts all 1,000.
Only the 20 kept rules are shuffled; the other 980 count as p = 1, with no extra
rows. The allowed rules depend on the maximum rule size and minimum cell-count/share
settings. Counting only the kept rules would ignore the wider search that found them.
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

A rule with 3 or more items is asked whether it adds anything its shorter parts did
not. Nothing is dropped — four columns are added:

- `rule_type` — `pairwise`, `ant-complex`, `con-complex`, `both-complex`
- `complex_class` — why the rule was kept or dismissed, `None` for pairwise
- `adds_information` — the one column to filter on
- `simpler_rules` — exactly what it was weighed against

Classification reads the corrected p-values from `add_p_values()`.
*Significant* below means `individual_fdr ≤ max_individual_fdr`.
With a cutoff set, a shorter rule with a missing value (`None`/`NaN`) cannot
dismiss a longer rule. Classification still runs and may assign `simpler_are_noise`
or `consequent_is_noise`, both with `adds_information=True`. A longer rule's own
missing FDR does not stop classification either: shorter rules with passing FDR
can still mark it redundant. If the cutoff is `None` or the whole column is absent,
only lift (effect strength) is used. All rows are returned; each rule's own
corrected p-value still needs to be checked before calling it significant.

### The decision tree

**Shortest rules first**, so a rule is only ever weighed against shorter ones already
judged:

```
2 items ....................................... pairwise, keep. done.

do the consequents already do this to each other?
├─ every consequent pair backed by a two-item rule
│  of the same kind, at least as strong?
│  ├─ yes, and every backing rule significant . consequent_driven    DROP
│  ├─ yes, but one rests on noise ............. consequent_is_noise  KEEP
│  └─ no ...................................... fall through
└─ one consequent only ........................ fall through

shorter rules = every rule contained in this one,
                one item left on each side

├─ none were mined ............................ new                  KEEP
├─ beats every one by min_lift_gain ........... stronger_effect      KEEP
└─ matched at least one
   ├─ any matched one is significant .......... redundant_by_simpler DROP
   └─ none is ................................. simpler_are_noise    KEEP
```

`A + B → C + D` answers to `A → C`, `A + B → C`, `A → C + D` and the rest — every rule
inside it, not only the next size down, since a rule two sizes down can be the strongest
while the one between collapsed. A dismissed shorter rule still counts: it is a yardstick,
not a verdict to inherit, or `new` stops meaning "nothing shorter was mined".

The correction reads no class, so it runs first and nothing goes in a circle.

`adds_information` is `False` for the two classes marked DROP, and `True` for everything
else, pairwise rules included.

### The consequent question, both ways

Whether the consequents already do to each other what the rule claims the antecedent
does to them. Every pair of consequent types must be backed, not just one — a niche
means the whole group hangs together.

| rule is | backing rule must be | at least as strong means |
| --- | --- | --- |
| `attracts` | `attracts` | pair lift ≥ this rule's — they always cluster, so finding them by the antecedent is not news |
| `avoids` | `avoids` | pair lift ≤ this rule's — they already exclude each other, so nothing sitting by both is not news |

### Counted by item, compared by type

Two different questions, so two different ways of matching:

- **How long is this rule?** By item, roles included. `Paneth_CENTER +
  Paneth_NEIGHBOR → Epithelial_NEIGHBOR` is three items — the cell in the middle and
  the cell beside it are two different observations.
- **Which rule is it up against?** By cell type, role dropped, duplicates kept.
  That rule answers to `Paneth → Epithelial`, whichever way round the roles fall.

Keeping duplicates is what makes the two agree: the type list is as long as the item
list, so dropping an item always lands on a genuinely shorter rule.

Several arrangements share one signature — `Muscle_NEIGHBOR + Paneth_CENTER →
Paneth_NEIGHBOR` and `Muscle_CENTER + Paneth_NEIGHBOR → Paneth_NEIGHBOR` both read
`Muscle, Paneth → Paneth`. One speaks for the group: **rules that earned their place
first, then the significant ones, then the strongest of those.**

Direction stays in the signature, so `C → A` is not a shorter version of `A → C`. It is
ignored only in the consequent question, where the rule joining two cell types always
has one of them as its center.

### Notes

- **`adds_information` tells you whether shorter rules already explain a rule.**
  `simpler_are_noise` means the shorter rules failed the cutoff or had missing
  corrected p-values. Check the longer rule's own `individual_fdr` separately.
- **A rule the consequent question claimed is not re-asked** the shorter-rule
  question, so a few rules that question would have caught are kept instead.
- For the shuffle count and statistical assumptions, see
  [Testing many rules at once](#testing-many-rules-at-once).

Reference: [Bayardo et al., *Constraint-Based Rule Mining in Large, Dense Databases*](https://www.bayardo.org/ps/icde99.pdf)
