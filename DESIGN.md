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
raw `p_value` and `individual_fdr`, corrected with Benjamini-Hochberg for each sample.

The correction counts all rule combinations allowed by the maximum rule size and
minimum cell-count/share settings, before support or effect filtering. Attraction
and avoidance count separately when both searches are enabled.

For example, if 20 rules pass mining out of 1,000 candidates, FDR counts all 1,000.
Only the 20 mined rules are shuffled; the other 980 count as p = 1 without extra
shuffles or rows. Passing a smaller list with `rules=` keeps the same total count.
Omitted candidates and supplied rules not mined in this sample count as p = 1.

This correction supports claims about one sample. Claims about patterns recurring
across a study need a separate analysis and correction. The shuffle assumptions
and BH's assumptions about dependence between rules still need to hold.

## Complex rules classification

A rule with 3 or more items is asked whether it adds anything its shorter parts did
not. Nothing is dropped — four columns are added:

- `rule_type` — `pairwise`, `ant-complex`, `con-complex`, `both-complex`
- `complex_class` — why the rule was kept or dismissed, `None` for pairwise
- `adds_information` — the one column to filter on
- `simpler_rules` — exactly what it was weighed against

Classification uses `individual_fdr` from `add_p_values()`, which counts all allowed
rules regardless of class. *Significant* below means
`individual_fdr ≤ max_individual_fdr`. Without that column, lift decides alone.

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

- **`adds_information` is about redundancy, not evidence.** `simpler_are_noise` means
  the *shorter* rules failed the threshold, not that this rule is weak. Filter
  `individual_fdr` separately — it applies to every class alike.
- **A rule the consequent question claimed is not re-asked** the shorter-rule
  question, so a few rules that question would have caught are kept instead.
- **`max_individual_fdr=None`** takes lift at its word: every rule counts as
  convincing. Same when there are no p-values.
- **`n_shuffles` has to be large enough.** With 1000 shuffles, the smallest possible
  p-value is about 0.001. Too few shuffles can prevent rules from passing FDR,
  even when no shuffle passes their thresholds.
- **Sub-rules and their longer rules are positively correlated**, not independent. BH
  holds under positive dependence (PRDS) — an assumption, not a free lunch.

Reference: [Bayardo et al., *Constraint-Based Rule Mining in Large, Dense Databases*](https://www.bayardo.org/ps/icde99.pdf)
