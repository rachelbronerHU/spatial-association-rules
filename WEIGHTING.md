# The neighbour weight, and why it should change

Written 2026-09-06, from measurements on the MIBI gut data (288 fields, 713,372 cells,
32 cell types, `radius=25`, `bandwidth=15`, `CN`). Every number below was measured, not
estimated.

**One line changes, in `transactions.py:114`. Nothing else.**

---

## The problem

A patch is a cell plus everything within 25µm — about **25 neighbours** in this tissue,
with a total kernel weight around 12 to 17.

For each cell type the neighbour weights are added up, then chopped at 1.0. That chop
is what breaks it. Three very different patches record the same number:

| epithelium nearby | recorded now |
|---|---|
| 6.0 — packed with it | 1.0 |
| 3.75 — an ordinary amount | 1.0 |
| 1.5 — noticeably little | 1.0 |

The chop bites in **39% to 70% of patches** for a common cell type, and in under 6% for
a rare one. So the distance weighting is switched off for common types and left on for
rare ones, and the weighted run is quietly a binary run wherever it bites.

### It puts a ceiling on lift

Under min-based support the joint can never exceed either side, so:

```
lift  =  support / (ant_support * con_support)  <=  1 / max(ant_support, con_support)
```

Both supports climb toward 1 as a cell type gets common. The ceiling falls with them:

| share of the field | highest lift the maths allows |
|---|---|
| under 2% | 4.64 |
| 5–10% | 1.99 |
| 10–20% | **1.36** |
| over 20% | **1.39** |

**35% of pairs involving a type above 20% abundance have a ceiling below
`min_lift = 1.2`.** Those rules cannot be found at all.

In `GVHD_01_FOV_1`, epithelium clustering with itself is the strongest signal in the
field — 59 standard deviations above 200 label shuffles — and scores lift **1.16**,
pinned exactly at its own ceiling, so it is dropped. Endothelial, at 0.6% of the field
and 2.2 standard deviations, passes at lift 3.48.

The threshold was not selecting for evidence. It was selecting for rarity.

---

## The change

Ask a different question before bounding: not *how much is here*, but *how much is here
compared with how much should be here*.

```python
# now — transactions.py:114
transaction = {item: min(weight, 1.0) for item, weight in transaction.items()}

# proposed
total = sum(transaction.values())              # patch's own kernel mass, label-free
transaction = {item: (lambda z: z / (1.0 + z))(
                   weight / (share[strip_role(item)] * total))
               for item, weight in transaction.items()}
```

where `share[type]` is that type's fraction of the sample. Written out:

```
expected = share(type) * patch's total neighbour weight
z        = observed / expected
weight   = z / (1 + z)
```

The same three patches, now:

| epithelium nearby | expected | z | recorded |
|---|---|---|---|
| 6.0 | 3.75 | 1.6 | 0.62 |
| 3.75 | 3.75 | 1.0 | 0.50 |
| 1.5 | 3.75 | 0.4 | 0.29 |

### Why the expectation is `share × patch total`

Keep every cell where it is and shuffle the labels. Each neighbour then has probability
`share` of being that type, so the expected mass is `share × (sum of that patch's
neighbour weights)`. Exact under a label permutation, and computed **per patch** — a
crowded patch is owed more than a sparse one, so patch density corrects itself.

One refinement: when conditioning on a centre of type A, `share` should exclude the
centre cell (`(n_j − [j == A]) / (n − 1)`). It moves the number by about `1/n` and the
permutation test absorbs it either way.

### Why `z / (1 + z)`

Four things are needed, and this is the simplest function that does all four: nothing
there gives 0, more always gives more, it approaches 1 without ever reaching it (so
nothing is ever chopped), and exactly-average gives exactly **0.5**, for every cell type.

It is not arbitrary. The algebra collapses to:

```
   z                observed
 ─────   =   ────────────────────────
  1 + z       observed + expected
```

A proportion, which is why it lands in `[0, 1]` for free. Same map that turns odds into
a probability. `1 − exp(−z)` also works; it just puts average at 0.63 instead of 0.5 and
costs an exponent.

### Why the metrics do not change

The weight stays in `[0, 1)`, so min-based support, confidence, lift, leverage and
conviction are all still defined and still mean what they meant. FP-growth, `min_support`,
`min_patches`, both searches, the shuffle test, the complex-rule classification, the FDR
and everything built on top keep working untouched.

---

## What it buys

12 fields (6 GVHD, 6 control), 200 label shuffles each, 1,269 qualifying pairs. Every
candidate's threshold is set so all of them make the **same 2% false-positive rate** —
otherwise a lenient measure only looks sensitive.

Truth is defined two ways. The one below is deliberately **independent of the proposal**:
observed-over-expected on plain neighbour *counts*, no kernel and no distance, so nothing
the proposal uses. A pair counts as real at z ≥ 3, and as empty at z ≤ 1.

| | real signals found | of those, types over 20% | cells with their own kind, 10–20% |
|---|---|---|---|
| binary | 52% | 54% | 21% |
| weighted + cap (now) | 69% | 60% | 63% |
| **proposed** | **83%** | **83%** | **100%** |

Cells sitting near their own kind is the second definition of truth: it is true in every
tissue, needs no metric to establish, and is therefore a fair test of what a measure can
see. The current weighting misses a third of it, and only for common types.

Lift ceilings after the change: 4.11 under 5%, 3.40 at 10–20%, 2.77 over 20%. The spread
across cell types narrows from 3.4× to 1.8×, and every type clears a 1.2 bar comfortably.

### Rare cell types are not harmed

The obvious worry, checked directly on the 542 pairs whose consequent is under 5% of the
field:

| | |
|---|---|
| pass now | 166 |
| pass after the change | 180 |
| lost | **1** — and it scored z = 1.97, so it was not real |
| newly found | 15 |
| rule strength before vs after | correlated **0.995** |

On rare pairs that are genuinely real, 89.5% found now, 92.5% after. Their ceiling
*rises*, 3.28 → 4.11. The change acts almost entirely on the common types that were
being blocked.

---

## Optional, one line more

A flat `min_lift = 1.2` still asks more of a type whose ceiling is 2.5 than of one whose
ceiling is 4.5. Judging each rule against its own ceiling costs one line in `attracts()`
and takes recovery from 83% to 93%:

```
ceiling  = 1 / max(ant_support, con_support)
headroom = (lift - 1) / (ceiling - 1)        # keep rules above about 0.08
```

Take it or leave it. The weight change alone is the fix; this is a refinement on top,
easy to add once a full run has been seen.

---

## Roads not taken

| option | what happens | why not |
|---|---|---|
| **Plain sum, no cap** | consequent support becomes a mass of 2–4 while the joint stays truncated at 1 | lift lands near 0.25 for the commonest types. Epithelium clustering with itself reads *avoids*. Direction inverted. |
| **Binary** | presence only | worst of the three: 52% of real signals, and 21% of self-attraction among 10–20% types. With 25 neighbours per patch, "is there one nearby" is almost always yes, so it carries no information. |
| **Softer cap** (noisy-OR, `1 − ∏(1 − w)`) | 0.771 where the hard cap gives 0.798 | any measure of *presence* saturates at 25 neighbours. What has to change is what goes into the bound, not the shape of the curve. |
| **Replace lift with observed/expected mass** | most accurate of all | it *is* the reference statistic, so its win is circular. It also has no confidence and no conviction, and makes A→B and B→A 98% correlated against 90% today. Right thing to validate against, wrong thing to ship. |

### Three numbers, three jobs

There is no single number that does everything, and the measurements say why rather than
guessing. Correlation between the two directions of the same pair, 992 ordered pairs:

| measure | A→B vs B→A |
|---|---|
| confidence | **0.25** |
| lift now | 0.90 |
| lift after the change | 0.96 |
| observed/expected mass | 0.98 |

Any enrichment measure is close to symmetric, because it counts the same neighbourhood
from both ends. **Direction has always lived in confidence, not in lift.** That is the
strongest argument for fixing this in the weight: confidence survives the change and
keeps its plain reading — *given a CD8T centre, how much macrophage sits around it*.

```
confidence   -> is the rule directional, and how strong is the rate
lift         -> is it more than the cell types simply being common
permutation p -> is there enough evidence to say so at all
```

---

## Known limits

- **The ceiling narrows, it does not vanish.** 2.8× to 4.1× across cell types instead of
  1.4× to 4.6×. Rank on the permutation result, report lift as effect size, and never
  compare a rare type's lift with a common type's.
- **The zero point is comparable across patches, the spread is not quite.** Patches
  scoring 0.48–0.52 sit at 1.00× their own expected amount for every cell type checked.
  But the score still carries roughly 0.25 correlation with how crowded a patch is
  (0.27 before). Partly real tissue structure, not purely artifact. Removing it entirely
  would mean normalising against patches of similar density rather than the field average.
- **A weight is a rate, so cell size and segmentation ride along with it.** One touching
  cell and three at 20µm give the same number. Dividing by the patch's own total cancels
  the patch-level density effect, not the type-level one. Worth one check against a
  count-based version.
- **Bandwidth becomes a real parameter.** The cap currently freezes 39–70% of common-type
  patches at exactly 1.0, so bandwidth has no effect on them. Afterwards it acts
  everywhere. One sweep at 10 / 15 / 20µm, reported, closes the question.
- **True is not the same as interesting.** Epithelium-with-epithelium will now pass in
  every field with a large z, and so will most self-pairs. Correct and dull. For rule
  fingerprints across fields, the meaningful comparison is a rule's strength against its
  own distribution across fields, not against 1.
- **`max_one_type_share = 0.9` drops only 1.7% of patches**, but disproportionately the
  epithelial and muscle patches where the signal is strongest. Once the weight is
  abundance-aware there is little reason to keep it.

---

## References

The change is not novel — it is standard practice in spatial omics, arriving in this
library late. These are the ones that validate each piece.

**The expectation, and observed-over-expected from label permutation**

- Dries R. et al. (2021). *Giotto: a toolbox for integrative analysis and visualization
  of spatial expression data.* Genome Biology 22:78.
  [doi:10.1186/s13059-021-02286-2](https://doi.org/10.1186/s13059-021-02286-2) — the
  `cellProximityEnrichment` score is observed over expected from permuting cell type
  labels on a fixed spatial network. The direct precedent for the expectation used here.
- Palla G. et al. (2022). *Squidpy: a scalable framework for spatial omics analysis.*
  Nature Methods 19:171–178.
  [doi:10.1038/s41592-021-01358-2](https://doi.org/10.1038/s41592-021-01358-2) —
  neighbourhood enrichment reported as a permutation z-score, not a raw ratio. Supports
  ranking on the permutation result rather than on lift.
- Baddeley A., Rubak E., Turner R. (2015). *Spatial Point Patterns: Methodology and
  Applications with R.* CRC Press, chapters on marked point patterns — the mark
  connection function. The statistic here is its kernel-smoothed form, under a name and
  a literature.

**Why lift behaves this way**

- Tan P.-N., Kumar V., Srivastava J. (2002). *Selecting the right interestingness measure
  for association patterns.* KDD '02, 32–41.
  [doi:10.1145/775047.775053](https://doi.org/10.1145/775047.775053) — lift is not
  null-invariant: its value moves with item frequency. The diagnosis above, stated
  generally, twenty years earlier.
- Omiecinski E. (2003). *Alternative interest measures for mining associations in
  databases.* IEEE TKDE 15(1):57–69.
  [doi:10.1109/TKDE.2003.1161582](https://doi.org/10.1109/TKDE.2003.1161582) —
  all-confidence and bond: frequency-stable measures that keep an anti-monotone prune.
  The fallback if this library ever needs to stay inside classical ARM.

**Weights between 0 and 1**

- Delgado M., Marín N., Sánchez D., Vila M.-A. (2003). *Fuzzy association rules: general
  model and applications.* IEEE Trans. Fuzzy Systems 11(2):214–225.
  [doi:10.1109/TFUZZ.2003.809896](https://doi.org/10.1109/TFUZZ.2003.809896) — the
  formal treatment of `[0,1]` item weights under a min t-norm, which is what these
  transactions already are. Says plainly what such a support does and does not mean.
- Tao F., Murtagh F., Farid M. (2003). *Weighted association rule mining using weighted
  support and significance framework.* KDD '03, 661–666.
  [doi:10.1145/956750.956836](https://doi.org/10.1145/956750.956836) — how to keep
  downward closure once weights are involved. The licence for pruning on a weighted
  support.

**Thresholds and permutations**

- Terada A., Okada-Hatakeyama M., Tsuda K., Sese J. (2013). *Statistical significance of
  combinatorial regulations.* PNAS 110(32):12996–13001.
  [doi:10.1073/pnas.1302233110](https://doi.org/10.1073/pnas.1302233110) — testability
  bounds allow correction across a very large rule family without losing everything.
- Llinares-López F., Sugiyama M., Papaxanthos L., Borgwardt K. (2015). *Fast and
  memory-efficient significant pattern mining via permutation testing.* KDD '15, 725–734.
  [doi:10.1145/2783258.2783363](https://doi.org/10.1145/2783258.2783363) — Westfall–Young
  at pattern-mining scale. The principled answer to the across-the-study claim `DESIGN.md`
  deliberately leaves open.

**The data**

- Schürch C.M. et al. (2020). *Coordinated cellular neighborhoods orchestrate antitumoral
  immunity at the colorectal cancer invasive front.* Cell 182(5):1341–1359.
  [doi:10.1016/j.cell.2020.07.005](https://doi.org/10.1016/j.cell.2020.07.005) — the
  compositional framing: a neighbourhood is a vector of proportions, not a set of present
  items. Why the shopping-basket model does not survive 25 neighbours.

---

## Order of work

1. Change `transactions.py:114`. Add `share` to what `build_transactions` receives.
2. Re-run one field and confirm epithelium clustering with itself now appears.
   **Check it with a permutation on the measured lift, not with `p_value`** — see below.
3. Full run. Compare rule counts per cell type against abundance; the correlation should
   drop sharply.
4. Bandwidth sweep at 10 / 15 / 20µm.
5. Optionally, the headroom threshold in `attracts()`.

### Do not check this with `p_value`

Measured over 8 fields at 500 shuffles: the share of rules at p ≤ 0.05, before and after
the change.

| share of the field | before | after |
|---|---|---|
| under 5% | 14% | 19% |
| 5–10% | 68% | 72% |
| 10–20% | 84% | 88% |
| over 20% | **100%** | **100%** |

The change does not disturb these — the 37 rules it rescues sit at p = 0.002 both before
and after — but they cannot confirm it either. Every pair involving a common type is
already at the floor. `p_value` counts how often a shuffle clears `min_lift`; for a
common type a shuffle essentially never reaches 1.2, so the count is zero and p lands at
`1/(n_shuffles+1)` whether or not the rule is real. Permute and compare the measured
lift instead.
