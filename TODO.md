# Things to fix

Written 2026-09-06, from a review of the code as published in `v0.1.0`.
Everything here was checked by running it, not just by reading.

Order matters: section 1 first, then 2, then the rest.

---

## 1. Wrong answers or crashes

| # | Problem | Where |
|---|---|---|
| 1 | Validate the retained **threshold-based** shuffle test on pre-specified rules under the null. Full-family FDR is now implemented (see section 6). The original 65% at p <= 0.05 was measured among mined rules over 12 random datasets, so it is not by itself a calibration check. | `validation/significance.py`, `validation/false_discovery.py` |
| 2 | **Cells sharing the same position, under KNN.** The centre cell can be missing from its own patch, so the patch gets k+1 neighbours instead of k, and the crowding check never sees the centre's own label. Only affects `KNN_R`. Runs using `CN` are not touched. | `transactions.py:55-61`, `:81` |
| 3 | **Attraction-only runs crash.** `include_avoidance_rules=False` is accepted and mines fine, then `add_p_values` raises `TypeError`. The avoidance check runs on every rule even when no rule is an avoidance rule. | `validation/significance.py:314-316` |
| 4 | **Plain text settings quietly pick the wrong method.** `weighting="weighted"` runs BINARY, and `method="CN"` runs KNN_R. The settings are compared by identity, and plain text is never converted. | `settings.py:64` |
| 5 | `min_label_share` rounds **down**. With 105 cells, a label with 10 cells (9.5%) passes a 10% bar. Should round up. | `rules.py:181` |

## 2. Name changes — cheap now, painful after release

| # | Problem | Where |
|---|---|---|
| 6 | `filter_rules` **does not filter**. It labels every rule and returns all of them. Either rename it to `classify_rules`, or make it really drop the rules that add nothing. | `rules.py:199` |
| 7 | Too little is offered to users. `is_crowded_by_one_type` is used by one of the notebooks and raises `ImportError`. | `__init__.py` |

## 3. Checking what comes in

| # | Problem | Where |
|---|---|---|
| 8 | Accepts `k_neighbors=-5`, `min_label_share=2.0`, `max_items_per_rule=3.7`, `min_support_when_strong=0` | `settings.py:64` |
| 9 | A negative `n_shuffles` quietly returns p = 1.0 | `validation/significance.py:38` |
| 10 | Nothing checks the shape of the positions, flat labels, empty or infinite positions, or missing labels | `mine.py:84` |
| 11 | `min_lift_gain` and `max_individual_fdr` are never checked | `complex_rules.py:42` |
| 12 | `strip_role` mangles a label that itself contains `_CENTER` or `_NEIGHBOR` | `transactions.py:25` |
| 13 | `run_samples` sets up logging when called. A library should leave that to the caller. | `runner.py:60` |
| 15 | An import sits inside a function to dodge a loop between two files | `rules.py:203` |

## 4. Writing

- The quick start in the README shows **3 rules**. Running it gives **14**.
- The README says positions are `(n_cells, 2)`. It works in any number of dimensions (checked in 3D).
- Nothing explains what `p_value` actually measures, or that it should never be read on its own.

## 5. Tests

- Nothing covers `run_samples`, running across several processes, reporting failed samples, or `run_config.json`
- No null-calibration test on pre-specified rules: small p-values should occur no more often than their cutoff; the pass/fail test need not be uniform
- The weighted search has no test against a known-correct answer. The one golden test uses binary weights.
- No test for cells sharing the same position
- No automatic test run across Python 3.10 to 3.14
- No check that the package installs cleanly into a fresh, empty setup

## 6. For a statistician, not for code

- Check BH's dependence assumptions for overlapping rules; the correction now counts the complete candidate family
- The test assumes any cell label could sit anywhere in the tissue
- Whether the shuffling should be held within regions or cell groups instead

Implemented correction (2026-09-10), retaining the threshold-based shuffle test:

1. Define and count the complete candidate-rule family.
2. Run permutations only for rules that pass the observed thresholds.
3. Give rules that did not pass a p-value of 1.
4. Apply FDR using the complete family, including those p-values of 1.

Wait for this before putting the package on PyPI. It does not need to hold up anything else.

---

## Already done

- Split into its own repository and renamed to `spatial_association_rules`
- `pyproject.toml` cleaned up: 10 requirements down to 5, project-only files removed
- MIT licence added
- Checked that the built package carries only the 13 library files
