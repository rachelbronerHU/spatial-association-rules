# Things to fix

Written 2026-09-06, from a review of the code as published in `v0.1.0`.
Everything here was checked by running it, not just by reading.

Order matters: section 1 first, then 2, then the rest.

---

## 1. Wrong answers or crashes

| # | Problem | Where |
|---|---|---|
| 1 | Validate the retained **threshold-based** shuffle test on pre-specified rules under the null. Full-family FDR is now implemented (see section 5). The original 65% at p <= 0.05 was measured among mined rules over 12 random datasets, so it is not by itself a calibration check. | `validation/significance.py`, `validation/false_discovery.py` |
| 2 | **Cells sharing the same position, under KNN.** The centre cell can be missing from its own patch, so the patch gets k+1 neighbours instead of k, and the crowding check never sees the centre's own label. Only affects `KNN_R`. Runs using `CN` are not touched. | `transactions.py:55-61`, `:81` |
| 4 | **Plain text settings quietly pick the wrong method.** `weighting="weighted"` runs BINARY, and `method="CN"` runs KNN_R. The settings are compared by identity, and plain text is never converted. | `settings.py:64` |
| 5 | `min_label_share` rounds **down**. With 105 cells, a label with 10 cells (9.5%) passes a 10% bar. Should round up. | `rules.py:181` |

## 2. Checking what comes in

| # | Problem | Where |
|---|---|---|
| 8 | Accepts `k_neighbors=-5`, `min_label_share=2.0`, `max_items_per_rule=3.7`, `min_support_when_strong=0` | `settings.py:64` |
| 9 | A negative `n_shuffles` quietly returns p = 1.0 | `validation/significance.py:38` |
| 10 | Nothing checks the shape of the positions, flat labels, empty or infinite positions, or missing labels | `mine.py:84` |
| 13 | `run_samples` sets up logging when called. A library should leave that to the caller. | `runner.py:60` |

## 3. Writing

- The quick start in the README shows **3 rules**. Running it gives **14**.
- The README says positions are `(n_cells, 2)`. It works in any number of dimensions (checked in 3D).
- Nothing explains what `regular_p_value` actually measures, or that it should never be read on its own.

## 4. Tests

- Nothing covers `run_samples` reporting failed samples
- No null-calibration test on pre-specified rules: small p-values should occur no more often than their cutoff; the pass/fail test need not be uniform
- The weighted search has no test against a known-correct answer. The one golden test uses binary weights.
- No test for cells sharing the same position
- No automatic test run across Python 3.10 to 3.14
- No check that the package installs cleanly into a fresh, empty setup

## 5. For a statistician, not for code

- Check BH's dependence assumptions for overlapping rules; the correction now counts the complete candidate family
- Validate conditional FDR on simulated tissues: preserve simpler associations, add no complex-rule effect, rerun the full pipeline, and check false discoveries at the chosen cutoff.
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
