"""Conditional tests checked against explicit label permutations and patch counts."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from spatial_association_rules import Method, Settings, Weighting, classify_rules
from spatial_association_rules.mine import Result
from spatial_association_rules.transactions import Patch, build_transactions
from spatial_association_rules.validation import conditional


def rule(ants, cons, kind="attracts", fdr=0.01):
    return dict(antecedents=tuple(ants), consequents=tuple(cons), kind=kind,
                individual_fdr=fdr, p_value=0.001)


def patch(center, neighbors, weights):
    return Patch(center, np.array([center, *neighbors]), np.array(neighbors), np.array(weights))


def sample(weighting=Weighting.BINARY, consequent=False, two_parents=False):
    # C at position 5 need not share a patch with A or B. It must still stay fixed.
    labels = list("AAAACCBDDD")
    weights = [1, 1, 1, 1] if weighting is Weighting.BINARY else [0.4, 0.8, 0.6, 0.3]
    c_weight = 1 if weighting is Weighting.BINARY else 0.6
    patches = [patch(0, [4, 6], [c_weight, weights[0]]),
               patch(1, [4, 7], [c_weight, weights[1]]),
               patch(2, [8], [weights[2]]), patch(3, [9], [weights[3]])]
    rows = [rule(["A_CENTER"], ["C_NEIGHBOR"])]
    if consequent:
        labels += list("EEEE")
        patches += [patch(center, [5], [c_weight]) for center in range(10, 14)]
        rows += [rule(["A_CENTER"], ["B_NEIGHBOR", "C_NEIGHBOR"])]
    else:
        if two_parents:
            labels += list("BD")
            patches += [patch(6, [4, 0], [c_weight, 1]), patch(10, [11], [1])]
            rows += [rule(["B_CENTER"], ["C_NEIGHBOR"])]
        rows += [rule(["A_CENTER", "B_NEIGHBOR"], ["C_NEIGHBOR"])]
    settings = Settings(weighting=weighting, method=Method.CN, radius=1,
                        min_support=0.1, min_lift=1.2, max_items_per_rule=4,
                        avoidance_max_lift=0.8)
    rules = pd.DataFrame(rows)
    rules.attrs["labels_kept_fixed"] = ("E",) if consequent else ()
    result = Result(rules, {"patches_kept": len(patches)}, patches, np.array(labels), settings)
    return result, rules


def direct_score(result, rules, labels):
    """Plain patch arithmetic, independent of the vectorized conditional scorer."""
    transactions, _ = build_transactions(result.patches, labels, result.settings)
    if not transactions:
        return 0

    def measure(row):
        def support(items):
            return sum(min(t.get(item, 0) for item in items) for t in transactions) / len(transactions)
        ant, con = support(row.antecedents), support(row.consequents)
        if ant == 0 or con == 0:
            return None
        confidence = support(row.antecedents + row.consequents) / ant
        if len(rules.iloc[-1].consequents) == 1:
            return confidence / con
        if con == 1:
            return None
        return (1 - con) / (1 - confidence) if confidence < 1 else np.inf

    child, parent = measure(rules.iloc[-1]), measure(rules.iloc[0])
    if child is None or parent is None:
        return 0
    if child == parent:
        return 1
    numerator, denominator = (parent, child) if rules.iloc[-1].kind == "avoids" else (child, parent)
    return numerator / denominator if denominator else np.inf


def enumerate_four_shuffles(monkeypatch):
    def generator(seed):
        orders = iter(np.roll(np.arange(6, 10), shift) for shift in range(4))
        return SimpleNamespace(permutation=lambda movable: next(orders))
    monkeypatch.setattr(conditional.np.random, "default_rng", generator)


def test_one_sample_workflow_from_p_values_to_classification():
    # Two known rules make every stage easy to inspect in a debugger.
    result, rules = sample()
    rules["lift"] = [1.0, 2.0]  # A -> C, then A + B -> C.

    # A cutoff of 1 keeps the example short; this is a workflow check.
    original = result.add_p_values(19, random_seed=7, max_individual_fdr=1)
    assert original.p_value.tolist() == pytest.approx([0.2, 0.1])
    assert original.individual_fdr.tolist() == [1.0, 1.0]

    plan = result.conditional_test_plan(original, max_individual_fdr=1)
    assert plan.fixed_types.tolist() == [("A", "C")]
    assert plan.status.tolist() == ["ready"]

    tested, comparisons = result.add_conditional_p_values(
        original, n_shuffles=19, random_seed=7, max_individual_fdr=1,
    )
    assert tested.conditional_p_value.iloc[1] == pytest.approx(0.3)
    assert tested.conditional_fdr.iloc[1] == 1.0
    assert "conditional_tests" not in tested
    assert comparisons.p_value.tolist() == pytest.approx([0.3])
    assert comparisons.observed_gain.tolist() == pytest.approx([2.0])

    classified = classify_rules(tested, max_individual_fdr=1)
    assert classified.rule_type.tolist() == ["pairwise", "complex-antecedents"]
    assert classified.complex_class.iloc[1] == "stronger_than_simpler"
    assert classified.adds_information.tolist() == [True, True]


@pytest.mark.parametrize("weighting", [Weighting.BINARY, Weighting.WEIGHTED])
@pytest.mark.parametrize("consequent", [False, True])
@pytest.mark.parametrize("kind", ["attracts", "avoids"])
def test_p_value_matches_explicit_patch_arithmetic(monkeypatch, weighting, consequent, kind):
    result, rules = sample(weighting, consequent)
    rules.loc[rules.index[-1], "kind"] = kind
    if kind == "avoids":
        result.labels[[6, 8]] = result.labels[[8, 6]]
        if consequent:
            # B and C co-occur outside A patches, so the consequent still exists.
            result.patches[-4:] = [patch(10, [5, 8], [1, 1]),
                                   patch(11, [6], [1]), patch(12, [7], [1]), patch(13, [9], [1])]
    observed = direct_score(result, rules, result.labels)
    shuffled_scores = []
    for shift in range(4):
        shuffled = result.labels.copy()
        shuffled[6:10] = result.labels[np.roll(np.arange(6, 10), shift)]
        shuffled_scores.append(direct_score(result, rules, shuffled))
    expected = (1 + sum(score >= observed or np.isclose(score, observed)
                        for score in shuffled_scores)) / 5 if observed > 1 else 1.0
    enumerate_four_shuffles(monkeypatch)
    tested, comparisons = result.add_conditional_p_values(rules, n_shuffles=4, random_seed=10)
    assert tested.conditional_p_value.iloc[-1] == pytest.approx(expected)
    assert comparisons.observed_gain.iloc[0] == pytest.approx(observed)
    assert tested.conditional_status.iloc[-1] == "tested"
    pd.testing.assert_frame_equal(tested[rules.columns], rules)


@pytest.mark.parametrize("weighting", [Weighting.BINARY, Weighting.WEIGHTED])
def test_fixed_comparison_is_conservative_over_its_entire_small_null(monkeypatch, weighting):
    p_values = []
    for b_position in range(6, 10):
        result, rules = sample(weighting)
        result.labels[[6, b_position]] = result.labels[[b_position, 6]]
        enumerate_four_shuffles(monkeypatch)
        tested, _ = result.add_conditional_p_values(rules, n_shuffles=4)
        p_values.append(tested.conditional_p_value.iloc[-1])
    for cutoff in [0.05, 0.2, 0.4, 0.6, 0.8]:
        assert np.mean(np.array(p_values) <= cutoff) <= cutoff


def test_all_parent_cells_and_configured_types_stay_fixed(monkeypatch):
    result, rules = sample(consequent=True)
    captured = []
    rebuild = conditional._transactions

    def capture(labels, adjacency, settings):
        captured.append(labels.copy())
        return rebuild(labels, adjacency, settings)

    monkeypatch.setattr(conditional, "_transactions", capture)
    plan = result.conditional_test_plan(rules)
    assert plan.fixed_types.tolist() == [("A", "C", "E")]
    assert plan.movable_cells.tolist() == [4]
    result.add_conditional_p_values(rules, n_shuffles=31, random_seed=5)
    fixed = np.isin(result.labels, ["A", "C", "E"])
    assert len(captured) == 32
    for shuffled in captured[1:]:
        np.testing.assert_array_equal(shuffled[fixed], captured[0][fixed])
        np.testing.assert_array_equal(shuffled.sum(axis=0), captured[0].sum(axis=0))
    assert any(not np.array_equal(labels, captured[0]) for labels in captured[1:])


def test_two_parents_use_separate_batches_and_combine_with_max():
    result, rules = sample(two_parents=True)
    plan = result.conditional_test_plan(rules)
    assert set(plan.fixed_types) == {("A", "C"), ("B", "C")}
    assert plan.attrs["shuffle_batches"] == 2
    # A cutoff of 1 needs no minimum shuffles, so conditional_fdr is calculated.
    tested, comparisons = result.add_conditional_p_values(
        rules, n_shuffles=39, random_seed=17, max_individual_fdr=1,
    )
    details = comparisons.loc[comparisons.rule_idx == len(rules) - 1]
    assert len(details) == 2
    assert details.status.tolist() == ["tested", "tested"]
    assert tested.conditional_p_value.iloc[-1] == details.p_value.max()
    # Four labels give 4 * C(4, 2) * 3 splits * 2 kinds = 144 candidates.
    assert tested.conditional_fdr.iloc[-1] == min(1, 144 * tested.conditional_p_value.iloc[-1])
    assert tested.conditional_fdr.iloc[:-1].isna().all()
    assert tested.attrs["conditional_test_summary"]["shuffled_tissues"] == 78


def test_rules_with_the_same_fixed_types_share_shuffles(monkeypatch):
    result, rules = sample()
    rules = pd.concat([rules, rules.iloc[[-1]]], ignore_index=True)
    rebuild = conditional._transactions
    calls = []

    def capture(*args):
        calls.append(1)
        return rebuild(*args)

    monkeypatch.setattr(conditional, "_transactions", capture)
    tested, _ = result.add_conditional_p_values(rules, n_shuffles=19, random_seed=6)
    assert len(calls) == 20  # One observed rebuild, then one shared batch.
    assert tested.conditional_p_value.iloc[1] == tested.conditional_p_value.iloc[2]
    assert tested.attrs["conditional_test_summary"]["shuffle_batches"] == 1


def test_reordering_rows_does_not_change_the_seeded_results():
    result, rules = sample(two_parents=True)
    first, _ = result.add_conditional_p_values(rules, n_shuffles=29, random_seed=13)
    reordered = rules.iloc[::-1]
    second, _ = result.add_conditional_p_values(reordered, n_shuffles=29, random_seed=13)
    pd.testing.assert_series_equal(first.conditional_p_value, second.conditional_p_value.iloc[::-1])
    pd.testing.assert_series_equal(first.conditional_fdr, second.conditional_fdr.iloc[::-1])


def test_comparisons_use_stored_rule_ids_not_row_positions():
    result, rules = sample(two_parents=True)
    rules = rules.assign(rule_idx=[10, 11, 12]).iloc[::-1]
    _, comparisons = result.add_conditional_p_values(rules, n_shuffles=9, random_seed=13)
    assert comparisons.rule_idx.tolist() == [12, 12]
    assert set(comparisons.simpler_idx) == {10, 11}


@pytest.mark.parametrize("fdr", [0.9, np.nan])
def test_only_significant_parents_are_tested(fdr):
    result, rules = sample()
    rules.loc[0, "individual_fdr"] = fdr
    assert result.conditional_test_plan(rules).empty
    tested, comparisons = result.add_conditional_p_values(rules, n_shuffles=9)
    assert tested.conditional_status.iloc[-1] == "no_significant_simpler"
    assert pd.isna(tested.conditional_p_value.iloc[-1])
    assert comparisons.empty


def test_parent_kind_does_not_restrict_the_comparison():
    result, rules = sample()
    rules.loc[0, "kind"] = "avoids"
    tested, _ = result.add_conditional_p_values(rules, n_shuffles=9, random_seed=2)
    assert tested.conditional_status.iloc[-1] == "tested"


@pytest.mark.parametrize("case,reason", [
    ("configured", "all_rule_types_fixed"), ("repeated", "all_rule_types_fixed"),
    ("one_label", "no_exchangeable_labels"), ("missing", "missing_cell_type"),
])
def test_impossible_comparisons_are_missing_not_significant(case, reason):
    result, rules = sample()
    if case == "configured":
        rules.attrs["labels_kept_fixed"] = ("B",)
    elif case == "repeated":
        rules.at[1, "antecedents"] = ("A_CENTER", "A_NEIGHBOR")
    elif case == "one_label":
        result.labels[7:10] = "B"
    else:
        rules.at[1, "antecedents"] = ("A_CENTER", "Missing_NEIGHBOR")
    plan = result.conditional_test_plan(rules)
    assert plan.status.tolist() == [reason]
    tested, _ = result.add_conditional_p_values(rules, n_shuffles=9)
    assert tested.conditional_status.iloc[-1] == "untestable"
    assert pd.isna(tested.conditional_p_value.iloc[-1])
    assert tested.attrs["conditional_test_summary"]["shuffle_batches"] == 0


def test_one_untestable_parent_prevents_a_partial_combined_p_value():
    result, rules = sample(two_parents=True)
    rules.attrs["labels_kept_fixed"] = ("A",)
    tested, comparisons = result.add_conditional_p_values(rules, n_shuffles=9, random_seed=4)
    assert set(comparisons.status) == {"tested", "all_rule_types_fixed"}
    assert pd.isna(tested.conditional_p_value.iloc[-1])


def test_crowding_filter_is_reapplied_for_observed_and_shuffled_tissues(monkeypatch):
    result, rules = sample(Weighting.WEIGHTED)
    result.settings = result.settings.replace(max_one_type_share=0.5)
    result.patches.append(patch(7, [8], [0.7]))  # D,D is dropped; B,D is retained.
    enumerate_four_shuffles(monkeypatch)
    observed = direct_score(result, rules, result.labels)
    scores = []
    for shift in range(4):
        shuffled = result.labels.copy()
        shuffled[6:10] = result.labels[np.roll(np.arange(6, 10), shift)]
        scores.append(direct_score(result, rules, shuffled))
    expected = (1 + sum(score >= observed for score in scores)) / 5
    tested, _ = result.add_conditional_p_values(rules, n_shuffles=4)
    assert tested.conditional_p_value.iloc[-1] == pytest.approx(expected)


def test_pairwise_mixed_and_empty_frames_are_not_tested():
    result, rules = sample()
    rules = pd.DataFrame([rules.iloc[0].to_dict(), rule(["A_CENTER", "B_NEIGHBOR"], ["C_NEIGHBOR", "D_NEIGHBOR"])])
    tested, comparisons = result.add_conditional_p_values(rules, n_shuffles=9)
    assert tested.conditional_status.tolist() == ["not_applicable", "not_applicable"]
    assert tested.conditional_p_value.isna().all()
    assert tested.conditional_fdr.isna().all()
    assert comparisons.empty
    empty, empty_comparisons = result.add_conditional_p_values(rules.iloc[:0], n_shuffles=9)
    assert empty.empty
    assert empty_comparisons.empty
    assert "conditional_p_value" in empty
    assert "conditional_fdr" in empty


def test_duplicate_indices_and_original_columns_are_preserved():
    result, rules = sample()
    rules.index = [8, 8]
    original = rules.copy(deep=True)
    tested, _ = result.add_conditional_p_values(rules, n_shuffles=9, random_seed=2,
                                                max_individual_fdr=1)
    pd.testing.assert_frame_equal(rules, original)
    pd.testing.assert_frame_equal(tested[original.columns], original)
    assert tested.conditional_status.tolist() == ["not_applicable", "tested"]
    assert pd.isna(tested.conditional_fdr.iloc[0])
    assert np.isfinite(tested.conditional_fdr.iloc[1])


@pytest.mark.parametrize("value", [None, 0, -1, 2, np.nan, np.inf])
def test_a_valid_parent_fdr_cutoff_is_required(value):
    result, rules = sample()
    with pytest.raises(ValueError, match="max_individual_fdr"):
        result.conditional_test_plan(rules, max_individual_fdr=value)


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_shuffle_count_must_be_a_positive_integer(value):
    result, rules = sample()
    with pytest.raises(ValueError, match="n_shuffles"):
        result.add_conditional_p_values(rules, n_shuffles=value)


def test_missing_fdr_column_does_not_silently_enable_all_parents():
    result, rules = sample()
    with pytest.raises(ValueError, match="individual_fdr"):
        result.conditional_test_plan(rules.drop(columns="individual_fdr"))


def test_undefined_observed_metric_is_untestable():
    result, rules = sample(consequent=True)
    result.labels[[6, 8]] = result.labels[[8, 6]]  # B and C never co-occur anywhere.
    tested, comparisons = result.add_conditional_p_values(rules, n_shuffles=9)
    assert tested.conditional_status.iloc[-1] == "untestable"
    assert comparisons.status.tolist() == ["undefined_metric"]
    assert pd.isna(tested.conditional_p_value.iloc[-1])


def test_original_fixed_label_configuration_is_recorded_and_inherited():
    result, _ = sample(consequent=True)
    tested = result.add_p_values(n_shuffles=0, labels_kept_fixed=("E",))
    tested["individual_fdr"] = 0.01
    assert tested.attrs["labels_kept_fixed"] == ("E",)
    plan = result.conditional_test_plan(tested)
    assert plan.fixed_types.tolist() == [("A", "C", "E")]
    extra = result.conditional_test_plan(tested, labels_kept_fixed=("B",))
    assert extra.fixed_types.tolist() == [("A", "B", "C", "E")]
    assert extra.status.tolist() == ["all_rule_types_fixed"]


def test_parent_type_names_are_exact_even_when_they_contain_a_star():
    result, rules = sample()
    result.labels = np.array(["A*"] * 4 + list("CCBDDD") + ["Ax"], dtype=object)
    rules.at[0, "antecedents"] = ("A*_CENTER",)
    rules.at[1, "antecedents"] = ("A*_CENTER", "B_NEIGHBOR")
    plan = result.conditional_test_plan(rules)
    assert plan.fixed_types.tolist() == [("A*", "C")]
    assert plan.movable_cells.tolist() == [5]  # Ax stays movable.


def test_score_handles_zero_and_infinite_metrics_without_nan():
    values = np.array([[np.inf, np.inf], [np.inf, np.inf], [0, 0], [0, 0], [2, 2]])
    comparisons = pd.DataFrame(dict(rule_pos=[0, 2, 0, 2, 4], simpler_pos=[1, 3, 4, 4, 0],
                                   metric=["conviction"] * 5,
                                   kind=["attracts", "avoids", "attracts", "avoids", "attracts"]))
    scores = conditional._scores(values, comparisons, np.arange(5))
    np.testing.assert_array_equal(scores, [1, 1, np.inf, np.inf, 0])


@pytest.mark.parametrize("one_sided,four_item_candidates", [(True, 128), (False, 224)])
def test_conditional_fdr_groups_sizes_and_pads_all_candidates(one_sided, four_item_candidates):
    result, _ = sample()
    settings = result.settings.replace(one_sided_complex_rules=one_sided)
    rules = pd.DataFrame([
        rule(["A_CENTER", "B_NEIGHBOR"], ["C_NEIGHBOR"]),
        rule(["A_CENTER"], ["B_NEIGHBOR", "C_NEIGHBOR"], kind="avoids"),
        rule(["B_CENTER", "C_NEIGHBOR"], ["D_NEIGHBOR"]),
        rule(["A_CENTER", "B_NEIGHBOR", "D_NEIGHBOR"], ["C_NEIGHBOR"]),
        rule(["C_CENTER"], ["A_NEIGHBOR", "D_NEIGHBOR"]),
        rule(["A_CENTER", "B_NEIGHBOR"], ["C_NEIGHBOR", "D_NEIGHBOR"]),
    ], index=[5] * 6)
    rules["conditional_p_value"] = [0.0001, 0.0004, 0.03, 0.0002, np.nan, np.nan]
    # BH ranks the three available 3-item values together, with 144 total candidates.
    expected = [0.0144, 0.0288, 1, four_item_candidates * 0.0002, np.nan, np.nan]
    actual = conditional._conditional_fdr(rules, result.labels, settings, n_shuffles=1, max_fdr=None)
    np.testing.assert_allclose(actual, expected, equal_nan=True)


def test_conditional_fdr_counts_only_enabled_kinds():
    result, rules = sample()
    settings = result.settings.replace(include_avoidance_rules=False)
    rules["conditional_p_value"] = [np.nan, 0.0001]
    # Attraction only halves the 3-item family from 144 to 72 candidates.
    actual = conditional._conditional_fdr(rules, result.labels, settings, n_shuffles=1, max_fdr=None)
    np.testing.assert_allclose(actual, [np.nan, 0.0072], equal_nan=True)


def test_too_few_shuffles_warns_and_leaves_conditional_fdr_missing(caplog):
    result, rules = sample()
    tested, _ = result.add_conditional_p_values(rules, n_shuffles=9, sample_id="FOV1")
    # 144 candidates and one tested rule need 2879 shuffles to reach BH <= 0.05.
    assert "[FOV1] Insufficient permutation resolution for 3-item" in caplog.text
    assert "At least 2879 shuffles" in caplog.text
    assert np.isfinite(tested.conditional_p_value.iloc[1])
    assert tested.conditional_fdr.isna().all()


def test_conditional_p_values_can_be_requested_without_fdr(caplog):
    result, rules = sample()
    tested, _ = result.add_conditional_p_values(rules, n_shuffles=9, random_seed=4,
                                                calculate_fdr=False)
    assert np.isfinite(tested.conditional_p_value.iloc[1])
    assert "conditional_fdr" not in tested
    assert "Insufficient permutation resolution" not in caplog.text

    raw = rules.drop(columns="individual_fdr")
    with pytest.raises(ValueError, match="individual_fdr"):
        result.add_conditional_p_values(raw, n_shuffles=9, calculate_fdr=False)


def test_more_rules_than_candidates_warns_instead_of_raising(caplog):
    result, rules = sample()
    rules = pd.concat([rules.iloc[[1]]] * 145)
    rules["conditional_p_value"] = 0.01
    actual = conditional._conditional_fdr(rules, result.labels, result.settings,
                                          n_shuffles=1, max_fdr=None)
    assert "145 3-item rules but only 144 candidates" in caplog.text
    assert np.isnan(actual).all()


def test_lists_and_tuples_can_be_mixed_in_rule_sides():
    result, rules = sample()
    rules["antecedents"] = rules["antecedents"].map(list)
    plan = result.conditional_test_plan(rules)
    assert plan.status.tolist() == ["ready"]


def test_numpy_integer_shuffle_count_is_accepted():
    conditional.check_shuffle_count(np.int64(9), "n_shuffles")
    with pytest.raises(ValueError, match="n_conditional_shuffles"):
        conditional.check_shuffle_count(np.True_, "n_conditional_shuffles")
