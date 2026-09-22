"""
Check corrected p-values by sample and rule size, with attraction and avoidance together.
"""

from importlib import import_module

import numpy as np
import pandas as pd
import pytest

from spatial_association_rules import Method, Settings, Weighting
from spatial_association_rules.mine import Result
from spatial_association_rules.rules import count_candidate_rules
from spatial_association_rules.validation.false_discovery import (
    false_discovery_rates, minimum_shuffles_for_fdr,
)


def test_false_discovery_rates_are_not_simply_stricter_when_wider():
    """BH divides by rank as well as count, so a wider family can lower an answer."""
    narrow = false_discovery_rates([0.04] + [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])[0]
    wide = false_discovery_rates([0.04] + [0.001] * 700 + [0.5] * 299)[0]
    assert wide < narrow


def test_the_correction_only_ever_raises_a_p_value():
    """Within one family, no rule ends up looking better than its raw p-value."""
    raw = [0.01, 0.02]
    corrected = false_discovery_rates(raw)
    assert (corrected >= raw).all()
    assert corrected[-1] == pytest.approx(0.02), "the largest is never raised"


def test_full_family_matches_the_worked_example():
    raw = [0.001, 0.009, 0.02, 0.2]
    assert false_discovery_rates(raw, n_tests=20) == pytest.approx([0.02, 0.09, 2 / 15, 1.0])


@pytest.mark.parametrize("raw", [[0.02, 0.001, 0.009, 0.2], [0.04, 0.04, 1], [1, 1]])
def test_counting_omitted_rules_matches_explicit_p_values_of_one(raw):
    padded = false_discovery_rates(raw + [1.0] * 20)[:len(raw)]
    np.testing.assert_allclose(false_discovery_rates(raw, n_tests=len(raw) + 20), padded)


def test_an_empty_family_needs_no_correction():
    assert false_discovery_rates([], n_tests=0).size == 0
    assert false_discovery_rates([], n_tests=100).size == 0


@pytest.mark.parametrize("n_tests", [0, -1, 2.5])
def test_invalid_family_size_is_rejected(n_tests):
    with pytest.raises(ValueError, match="n_tests"):
        false_discovery_rates([0.01], n_tests=n_tests)


@pytest.mark.parametrize("n_labels,max_items,per_kind", [
    (0, 4, 0), (1, 4, 1), (2, 2, 4), (2, 3, 10), (3, 4, 57), (4, 5, 260),
])
@pytest.mark.parametrize("avoidance", [False, True])
def test_candidate_family_counts_centers_neighbor_splits_and_kinds(n_labels, max_items, per_kind, avoidance):
    settings = Settings(weighting=Weighting.BINARY, method=Method.CN, radius=1,
                        min_support=0.1, min_lift=1.2, max_items_per_rule=max_items,
                        include_avoidance_rules=avoidance, avoidance_max_lift=0.8)
    labels = [str(i) for i in range(n_labels)]
    assert count_candidate_rules(labels, settings) == per_kind * (2 if avoidance else 1)


def test_candidate_family_uses_label_eligibility_but_not_effect_thresholds():
    labels = ["A"] * 5 + ["B"] * 5 + ["rare"]
    settings = Settings(weighting=Weighting.BINARY, method=Method.CN, radius=1,
                        min_support=0.1, min_lift=1.2, max_items_per_rule=3,
                        min_label_count=5, avoidance_max_lift=0.8)
    assert count_candidate_rules(labels, settings) == 20
    assert count_candidate_rules(labels, settings.replace(min_lift=100, min_support=0.9)) == 20
    assert count_candidate_rules(labels, settings.replace(min_label_count=None, min_label_share=0.4)) == 20


def test_candidate_counts_by_size_partition_the_full_family():
    settings = Settings(weighting=Weighting.BINARY, method=Method.CN, radius=1,
                        min_support=0.1, min_lift=1.2, max_items_per_rule=5,
                        avoidance_max_lift=0.8)
    labels = [str(i) for i in range(32)]
    counts = [count_candidate_rules(labels, settings, n_items=k) for k in range(2, 6)]
    assert counts == [2048, 95232, 2222080, 34521600]  # Both kinds together.
    assert sum(counts) == count_candidate_rules(labels, settings)
    assert count_candidate_rules(labels, settings.replace(max_items_per_rule=3), n_items=4) == 0


def test_minimum_shuffle_budget_includes_the_equality_boundary():
    needed = minimum_shuffles_for_fdr(18, 2, 0.05)
    assert needed == 179
    assert (false_discovery_rates([1 / (needed + 1)] * 2, n_tests=18) <= 0.05).all()
    assert (false_discovery_rates([1 / needed] * 2, n_tests=18) > 0.05).all()
    assert minimum_shuffles_for_fdr(18, 2, 1) == 0


@pytest.mark.parametrize("shuffles,cutoff,blocked", [
    (178, 0.05, {2, 3}), (179, 0.05, {3}), (199, 0.05, {3}),
    (1079, 0.05, set()), (199, None, set()), (199, 0.5, set()),
    (0, 0.05, {2, 3}), (0, 1, set()),
])
def test_resolution_check_precedes_shuffles_and_only_skips_impossible_sizes(
        monkeypatch, caplog, shuffles, cutoff, blocked):
    settings = Settings(weighting=Weighting.BINARY, method=Method.CN, radius=1,
                        min_support=0.1, min_lift=1.2, max_items_per_rule=3,
                        avoidance_max_lift=0.8)
    mined = pd.DataFrame([
        (("A_CENTER",), ("B_NEIGHBOR",), "attracts"),
        (("B_CENTER",), ("A_NEIGHBOR",), "avoids"),
        (("A_CENTER",), ("B_NEIGHBOR", "C_NEIGHBOR"), "attracts"),
    ], columns=["antecedents", "consequents", "kind"], index=[42, 7, 9])
    result = Result(mined, {}, [], np.array(["A", "B", "C"]), settings)

    def shuffling(rules, *args):
        # A blocked size still gets its raw p-values, so every mined rule is shuffled.
        assert len(rules) == 3
        assert len(caplog.records) == len(blocked), "warnings must precede shuffling"
        for size in blocked:
            assert f"[FOV1] Insufficient permutation resolution for {size}-item" in caplog.text
        return np.full(len(rules), 1 / (shuffles + 1))

    monkeypatch.setattr(import_module("spatial_association_rules.mine"), "p_values_for", shuffling)
    tested = result.add_p_values(shuffles, max_individual_fdr=cutoff, sample_id="FOV1")
    assert tested.index.tolist() == [42, 7, 9]
    assert tested.p_value.tolist() == [1 / (shuffles + 1)] * 3
    for pos, size in enumerate([2, 2, 3]):
        adjusted = tested.iloc[pos].individual_fdr
        if size in blocked:
            assert np.isnan(adjusted)
        else:
            # For pairs: 18 candidates / 2 mined, including opposite kinds.
            # For triples: 54 candidates / 1 mined, independent of the pair ranks.
            factor = 9 if size == 2 else 54
            assert adjusted == pytest.approx(min(1, factor / (shuffles + 1)))


@pytest.mark.parametrize("cutoff", [0, -0.1, 1.1, float("nan"), float("inf")])
def test_invalid_fdr_cutoff_is_rejected_before_shuffling(cutoff):
    result = Result(pd.DataFrame(), {}, [], np.array([]), None)
    with pytest.raises(ValueError, match="max_individual_fdr"):
        result.add_p_values(100, max_individual_fdr=cutoff)
