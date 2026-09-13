"""
Tests for the per-sample correction: many rules were tested at once, so correct
across them together. Attached as `individual_fdr` by `Result.add_p_values`.
"""

import numpy as np
import pytest

from spatial_association_rules import Method, Settings, Weighting
from spatial_association_rules.rules import count_candidate_rules
from spatial_association_rules.validation.false_discovery import false_discovery_rates


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
