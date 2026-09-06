"""
Tests for the per-sample correction: many rules were tested at once, so correct
across them together. Attached as `individual_fdr` by `Result.add_p_values`.
"""

import pytest

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
