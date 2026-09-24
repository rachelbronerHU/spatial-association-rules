"""Classification decisions with independently chosen effects and FDR values."""

import pandas as pd
import pytest

from spatial_association_rules import classify_rules, filter_rules
from spatial_association_rules.complex_rules import classify_complex_rules


def rule(ants, cons, lift=2.0, conviction=2.0, fdr=0.01, kind="attracts"):
    return dict(antecedents=tuple(ants), consequents=tuple(cons), kind=kind,
                lift=lift, conviction=conviction, individual_fdr=fdr)


def classify(*rows, **options):
    options.setdefault("min_lift_gain", 1.1)
    options.setdefault("max_individual_fdr", 0.05)
    return classify_rules(pd.DataFrame(rows), **options)


def test_types_and_unclassified_mixed_rules():
    result = classify(
        rule(["A_CENTER"], ["C_NEIGHBOR"]),
        rule(["A_CENTER", "B_NEIGHBOR"], ["C_NEIGHBOR"]),
        rule(["A_CENTER"], ["C_NEIGHBOR", "D_NEIGHBOR"]),
        rule(["A_CENTER", "B_NEIGHBOR"], ["C_NEIGHBOR", "D_NEIGHBOR"]),
    )
    assert result.rule_type.tolist() == ["pairwise", "complex-antecedents",
                                         "complex-consequents", "complex-mixed"]
    assert result.complex_class.iloc[0] is None
    assert result.adds_information.iloc[0]
    assert result.complex_class.iloc[3] is None
    assert pd.isna(result.adds_information.iloc[3])
    assert result.simpler_rules.iloc[3] == []


def test_empty_frame_has_nullable_information_column():
    result = classify_rules(pd.DataFrame())
    assert set(result.columns) == {"rule_type", "complex_class", "adds_information", "simpler_rules"}
    assert str(result.adds_information.dtype) == "boolean"


@pytest.mark.parametrize("kind", ["attracts", "avoids"])
def test_associations_between_consequents_do_not_classify_the_rule(kind):
    result = classify(
        rule(["A_CENTER"], ["B_NEIGHBOR", "C_NEIGHBOR"], kind=kind),
        rule(["B_CENTER"], ["C_NEIGHBOR"], kind=kind),
    )
    assert result.complex_class.iloc[0] == "no_simpler"
    assert result.adds_information.iloc[0]


@pytest.mark.parametrize("fdr", [0.9, float("nan"), None, pd.NA])
@pytest.mark.parametrize("complex_side", ["antecedents", "consequents"])
def test_no_significant_simpler_precedes_effect_comparison(fdr, complex_side):
    ants = ["A_CENTER", "B_NEIGHBOR"] if complex_side == "antecedents" else ["A_CENTER"]
    cons = ["C_NEIGHBOR"] if complex_side == "antecedents" else ["C_NEIGHBOR", "D_NEIGHBOR"]
    result = classify(
        rule(ants, cons, lift=10, conviction=10),
        rule(["A_CENTER"], ["C_NEIGHBOR"], fdr=fdr),
    )
    assert result.complex_class.iloc[0] == "no_significant_simpler"
    assert result.adds_information.iloc[0]
    assert len(result.simpler_rules.iloc[0]) == 1


@pytest.mark.parametrize("kind,value,expected", [
    ("attracts", 2.2, "stronger_than_simpler"),
    ("attracts", 2.19, "redundant_by_simpler"),
    ("avoids", 0.4 / 1.1, "stronger_than_simpler"),
    ("avoids", 0.4, "redundant_by_simpler"),
])
def test_antecedents_use_lift_not_conviction(kind, value, expected):
    result = classify(
        rule(["A_CENTER", "B_NEIGHBOR"], ["C_NEIGHBOR"], lift=value, conviction=100, kind=kind),
        rule(["A_CENTER"], ["C_NEIGHBOR"], lift=2 if kind == "attracts" else 0.4, kind=kind),
    )
    assert result.complex_class.iloc[0] == expected
    assert result.adds_information.iloc[0] == (expected == "stronger_than_simpler")


@pytest.mark.parametrize("kind,value,expected", [
    ("attracts", 2.2, "stronger_than_simpler"),
    ("attracts", 2.19, "redundant_by_simpler"),
    ("avoids", 0.8 / 1.1, "stronger_than_simpler"),
    ("avoids", 0.8, "redundant_by_simpler"),
])
def test_consequents_use_default_conviction_gain_not_lift(kind, value, expected):
    result = classify(
        rule(["A_CENTER"], ["C_NEIGHBOR", "D_NEIGHBOR"], conviction=value, lift=100, kind=kind),
        rule(["A_CENTER"], ["C_NEIGHBOR"], conviction=2 if kind == "attracts" else 0.8, kind=kind),
        min_lift_gain=100,
    )
    assert result.complex_class.iloc[0] == expected


@pytest.mark.parametrize("gain", [None, 0, 1])
@pytest.mark.parametrize("kind,value,simpler,expected", [
    ("attracts", 2.001, 2, True), ("attracts", 2, 2, False),
    ("avoids", 0.799, 0.8, True), ("avoids", 0.8, 0.8, False),
    ("attracts", float("inf"), float("inf"), False),
    ("attracts", float("inf"), 2, True), ("avoids", 0, 0, False),
])
def test_no_minimum_still_requires_strict_improvement(gain, kind, value, simpler, expected):
    result = classify(
        rule(["A_CENTER"], ["C_NEIGHBOR", "D_NEIGHBOR"], conviction=value, kind=kind),
        rule(["A_CENTER"], ["C_NEIGHBOR"], conviction=simpler, kind=kind),
        min_consequent_conviction_gain=gain,
    )
    assert result.adds_information.iloc[0] == expected


@pytest.mark.parametrize("kind,complex_value,strongest,weakest", [
    ("attracts", 3, 2.9, 1.5), ("avoids", 0.5, 0.4, 0.9),
])
def test_must_beat_every_significant_simpler_rule(kind, complex_value, strongest, weakest):
    result = classify(
        rule(["A_CENTER"], ["C_NEIGHBOR", "D_NEIGHBOR"], conviction=complex_value, kind=kind),
        rule(["A_CENTER"], ["C_NEIGHBOR"], conviction=strongest, kind=kind),
        rule(["A_CENTER"], ["D_NEIGHBOR"], conviction=weakest, kind=kind),
    )
    assert result.complex_class.iloc[0] == "redundant_by_simpler"


def test_strong_insignificant_simpler_does_not_block_improvement():
    result = classify(
        rule(["A_CENTER", "B_NEIGHBOR"], ["C_NEIGHBOR"], lift=3, fdr=float("nan")),
        rule(["A_CENTER"], ["C_NEIGHBOR"], lift=10, fdr=0.9),
        rule(["B_CENTER"], ["C_NEIGHBOR"], lift=2, fdr=0.05),
    )
    assert result.complex_class.iloc[0] == "stronger_than_simpler"
    assert pd.isna(result.individual_fdr.iloc[0])  # Its own significance is a separate question.


def test_every_shorter_size_counts_even_when_the_intermediate_rule_is_redundant():
    result = classify(
        rule(["A_CENTER", "B_NEIGHBOR", "D_NEIGHBOR"], ["C_NEIGHBOR"], lift=2.5),
        rule(["A_CENTER", "B_NEIGHBOR"], ["C_NEIGHBOR"], lift=2),
        rule(["A_CENTER"], ["C_NEIGHBOR"], lift=3),
    )
    assert result.complex_class.tolist() == ["redundant_by_simpler", "redundant_by_simpler", None]
    assert "A_CENTER -> C_NEIGHBOR" in result.simpler_rules.iloc[0]


def test_matching_preserves_rule_direction_and_repeated_types():
    result = classify(
        rule(["A_CENTER", "A_NEIGHBOR"], ["C_NEIGHBOR"]),
        rule(["C_CENTER"], ["A_NEIGHBOR"]),
    )
    assert result.rule_type.iloc[0] == "complex-antecedents"
    assert result.complex_class.iloc[0] == "no_simpler"


def test_all_role_arrangements_are_compared_by_type():
    result = classify(
        rule(["A_CENTER", "A_NEIGHBOR", "B_NEIGHBOR"], ["C_NEIGHBOR"], lift=3),
        rule(["A_CENTER", "B_NEIGHBOR"], ["C_NEIGHBOR"], lift=2),
        rule(["B_CENTER", "A_NEIGHBOR"], ["C_NEIGHBOR"], lift=2.9),
    )
    assert result.complex_class.iloc[0] == "redundant_by_simpler"


def test_effect_only_mode_with_no_cutoff_or_no_fdr_column():
    rows = [rule(["A_CENTER", "B_NEIGHBOR"], ["C_NEIGHBOR"], fdr=0.9),
            rule(["A_CENTER"], ["C_NEIGHBOR"], fdr=0.9)]
    assert classify(*rows, max_individual_fdr=None).complex_class.iloc[0] == "redundant_by_simpler"
    frame = pd.DataFrame(rows).drop(columns="individual_fdr")
    assert classify_rules(frame, max_individual_fdr=0.05).complex_class.iloc[0] == "redundant_by_simpler"


def test_input_values_order_and_duplicate_indices_are_preserved():
    frame = pd.DataFrame([
        rule(["A_CENTER", "B_NEIGHBOR"], ["C_NEIGHBOR"], lift=3),
        rule(["A_CENTER"], ["C_NEIGHBOR"]),
    ], index=[7, 7])
    original = frame.copy(deep=True)
    result = classify_rules(frame, max_individual_fdr=0.05)
    pd.testing.assert_frame_equal(frame, original)
    pd.testing.assert_frame_equal(result[original.columns], original)
    assert result.complex_class.tolist() == ["stronger_than_simpler", None]


@pytest.mark.parametrize("parameter", ["min_lift_gain", "min_consequent_conviction_gain"])
@pytest.mark.parametrize("value", [-1, 0.5, float("nan"), float("inf")])
def test_invalid_gain_is_rejected(parameter, value):
    with pytest.raises(ValueError, match=parameter):
        classify_rules(pd.DataFrame(), **{parameter: value})


@pytest.mark.parametrize("value", [0, -1, 1.1, float("nan"), float("inf")])
def test_invalid_fdr_cutoff_is_rejected(value):
    with pytest.raises(ValueError, match="max_individual_fdr"):
        classify_rules(pd.DataFrame(), max_individual_fdr=value)


def test_filter_rules_remains_an_alias():
    assert filter_rules is classify_rules


@pytest.mark.parametrize("classify_function", [classify_rules, classify_complex_rules])
@pytest.mark.parametrize("value,expected", [(2.1, False), (2.2, True)])
def test_both_default_gains_require_a_tenth_more(classify_function, value, expected):
    frame = pd.DataFrame([
        rule(["A_CENTER"], ["C_NEIGHBOR"], lift=2, conviction=2),
        rule(["A_CENTER", "B_NEIGHBOR"], ["C_NEIGHBOR"], lift=value),
        rule(["A_CENTER"], ["C_NEIGHBOR", "D_NEIGHBOR"], conviction=value),
    ])
    result = classify_function(frame)
    assert result.adds_information.tolist() == [True, expected, expected]


@pytest.mark.parametrize("gain", [None, 0])
def test_explicit_no_minimum_overrides_both_defaults(gain):
    frame = pd.DataFrame([
        rule(["A_CENTER"], ["C_NEIGHBOR"], lift=2, conviction=2),
        rule(["A_CENTER", "B_NEIGHBOR"], ["C_NEIGHBOR"], lift=2.001),
        rule(["A_CENTER"], ["C_NEIGHBOR", "D_NEIGHBOR"], conviction=2.001),
    ])
    result = classify_rules(frame, min_lift_gain=gain, min_consequent_conviction_gain=gain)
    assert result.adds_information.all()


@pytest.mark.parametrize("complex_side", ["antecedents", "consequents"])
@pytest.mark.parametrize("kind,value,simpler,expected", [
    ("attracts", 1.4, 0.8, "stronger_than_simpler"),
    ("avoids", 0.8, 1.4, "stronger_than_simpler"),
    ("attracts", 1.01, 0.99, "redundant_by_simpler"),
    ("avoids", 0.99, 1.01, "redundant_by_simpler"),
])
def test_opposite_kinds_use_the_complex_direction_and_still_require_gain(
        complex_side, kind, value, simpler, expected):
    ants = ["A_CENTER", "B_NEIGHBOR"] if complex_side == "antecedents" else ["A_CENTER"]
    cons = ["C_NEIGHBOR"] if complex_side == "antecedents" else ["C_NEIGHBOR", "D_NEIGHBOR"]
    opposite = "avoids" if kind == "attracts" else "attracts"
    result = classify(
        rule(ants, cons, lift=value, conviction=value, kind=kind),
        rule(["A_CENTER"], ["C_NEIGHBOR"], lift=simpler, conviction=simpler, kind=opposite),
    )
    assert result.complex_class.iloc[0] == expected
    assert result.adds_information.iloc[0] == (expected == "stronger_than_simpler")
    assert result.simpler_rules.iloc[0] == ["A_CENTER -> C_NEIGHBOR"]
    assert result.kind.tolist() == [kind, opposite]


@pytest.mark.parametrize("complex_side", ["antecedents", "consequents"])
@pytest.mark.parametrize("fdr", [0.9, float("nan")])
def test_opposite_kind_still_needs_to_pass_fdr(complex_side, fdr):
    ants = ["A_CENTER", "B_NEIGHBOR"] if complex_side == "antecedents" else ["A_CENTER"]
    cons = ["C_NEIGHBOR"] if complex_side == "antecedents" else ["C_NEIGHBOR", "D_NEIGHBOR"]
    result = classify(
        rule(ants, cons, lift=1.4, conviction=1.4),
        rule(["A_CENTER"], ["C_NEIGHBOR"], lift=0.8, conviction=0.8, kind="avoids", fdr=fdr),
    )
    assert result.complex_class.iloc[0] == "no_significant_simpler"
    assert result.adds_information.iloc[0]


@pytest.mark.parametrize("complex_side", ["antecedents", "consequents"])
@pytest.mark.parametrize("kind,value,same,opposite", [
    ("attracts", 1.4, 1.5, 0.8), ("avoids", 0.7, 0.6, 1.4),
])
def test_beating_opposite_kind_does_not_skip_a_stronger_same_kind_rule(
        complex_side, kind, value, same, opposite):
    ants = ["A_CENTER", "B_NEIGHBOR"] if complex_side == "antecedents" else ["A_CENTER"]
    cons = ["C_NEIGHBOR"] if complex_side == "antecedents" else ["C_NEIGHBOR", "D_NEIGHBOR"]
    other_ants = ["B_CENTER"] if complex_side == "antecedents" else ["A_CENTER"]
    other_cons = ["C_NEIGHBOR"] if complex_side == "antecedents" else ["D_NEIGHBOR"]
    result = classify(
        rule(ants, cons, lift=value, conviction=value, kind=kind),
        rule(["A_CENTER"], ["C_NEIGHBOR"], lift=opposite, conviction=opposite,
             kind="avoids" if kind == "attracts" else "attracts"),
        rule(other_ants, other_cons, lift=same, conviction=same, kind=kind),
    )
    assert result.complex_class.iloc[0] == "redundant_by_simpler"
    assert len(result.simpler_rules.iloc[0]) == 2
