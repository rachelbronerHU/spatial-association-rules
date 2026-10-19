"""Which shorter rules a complex rule is matched with."""

import pandas as pd
import pytest

from spatial_association_rules.complex_rules import simpler_rules


def rule(ants, cons, kind="attracts"):
    return dict(antecedents=tuple(ants), consequents=tuple(cons), kind=kind)


def frame(*rows):
    return pd.DataFrame(rows)


def test_pairwise_and_mixed_rules_have_no_simpler_rules():
    rules = frame(rule(["A_CENTER"], ["C_NEIGHBOR"]),
                  rule(["A_CENTER", "B_NEIGHBOR"], ["C_NEIGHBOR", "D_NEIGHBOR"]))
    raw = frame(rule(["A_CENTER"], ["C_NEIGHBOR"]), rule(["A_CENTER", "B_NEIGHBOR"], ["C_NEIGHBOR"]))
    assert simpler_rules(rules, raw) == [[], []]


@pytest.mark.parametrize("kind", ["attracts", "avoids"])
def test_any_measured_raw_rule_counts_whatever_its_kind(kind):
    rules = frame(rule(["A_CENTER", "B_NEIGHBOR"], ["C_NEIGHBOR"]))
    raw = frame(rule(["A_CENTER"], ["C_NEIGHBOR"], kind=kind))
    assert simpler_rules(rules, raw) == [[(("A",), ("C",))]]


def test_only_raw_rules_count_not_the_mined_table():
    rules = frame(rule(["A_CENTER", "B_NEIGHBOR"], ["C_NEIGHBOR"]), rule(["A_CENTER"], ["C_NEIGHBOR"]))
    assert simpler_rules(rules, frame(rule(["B_CENTER"], ["C_NEIGHBOR"]))) == [[(("B",), ("C",))], []]
    assert simpler_rules(rules, pd.DataFrame()) == [[], []]


def test_every_shorter_size_counts_and_the_other_side_stays_fixed():
    rules = frame(rule(["A_CENTER", "B_NEIGHBOR", "D_NEIGHBOR"], ["C_NEIGHBOR"]),
                  rule(["A_CENTER"], ["B_NEIGHBOR", "C_NEIGHBOR"]))
    raw = frame(rule(["A_CENTER"], ["C_NEIGHBOR"]), rule(["A_CENTER", "B_NEIGHBOR"], ["C_NEIGHBOR"]),
                rule(["A_CENTER"], ["B_NEIGHBOR"]), rule(["C_CENTER"], ["A_NEIGHBOR"]))
    assert simpler_rules(rules, raw) == [
        [(("A",), ("C",)), (("A", "B"), ("C",))],
        [(("A",), ("B",)), (("A",), ("C",))],
    ]


def test_matching_preserves_rule_direction_and_repeated_types():
    rules = frame(rule(["A_CENTER", "A_NEIGHBOR"], ["C_NEIGHBOR"]))
    assert simpler_rules(rules, frame(rule(["C_CENTER"], ["A_NEIGHBOR"]))) == [[]]
    assert simpler_rules(rules, frame(rule(["A_CENTER"], ["C_NEIGHBOR"]))) == [[(("A",), ("C",))]]


def test_role_arrangements_of_the_same_types_are_one_simpler_rule():
    rules = frame(rule(["A_CENTER", "A_NEIGHBOR", "B_NEIGHBOR"], ["C_NEIGHBOR"]))
    raw = frame(rule(["A_CENTER", "B_NEIGHBOR"], ["C_NEIGHBOR"]),
                rule(["B_CENTER", "A_NEIGHBOR"], ["C_NEIGHBOR"]),
                rule(["A_CENTER", "B_NEIGHBOR"], ["C_NEIGHBOR"], kind="avoids"))
    assert simpler_rules(rules, raw) == [[(("A", "B"), ("C",))]]
