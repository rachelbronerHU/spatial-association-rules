"""Every shorter cell-type combination used to test a complex rule."""

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
    assert simpler_rules(rules) == [[], []]


@pytest.mark.parametrize("kind", ["attracts", "avoids"])
def test_every_shorter_combination_counts_for_either_kind(kind):
    rules = frame(rule(["A_CENTER", "B_NEIGHBOR"], ["C_NEIGHBOR"], kind=kind))
    assert simpler_rules(rules) == [[(("A",), ("C",)), (("B",), ("C",))]]


def test_every_shorter_size_counts_and_the_other_side_stays_fixed():
    rules = frame(rule(["A_CENTER", "B_NEIGHBOR", "D_NEIGHBOR"], ["C_NEIGHBOR"]),
                  rule(["A_CENTER"], ["B_NEIGHBOR", "C_NEIGHBOR"]))
    assert simpler_rules(rules) == [
        [(("A",), ("C",)), (("B",), ("C",)), (("D",), ("C",)),
         (("A", "B"), ("C",)), (("A", "D"), ("C",)), (("B", "D"), ("C",))],
        [(("A",), ("B",)), (("A",), ("C",))],
    ]


def test_subsets_preserve_rule_direction_and_repeated_types():
    rules = frame(rule(["A_CENTER", "A_NEIGHBOR", "B_NEIGHBOR"], ["C_NEIGHBOR"]))
    assert simpler_rules(rules) == [[
        (("A",), ("C",)), (("B",), ("C",)), (("A", "A"), ("C",)), (("A", "B"), ("C",)),
    ]]


def test_role_arrangements_do_not_change_the_subsets():
    rules = frame(rule(["A_CENTER", "B_NEIGHBOR"], ["C_NEIGHBOR"]),
                  rule(["B_CENTER", "A_NEIGHBOR"], ["C_NEIGHBOR"]))
    assert simpler_rules(rules) == [[(("A",), ("C",)), (("B",), ("C",))]] * 2
