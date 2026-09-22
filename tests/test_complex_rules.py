"""
Tests for "does a longer rule earn its place next to its shorter parts".

Every frame is built by hand so the lift and p-value of each rule are known before
anything runs, and each test pins one branch of the decision tree.
"""

import pandas as pd
import pytest

from spatial_association_rules.complex_rules import classify_complex_rules
from spatial_association_rules.validation.false_discovery import false_discovery_rates

ATTRACTS = "attracts"
AVOIDS = "avoids"

# Well under the 0.05 gate, and well over it. Nothing is shuffled here, so these are
# the raw p-values a permutation test would have handed back.
CONVINCING = 0.0001
NOISE = 0.9

FDR = 0.05
GAIN = 1.1


def rule(antecedents, consequents, lift, kind=ATTRACTS, p_value=None):
    row = {
        "antecedents": tuple(antecedents),
        "consequents": tuple(consequents),
        "lift": lift,
        "kind": kind,
    }
    if p_value is not None:
        row["p_value"] = p_value
    return row


def classify(*rows, min_lift_gain=GAIN, max_individual_fdr=FDR):
    """Classify example rules after correcting their p-values as one test group."""
    frame = pd.DataFrame(list(rows))
    if "p_value" in frame.columns:
        frame["individual_fdr"] = false_discovery_rates(frame["p_value"].values)
    return classify_complex_rules(frame, min_lift_gain, max_individual_fdr)


def class_of(frame, position):
    return frame.iloc[position]["complex_class"]


# --- shape --------------------------------------------------------------------

def test_pairwise_rules_are_left_alone():
    """Nothing simpler exists to compare a two-item rule against."""
    frame = classify(rule(["A_CENTER"], ["B_NEIGHBOR"], lift=2.0, p_value=CONVINCING))
    assert frame.iloc[0]["rule_type"] == "pairwise"
    assert frame.iloc[0]["complex_class"] is None
    assert frame.iloc[0]["adds_information"]


def test_an_empty_frame_still_gets_the_columns():
    """The four this pass adds. individual_fdr comes from add_p_values, not from here."""
    frame = classify_complex_rules(pd.DataFrame(), GAIN, FDR)
    for column in ("rule_type", "complex_class", "adds_information", "simpler_rules"):
        assert column in frame.columns


# --- settled on lift alone ----------------------------------------------------

def test_a_rule_with_no_simpler_parts_found_is_new():
    """A + B -> C, where neither A -> C nor B -> C was mined."""
    frame = classify(
        rule(["A_CENTER", "B_CENTER"], ["C_NEIGHBOR"], lift=3.0, p_value=CONVINCING),
    )
    assert frame.iloc[0]["rule_type"] == "ant-complex"
    assert class_of(frame, 0) == "new"
    assert frame.iloc[0]["adds_information"]


def test_beating_every_simpler_rule_is_a_stronger_effect():
    """3.0 clears both 2.0 and 1.5 by the 1.1 gain, so the pair says something new."""
    frame = classify(
        rule(["A_CENTER", "B_CENTER"], ["C_NEIGHBOR"], lift=3.0, p_value=CONVINCING),
        rule(["A_CENTER"], ["C_NEIGHBOR"], lift=2.0, p_value=CONVINCING),
        rule(["B_CENTER"], ["C_NEIGHBOR"], lift=1.5, p_value=CONVINCING),
    )
    assert class_of(frame, 0) == "stronger_effect"
    assert frame.iloc[0]["adds_information"]


def test_avoidance_must_clear_the_strongest_simpler_rule_not_the_weakest():
    """
    For avoids, lower lift is stronger, so clearing every sub-rule comes down to the
    *lowest* of them — the strongest avoidance. Here 0.5 clears B -> C (0.9/1.1 = 0.82)
    but not A -> C (0.4/1.1 = 0.36), so A -> C is still standing and the rule is left
    open rather than called an improvement.
    """
    frame = classify(
        rule(["A_CENTER", "B_CENTER"], ["C_NEIGHBOR"], lift=0.5, kind=AVOIDS, p_value=CONVINCING),
        rule(["A_CENTER"], ["C_NEIGHBOR"], lift=0.4, kind=AVOIDS, p_value=CONVINCING),
        rule(["B_CENTER"], ["C_NEIGHBOR"], lift=0.9, kind=AVOIDS, p_value=CONVINCING),
    )
    assert class_of(frame, 0) == "redundant_by_simpler"


def test_avoidance_below_every_simpler_rule_is_a_stronger_effect():
    """0.2 clears both 0.36 and 0.82, so it beats the whole field."""
    frame = classify(
        rule(["A_CENTER", "B_CENTER"], ["C_NEIGHBOR"], lift=0.2, kind=AVOIDS, p_value=CONVINCING),
        rule(["A_CENTER"], ["C_NEIGHBOR"], lift=0.4, kind=AVOIDS, p_value=CONVINCING),
        rule(["B_CENTER"], ["C_NEIGHBOR"], lift=0.9, kind=AVOIDS, p_value=CONVINCING),
    )
    assert class_of(frame, 0) == "stronger_effect"


def test_consequents_that_already_sit_together_for_real_make_the_rule_redundant():
    """A -> B + C explains nothing when B and C sit together this strongly, really."""
    frame = classify(
        rule(["A_CENTER"], ["B_NEIGHBOR", "C_NEIGHBOR"], lift=2.0, p_value=CONVINCING),
        rule(["B_NEIGHBOR"], ["C_NEIGHBOR"], lift=5.0, p_value=CONVINCING),
    )
    assert frame.iloc[0]["rule_type"] == "con-complex"
    assert class_of(frame, 0) == "consequent_driven"
    assert not frame.iloc[0]["adds_information"]


def test_one_arrangement_is_enough_for_a_pair_to_count():
    """Roles are stripped, so C -> B alone still shows B and C sit together."""
    frame = classify(
        rule(["A_CENTER"], ["B_NEIGHBOR", "C_NEIGHBOR"], lift=2.0, p_value=CONVINCING),
        rule(["C_CENTER"], ["B_NEIGHBOR"], lift=5.0, p_value=CONVINCING),
    )
    assert class_of(frame, 0) == "consequent_driven"


# --- the consequent link has to be real ---------------------------------------

def test_a_link_that_is_only_noise_lets_the_rule_stand():
    """B and C look tightly packed, but that packing is a fluke, so it proves nothing."""
    frame = classify(
        rule(["A_CENTER"], ["B_NEIGHBOR", "C_NEIGHBOR"], lift=2.0, p_value=CONVINCING),
        rule(["B_NEIGHBOR"], ["C_NEIGHBOR"], lift=5.0, p_value=NOISE),
    )
    assert class_of(frame, 0) == "consequent_is_noise"
    assert frame.iloc[0]["adds_information"]


def test_a_noisy_link_is_not_re_asked_the_simpler_question():
    """
    A -> B would have condemned this rule, but the consequent check already claimed it
    and does not hand it back. Deliberate: the rule keeps consequent_is_noise.
    """
    frame = classify(
        rule(["A_CENTER"], ["B_NEIGHBOR", "C_NEIGHBOR"], lift=2.0, p_value=CONVINCING),
        rule(["B_NEIGHBOR"], ["C_NEIGHBOR"], lift=5.0, p_value=NOISE),
        rule(["A_CENTER"], ["B_NEIGHBOR"], lift=2.0, p_value=CONVINCING),
    )
    assert class_of(frame, 0) == "consequent_is_noise"


def test_a_pair_cannot_borrow_strength_from_one_side_and_evidence_from_the_other():
    """
    B -> C is strong but a fluke; C -> B is real but too weak to dismiss anything.
    Neither arrangement carries both, so the pair does not hold.
    """
    frame = classify(
        rule(["A_CENTER"], ["B_NEIGHBOR", "C_NEIGHBOR"], lift=2.0, p_value=CONVINCING),
        rule(["B_NEIGHBOR"], ["C_NEIGHBOR"], lift=5.0, p_value=NOISE),
        rule(["C_CENTER"], ["B_NEIGHBOR"], lift=0.5, p_value=CONVINCING),
    )
    assert class_of(frame, 0) == "consequent_is_noise"


def test_every_pair_of_consequent_types_has_to_be_linked():
    """
    B and C sit together, but nothing links either of them to D, so the three of them
    are not a niche and the rule goes to the ordinary comparison instead.
    """
    frame = classify(
        rule(["A_CENTER"], ["B_NEIGHBOR", "C_NEIGHBOR", "D_NEIGHBOR"], lift=2.0, p_value=CONVINCING),
        rule(["B_NEIGHBOR"], ["C_NEIGHBOR"], lift=5.0, p_value=CONVINCING),
    )
    assert class_of(frame, 0) == "new"
    assert frame.iloc[0]["adds_information"]


def test_one_noisy_pair_out_of_three_is_enough_to_let_the_rule_stand():
    frame = classify(
        rule(["A_CENTER"], ["B_NEIGHBOR", "C_NEIGHBOR", "D_NEIGHBOR"], lift=2.0, p_value=CONVINCING),
        rule(["B_NEIGHBOR"], ["C_NEIGHBOR"], lift=5.0, p_value=CONVINCING),
        rule(["B_NEIGHBOR"], ["D_NEIGHBOR"], lift=5.0, p_value=CONVINCING),
        rule(["C_NEIGHBOR"], ["D_NEIGHBOR"], lift=5.0, p_value=NOISE),
    )
    assert class_of(frame, 0) == "consequent_is_noise"


def test_all_three_pairs_real_makes_it_consequent_driven():
    frame = classify(
        rule(["A_CENTER"], ["B_NEIGHBOR", "C_NEIGHBOR", "D_NEIGHBOR"], lift=2.0, p_value=CONVINCING),
        rule(["B_NEIGHBOR"], ["C_NEIGHBOR"], lift=5.0, p_value=CONVINCING),
        rule(["B_NEIGHBOR"], ["D_NEIGHBOR"], lift=5.0, p_value=CONVINCING),
        rule(["C_NEIGHBOR"], ["D_NEIGHBOR"], lift=5.0, p_value=CONVINCING),
    )
    assert class_of(frame, 0) == "consequent_driven"
    assert not frame.iloc[0]["adds_information"]


# --- what lift left open, significance settles --------------------------------

def test_matching_a_convincing_simpler_rule_is_redundant():
    """2.0 does not clear 2.0 by the gain, and A -> C stands up on its own."""
    frame = classify(
        rule(["A_CENTER", "B_CENTER"], ["C_NEIGHBOR"], lift=2.0, p_value=CONVINCING),
        rule(["A_CENTER"], ["C_NEIGHBOR"], lift=2.0, p_value=CONVINCING),
    )
    assert class_of(frame, 0) == "redundant_by_simpler"
    assert not frame.iloc[0]["adds_information"]


def test_matching_only_noise_keeps_the_complex_rule():
    """Matching a sub-rule that is itself noise is no reason to throw the pair away."""
    frame = classify(
        rule(["A_CENTER", "B_CENTER"], ["C_NEIGHBOR"], lift=2.0, p_value=CONVINCING),
        rule(["A_CENTER"], ["C_NEIGHBOR"], lift=2.0, p_value=NOISE),
        rule(["B_CENTER"], ["C_NEIGHBOR"], lift=2.0, p_value=NOISE),
    )
    assert class_of(frame, 0) == "simpler_are_noise"
    assert frame.iloc[0]["adds_information"]


def test_one_convincing_match_is_enough_to_make_it_redundant():
    frame = classify(
        rule(["A_CENTER", "B_CENTER"], ["C_NEIGHBOR"], lift=2.0, p_value=CONVINCING),
        rule(["A_CENTER"], ["C_NEIGHBOR"], lift=2.0, p_value=NOISE),
        rule(["B_CENTER"], ["C_NEIGHBOR"], lift=2.0, p_value=CONVINCING),
    )
    assert class_of(frame, 0) == "redundant_by_simpler"


def test_without_a_threshold_everything_left_open_reads_redundant():
    """max_individual_fdr=None is the lift-only behaviour."""
    frame = classify(
        rule(["A_CENTER", "B_CENTER"], ["C_NEIGHBOR"], lift=2.0, p_value=NOISE),
        rule(["A_CENTER"], ["C_NEIGHBOR"], lift=2.0, p_value=NOISE),
        max_individual_fdr=None,
    )
    assert class_of(frame, 0) == "redundant_by_simpler"


def test_rules_never_corrected_fall_back_to_lift_instead_of_raising():
    """filter_rules is public, and mine() hands back rules before add_p_values runs."""
    frame = classify(
        rule(["A_CENTER", "B_CENTER"], ["C_NEIGHBOR"], lift=2.0),
        rule(["A_CENTER"], ["C_NEIGHBOR"], lift=2.0),
    )
    assert class_of(frame, 0) == "redundant_by_simpler"
    assert "individual_fdr" not in frame.columns, "nothing was tested, so nothing to correct"


@pytest.mark.parametrize("cutoff,expected", [(FDR, "simpler_are_noise"), (None, "redundant_by_simpler")])
def test_missing_adjustment_does_not_pass_an_enabled_fdr_gate(cutoff, expected):
    frame = pd.DataFrame([
        rule(["A_CENTER", "B_NEIGHBOR"], ["C_NEIGHBOR"], lift=2.0),
        rule(["A_CENTER"], ["C_NEIGHBOR"], lift=2.0),
    ])
    frame["individual_fdr"] = [0.01, float("nan")]
    classified = classify_complex_rules(frame, GAIN, cutoff)
    assert class_of(classified, 0) == expected
    assert pd.isna(classified.iloc[1].individual_fdr)


# --- the correction is an input, and every class keeps it ----------------------

def test_a_dismissed_rule_keeps_its_corrected_value():
    """
    Being dismissed settles nothing permanently — a consequent-driven rule can come
    back once the link is questioned — so its own FDR must survive the pass.
    """
    frame = classify(
        rule(["A_CENTER"], ["B_NEIGHBOR", "C_NEIGHBOR"], lift=2.0, p_value=CONVINCING),
        rule(["B_NEIGHBOR"], ["C_NEIGHBOR"], lift=5.0, p_value=CONVINCING),
    )
    assert class_of(frame, 0) == "consequent_driven"
    assert not frame["individual_fdr"].isna().any()


def test_everything_worth_keeping_carries_a_corrected_value():
    """
    A rule left open by lift keeps its FDR too, so one promoted to simpler_are_noise
    is never left without a number of its own.
    """
    frame = classify(
        rule(["A_CENTER", "B_CENTER"], ["C_NEIGHBOR"], lift=2.0, p_value=CONVINCING),
        rule(["A_CENTER"], ["C_NEIGHBOR"], lift=2.0, p_value=NOISE),
        rule(["B_CENTER"], ["C_NEIGHBOR"], lift=2.0, p_value=NOISE),
    )
    assert class_of(frame, 0) == "simpler_are_noise"
    kept = frame[frame["adds_information"]]
    assert not kept["individual_fdr"].isna().any()


# --- item counts, roles included ----------------------------------------------

def test_the_same_cell_type_twice_is_still_two_items():
    """
    Paneth in the middle AND Paneth around it is a three-item rule, not a pairwise
    one. It has two shorter versions to answer to.
    """
    frame = classify(
        rule(["Paneth_CENTER", "Paneth_NEIGHBOR"], ["Epithelial_NEIGHBOR"], lift=3.0, p_value=CONVINCING),
        rule(["Paneth_CENTER"], ["Epithelial_NEIGHBOR"], lift=2.0, p_value=CONVINCING),
    )
    assert frame.iloc[0]["rule_type"] == "ant-complex"
    assert class_of(frame, 0) == "stronger_effect"


# --- many items on both sides -------------------------------------------------

def test_a_rule_long_on_both_sides_is_classified_too():
    """A + B -> C + D is long on both sides at once, and still gets a class."""
    frame = classify(
        rule(["A_CENTER", "B_NEIGHBOR"], ["C_NEIGHBOR", "D_NEIGHBOR"], lift=2.0, p_value=CONVINCING),
    )
    assert frame.iloc[0]["rule_type"] == "both-complex"
    assert class_of(frame, 0) == "new"


def test_it_is_weighed_against_shorter_rules_from_either_side():
    """Items come off the antecedent or the consequent, not just one side."""
    frame = classify(
        rule(["A_CENTER", "B_NEIGHBOR"], ["C_NEIGHBOR", "D_NEIGHBOR"], lift=2.0, p_value=CONVINCING),
        rule(["A_CENTER"], ["C_NEIGHBOR", "D_NEIGHBOR"], lift=2.0, p_value=CONVINCING),
    )
    assert class_of(frame, 0) == "redundant_by_simpler"
    assert frame.iloc[0]["simpler_rules"] == ["A_CENTER -> C_NEIGHBOR + D_NEIGHBOR"]


def test_a_dismissed_shorter_rule_is_still_something_to_answer_to():
    """
    A -> C + D is redundant, but it exists, so the longer rule is not 'new'.

    Calling it new would let a rule escape by the accident of its parent being
    dismissed. It is still weighed on lift, and here it adds nothing.
    """
    frame = classify(
        rule(["A_CENTER", "B_NEIGHBOR"], ["C_NEIGHBOR", "D_NEIGHBOR"], lift=2.0, p_value=CONVINCING),
        rule(["A_CENTER"], ["C_NEIGHBOR", "D_NEIGHBOR"], lift=2.0, p_value=CONVINCING),
        rule(["A_CENTER"], ["C_NEIGHBOR"], lift=2.0, p_value=CONVINCING),
        rule(["A_CENTER"], ["D_NEIGHBOR"], lift=2.0, p_value=CONVINCING),
    )
    assert class_of(frame, 1) == "redundant_by_simpler"     # A -> C + D falls first
    assert class_of(frame, 0) == "redundant_by_simpler"     # and still counts against A + B
    assert frame.iloc[0]["simpler_rules"] != []


def test_beating_a_dismissed_shorter_rule_still_earns_a_place():
    """
    The dismissed parent is a yardstick, not a verdict to inherit.

    Same rules as above, but the long one triples the lift. Inheriting the parent's
    redundancy would have thrown away the strongest finding in the sample.
    """
    frame = classify(
        rule(["A_CENTER", "B_NEIGHBOR"], ["C_NEIGHBOR", "D_NEIGHBOR"], lift=6.0, p_value=CONVINCING),
        rule(["A_CENTER"], ["C_NEIGHBOR", "D_NEIGHBOR"], lift=2.0, p_value=CONVINCING),
        rule(["A_CENTER"], ["C_NEIGHBOR"], lift=2.0, p_value=CONVINCING),
        rule(["A_CENTER"], ["D_NEIGHBOR"], lift=2.0, p_value=CONVINCING),
    )
    assert class_of(frame, 1) == "redundant_by_simpler"
    assert class_of(frame, 0) == "stronger_effect"


def test_a_rule_answers_to_every_shorter_rule_inside_it_not_only_the_next_one_down():
    """
    A + B -> C + D beats A -> C + D, but A -> C beats them both.

    One level down is not enough: the middle rule collapsed to lift 1.0, so beating it
    proves nothing. The two-item rule is inside the four-item rule and must be asked.
    """
    frame = classify(
        rule(["A_CENTER", "B_NEIGHBOR"], ["C_NEIGHBOR", "D_NEIGHBOR"], lift=1.5, p_value=CONVINCING),
        rule(["A_CENTER"], ["C_NEIGHBOR", "D_NEIGHBOR"], lift=1.0, p_value=CONVINCING),
        rule(["A_CENTER"], ["C_NEIGHBOR"], lift=2.0, p_value=CONVINCING),
    )
    assert class_of(frame, 1) == "redundant_by_simpler"
    assert class_of(frame, 0) == "redundant_by_simpler"
    assert "A_CENTER -> C_NEIGHBOR" in frame.iloc[0]["simpler_rules"]


def test_shorter_rules_are_judged_before_longer_ones():
    """The order is by item count, not by row order."""
    frame = classify(
        rule(["A_CENTER", "B_CENTER"], ["C_NEIGHBOR"], lift=3.0, p_value=CONVINCING),
        rule(["A_CENTER"], ["C_NEIGHBOR"], lift=2.0, p_value=CONVINCING),
    )
    assert class_of(frame, 1) is None                       # pairwise, judged first
    assert class_of(frame, 0) == "stronger_effect"


# --- avoidance asks the same question, mirrored -------------------------------

def test_consequents_that_already_exclude_each_other_explain_an_avoidance_rule():
    """
    'A keeps away from B and C together' is not news when B and C already keep away
    from each other — almost nothing sits by both, with or without A.
    """
    frame = classify(
        rule(["A_CENTER"], ["B_NEIGHBOR", "C_NEIGHBOR"], lift=0.5, kind=AVOIDS, p_value=CONVINCING),
        rule(["B_NEIGHBOR"], ["C_NEIGHBOR"], lift=0.2, kind=AVOIDS, p_value=CONVINCING),
    )
    assert class_of(frame, 0) == "consequent_driven"


def test_consequents_that_attract_do_not_explain_an_avoidance_rule():
    """The backing rule has to do the same thing, not the opposite one."""
    frame = classify(
        rule(["A_CENTER"], ["B_NEIGHBOR", "C_NEIGHBOR"], lift=0.5, kind=AVOIDS, p_value=CONVINCING),
        rule(["B_NEIGHBOR"], ["C_NEIGHBOR"], lift=5.0, p_value=CONVINCING),
    )
    assert class_of(frame, 0) == "new"


def test_a_weaker_avoidance_between_consequents_explains_nothing():
    """0.9 is barely avoidance; it cannot account for a rule at 0.2."""
    frame = classify(
        rule(["A_CENTER"], ["B_NEIGHBOR", "C_NEIGHBOR"], lift=0.2, kind=AVOIDS, p_value=CONVINCING),
        rule(["B_NEIGHBOR"], ["C_NEIGHBOR"], lift=0.9, kind=AVOIDS, p_value=CONVINCING),
    )
    assert class_of(frame, 0) == "new"


def test_a_noisy_exclusion_lets_the_avoidance_rule_stand():
    frame = classify(
        rule(["A_CENTER"], ["B_NEIGHBOR", "C_NEIGHBOR"], lift=0.5, kind=AVOIDS, p_value=CONVINCING),
        rule(["B_NEIGHBOR"], ["C_NEIGHBOR"], lift=0.2, kind=AVOIDS, p_value=NOISE),
    )
    assert class_of(frame, 0) == "consequent_is_noise"
    assert frame.iloc[0]["adds_information"]


# --- compared by cell type, counted by item -----------------------------------

def test_a_type_named_twice_does_not_let_a_rule_match_itself():
    """
    Paneth in the middle and Paneth around it: three items, two of the same type.
    Dropping one leaves a genuinely shorter rule, never this one again.
    """
    frame = classify(
        rule(["Paneth_CENTER", "Paneth_NEIGHBOR"], ["Epithelial_NEIGHBOR"], lift=3.0, p_value=CONVINCING),
    )
    assert frame.iloc[0]["rule_type"] == "ant-complex"
    assert class_of(frame, 0) == "new"


def test_dropping_a_repeated_type_lands_on_the_shorter_rule():
    frame = classify(
        rule(["Paneth_CENTER", "Paneth_NEIGHBOR"], ["Epithelial_NEIGHBOR"], lift=3.0, p_value=CONVINCING),
        rule(["Paneth_CENTER"], ["Epithelial_NEIGHBOR"], lift=2.0, p_value=CONVINCING),
    )
    assert class_of(frame, 0) == "stronger_effect"


def test_the_shorter_rule_is_matched_by_type_whatever_its_roles():
    """Same two cell types, other way round on centre and neighbour. Still counts."""
    frame = classify(
        rule(["A_CENTER", "B_NEIGHBOR"], ["C_NEIGHBOR"], lift=2.0, p_value=CONVINCING),
        rule(["A_NEIGHBOR"], ["C_CENTER"], lift=2.0, p_value=CONVINCING),
    )
    assert class_of(frame, 0) == "redundant_by_simpler"


def test_direction_still_matters_for_the_shorter_rule():
    """C -> A is not a shorter version of a rule that reads A -> C."""
    frame = classify(
        rule(["A_CENTER", "B_NEIGHBOR"], ["C_NEIGHBOR"], lift=2.0, p_value=CONVINCING),
        rule(["C_CENTER"], ["A_NEIGHBOR"], lift=2.0, p_value=CONVINCING),
    )
    assert class_of(frame, 0) == "new"


# --- one rule speaks for arrangements sharing a type signature ----------------

def test_a_convincing_arrangement_speaks_over_a_louder_fluke():
    """
    Two arrangements of the same two types. The loud one is noise, so the believable
    one speaks for the group — and 3.0 clears its 2.0 comfortably.
    """
    frame = classify(
        rule(["A_CENTER", "B_NEIGHBOR"], ["C_NEIGHBOR"], lift=3.0, p_value=CONVINCING),
        rule(["A_CENTER"], ["C_NEIGHBOR"], lift=10.0, p_value=NOISE),
        rule(["A_NEIGHBOR"], ["C_CENTER"], lift=2.0, p_value=CONVINCING),
    )
    assert class_of(frame, 0) == "stronger_effect"


def test_the_strongest_convincing_arrangement_is_the_one_to_beat():
    """Both are believable, so the longer rule has to clear the stronger of them."""
    frame = classify(
        rule(["A_CENTER", "B_NEIGHBOR"], ["C_NEIGHBOR"], lift=3.0, p_value=CONVINCING),
        rule(["A_CENTER"], ["C_NEIGHBOR"], lift=2.0, p_value=CONVINCING),
        rule(["A_NEIGHBOR"], ["C_CENTER"], lift=2.9, p_value=CONVINCING),
    )
    assert class_of(frame, 0) == "redundant_by_simpler"
