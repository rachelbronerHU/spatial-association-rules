"""
Tests for spatial_association_rules.

Every input here is built by arithmetic, never by a random number generator and
never from a file, so the expected values can be checked by hand and cannot drift
when a dependency is upgraded.

Each test asserts a fact about the algorithm. None of them test the shape of the API.
"""

import logging
import math
from importlib import import_module
from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from spatial_association_rules import Method, Settings, Weighting, mine
from spatial_association_rules.attraction import attracts
from spatial_association_rules.avoidance import (
    avoids,
    extend,
    items_of,
    items_worth_combining,
    mine_avoidance,
    sides_worth_pairing,
)
from spatial_association_rules.mine import Result, mine_rules
from spatial_association_rules.rules import (
    metrics,
    packed,
    splits_of,
    support_of,
    support_of_many,
    support_of_many_packed,
    weight_matrix,
)
from spatial_association_rules.validation.significance import (
    _rule_columns,
    not_crowded,
    p_values_for,
    survives_shuffle,
)
from spatial_association_rules.validation.false_discovery import false_discovery_rates
from spatial_association_rules.transactions import (
    build_transactions,
    find_patches,
    measure_patches,
    strip_role,
)
from spatial_association_rules.tree import find_itemsets


def binary(*items):
    return {item: 1.0 for item in items}


def base(**changes):
    """
    Settings for the tests, stated in full here on purpose.

    Nothing is read from constants.py and nothing is left to a library default, so a
    change to either cannot quietly change what these tests mean.
    """
    settings = Settings(weighting=Weighting.WEIGHTED, method=Method.CN,
                        radius=25.0, min_support=0.01, min_lift=1.0,
                        max_items_per_rule=2,
                        min_cells_per_patch=2, max_one_type_share=1.0, min_patches=0,
                        include_avoidance_rules=True, avoidance_max_lift=0.999,
                        avoidance_min_expected_meetings=1)
    return settings.replace(**changes) if changes else settings


def grid_tissue(side=14, spacing=10.0):
    """
    A square grid of cells, labelled by position so the layout is fully determined.

    Every third column is B, every fifth row is C, the rest are A. No randomness,
    so any test built on it means the same thing forever.
    """
    coords, labels = [], []
    for row in range(side):
        for col in range(side):
            coords.append([col * spacing, row * spacing])
            labels.append("B" if col % 3 == 0 else "C" if row % 5 == 0 else "A")
    return np.array(coords, dtype=float), np.array(labels, dtype=object)


# --- support: the definition, checkable by eye --------------------------------

FIVE = [
    binary("A_CENTER", "B_NEIGHBOR"),
    binary("A_CENTER", "B_NEIGHBOR"),
    binary("A_CENTER", "C_NEIGHBOR"),
    binary("D_CENTER", "B_NEIGHBOR"),
    binary("D_CENTER", "C_NEIGHBOR"),
]


def test_binary_support_is_plain_counting():
    """With weights of 1.0, support is just the fraction of transactions."""
    matrix, index = weight_matrix(FIVE)

    assert support_of(frozenset({"A_CENTER"}), matrix, index) == pytest.approx(3 / 5)
    assert support_of(frozenset({"B_NEIGHBOR"}), matrix, index) == pytest.approx(3 / 5)
    assert support_of(frozenset({"A_CENTER", "B_NEIGHBOR"}), matrix, index) == pytest.approx(2 / 5)


def test_weighted_support_is_min_based():
    """A pattern is only as strong as its weakest member in each transaction."""
    transactions = [
        {"A_CENTER": 1.0, "B_NEIGHBOR": 0.5},
        {"A_CENTER": 1.0, "B_NEIGHBOR": 0.2},
        {"A_CENTER": 1.0},
    ]
    matrix, index = weight_matrix(transactions)

    both = support_of(frozenset({"A_CENTER", "B_NEIGHBOR"}), matrix, index)
    assert both == pytest.approx((0.5 + 0.2 + 0.0) / 3)
    # Downward closure: adding an item can never raise support.
    assert both <= support_of(frozenset({"A_CENTER"}), matrix, index)


def test_packed_support_answers_exactly_what_the_float_path_answers():
    """
    With 0/1 weights the bitset path is an optimisation, not a second definition of support.

    130 transactions on purpose: bitsets hold 64 to a word, so this crosses a word
    boundary and leaves a part-full last word whose padding must not be counted.
    """
    transactions = [binary("A_CENTER", "B_NEIGHBOR") if row % 2
                    else binary("A_CENTER", "C_NEIGHBOR") if row % 3
                    else binary("D_CENTER", "B_NEIGHBOR")
                    for row in range(130)]
    matrix, item_index = weight_matrix(transactions)
    bits = packed(matrix)
    items = sorted(item_index.values())

    for width in (1, 2, 3):
        columns = np.array([list(group) for group in combinations(items, width)])
        assert support_of_many_packed(bits, columns, len(transactions)) == pytest.approx(
            support_of_many(matrix, columns))

    # A pair that never co-occurs and one that always does: neither is all this compares.
    assert support_of_many_packed(bits, np.array([[item_index["D_CENTER"],
                                                   item_index["C_NEIGHBOR"]]]), 130) == 0.0
    assert support_of_many_packed(bits, np.array([[item_index["A_CENTER"]]]), 130) > 0.0


def test_binary_rule_metrics_are_hand_checkable():
    rules = mine_rules(FIVE, base(weighting=Weighting.BINARY, min_support=0.1, min_patches=0))
    rule = rules[(rules["antecedents"] == ("A_CENTER",))
                 & (rules["consequents"] == ("B_NEIGHBOR",))].iloc[0]

    assert rule["support"] == pytest.approx(0.4)          # 2 of 5
    assert rule["confidence"] == pytest.approx(0.4 / 0.6)  # 2 of the 3 A patches
    assert rule["lift"] == pytest.approx((0.4 / 0.6) / 0.6)
    assert rule["leverage"] == pytest.approx(0.4 - 0.6 * 0.6)


# --- the golden set: frozen output of a fixed input ---------------------------

def repeating_transactions(n=210):
    """
    Transaction i holds items decided by arithmetic on i, so the supports below are
    exact fractions: A every 2nd, B every 3rd, C every 5th, D every 7th.
    """
    transactions = []
    for i in range(n):
        items = ["X_CENTER"]
        if i % 2 == 0:
            items.append("A_NEIGHBOR")
        if i % 3 == 0:
            items.append("B_NEIGHBOR")
        if i % 5 == 0:
            items.append("C_NEIGHBOR")
        if i % 7 == 0:
            items.append("D_NEIGHBOR")
        transactions.append(binary(*items))
    return transactions


# X is in every transaction; the rest appear on their own multiples of i.
EVERY = {"X_CENTER": 1, "A_NEIGHBOR": 2, "B_NEIGHBOR": 3, "C_NEIGHBOR": 5, "D_NEIGHBOR": 7}


def expected_itemsets(n, min_support):
    """
    What FP-growth must find, worked out by counting instead of by mining.

    Items co-occur exactly when i divides by all of their numbers at once, so an
    itemset appears once per multiple of their lowest common multiple. This never
    touches the tree, so agreeing with it is real evidence rather than circular.
    """
    expected = {}
    items = list(EVERY)
    for size in range(1, len(items) + 1):
        for combo in combinations(items, size):
            step = math.lcm(*(EVERY[item] for item in combo))
            count = len(range(0, n, step))
            if count >= min_support * n:
                expected[frozenset(combo)] = count / n
    return expected


def test_mining_finds_exactly_the_itemsets_counting_predicts():
    """
    Not a sample of them: exactly them, with exactly those supports.

    Verified against mlxtend once on this input. The counting oracle replaces that
    dependency, and unlike a frozen list it says *why* each number is what it is.
    """
    n, min_support = 210, 0.02
    transactions = repeating_transactions(n)
    matrix, index = weight_matrix(transactions)

    found = set(find_itemsets(transactions, min_support * n, max_items=len(EVERY)))
    expected = expected_itemsets(n, min_support)

    assert found == set(expected), (
        f"missing {sorted(map(sorted, set(expected) - found))}, "
        f"unexpected {sorted(map(sorted, found - set(expected)))}"
    )
    for itemset, support in expected.items():
        assert support_of(itemset, matrix, index) == pytest.approx(support, abs=1e-12), sorted(itemset)


# --- the two bugs the review found --------------------------------------------

def test_null_uses_the_same_transactions_the_run_mined():
    """
    The significance test must judge rules against the same transaction set that
    produced them. Crowded patches are dropped from both, by the same rule.
    """
    coords, labels = grid_tissue()
    settings = base(max_one_type_share=0.5, min_patches=0)

    patches = measure_patches(find_patches(coords, settings), coords, settings)
    _, stats = build_transactions(patches, labels, settings)

    # The mask the null builds from the real labels must agree with what mining kept.
    names = sorted(set(labels))
    onehot = np.zeros((len(labels), len(names)), dtype=np.float32)
    for cell, label in enumerate(labels):
        onehot[cell, names.index(label)] = 1.0

    from spatial_association_rules.validation.significance import _adjacency
    _, _, membership, sizes = _adjacency(patches, len(labels))
    kept_by_null = int(not_crowded(membership, sizes, onehot, settings.max_one_type_share).sum())

    assert kept_by_null == stats["patches_kept"]
    assert kept_by_null < len(patches), "this tissue should have crowded patches, or the test proves nothing"


def test_decay_distance_follows_radius():
    """Changing the radius must change how far the weights reach."""
    settings = base()                       # no bandwidth given
    assert settings.decay_distance == 25.0

    wider = settings.replace(radius=50.0)
    assert wider.decay_distance == 50.0, "the decay scale went stale when the radius changed"

    explicit = base(bandwidth=15.0).replace(radius=50.0)
    assert explicit.decay_distance == 15.0, "an explicit bandwidth must be left alone"


# --- end to end ---------------------------------------------------------------

def test_a_patch_of_one_label_is_skipped():
    """Nothing can be learned from a patch where every cell is the same type."""
    coords = np.array([[x * 5.0, 0.0] for x in range(20)])
    one_type = np.array(["Epithelial"] * 20, dtype=object)
    assert mine(coords, one_type, base(max_one_type_share=0.9)).stats["patches_kept"] == 0


def test_the_centre_never_appears_on_the_right():
    coords, labels = grid_tissue()
    for rule in mine(coords, labels, base(min_patches=0)).rules.itertuples():
        assert all("_CENTER" not in item for item in rule.consequents)
        assert any("_CENTER" in item for item in rule.antecedents)


def test_same_seed_gives_the_same_p_values():
    coords, labels = grid_tissue()
    result = mine(coords, labels, base(min_patches=0))
    assert not result.rules.empty

    first = result.add_p_values(n_shuffles=10, random_seed=99)
    second = result.add_p_values(n_shuffles=10, random_seed=99)
    pd.testing.assert_series_equal(first["p_value"], second["p_value"])


def test_p_values_are_between_zero_and_one():
    coords, labels = grid_tissue()
    tested = mine(coords, labels, base(min_patches=0)).add_p_values(n_shuffles=5, random_seed=1)
    assert tested["p_value"].between(0, 1).all()
    assert "p_value_adj" not in tested.columns, "the pipeline must not correct anything"


def test_no_shuffles_means_no_claim():
    """With nothing tested, every p-value is 1: the test was not run."""
    coords, labels = grid_tissue()
    result = mine(coords, labels, base(min_patches=0))
    p_values = p_values_for(result.rules, result.patches, labels, result.settings,
                            n_shuffles=0, random_seed=42, labels_kept_fixed=())
    assert (p_values == 1).all()


def test_individual_fdr_includes_unmined_candidates_without_changing_raw_p_values():
    coords, labels = grid_tissue()
    result = mine(coords, labels, base())
    tested = result.add_p_values(n_shuffles=20, random_seed=1)
    raw = p_values_for(result.rules, result.patches, labels, result.settings,
                       n_shuffles=20, random_seed=1, labels_kept_fixed=())
    # Three centers x three neighbors x two kinds, including unmined rules.
    assert 0 < len(raw) < 18
    expected = false_discovery_rates(np.r_[raw, np.ones(18 - len(raw))])[:len(raw)]
    np.testing.assert_allclose(tested["p_value"], raw)
    np.testing.assert_allclose(tested["individual_fdr"], expected)


def test_requested_subset_keeps_the_full_candidate_count():
    coords, labels = grid_tissue()
    result = mine(coords, labels, base())
    requested = result.rules.iloc[[0]].copy()
    requested.index = [42]
    tested = result.add_p_values(n_shuffles=20, rules=requested, random_seed=1)
    assert tested.index.tolist() == [42]
    assert tested.iloc[0]["individual_fdr"] == pytest.approx(min(1, tested.iloc[0]["p_value"] * 18))


def test_supplied_rules_that_failed_mining_get_one_without_shuffling(monkeypatch):
    coords, labels = grid_tissue()
    loose = mine(coords, labels, base())
    strict = mine(coords, labels, base(min_lift=100))
    failed = loose.rules[loose.rules["kind"] == "attracts"]
    assert not failed.empty
    assert not (strict.rules["kind"] == "attracts").any()

    def check_no_rules_to_shuffle(rules, *args):
        assert rules.empty
        return np.ones(0)

    monkeypatch.setattr(import_module("spatial_association_rules.mine"),
                        "p_values_for", check_no_rules_to_shuffle)
    tested = strict.add_p_values(n_shuffles=20, rules=failed, random_seed=1)
    assert (tested[["p_value", "individual_fdr"]] == 1).all().all()


def test_empty_mining_result_has_empty_p_values_and_fdr():
    coords, labels = grid_tissue()
    result = mine(coords, labels, base(min_label_count=len(labels) + 1))
    tested = result.add_p_values(n_shuffles=20, random_seed=1)
    assert tested.empty
    assert {"p_value", "individual_fdr"} <= set(tested.columns)


# --- settings -----------------------------------------------------------------

@pytest.mark.parametrize("bad", [
    {"min_support": 1.5},
    {"min_support": 0.0},
    {"radius": 0},
    {"method": Method.KNN_R},              # without k_neighbors
    {"max_items_per_rule": 1},
    {"max_items_per_rule": 11},            # past LONGEST_RULE
    {"strong_confidence": 0.9},            # without its partner
    {"avoidance_max_lift": None},          # while the avoidance search is on
    {"avoidance_max_lift": 1.0},           # not below 1, so not avoidance
    {"avoidance_max_leverage": 0.5},       # not below 0, so not avoidance
    {"min_lift": 0.8},                     # below 1, so not attraction
    {"avoidance_min_expected_meetings": 0},
])
def test_settings_reject_impossible_values(bad):
    with pytest.raises(ValueError):
        base(**bad)


def test_the_thresholds_that_make_a_p_value_mean_something_are_required():
    """
    Without them a shuffled tissue passes as often as the real one.

    min_lift always, and avoidance_max_lift whenever that search is on. Saying so in
    the type is the only way a caller cannot quietly end up with meaningless p-values.
    """
    with pytest.raises(TypeError):                       # min_lift has no default
        Settings(weighting=Weighting.BINARY, method=Method.CN, radius=25.0,
                 min_support=0.01, max_items_per_rule=2)

    # Attraction alone needs no avoidance threshold.
    assert base(include_avoidance_rules=False, avoidance_max_lift=None)


def test_unset_thresholds_do_not_filter():
    """A threshold nobody set must not quietly remove rules."""
    coords, labels = grid_tissue()
    everything = mine(coords, labels, base(min_patches=0)).rules
    stricter = mine(coords, labels, base(min_patches=0, min_lift=1.2)).rules

    assert len(everything) > len(stricter)


def test_attraction_thresholds_do_not_reach_avoidance():
    """min_lift says what counts as attraction, and nothing about keeping apart."""
    coords, labels = grid_tissue()
    settings = base(min_patches=0, min_lift=1.2)

    both = mine(coords, labels, settings).rules
    assert (both[both["kind"] == "attracts"]["lift"] >= 1.2).all()
    assert (both[both["kind"] == "avoids"]["lift"] < 1.0).all()

    # Turning the avoidance search off is what leaves only attraction.
    only_attraction = mine(coords, labels,
                           settings.replace(include_avoidance_rules=False)).rules
    assert (only_attraction["kind"] == "attracts").all()
    assert (only_attraction["lift"] >= 1.2).all()


# --- avoidance: the opposite claim, searched for separately --------------------

def apart(n=100):
    """
    Two pairs that each stick together, so across the pairs nothing ever co-occurs.

    A_CENTER and D_NEIGHBOR are each in half the transactions and never in the same
    one: the strongest avoidance there is, and it carries no joint support at all.
    Expected meetings 0.5 * 0.5 * 100 = 25, so there was plenty to deplete.
    """
    return ([binary("A_CENTER", "B_NEIGHBOR")] * (n // 2)
            + [binary("C_CENTER", "D_NEIGHBOR")] * (n // 2))


def rule_named(rules, antecedents, consequents):
    found = rules[(rules["antecedents"] == antecedents) & (rules["consequents"] == consequents)]
    assert len(found) == 1, f"expected one {antecedents} -> {consequents}, got {len(found)}"
    return found.iloc[0]


def test_a_pair_that_never_co_occurs_is_found():
    """Joint support of zero is the finding, not a reason to prune it away."""
    rules = mine_rules(apart(), base(min_support=0.1))

    rule = rule_named(rules, ("A_CENTER",), ("D_NEIGHBOR",))
    assert rule["kind"] == "avoids"
    assert rule["support"] == 0.0
    assert rule["lift"] == 0.0
    assert rule["leverage"] == pytest.approx(-0.25)      # 0 - 0.5 * 0.5


def test_a_pair_that_never_co_occurs_survives_no_shuffle():
    """The null must judge it as avoidance too, or its p-value is about another rule."""
    transactions = apart()
    settings = base(min_support=0.1)
    rules = mine_rules(transactions, settings)

    matrix, item_index = weight_matrix(transactions)
    layout = _rule_columns(rules, item_index)
    # The real tissue, unshuffled: every mined rule must still pass its own thresholds.
    assert survives_shuffle(layout, matrix, settings).all()


def test_a_meeting_nobody_expected_is_not_a_finding():
    """
    Two cell types that were never going to meet anyway say nothing by not meeting.

    D_NEIGHBOR is in 4 of 100 transactions, so A_CENTER (96) and D_NEIGHBOR would meet
    3.8 times by chance. Seeing none of those happens easily; at a bar of 10 expected
    meetings the rule is not built, and at a bar of 2 it is.
    """
    transactions = [binary("A_CENTER", "B_NEIGHBOR")] * 96 + [binary("C_CENTER", "D_NEIGHBOR")] * 4
    settings = base(min_support=0.02)

    strict = mine_rules(transactions, settings.replace(avoidance_min_expected_meetings=10))
    avoiding = strict[strict["kind"] == "avoids"]
    assert not ((avoiding["antecedents"] == ("A_CENTER",))
                & (avoiding["consequents"] == ("D_NEIGHBOR",))).any()

    lenient = mine_rules(transactions, settings.replace(avoidance_min_expected_meetings=2))
    assert rule_named(lenient, ("A_CENTER",), ("D_NEIGHBOR",))["kind"] == "avoids"


def test_a_rate_measured_on_too_few_patches_is_not_a_finding():
    """
    min_patches is about the antecedent here: a confidence over 6 patches means nothing.

    C_CENTER is in 6 of 100 transactions and B_NEIGHBOR in 94, so 5.6 meetings were
    expected — enough to deplete, but not enough patches to measure a rate on.
    """
    transactions = [binary("A_CENTER", "B_NEIGHBOR")] * 94 + [binary("C_CENTER", "D_NEIGHBOR")] * 6
    settings = base(min_support=0.02, avoidance_min_expected_meetings=5)

    assert rule_named(mine_rules(transactions, settings.replace(min_patches=5)),
                      ("C_CENTER",), ("B_NEIGHBOR",))["kind"] == "avoids"

    strict = mine_rules(transactions, settings.replace(min_patches=10))
    assert not ((strict["antecedents"] == ("C_CENTER",)).any())


def test_a_rare_cell_type_can_still_be_the_centre():
    """
    The reason a support bar on each side was the wrong gate.

    A_CENTER is in 2 of 100 transactions — far too rare to clear any sensible support
    bar — but B_NEIGHBOR is in 90, so 1.8 meetings were expected and none happened.
    With enough patches behind it that is a real claim about a rare cell type.
    """
    transactions = ([binary("A_CENTER", "C_NEIGHBOR")] * 20
                    + [binary("D_CENTER", "B_NEIGHBOR")] * 900)
    settings = base(min_support=0.01, min_patches=10, avoidance_min_expected_meetings=10)

    rule = rule_named(mine_rules(transactions, settings), ("A_CENTER",), ("B_NEIGHBOR",))
    assert rule["kind"] == "avoids"
    assert rule["antecedent support"] == pytest.approx(20 / 920)   # about 2%
    assert rule["support"] == 0.0


def test_the_avoidance_thresholds_remove_rules():
    """A threshold that never removes anything is not a threshold."""
    transactions = ([binary("A_CENTER", "B_NEIGHBOR")] * 40
                    + [binary("A_CENTER", "C_NEIGHBOR")] * 10
                    + [binary("D_CENTER", "B_NEIGHBOR")] * 50)
    settings = base(min_support=0.05, avoidance_min_expected_meetings=5)

    # A_CENTER -> B_NEIGHBOR: support 0.4, expected 0.5 * 0.9 = 0.45, so lift is 0.888.
    loose = mine_rules(transactions, settings.replace(avoidance_max_lift=0.9))
    assert rule_named(loose, ("A_CENTER",), ("B_NEIGHBOR",))["lift"] == pytest.approx(0.4 / 0.45)

    tighter = mine_rules(transactions, settings.replace(avoidance_max_lift=0.8))
    assert not ((tighter["antecedents"] == ("A_CENTER",))
                & (tighter["consequents"] == ("B_NEIGHBOR",))).any()

    # leverage here is 0.4 - 0.45 = -0.05, so a bar below that removes it too.
    by_leverage = mine_rules(transactions, settings.replace(avoidance_max_lift=0.9,
                                                            avoidance_max_leverage=-0.1))
    assert not ((by_leverage["antecedents"] == ("A_CENTER",))
                & (by_leverage["consequents"] == ("B_NEIGHBOR",))).any()


def test_a_cell_type_can_avoid_itself():
    """A_CENTER -> A_NEIGHBOR is a real claim: these cells do not sit next to their own kind."""
    transactions = ([binary("A_CENTER", "B_NEIGHBOR")] * 45
                    + [binary("B_CENTER", "A_NEIGHBOR")] * 45
                    + [binary("A_CENTER", "A_NEIGHBOR")] * 10)
    rules = mine_rules(transactions, base(min_support=0.05, avoidance_min_expected_meetings=5))

    rule = rule_named(rules, ("A_CENTER",), ("A_NEIGHBOR",))
    assert rule["kind"] == "avoids"
    assert rule["support"] == pytest.approx(0.1)          # 10 of 100
    assert rule["lift"] == pytest.approx((0.1 / 0.55) / 0.55)


def test_avoidance_finds_rules_longer_than_a_pair():
    """
    Three items, and the metrics are the ordinary ones measured on the same halves.

    A_CENTER with B_NEIGHBOR present avoids C_NEIGHBOR: the pair holds in 30 of 100
    transactions, C_NEIGHBOR in 40, so 12 meetings were expected and 2 happened.
    """
    transactions = ([binary("A_CENTER", "B_NEIGHBOR")] * 28
                    + [binary("A_CENTER", "B_NEIGHBOR", "C_NEIGHBOR")] * 2
                    + [binary("A_CENTER", "C_NEIGHBOR")] * 20
                    + [binary("D_CENTER", "C_NEIGHBOR")] * 18
                    + [binary("D_CENTER", "B_NEIGHBOR")] * 32)
    rules = mine_rules(transactions, base(min_support=0.02, max_items_per_rule=3,
                                          avoidance_min_expected_meetings=5))

    rule = rule_named(rules, ("A_CENTER", "B_NEIGHBOR"), ("C_NEIGHBOR",))
    assert rule["kind"] == "avoids"
    assert rule["support"] == pytest.approx(0.02)
    assert rule["antecedent support"] == pytest.approx(0.30)
    assert rule["consequent support"] == pytest.approx(0.40)
    assert rule["confidence"] == pytest.approx(0.02 / 0.30)
    assert rule["lift"] == pytest.approx((0.02 / 0.30) / 0.40)
    assert rule["leverage"] == pytest.approx(0.02 - 0.30 * 0.40)


def test_avoidance_reads_the_weights_when_they_are_not_all_one():
    """
    Weighted support is min-based, and avoidance measures it the same way as attraction.

    A far B neighbour counts 0.25, so support({A_CENTER, B_NEIGHBOR}) is 0.25 per such
    transaction, not 1. Every number below is worked out by hand from that.
    """
    transactions = ([{"A_CENTER": 1.0, "B_NEIGHBOR": 0.25}] * 40
                    + [{"A_CENTER": 1.0, "C_NEIGHBOR": 1.0}] * 10
                    + [{"D_CENTER": 1.0, "B_NEIGHBOR": 1.0}] * 50)
    rules = mine_rules(transactions, base(min_support=0.05,
                                          avoidance_min_expected_meetings=5))

    #  A_CENTER 0.50,  B_NEIGHBOR (40 * 0.25 + 50) / 100 = 0.60,  joint 40 * 0.25 / 100 = 0.10
    rule = rule_named(rules, ("A_CENTER",), ("B_NEIGHBOR",))
    assert rule["kind"] == "avoids"
    assert rule["support"] == pytest.approx(0.10)
    assert rule["antecedent support"] == pytest.approx(0.50)
    assert rule["consequent support"] == pytest.approx(0.60)
    assert rule["lift"] == pytest.approx((0.10 / 0.50) / 0.60)


def test_two_centres_are_never_combined():
    """
    A patch has one centre cell, so a side with two centres can only ever measure zero.

    A side carries the centre it started with and is only ever grown by neighbours, so
    two centres cannot meet — it is impossible by shape, not filtered out afterwards.
    """
    transactions = ([binary("A_CENTER", "X_NEIGHBOR")] * 40
                    + [binary("B_CENTER", "Y_NEIGHBOR")] * 40
                    + [binary("A_CENTER", "Y_NEIGHBOR")] * 20)
    matrix, item_index = weight_matrix(transactions)
    settings = base(max_items_per_rule=4, min_patches=0, avoidance_min_expected_meetings=1)

    sides = sides_worth_pairing(matrix, item_index, settings, len(transactions))
    assert sides, "no sides survived, so this test proves nothing"
    for side in sides:
        centres = [item for item in items_of(side) if item.endswith("_CENTER")]
        assert len(centres) <= 1, f"{items_of(side)} has {len(centres)} centres"


def test_nothing_with_two_centres_is_even_measured():
    """
    Not just unused — never enumerated. At length 4 that would be most of the search.

    extend() adds neighbours only, and orders them so each longer side is built once.
    """
    neighbours = ["X_NEIGHBOR", "Y_NEIGHBOR", "Z_NEIGHBOR"]
    sides = [("A_CENTER", ()), ("B_CENTER", ()), (None, ("X_NEIGHBOR",))]

    for _ in range(3):
        sides = list(extend(sides, neighbours))
        assert len(sides) == len(set(sides)), "a side was built twice"
        for side in sides:
            assert sum(item.endswith("_CENTER") for item in items_of(side)) <= 1


def test_a_centre_that_sorts_after_a_neighbour_still_grows():
    """
    Only the neighbours are ordered, never the centre.

    Order the whole side instead and "Epithelial_NEIGHBOR" > "Muscle_CENTER" is False, so
    that side is never built — and no other path builds it, since a centre is only ever
    a seed. The rule would vanish with no error.
    """
    grown = set(extend([("Muscle_CENTER", ())], ["Epithelial_NEIGHBOR", "Zeta_NEIGHBOR"]))
    assert grown == {("Muscle_CENTER", ("Epithelial_NEIGHBOR",)),
                     ("Muscle_CENTER", ("Zeta_NEIGHBOR",))}

    transactions = ([binary("Muscle_CENTER", "Epithelial_NEIGHBOR")] * 50
                    + [binary("Muscle_CENTER", "Zeta_NEIGHBOR")] * 50)
    matrix, item_index = weight_matrix(transactions)
    settings = base(max_items_per_rule=3, min_patches=0, avoidance_min_expected_meetings=1)

    sides = sides_worth_pairing(matrix, item_index, settings, len(transactions))
    assert ("Muscle_CENTER", ("Epithelial_NEIGHBOR",)) in sides


def brute_force_avoidance(transactions, settings):
    """
    Every rule the avoidance search could produce, with no prefilter at all.

    Deliberately slow and obvious: every combination of every item, measured one at a
    time. It exists to disagree with the real search, so it shares none of its code
    beyond the definition of support and the metrics.
    """
    matrix, item_index = weight_matrix(transactions)
    n = len(transactions)
    supports = {
        frozenset(combo): support_of(frozenset(combo), matrix, item_index)
        for size in range(1, settings.max_items_per_rule + 1)
        for combo in combinations(sorted(item_index), size)
    }

    found = set()
    for itemset, support in supports.items():
        if len(itemset) < 2:
            continue
        for antecedent, consequent in splits_of(itemset):
            ant_support, con_support = supports[antecedent], supports[consequent]
            if ant_support <= 0 or con_support <= 0:
                continue
            measures = metrics(support, ant_support, con_support)
            if avoids(settings, support, ant_support, con_support, measures, n):
                found.add((tuple(sorted(antecedent)), tuple(sorted(consequent))))
    return found


def test_the_prefilter_cannot_change_which_rules_come_out():
    """
    The prefilters only save work. The rules must be the ones brute force finds.

    items_worth_combining bars on expected meetings alone, and sides_worth_pairing drops
    a side only when it is too rare for any rule to pass — so neither can lose a rule a
    surviving one would have named. A rare neighbour still counts when a common centre
    supplies the expected meetings.
    """
    transactions = ([binary("A_CENTER", "B_NEIGHBOR")] * 60
                    + [binary("A_CENTER", "C_NEIGHBOR")] * 12
                    + [binary("D_CENTER", "B_NEIGHBOR")] * 25
                    + [binary("D_CENTER", "E_NEIGHBOR")] * 3)
    settings = base(min_support=0.05, max_items_per_rule=3,
                    min_patches=10, avoidance_min_expected_meetings=5)

    mined = mine_rules(transactions, settings)
    mined = mined[mined["kind"] == "avoids"]
    searched = {(row.antecedents, row.consequents) for row in mined.itertuples()}

    expected = brute_force_avoidance(transactions, settings)
    assert expected, "this tissue must produce avoidance rules, or the test proves nothing"
    assert searched == expected, (
        f"prefilter lost {sorted(expected - searched)}, invented {sorted(searched - expected)}")

    # And it really was filtering, so the agreement above is not vacuous.
    matrix, item_index = weight_matrix(transactions)
    kept, _ = items_worth_combining(matrix, item_index, settings, len(transactions))
    assert set(kept) < set(item_index), "nothing was dropped, so this proves nothing"


def test_the_two_kinds_never_describe_the_same_rule():
    """Attraction and avoidance are opposite claims, so no rule may be given both."""
    coords, labels = grid_tissue()
    rules = mine(coords, labels, base(min_patches=0)).rules

    named = rules[["antecedents", "consequents"]]
    assert not named.duplicated().any(), "a rule came out of both searches"
    assert (rules[rules["kind"] == "attracts"]["lift"] >= 1.0).all()
    assert (rules[rules["kind"] == "avoids"]["lift"] < 1.0).all()


def test_avoidance_can_be_turned_off():
    rules = mine_rules(apart(), base(min_support=0.1, include_avoidance_rules=False))
    assert rules.empty or (rules["kind"] == "attracts").all()


# --- a rule this sample could not have been asked about ------------------------

def test_a_rule_naming_a_missing_cell_type_is_untested_not_perfect():
    """
    Never surviving a shuffle and never being tested are opposites, not the same zero.

    A rule about a cell type this sample does not have cannot pass here or in any
    shuffle, and (0 + 1) / (n + 1) would report that as the strongest result the run
    can produce. No evidence is 1.0.
    """
    coords, labels = grid_tissue()
    result = mine(coords, labels, base(min_patches=0))

    elsewhere = result.rules.head(3).copy()
    elsewhere["consequents"] = [("Nowhere_NEIGHBOR",)] * len(elsewhere)

    tested = result.add_p_values(n_shuffles=20, rules=elsewhere, random_seed=1)
    assert (tested["p_value"] == 1.0).all()
    # The real rules still get real p-values, so the fix has not flattened everything.
    assert result.add_p_values(n_shuffles=20, random_seed=1)["p_value"].min() < 1.0


# --- the two weightings are the same code -------------------------------------

def test_binary_and_weighted_agree_when_every_weight_is_one():
    """
    Nothing after the transactions looks at `weighting`, and this says so.

    The two modes differ only in how a neighbour is weighed. Give them identical
    transactions and every rule, threshold and metric must come out the same.
    """
    transactions = repeating_transactions(210)
    settings = base(min_support=0.02)

    as_binary = mine_rules(transactions, settings.replace(weighting=Weighting.BINARY))
    as_weighted = mine_rules(transactions, settings.replace(weighting=Weighting.WEIGHTED))

    assert not as_binary.empty
    pd.testing.assert_frame_equal(as_binary, as_weighted)


# --- one rule or a million: the same arithmetic --------------------------------

def scalar_metrics(support, ant_support, con_support):
    """
    The metrics written out one rule at a time, the plain way.

    The real one is elementwise so the shuffle test can judge every rule at once.
    This is what it has to keep agreeing with.
    """
    if ant_support == 0:
        return 0.0, 0.0, 0.0, float("inf")
    confidence = support / ant_support
    lift = confidence / con_support if con_support > 0 else 0.0
    leverage = support - ant_support * con_support
    conviction = float("inf") if confidence >= 1.0 else (1 - con_support) / (1 - confidence)
    return confidence, lift, leverage, conviction


TRICKY = [
    (0.0, 0.5, 0.5),      # never meet: the strongest avoidance there is
    (0.5, 0.5, 1.0),      # confidence exactly 1, so conviction is infinite
    (0.0, 0.0, 0.5),      # no antecedent: nothing was measured
    (0.3, 0.6, 0.0),      # no consequent: lift undefined
    (0.25, 0.5, 0.5),     # lift exactly 1, the line between the two searches
    (0.4, 0.5, 0.6),      # ordinary attraction
    (0.1, 0.5, 0.6),      # ordinary avoidance
]


@pytest.mark.parametrize("support,ant,con", TRICKY)
def test_the_metrics_agree_with_the_one_at_a_time_version(support, ant, con):
    """Every edge the division can fall off, checked against the plain arithmetic."""
    assert metrics(support, ant, con) == scalar_metrics(support, ant, con)


def test_judging_every_rule_at_once_matches_judging_them_one_by_one():
    """
    The shuffle test judges rules in bulk. It must decide exactly what a loop decided.

    Nothing here is random: the supports walk a grid that includes zero, one, and the
    boundaries the two searches divide on.
    """
    settings = base(min_support=0.05, min_lift=1.2, min_patches=10,
                    avoidance_max_lift=0.8, avoidance_min_expected_meetings=5,
                    strong_confidence=0.9, min_support_when_strong=0.02)
    n = 1000
    grid = [i / 20 for i in range(21)]
    joint, ant, con = map(np.array, zip(*[(min(s, a, c), a, c)
                                          for s in grid for a in grid for c in grid]))

    bulk = metrics(joint, ant, con)
    for judge in (attracts, avoids):
        together = judge(settings, joint, ant, con, bulk, n)
        apart = [judge(settings, joint[i], ant[i], con[i],
                       metrics(joint[i], ant[i], con[i]), n) for i in range(len(joint))]
        assert list(np.asarray(together)) == [bool(x) for x in apart], judge.__name__


# --- holding labels still ------------------------------------------------------

def test_pinning_every_label_is_refused():
    """
    With nothing left to move, no shuffle changes anything and every p-value is 1.0 —
    which reads exactly like a real "nothing is significant".
    """
    coords, labels = grid_tissue()
    result = mine(coords, labels, base(min_patches=0))

    with pytest.raises(ValueError, match="free to move"):
        result.add_p_values(n_shuffles=5, labels_kept_fixed=("A", "B", "C"))


def test_pinning_most_labels_warns(caplog):
    """Barely any movement is not an error, but the p-values stop discriminating."""
    coords = np.array([[x * 10.0, y * 10.0] for y in range(10) for x in range(10)])
    labels = np.array(["A"] * 95 + ["B"] * 5, dtype=object)   # pinning A leaves 5%
    result = mine(coords, labels, base(min_patches=0))
    assert not result.rules.empty

    with caplog.at_level(logging.WARNING):
        result.add_p_values(n_shuffles=2, labels_kept_fixed=("A",), random_seed=1)
    assert any("free to move" in message for message in caplog.messages)


# --- item names ----------------------------------------------------------------

def test_exactly_one_role_is_stripped_and_only_from_the_end():
    """
    An item carries one role, so only one comes off.

    A cell type may have CENTER or NEIGHBOR anywhere in its own name, the end included.
    """
    assert strip_role("CD8T_CENTER") == "CD8T"
    assert strip_role("CD8T_NEIGHBOR") == "CD8T"
    assert strip_role("CENTER_CD8T_CENTER") == "CENTER_CD8T"
    assert strip_role("Tumor_CENTER_high_NEIGHBOR") == "Tumor_CENTER_high"
    # A cell type literally called "X_NEIGHBOR" keeps its name.
    assert strip_role("X_NEIGHBOR_CENTER") == "X_NEIGHBOR"
    assert strip_role("X_CENTER_NEIGHBOR") == "X_CENTER"


def test_avoidance_rules_cannot_be_tested_without_their_threshold():
    """
    Judging them by a threshold other than the one that found them answers another question.

    Reachable by turning the avoidance search off and handing avoidance rules in, which
    is the same path as testing one sample's rules against another.
    """
    coords, labels = grid_tissue()
    result = mine(coords, labels, base(min_patches=0))
    avoiding = result.rules[result.rules["kind"] == "avoids"]
    assert not avoiding.empty

    without = result.settings.replace(include_avoidance_rules=False, avoidance_max_lift=None)
    blind = Result(rules=result.rules, stats=result.stats, patches=result.patches,
                   labels=result.labels, settings=without)
    with pytest.raises(ValueError, match="avoidance_max_lift"):
        blind.add_p_values(n_shuffles=5, rules=avoiding)
