"""
Search for cell types that keep apart. See README, "Attraction and avoidance".
"""

import logging

import numpy as np

from .rules import AVOIDS, frame_of, rules_from, support_of_many
from .transactions import is_center

logger = logging.getLogger(__name__)


def enough_to_judge_avoidance(settings, ant_support, con_support, n_transactions):
    """Enough patches to measure a rate on, and enough expected meetings to miss."""
    return ((ant_support * n_transactions >= settings.min_patches)
            & (ant_support * con_support * n_transactions >= settings.avoidance_min_expected_meetings))


def is_weak_enough(settings, lift, leverage):
    """The avoidance thresholds."""
    weak = lift <= settings.avoidance_max_lift
    if settings.avoidance_max_leverage is not None:
        weak = weak & (leverage <= settings.avoidance_max_leverage)
    return weak


def avoids(settings, support, ant_support, con_support, measures, n_transactions):
    """Do these cell types clearly keep apart? Same signature as attracts()."""
    _, lift, leverage, _ = measures
    return ((lift < 1.0)
            & enough_to_judge_avoidance(settings, ant_support, con_support, n_transactions)
            & is_weak_enough(settings, lift, leverage))


def items_worth_combining(matrix, item_index, settings, n_transactions):
    """
    Items worth combining, and the support of each. A speed filter only.

    Only the expected meetings bar can be asked of a single item: no side of a rule is
    more common than its rarest item, and neither side's share exceeds 1, so an item too
    rare here can never sit in a rule that passes. min_patches is not asked here — it
    constrains the antecedent only, and enough_to_judge_avoidance() applies it there.
    """
    if matrix.size == 0:
        return [], {}

    bar = settings.avoidance_min_expected_meetings / n_transactions
    supports = matrix.sum(axis=0) / n_transactions
    kept = sorted(item for item, column in item_index.items() if supports[column] >= bar)
    return kept, {item: float(supports[item_index[item]]) for item in kept}


def by_role(items):
    """The items split into centers and neighbors."""
    return ([item for item in items if is_center(item)],
            [item for item in items if not is_center(item)])


def items_of(side):
    """The items of a side: its center, if it has one, then its neighbors."""
    center, neighbors = side
    return (center,) + neighbors if center is not None else neighbors


def extend(sides, neighbors):
    """
    Each side grown by one more neighbor, every longer side built exactly once.

    Only the neighbors are kept in order, never the center — so a center that sorts
    late (Muscle_CENTER) can still be joined by a neighbor that sorts early
    (Epithelial_NEIGHBOR), and no side can ever collect a second center.
    """
    for center, so_far in sides:
        for item in neighbors:
            if not so_far or item > so_far[-1]:
                yield center, so_far + (item,)


def supports_of(sides, matrix, item_index):
    """The support of every side in a level, in one batched pass."""
    columns = np.asarray([[item_index[item] for item in items_of(side)] for side in sides])
    return support_of_many(matrix, columns)


def joint_supports(wholes, matrix, item_index):
    """Measures support of unique itemsets, grouping by item count for efficient calculation."""

    by_size = {}
    for whole in wholes:
        by_size.setdefault(len(whole), []).append(whole)

    supports = {}
    for group in by_size.values():
        columns = np.asarray([[item_index[item] for item in whole] for whole in group])
        for whole, support in zip(group, support_of_many(matrix, columns)):
            supports[whole] = float(support)

    return supports


def sides_worth_pairing(matrix, item_index, settings, n_transactions):
    """
    Every side of a rule common enough to be worth pairing, and its support.

    returns: a dictionary containing the sides of potential rules that meet the minimum support thresholds, 
                along with their respective support values

    A rule needs ant_support * con_support * n_transactions expected meetings, and
    neither share exceeds 1, so each side alone must clear that bar. An antecedent must
    also cover min_patches by itself. A side is never more common than the shorter side
    it grew from, so one that fails is dropped and never extended again — the whole
    branch above it disappears with it.
    """
    items, single_supports = items_worth_combining(matrix, item_index, settings, n_transactions)
    centers, neighbors = by_role(items)
    bar = settings.avoidance_min_expected_meetings / n_transactions
    center_bar = max(bar, settings.min_patches / n_transactions)

    # Level one is single items, and items_worth_combining has already measured them.
    kept = {}
    level = {(center, ()): single_supports[center] for center in centers}
    level.update({(None, (item,)): single_supports[item] for item in neighbors})

    # A side can hold at most max_items_per_rule - 1 items: the other side needs one.
    longest = settings.max_items_per_rule - 1
    for size in range(1, longest + 1):
        survivors = []
        for side, support in level.items():
            if support >= (center_bar if side[0] is not None else bar):
                kept[side] = float(support)
                survivors.append(side)

        if size == longest or not survivors:
            break        # no round left to measure a longer side, or nothing left to grow
        grown = list(extend(survivors, neighbors))
        level = dict(zip(grown, supports_of(grown, matrix, item_index)))

    return kept


def mine_avoidance(matrix, item_index, settings, sample_id: str = ""):
    """Every side worth pairing, paired into rules, keeping what keeps apart."""

    n = matrix.shape[0]
    if n == 0:
        return frame_of([])

    sides = sides_worth_pairing(matrix, item_index, settings, n)
    # A side of no support divides nothing and judges nothing, so it never pairs.
    antecedents = [(items_of(side), support)
                   for side, support in sides.items() if side[0] is not None and support > 0]
    consequents = sorted(((items_of(side), support)
                          for side, support in sides.items() if side[0] is None and support > 0),
                         key=lambda pair: -pair[1])  # most common first, so the pairing stops early

    floor = settings.avoidance_min_expected_meetings
    pairs = []
    for ant_items, ant_support in antecedents:
        needed = floor / (ant_support * n)
        for con_items, con_support in consequents:
            if con_support < needed:
                break
            if len(ant_items) + len(con_items) > settings.max_items_per_rule:
                continue
            if settings.one_sided_complex_rules and len(ant_items) > 1 and len(con_items) > 1:
                continue
            if set(ant_items) & set(con_items):
                continue
            pairs.append((frozenset(ant_items), frozenset(con_items), ant_support, con_support))

    joint = joint_supports({ant | con for ant, con, _, _ in pairs}, matrix, item_index)

    rules = rules_from([(ant, con, joint[ant | con], ant_support, con_support)
                        for ant, con, ant_support, con_support in pairs],
                       settings, n, avoids, AVOIDS)

    prefix = f"[{sample_id}] " if sample_id else ""
    logger.info(f"{prefix}Avoidance: {len(sides)} sides worth pairing, "
                f"{len(joint)} joint supports measured, {len(rules)} rules")

    return rules
