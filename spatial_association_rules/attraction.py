"""
Search for cell types that turn up together. See README, "Attraction and avoidance".
"""

import logging

import numpy as np

from .rules import ATTRACTS, rules_from, splits_of, support_of
from .tree import find_itemsets

logger = logging.getLogger(__name__)


def passes_support_policy(settings, support, confidence, n_transactions):
    """Does this rule happen often enough to judge? A confident rule gets a lower bar."""
    floor = settings.min_patches / n_transactions
    needed = settings.min_support
    if settings.strong_confidence is not None:
        needed = np.where(confidence >= settings.strong_confidence,
                          settings.min_support_when_strong, settings.min_support)
    return support >= np.maximum(needed, floor)


def is_strong_enough(settings, confidence, lift, leverage, conviction):
    """The attraction thresholds."""
    enough = lift >= settings.min_lift
    for value, limit in [(leverage, settings.min_leverage),
                         (conviction, settings.min_conviction),
                         (confidence, settings.min_confidence)]:
        if limit is not None:
            enough = enough & (value >= limit)
    return enough


def attracts(settings, support, ant_support, con_support, measures, n_transactions):
    """Do these cell types clearly draw together? Same signature as avoids()."""
    confidence, lift, leverage, conviction = measures
    return ((lift >= 1.0)
            & passes_support_policy(settings, support, confidence, n_transactions)
            & is_strong_enough(settings, confidence, lift, leverage, conviction))


def mine_attraction(transactions, matrix, item_index, settings):
    """Itemsets that recur, split into rules, keeping the ones that clearly attract."""
    n = len(transactions)
    plain = max(settings.min_support * n, settings.min_patches)
    # Mine to the lower floor, so a confident rule is not dropped before its confidence is known.
    when_strong = (max(settings.min_support_when_strong * n, settings.min_patches)
                   if settings.min_support_when_strong is not None else plain)
    min_weight = min(plain, when_strong)

    candidates = find_itemsets(transactions, min_weight, settings.max_items_per_rule)

    # Tree weights are only an upper bound, so measure each candidate exactly.
    supports = {}
    for itemset in candidates:
        support = support_of(itemset, matrix, item_index)
        if support >= min_weight / n:
            supports[itemset] = support

    splits = []
    for itemset, support in supports.items():
        if len(itemset) < 2:
            continue
        for antecedent, consequent in splits_of(itemset, settings.one_sided_complex_rules):
            ant_support = supports.get(antecedent, 0.0)
            con_support = supports.get(consequent, 0.0)
            if ant_support <= 0 or con_support <= 0:
                continue
            splits.append((antecedent, consequent, support, ant_support, con_support))

    rules = rules_from(splits, settings, n, attracts, ATTRACTS)
    logger.debug(f"Attraction: {len(supports)} itemsets measured, {len(rules)} rules")
    return rules
