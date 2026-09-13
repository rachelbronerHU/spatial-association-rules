"""
What both searches share: support, the metrics, itemset -> rule, and the filters.

    support(I) = mean over transactions of min(weight of each item in I)
"""

from collections import Counter
from itertools import combinations
from math import comb

import numpy as np
import pandas as pd

from .transactions import is_center, strip_role

ATTRACTS = "attracts"
AVOIDS = "avoids"

COLUMNS = ["antecedents", "consequents", "kind", "support", "antecedent support",
           "consequent support", "confidence", "lift", "leverage", "conviction",
           "len_ant", "len_con"]

BLOCK = 4_000_000        # biggest temporary allowed while measuring, in floats


# --- support and metrics ---------------------------------------------------

def weight_matrix(transactions):
    """Transactions as a (transactions x items) matrix, plus the item order."""
    if not transactions:
        return np.zeros((0, 0)), {}

    items = sorted({item for transaction in transactions for item in transaction})
    item_index = {item: i for i, item in enumerate(items)}
    matrix = np.zeros((len(transactions), len(items)))
    for row, transaction in enumerate(transactions):
        for item, weight in transaction.items():
            matrix[row, item_index[item]] = weight
    return matrix, item_index


def support_of_columns(weights, n_transactions):
    """Calculates how often a group of cells appear together by finding their weakest link (minimum weight) in every patch, summing those minimums, and dividing by total patches."""
    if weights.size == 0 or n_transactions == 0:
        return 0.0
    present = np.all(weights > 0, axis=1)
    if not present.any():
        return 0.0
    return float(weights[present].min(axis=1).sum()) / n_transactions


def support_of(itemset, matrix, item_index):
    """Looks up a specific group of cells by their readable names and calculates their joint support."""
    if matrix.size == 0:
        return 0.0
    columns = [item_index[item] for item in itemset if item in item_index]
    if len(columns) < len(itemset):
        return 0.0
    return support_of_columns(matrix[:, columns], matrix.shape[0])


def support_of_many(matrix, columns):
    """Calculates the joint support for thousands of different cell combinations at the exact same time."""
    out = np.zeros(len(columns))
    n = matrix.shape[0]
    if n == 0 or len(columns) == 0:
        return out

    per_group = max(1, n * max(1, columns.shape[1]))
    step = max(1, BLOCK // per_group)
    for start in range(0, len(columns), step):
        block = columns[start:start + step]
        out[start:start + len(block)] = matrix[:, block].min(axis=2).sum(axis=0) / n
    return out


def metrics(support, ant_support, con_support):
    """confidence, lift, leverage, conviction. Takes single numbers or whole arrays."""
    support = np.asarray(support, dtype=float)
    ant_support = np.asarray(ant_support, dtype=float)
    con_support = np.asarray(con_support, dtype=float)

    # Denominators of 0 are swapped to 1 to prevent crashes; the resulting 0.0 dummy metrics are later eliminated by the judge() function.
    has_ant = ant_support > 0
    confidence = np.where(has_ant, support / np.where(has_ant, ant_support, 1.0), 0.0)

    has_con = con_support > 0
    lift = np.where(has_con, confidence / np.where(has_con, con_support, 1.0), 0.0)

    leverage = support - ant_support * con_support

    certain = confidence >= 1.0
    conviction = np.where(certain, np.inf,
                          (1 - con_support) / np.where(certain, 1.0, 1 - confidence))

    lift = np.where(has_ant, lift, 0.0)
    leverage = np.where(has_ant, leverage, 0.0)
    conviction = np.where(has_ant, conviction, np.inf)

    # If passed a single number instead of a list (0-D), unpack the NumPy result back into a plain Python float.
    if support.ndim == 0:
        return float(confidence), float(lift), float(leverage), float(conviction)
    return confidence, lift, leverage, conviction


# --- from itemset to rule --------------------------------------------------

def splits_of(itemset):
    """
    Every way of reading an itemset as a rule, as (antecedent, consequent) pairs.

    The center item goes on the left and never on the right.
    """
    for size in range(1, len(itemset)):
        for antecedent in combinations(sorted(itemset), size):
            antecedent = frozenset(antecedent)
            consequent = itemset - antecedent
            if any(is_center(item) for item in consequent):
                continue
            if not any(is_center(item) for item in antecedent):
                continue
            yield antecedent, consequent


def rule_row(antecedent, consequent, support, ant_support, con_support, measures, kind):
    """One row of the rules frame. measures is what metrics() returned."""
    confidence, lift, leverage, conviction = measures
    return {
        "antecedents": tuple(sorted(antecedent)),
        "consequents": tuple(sorted(consequent)),
        "kind": kind,
        "support": support,
        "antecedent support": ant_support,
        "consequent support": con_support,
        "confidence": confidence,
        "lift": lift,
        "leverage": leverage,
        "conviction": conviction,
        "len_ant": len(antecedent),
        "len_con": len(consequent),
    }


def rules_from(splits, settings, n_transactions, judge, kind):
    """
    Measure and judge every split at once, then build the rows that passed.

    splits: (antecedent, consequent, joint, ant_support, con_support) tuples.
    judge:  attracts() or avoids().
    """
    if not splits:
        return empty_rules()

    joint = np.fromiter((s[2] for s in splits), dtype=float, count=len(splits))
    ant_support = np.fromiter((s[3] for s in splits), dtype=float, count=len(splits))
    con_support = np.fromiter((s[4] for s in splits), dtype=float, count=len(splits))

    measures = metrics(joint, ant_support, con_support)
    kept = np.flatnonzero(judge(settings, joint, ant_support, con_support,
                                measures, n_transactions))

    rows = [rule_row(splits[i][0], splits[i][1], joint[i], ant_support[i], con_support[i],
                     tuple(measure[i] for measure in measures), kind)
            for i in kept]
    return frame_of(rows)


def frame_of(rows):
    return (pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)
            if rows else empty_rules())


def empty_rules():
    return pd.DataFrame(columns=COLUMNS)


# --- filters ---------------------------------------------------------------

def labels_with_enough_cells(labels, settings):
    """Labels this sample could produce a rule about: present, and common enough."""
    counts = Counter(str(label) for label in labels)
    threshold = max(settings.min_label_count or 0, int((settings.min_label_share or 0) * len(labels)))
    return frozenset(label for label, count in counts.items() if count >= threshold)


def count_candidate_rules(labels, settings):
    """All allowed rules before support/effect filtering, counting each search kind."""
    n_labels = len(labels_with_enough_cells(labels, settings))
    # Choose one center and r neighbor types. Split neighbors between the two
    # sides in 2**r ways, excluding the split with nothing on the right.
    per_center = sum(comb(n_labels, r) * (2**r - 1)
                     for r in range(1, min(n_labels, settings.max_items_per_rule - 1) + 1))
    n_kinds = 2 if settings.include_avoidance_rules else 1
    return n_labels * per_center * n_kinds


def drop_rare_labels(rules, labels, settings):
    """Remove rules naming a label too rare in this sample to say anything about."""
    if rules.empty or (settings.min_label_count is None and settings.min_label_share is None):
        return rules

    allowed = labels_with_enough_cells(labels, settings)
    keep = [
        all(strip_role(item) in allowed
            for item in tuple(row.antecedents) + tuple(row.consequents))
        for row in rules.itertuples()
    ]
    return rules[keep]


def filter_rules(rules, min_lift_gain=None, max_individual_fdr=None):
    """Classify complex rules to redundant (with no additional value to simpler rule) vs informative."""
    if min_lift_gain is None:
        min_lift_gain = 1.0
    from .complex_rules import classify_complex_rules
    return classify_complex_rules(rules, min_lift_gain, max_individual_fdr)

