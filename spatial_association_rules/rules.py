"""
What both searches share: support, the metrics, itemset -> rule, and the filters.

    support(I) = mean over transactions of min(weight of each item in I)
"""

from collections import Counter
from itertools import combinations
from math import comb

import numpy as np
import pandas as pd

from .complex_rules import DEFAULT_IMPROVEMENT_GAIN, classify_complex_rules
from .transactions import is_center, strip_role

ATTRACTS = "attracts"
AVOIDS = "avoids"

COLUMNS = ["antecedents", "consequents", "kind", "support", "antecedent support",
           "consequent support", "confidence", "lift", "leverage", "conviction",
           "len_ant", "len_con"]

BLOCK = 4_000_000        # biggest temporary allowed while measuring, in floats
MOSTLY_ABSENT = 0.5      # a table emptier than this is searched, not read right through
# How many 1s are in each of the 256 values a byte can hold.
ONES_IN_BYTE = np.array([bin(byte).count("1") for byte in range(256)], dtype=np.uint8)


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


def packed(matrix):
    """
    Which items each transaction holds, one bit apiece.

    An item is either in a transaction or it is not, so a single bit says all there is
    to say about that. Sixty-four transactions then fit inside one number, so they can
    be ruled in or out sixty-four at a time.
    """
    # One row of bits per item, laid end to end so a whole row can be packed at once.
    present = np.ascontiguousarray(np.asarray(matrix, dtype=bool).T)

    # Bits travel in groups of 64, so count the empty slots the last group still has.
    blanks = -present.shape[1] % 64
    if blanks:
        # Fill them with 0, meaning "item absent", so they can never add to a count.
        present = np.pad(present, ((0, 0), (0, blanks)))

    # Eight bits to a byte, eight bytes to a 64-bit number.
    return np.packbits(present, axis=1, bitorder="little").view(np.uint64)


def support_of_many(matrix, columns):
    """
    Calculates the joint support for thousands of different cell combinations at the exact same time.

    A transaction missing any item of a group adds 0 to that group, so when most weights
    are absent it pays to find the few transactions holding a whole group. A crowded
    table has nothing worth skipping, and is read straight through as it always was.
    """
    supports = np.zeros(len(columns))
    n = matrix.shape[0]
    if n == 0 or len(columns) == 0:
        return supports

    if np.count_nonzero(matrix) < MOSTLY_ABSENT * matrix.size:
        return _support_from_matching_rows(matrix, columns)

    per_group = max(1, n * max(1, columns.shape[1]))
    step = max(1, BLOCK // per_group)
    for start in range(0, len(columns), step):
        block = columns[start:start + step]
        supports[start:start + len(block)] = matrix[:, block].min(axis=2).sum(axis=0) / n
    return supports


def _support_from_matching_rows(matrix, columns):
    """
    The support of each group, weighed only where every item of it is present.

    In every other transaction the weakest item of the group weighs 0, and adding 0
    changes no total, so those transactions are skipped.
    """
    n = matrix.shape[0]
    bits = packed(matrix)

    supports = np.zeros(len(columns))
    for i, group in enumerate(columns):
        rows = _matching_rows(bits, group, n)
        if len(rows):
            # np.ix_ cuts out those rows and this group's columns, and nothing else.
            supports[i] = matrix[np.ix_(rows, group)].min(axis=1).sum() / n
    return supports


def _matching_rows(bits, group, n_transactions):
    """Which transactions hold every item of the group, as row numbers."""
    # Start from the first item, then drop any transaction missing one of the others.
    together = bits[group[0]].copy()
    for item in group[1:]:
        together &= bits[item]

    # Undo the packing, back to one bit per transaction, less the blanks packed() added.
    one_each = np.unpackbits(together.view(np.uint8), bitorder="little")
    return np.flatnonzero(one_each[:n_transactions])


def support_of_many_packed(bits, columns, n_transactions):
    """
    *For binary weighting only*
    What support_of_many() measures.

    With nothing but 0s and 1s, "the weakest item in this transaction" is really just
    "are they all here?". So joining two items is an and over 64 transactions at once,
    and the support is however many 1s are left over.
    """
    supports = np.zeros(len(columns))
    if n_transactions == 0 or len(columns) == 0:
        return supports

    # Each row of columns is one group of items. Start from the group's first item.
    together = bits[columns[:, 0]]

    # Keep a bit only where the group's next item is present too.
    for item in columns.T[1:]:
        together &= bits[item]

    # np.bitwise_count would count these in one step, but it needs NumPy 2. So read
    # each word as its 8 bytes instead, and look up how many 1s each byte holds.
    return ONES_IN_BYTE[together.view(np.uint8)].sum(axis=1) / n_transactions


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

def splits_of(itemset, one_sided_complex_rules=True):
    """Allowed (antecedent, consequent) splits, with the center on the left."""
    for size in range(1, len(itemset)):
        if one_sided_complex_rules and size > 1 and len(itemset) - size > 1:
            continue
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


def count_candidate_rules(labels, settings, n_items=None):
    """Count all possible rules before search filters, optionally for one item count."""
    n_labels = len(labels_with_enough_cells(labels, settings))
    # With one-sided complexity: all neighbors on the right, or just one.
    # For a single neighbor these are the same split.
    per_center = 0
    for r in range(1, min(n_labels, settings.max_items_per_rule - 1) + 1):
        if n_items is not None and r + 1 != n_items:
            continue
        splits = 2**r - 1
        if settings.one_sided_complex_rules and r > 1:
            splits = r + 1
        per_center += comb(n_labels, r) * splits
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


def classify_rules(rules, min_lift_gain=DEFAULT_IMPROVEMENT_GAIN, max_individual_fdr=None,
                   min_consequent_conviction_gain=DEFAULT_IMPROVEMENT_GAIN):
    """Classify all rows: antecedents by lift, consequents by conviction, mixed untested."""
    return classify_complex_rules(rules, min_lift_gain, max_individual_fdr,
                                  min_consequent_conviction_gain)


filter_rules = classify_rules  # Compatibility with the original public name.
