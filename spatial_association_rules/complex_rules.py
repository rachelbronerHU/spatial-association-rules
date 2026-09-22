"""
Does a longer rule earn its place next to its shorter parts?

Shortest rules first, so a rule is only ever weighed against shorter ones that have
already been judged.

Rules are counted by item but compared by cell type. 'Paneth_CENTER + Paneth_NEIGHBOR'
is two items, and both are Paneth — so the rule is complex, and it answers to
'Paneth -> ...' rather than only to the one arrangement that happens to match.

Read individual_fdr from add_p_values(), which corrects each rule size separately.
Classification does not change those values. See DESIGN.md, "Complex rules classification".
"""

import itertools
from collections import defaultdict
from functools import lru_cache

import pandas as pd

from .transactions import strip_role

ATTRACTS = "attracts"
AVOIDS = "avoids"

NEW = "new"
STRONGER_EFFECT = "stronger_effect"
SIMPLER_ARE_NOISE = "simpler_are_noise"
REDUNDANT_BY_SIMPLER = "redundant_by_simpler"
CONSEQUENT_DRIVEN = "consequent_driven"
CONSEQUENT_IS_NOISE = "consequent_is_noise"

# The classes where something convincing was shown against the rule. The "_is_noise"
# ones are not here: what would have dismissed them is itself too weak to judge by.
REDUNDANT_CLASSES = frozenset({REDUNDANT_BY_SIMPLER, CONSEQUENT_DRIVEN})

_ADDED_COLUMNS = {"rule_type": object, "complex_class": object, "adds_information": bool,
                  "simpler_rules": object}


def classify_complex_rules(rules, min_lift_gain, max_individual_fdr=None):
    """
    Adds classification columns to rules. Nothing is dropped.

    - rule_type:        'pairwise', 'ant-complex', 'con-complex', 'both-complex'
    - complex_class:    why the rule was kept or dismissed, None for pairwise
    - adds_information: False when something else already said it — filter on this
    - simpler_rules:    what the rule was weighed against

    min_lift_gain:      how much a longer rule must beat a shorter one by
    max_individual_fdr: cutoff a shorter rule must pass to dismiss a longer one.
                        Missing values fail this check. None, or an absent
                        individual_fdr column, means use lift alone.

    Classification still runs when values are missing. A rule's own FDR does not
    decide its class; check it separately before treating the rule as significant.
    """
    if rules.empty:
        rules = rules.copy()
        for column, dtype in _ADDED_COLUMNS.items():
            rules[column] = pd.Series(dtype=dtype)
        return rules

    rules = rules.copy()

    antecedents = list(rules["antecedents"])
    consequents = list(rules["consequents"])
    kinds = list(rules["kind"])
    lifts = list(rules["lift"])
    fdrs = (list(rules["individual_fdr"]) if "individual_fdr" in rules.columns
            else [float("nan")] * len(rules))

    ant_types = [_types(items) for items in antecedents]
    con_types = [_types(items) for items in consequents]
    sizes = [len(a) + len(c) for a, c in zip(antecedents, consequents)]

    rules["rule_type"] = [_rule_type(len(a), len(c))
                          for a, c in zip(antecedents, consequents)]
    rules["complex_class"] = None
    rules["simpler_rules"] = [[] for _ in range(len(rules))]

    # Several arrangements of the same cell types share one signature, so both maps
    # hold every rule that fits, never just the last one seen.
    same_types, pair_rules = defaultdict(list), defaultdict(list)
    for pos in range(len(rules)):
        same_types[(ant_types[pos], con_types[pos], kinds[pos])].append(pos)
        if sizes[pos] == 2:
            pair_rules[(frozenset(ant_types[pos] + con_types[pos]), kinds[pos])].append(pos)

    def pass_fdr_threshold(pos):
        """
        Can this rule's corrected p-value support dismissing a longer rule?
        Missing values fail. No cutoff or no FDR column skips this check.
        """
        return (max_individual_fdr is None or "individual_fdr" not in rules.columns
                or (pd.notna(fdrs[pos]) and fdrs[pos] <= max_individual_fdr))

    def best_of(group, kind):
        """
        One rule to stand for all the arrangements sharing a type signature (after removing suffixes like _CENTER, _NEIGHBOR).
        Rules that pass the FDR bar first, then the strongest of those.
        """
        passed_fdr = [pos for pos in group if pass_fdr_threshold(pos)]
        return _strongest(passed_fdr or group, lifts, kind)

    informative = [True] * len(rules)
    for pos in sorted(range(len(rules)), key=lambda p: sizes[p]):
        # A two-item rule has nothing shorter to answer to.
        if sizes[pos] == 2:
            continue

        idx = rules.index[pos]
        kind, lift = kinds[pos], lifts[pos]

        backing = _find_consequent_backing_rules(con_types[pos], lift, kind, pair_rules, lifts)
        if backing is not None:
            rules.at[idx, "simpler_rules"] = _named(
                [pos for pair in backing for pos in pair], antecedents, consequents)
            if all(any(pass_fdr_threshold(p) for p in pair) for pair in backing):
                complex_class = CONSEQUENT_DRIVEN
            else:
                complex_class = CONSEQUENT_IS_NOISE
        else:
            groups = (same_types.get(signature, [])
                      for signature in _every_shorter(ant_types[pos], con_types[pos], kind))
            # Prefer a parent that earned its place; if a group has none, weigh against
            # the dismissed one rather than pretending nothing shorter exists.
            live = [[p for p in group if informative[p]] or group
                    for group in groups if group]
            shorter = [best_of(group, kind) for group in live]
            matched = [p for p in shorter
                       if not _lift_beats(lift, lifts[p], min_lift_gain, kind)]
            rules.at[idx, "simpler_rules"] = _named(shorter, antecedents, consequents)
            if not shorter:
                complex_class = NEW
            elif not matched:
                complex_class = STRONGER_EFFECT
            elif any(pass_fdr_threshold(p) for p in matched):
                complex_class = REDUNDANT_BY_SIMPLER
            else:
                complex_class = SIMPLER_ARE_NOISE

        rules.at[idx, "complex_class"] = complex_class
        informative[pos] = complex_class not in REDUNDANT_CLASSES

    rules["adds_information"] = ~rules["complex_class"].isin(REDUNDANT_CLASSES)
    return rules

def _types(items):
    """
    The cell types a rule names, role dropped and duplicates kept.

    'Paneth_CENTER + Paneth_NEIGHBOR' -> ('Paneth', 'Paneth'), so the count still
    matches the item count and a rule can never match itself.
    """
    return tuple(sorted(strip_role(item) for item in items))


def _rule_type(n_ant, n_con):
    """Item counts, roles included — 'Paneth_CENTER + Paneth_NEIGHBOR' is two items."""
    if n_ant + n_con <= 2:
        return "pairwise"
    if n_con == 1:
        return "ant-complex"
    if n_ant == 1:
        return "con-complex"
    return "both-complex"


@lru_cache(maxsize=None)
def _every_shorter(ant_types, con_types, kind):
    """
    Every type signature this rule contains: any items dropped, one left on each side.

    Every one is looked up directly, not only the next size down. A rule two sizes down
    can be the strongest of the lot while the one between it collapsed, and it may be
    the only one that was mined at all. Cached because many rules share a signature.
    """
    shorter = set()
    for antecedent in _parts_of(ant_types):
        for consequent in _parts_of(con_types):
            if len(antecedent) + len(consequent) < len(ant_types) + len(con_types):
                shorter.add((antecedent, consequent, kind))
    return shorter


def _parts_of(types):
    """Every non-empty part of a signature, still sorted, since types is."""
    return {part for size in range(1, len(types) + 1)
            for part in itertools.combinations(types, size)}


def _lift_beats(lift, shorter_lift, min_lift_gain, kind):
    """True when the longer rule improves on a shorter one's lift enough."""
    if kind == AVOIDS:
        return lift < shorter_lift / min_lift_gain
    return lift >= shorter_lift * min_lift_gain


def _strongest(group, lifts, kind):
    """The strongest lift in a group — highest for attraction, lowest for avoidance."""
    pick = min if kind == AVOIDS else max
    return pick(group, key=lambda pos: lifts[pos])


def _at_least_as_strong(pair_lift, lift, kind):
    """A pair backs a rule when it does the same thing at least as hard."""
    return pair_lift <= lift if kind == AVOIDS else pair_lift >= lift


def _find_consequent_backing_rules(con_types, lift, kind, pair_rules, lifts):
    """
    Do the consequents already do this to each other, without the antecedent?

    - attraction: they always cluster, so finding them by the antecedent is not news
    - avoidance:  they already exclude each other, so nothing sitting by both is not news

    Every pair of consequent types must be backed by a two-item rule of the same kind,
    at least as strong as this one. Roles are ignored here: the rule joining two types
    always has one of them as its center, so either arrangement is the same evidence.

    Returns the backing rules per pair, or None if any pair has none.
    """
    types = sorted(set(con_types))
    if len(types) < 2:
        return None

    backing = []
    for pair in itertools.combinations(types, 2):
        found = [pos for pos in pair_rules.get((frozenset(pair), kind), [])
                 if _at_least_as_strong(lifts[pos], lift, kind)]
        if not found:
            return None
        backing.append(found)
    return backing


def _named(positions, antecedents, consequents):
    """The rules themselves, roles and all, so the comparison can be read back."""
    return [f"{' + '.join(sorted(antecedents[pos]))} -> {' + '.join(sorted(consequents[pos]))}"
            for pos in positions]
