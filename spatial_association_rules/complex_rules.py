"""Compare complex rules with their shorter parts. See DESIGN.md for the policy."""

from collections import defaultdict
from itertools import combinations
from math import isfinite

import pandas as pd

from .transactions import strip_role


DEFAULT_IMPROVEMENT_GAIN = 1.1

_ADDED_COLUMNS = {"rule_type": object, "complex_class": object,
                  "adds_information": "boolean", "simpler_rules": object}


def classify_complex_rules(rules, min_lift_gain=DEFAULT_IMPROVEMENT_GAIN, max_individual_fdr=None,
                           min_consequent_conviction_gain=DEFAULT_IMPROVEMENT_GAIN):
    """Classify by lift (antecedents) or conviction (consequents); keep every row.

    Both gains default to DEFAULT_IMPROVEMENT_GAIN; None or 0 requires only strict improvement.
    Simpler rules may have either kind; the complex rule sets the comparison direction.
    An FDR cutoff excludes missing values. No cutoff or no individual_fdr column
    means effect-only comparison. Mixed rules have no class or information flag.
    """
    lift_gain = _gain(min_lift_gain, "min_lift_gain")
    conviction_gain = _gain(min_consequent_conviction_gain, "min_consequent_conviction_gain")
    if max_individual_fdr is not None and not 0 < max_individual_fdr <= 1:
        raise ValueError("max_individual_fdr must be in (0, 1], or None")

    result = rules.copy()
    if result.empty:
        for column, dtype in _ADDED_COLUMNS.items():
            result[column] = pd.Series(index=result.index, dtype=dtype)
        return result

    antecedents = list(result["antecedents"])
    consequents = list(result["consequents"])
    kinds = list(result["kind"])
    ant_types = [_types(items) for items in antecedents]
    con_types = [_types(items) for items in consequents]
    rule_types = [_rule_type(len(a), len(c)) for a, c in zip(antecedents, consequents)]

    eligible = [True] * len(result)
    if max_individual_fdr is not None and "individual_fdr" in result:
        eligible = result["individual_fdr"].le(max_individual_fdr).fillna(False).tolist()

    by_signature = defaultdict(list)
    for pos, signature in enumerate(zip(ant_types, con_types)):
        by_signature[signature].append(pos)

    classes, information, simpler_rules = [], [], []
    for pos, rule_type in enumerate(rule_types):
        if rule_type in ("pairwise", "complex-mixed"):
            classes.append(None)
            information.append(True if rule_type == "pairwise" else pd.NA)
            simpler_rules.append([])
            continue

        shorter = sorted(p for signature in _shorter(ant_types[pos], con_types[pos])
                         for p in by_signature.get(signature, []))
        significant = [p for p in shorter if eligible[p]]
        simpler_rules.append([_name(antecedents[p], consequents[p]) for p in shorter])

        if not shorter:
            category = "no_simpler"
        elif not significant:
            category = "no_significant_simpler"
        else:
            metric, gain = (("lift", lift_gain) if rule_type == "complex-antecedents"
                            else ("conviction", conviction_gain))
            values = result[metric]
            stronger = all(_beats(values.iloc[pos], values.iloc[p], gain, kinds[pos])
                           for p in significant)
            category = "stronger_than_simpler" if stronger else "redundant_by_simpler"
        classes.append(category)
        information.append(category != "redundant_by_simpler")

    result["rule_type"] = rule_types
    result["complex_class"] = pd.Series(classes, index=result.index, dtype=object)
    result["adds_information"] = pd.array(information, dtype="boolean")
    result["simpler_rules"] = simpler_rules
    return result


def _gain(value, name):
    if value is None or value == 0:
        return 1.0
    if not isfinite(value) or value < 1:
        raise ValueError(f"{name} must be a finite ratio >= 1, or None or 0")
    return value


def _beats(value, simpler, gain, kind):
    """Require a strict improvement as well as the ratio, including at zero/infinity."""
    if pd.isna(value) or pd.isna(simpler):
        return False
    if kind == "avoids":
        return value < simpler and value <= simpler / gain
    return value > simpler and value >= simpler * gain


def _rule_type(n_ant, n_con):
    if n_ant == n_con == 1:
        return "pairwise"
    if n_con == 1:
        return "complex-antecedents"
    if n_ant == 1:
        return "complex-consequents"
    return "complex-mixed"


def _types(items):
    """Match cell types without roles; keep duplicates so item counts stay intact."""
    return tuple(sorted(strip_role(item) for item in items))


def _shorter(ants, cons):
    """Drop items from the complex side, keeping the other side fixed."""
    parts = ants if len(cons) == 1 else cons
    for size in range(1, len(parts)):
        for subset in sorted(set(combinations(parts, size))):
            yield (subset, cons) if len(cons) == 1 else (ants, subset)


def _name(antecedents, consequents):
    return f"{' + '.join(sorted(antecedents))} -> {' + '.join(sorted(consequents))}"
