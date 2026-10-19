"""Match complex rules with their shorter parts. See DESIGN.md, "Simpler rules"."""

from itertools import combinations

from .transactions import strip_role


def is_one_sided_complex(antecedents, consequents):
    """Several items on exactly one side. Only these rules get a conditional test."""
    return (len(antecedents) > 1) != (len(consequents) > 1)


def simpler_rules(rules, raw_rules):
    """For each rule, the shorter cell-type rules measured in raw_rules, of either kind."""
    measured = {(_types(a), _types(c))
                for a, c in zip(raw_rules.get("antecedents", ()), raw_rules.get("consequents", ()))}
    found = []
    for a, c in zip(rules["antecedents"], rules["consequents"]):
        shorter = _shorter(_types(a), _types(c)) if is_one_sided_complex(a, c) else ()
        found.append([rule for rule in shorter if rule in measured])
    return found


def rule_name(antecedents, consequents):
    return f"{' + '.join(sorted(antecedents))} -> {' + '.join(sorted(consequents))}"


def _types(items):
    """Cell types without roles; repeats are kept, so item counts stay intact."""
    return tuple(sorted(strip_role(item) for item in items))


def _shorter(ants, cons):
    """Drop items from the complex side, keeping the other side fixed."""
    parts = ants if len(cons) == 1 else cons
    for size in range(1, len(parts)):
        for subset in sorted(set(combinations(parts, size))):
            yield (subset, cons) if len(cons) == 1 else (ants, subset)
