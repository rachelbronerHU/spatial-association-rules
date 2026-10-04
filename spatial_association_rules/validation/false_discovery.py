"""
Correct p-values for testing many rules. The caller chooses which rules to group.
See DESIGN.md, "Testing many rules at once".
"""

import logging
from collections import Counter
from math import ceil

import numpy as np
from statsmodels.stats.multitest import multipletests

from ..rules import count_candidate_rules

logger = logging.getLogger(__name__)


def minimum_shuffles_for_fdr(n_tests, n_rules, max_individual_fdr):
    """Fewest shuffles that could let any rule pass, assuming all get the best result."""
    if n_rules == 0 or max_individual_fdr == 1:
        return 0  # Adjusted values are capped at 1.
    return max(0, ceil(n_tests / (max_individual_fdr * n_rules)) - 1)


def false_discovery_rates(p_values, n_tests=None):
    """Correct with Benjamini-Hochberg. Extra tests in n_tests count as p=1."""
    p_values = np.asarray(p_values, dtype=float)
    if n_tests is None:
        n_tests = p_values.size
    if not isinstance(n_tests, (int, np.integer)) or n_tests < p_values.size:
        raise ValueError("n_tests must be an integer at least as large as the number of p-values")
    if p_values.size == 0:
        return p_values
    adjusted = multipletests(p_values, method="fdr_bh")[1]
    # Count the missing tests as p=1 without adding rows.
    return np.minimum(1.0, adjusted * (n_tests / p_values.size))


def fdr_families(rules, labels, settings, *, n_shuffles, max_fdr, column, sample_id=""):
    """Candidates per rule size, or None where correction is skipped with a warning.

    Pass the rows that will get p-values. A size is skipped if it has more rows than
    candidates, or, with max_fdr set, too few shuffles for any row to reach max_fdr.
    """
    prefix = f"[{sample_id}] " if sample_id else ""
    families = {}
    for size, n_rules in sorted(Counter(_sizes(rules)).items()):
        n_tests = count_candidate_rules(labels, settings, n_items=size)
        families[size] = n_tests
        if n_rules > n_tests:
            logger.warning(
                f"{prefix}{n_rules} {size}-item rules but only {n_tests} candidates. "
                "Rules are repeated or come from another sample; "
                f"pass this sample's rules once each. {column} is NaN for this size."
            )
            families[size] = None
        elif max_fdr is not None:
            needed = minimum_shuffles_for_fdr(n_tests, n_rules, max_fdr)
            if n_shuffles < needed:
                logger.warning(
                    f"{prefix}Insufficient permutation resolution for {size}-item rules: "
                    f"{n_tests} candidates, {n_rules} rules, {n_shuffles} shuffles. "
                    f"At least {needed} shuffles are needed for any possibility of "
                    f"BH <= {max_fdr}, even with zero shuffle successes. "
                    f"Raw p-values are kept; {column} is NaN for this size."
                )
                families[size] = None
    return families


def fdr_by_size(rules, p_values, families):
    """Correct each size from fdr_families; missing p-values and skipped sizes stay NaN."""
    p_values = np.asarray(p_values, dtype=float)
    sizes = _sizes(rules)
    adjusted = np.full(len(p_values), np.nan)
    for size, n_tests in families.items():
        rows = (sizes == size) & ~np.isnan(p_values)
        if n_tests is not None and rows.any():
            adjusted[rows] = false_discovery_rates(p_values[rows], n_tests=n_tests)
    return adjusted


def _sizes(rules):
    return (rules["antecedents"].map(len) + rules["consequents"].map(len)).to_numpy()
