"""
Correct p-values for testing many rules. The caller chooses which rules to group.
See DESIGN.md, "Testing many rules at once".
"""

from math import ceil

import numpy as np
from statsmodels.stats.multitest import multipletests


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
