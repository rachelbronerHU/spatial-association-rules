"""
Many rules were tested at once, so correct for that. Nothing here runs on its own —
the caller picks the scope. See DESIGN.md, "Testing many rules at once".

    false_discovery_rates   many rules tested at once -> corrected p-values
"""

import numpy as np
from statsmodels.stats.multitest import multipletests


def false_discovery_rates(p_values, n_tests=None):
    """Benjamini-Hochberg; omitted tests count as p=1 when n_tests is larger."""
    p_values = np.asarray(p_values, dtype=float)
    if n_tests is None:
        n_tests = p_values.size
    if not isinstance(n_tests, (int, np.integer)) or n_tests < p_values.size:
        raise ValueError("n_tests must be an integer at least as large as the number of p-values")
    if p_values.size == 0:
        return p_values
    adjusted = multipletests(p_values, method="fdr_bh")[1]
    # Exactly equivalent to appending p=1 values, without allocating those rows.
    return np.minimum(1.0, adjusted * (n_tests / p_values.size))
