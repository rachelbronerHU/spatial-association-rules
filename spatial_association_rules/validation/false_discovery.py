"""
Many rules were tested at once, so correct for that. Nothing here runs on its own —
the caller picks the scope. See DESIGN.md, "Testing many rules at once".

    false_discovery_rates   many rules tested at once -> corrected p-values
"""

import numpy as np
from statsmodels.stats.multitest import multipletests


def false_discovery_rates(p_values):
    """Benjamini-Hochberg: of the results you would call significant, what share are wrong."""
    p_values = np.asarray(p_values, dtype=float)
    if p_values.size == 0:
        return p_values
    return multipletests(p_values, method="fdr_bh")[1]
