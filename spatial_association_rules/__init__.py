"""
Spatial association rule mining. See README.md.

    from spatial_association_rules import Settings, Weighting, Method, mine, classify_rules

    settings = Settings(weighting=Weighting.WEIGHTED, method=Method.CN,
                        radius=25.0, min_support=0.01,
                        min_lift=1.2, max_items_per_rule=4, avoidance_max_lift=0.8)

    result = mine(coords, labels, settings)
    tested = result.add_p_values(n_shuffles=1000, random_seed=42)
    rules  = classify_rules(tested, max_individual_fdr=0.05)
"""

from .mine import Result, mine
from .rules import classify_rules, filter_rules
from .runner import RunReport, SampleResult, run_samples
from .settings import Method, Settings, Weighting
from .validation.significance import seed_for
from .transactions import (CENTER, NEIGHBOR, find_patches, is_crowded_by_one_type,
                           strip_role)

# Small on purpose: this is the promise. Everything else is free to change.
__all__ = [
    "Settings", "Weighting", "Method",
    "mine", "Result", "classify_rules", "filter_rules",
    "run_samples", "RunReport", "SampleResult", "seed_for",
    "CENTER", "NEIGHBOR", "strip_role",
    "find_patches", "is_crowded_by_one_type",
]
