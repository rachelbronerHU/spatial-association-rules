"""Mining settings and their defaults."""

from dataclasses import dataclass, replace as _replace
from enum import Enum
from typing import Optional


class Weighting(str, Enum):
    """How much a neighbor counts."""
    WEIGHTED = "weighted"   # by distance: exp(-0.5 * (distance / bandwidth) ** 2)
    BINARY = "binary"       # by presence: 1.0, near or far

    # Print as "weighted", not "Weighting.WEIGHTED", so names reach filenames intact.
    __str__ = str.__str__


class Method(str, Enum):
    """How cells are grouped into patches."""
    CN = "CN"          # every cell inside the radius
    KNN_R = "KNN_R"    # the k nearest cells, capped by the radius

    __str__ = str.__str__


LONGEST_RULE = 5     # both searches prune by support; this caps how long a rule may get


@dataclass(frozen=True)
class Settings:
    """
    How to mine. Fields without defaults must be chosen. Optional thresholds set
    to None are not applied. See README, "Parameters".
    """

    weighting: Weighting
    method: Method
    radius: float
    min_support: float
    min_lift: float                        # what counts as attraction. >= 1
    max_items_per_rule: int                # longest rule to build. 2..LONGEST_RULE

    bandwidth: Optional[float] = None      # decay scale. Unset, it follows the radius
    k_neighbors: Optional[int] = None      # KNN_R only

    min_cells_per_patch: int = 2           # a patch of one cell has no neighbors
    max_one_type_share: float = 1.0        # skip a patch this dominated by one label

    min_patches: int = 0                   # patches backing a rule, counted in weight
    strong_confidence: Optional[float] = None       # above this, allow a lower support
    min_support_when_strong: Optional[float] = None

    min_confidence: Optional[float] = None
    min_leverage: Optional[float] = None
    min_conviction: Optional[float] = None

    include_avoidance_rules: bool = True
    avoidance_max_lift: Optional[float] = None      # what counts as avoidance. < 1
    avoidance_max_leverage: Optional[float] = None
    avoidance_min_expected_meetings: int = 10       # meetings expected before a miss counts

    min_label_count: Optional[int] = None  # drop rules naming a label this rare
    min_label_share: Optional[float] = None

    one_sided_complex_rules: bool = True      # at most one side may contain multiple items

    def __post_init__(self):
        checks = [
            (isinstance(self.one_sided_complex_rules, bool), "one_sided_complex_rules must be a bool"),
            (self.radius > 0, f"radius must be > 0, got {self.radius}"),
            (0 < self.min_support < 1, f"min_support must be in (0, 1), got {self.min_support}"),
            (self.bandwidth is None or self.bandwidth > 0, f"bandwidth must be > 0, got {self.bandwidth}"),
            (self.method is not Method.KNN_R or self.k_neighbors,
             "method KNN_R needs k_neighbors"),
            (self.min_cells_per_patch >= 2, f"min_cells_per_patch must be >= 2, got {self.min_cells_per_patch}"),
            (0 < self.max_one_type_share <= 1, f"max_one_type_share must be in (0, 1], got {self.max_one_type_share}"),
            (self.min_patches >= 0, f"min_patches must be >= 0, got {self.min_patches}"),
            ((self.strong_confidence is None) == (self.min_support_when_strong is None),
             "strong_confidence and min_support_when_strong are only meaningful together"),
            (self.min_support_when_strong is None or self.min_support_when_strong <= self.min_support,
             f"min_support_when_strong ({self.min_support_when_strong}) must be <= min_support "
             f"({self.min_support}): it lowers the bar for confident rules, it cannot raise it"),
            (self.min_lift >= 1, f"min_lift must be >= 1, got {self.min_lift}"),
            (not self.include_avoidance_rules or self.avoidance_max_lift is not None,
             "include_avoidance_rules needs avoidance_max_lift: without it a shuffled tissue "
             "passes as often as the real one and the p-values mean nothing. Pass "
             "include_avoidance_rules=False to search for attraction only"),
            (self.avoidance_max_lift is None or 0 <= self.avoidance_max_lift < 1,
             f"avoidance_max_lift must be in [0, 1), got {self.avoidance_max_lift}"),
            (self.avoidance_max_leverage is None or self.avoidance_max_leverage <= 0,
             f"avoidance_max_leverage must be <= 0, got {self.avoidance_max_leverage}"),
            (self.avoidance_min_expected_meetings > 0,
             f"avoidance_min_expected_meetings must be > 0, got {self.avoidance_min_expected_meetings}"),
            (2 <= self.max_items_per_rule <= LONGEST_RULE,
             f"max_items_per_rule must be between 2 and {LONGEST_RULE}, got {self.max_items_per_rule}"),
        ]
        for ok, message in checks:
            if not ok:
                raise ValueError(message)

    @property
    def decay_distance(self) -> float:
        """How far a neighbor's weight reaches: the bandwidth, or the radius if unset."""
        return self.bandwidth if self.bandwidth is not None else self.radius

    def replace(self, **changes) -> "Settings":
        """A copy with some fields changed."""
        return _replace(self, **changes)
