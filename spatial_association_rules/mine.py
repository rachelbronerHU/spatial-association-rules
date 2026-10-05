"""The entry point: coordinates and labels in, rules out."""

import logging
import time
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from .attraction import mine_attraction
from .avoidance import mine_avoidance
from .rules import drop_rare_labels, empty_rules, weight_matrix
from .settings import Settings
from .validation.significance import p_values_for
from .validation.conditional import conditional_p_values, conditional_test_plan
from .validation.false_discovery import fdr_by_size, fdr_families
from .transactions import Patch, build_transactions, find_patches, measure_patches


def mine_rules(transactions, settings: Settings, sample_id: str = ""):
    """Return passed rules and rules before the final filters."""
    if not transactions:
        return empty_rules(), empty_rules()

    matrix, item_index = weight_matrix(transactions)
    
    start_attraction = time.time()
    found = [mine_attraction(transactions, matrix, item_index, settings)]
    elapsed_att = time.time() - start_attraction
    str_attraction_time = f"Attraction search took {int(elapsed_att // 60)}m {elapsed_att % 60:.1f}s" if elapsed_att >= 60 else f"Attraction search took {elapsed_att:.2f}s"
    str_avoidance_time = ""

    if settings.include_avoidance_rules:
        start_avoidance = time.time()
        found.append(mine_avoidance(matrix, item_index, settings, sample_id=sample_id))
        elapsed_avo = time.time() - start_avoidance
        str_avoidance_time = f"Avoidance search took {int(elapsed_avo // 60)}m {elapsed_avo % 60:.1f}s" if elapsed_avo >= 60 else f"Avoidance search took {elapsed_avo:.2f}s"

    prefix = f"[{sample_id}] " if sample_id else ""
    logger.info(f"{prefix}{str_attraction_time} {'| ' + str_avoidance_time if str_avoidance_time else ''}")

    rules, raw_rules = ([frame for frame in frames if not frame.empty] for frames in zip(*found))
    return (pd.concat(rules, ignore_index=True) if rules else empty_rules(),
            pd.concat(raw_rules, ignore_index=True) if raw_rules else empty_rules())


@dataclass
class Result:
    """What one run produced, plus what it needs to test those rules later."""

    rules: pd.DataFrame
    stats: dict
    patches: List[Patch] = field(repr=False)
    labels: np.ndarray = field(repr=False)
    settings: Settings = field(repr=False)
    raw_rules: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)

    def add_p_values(self, n_shuffles, random_seed=None, labels_kept_fixed=(), sample_id="",
                     max_individual_fdr=None, calculate_fdr=True):
        """
        Return raw p-values and, if requested, individual_fdr for this sample.
        Correct each rule size separately, with attraction and avoidance together.

        Rules dropped by the search count as p=1, without adding rows. With a cutoff
        set, warn if too few shuffles are planned for any rule of a size to pass.
        Those sizes still get raw p-values, but individual_fdr is NaN (missing).
        A cutoff of None skips this check; calculate_fdr chooses whether to correct.
        """
        if not isinstance(calculate_fdr, bool):
            raise ValueError("calculate_fdr must be a bool")
        if max_individual_fdr is not None and not 0 < max_individual_fdr <= 1:
            raise ValueError("max_individual_fdr must be in (0, 1], or None")
        if not calculate_fdr and max_individual_fdr is not None:
            raise ValueError("max_individual_fdr requires calculate_fdr=True")
        labels_kept_fixed = tuple(labels_kept_fixed)
        rules = self.rules.copy()
        rules.drop(columns="individual_fdr", errors="ignore", inplace=True)
        if calculate_fdr:
            # Check before shuffling, so warnings come first.
            families = fdr_families(rules, self.labels, self.settings, n_shuffles=n_shuffles,
                                    max_fdr=max_individual_fdr, column="individual_fdr",
                                    sample_id=sample_id)
        rules["p_value"] = p_values_for(
            rules, self.patches, self.labels, self.settings,
            n_shuffles, random_seed, labels_kept_fixed, sample_id,
        )
        if calculate_fdr:
            rules["individual_fdr"] = fdr_by_size(rules, rules["p_value"], families)
        rules.attrs["labels_kept_fixed"] = labels_kept_fixed
        return rules

    def conditional_test_plan(self, tested, *, max_individual_fdr=0.05, labels_kept_fixed=()):
        """Preview comparisons and fixed-type sets. No shuffles are run."""
        return conditional_test_plan(tested, self.labels, max_individual_fdr, labels_kept_fixed)

    def add_conditional_p_values(self, tested, *, n_shuffles, max_individual_fdr=0.05,
                                 random_seed=None, labels_kept_fixed=(), sample_id="",
                                 calculate_fdr=True):
        """Return (rules, comparisons) with optional conditional_fdr.

        Inherits fixed labels recorded by add_p_values; labels_kept_fixed adds more.
        Requires individual_fdr to select parents. If correction is requested, warns
        when too few shuffles can reach the cutoff. See DESIGN.md.
        """
        return conditional_p_values(
            tested, self.patches, self.labels, self.settings, n_shuffles=n_shuffles,
            max_individual_fdr=max_individual_fdr, random_seed=random_seed,
            labels_kept_fixed=labels_kept_fixed, sample_id=sample_id,
            calculate_fdr=calculate_fdr,
        )


def mine(coords, labels, settings: Settings, sample_id: str = "") -> Result:
    """
    Mine spatial association rules.

    coords:  (n_cells, 2) positions
    labels:  (n_cells,) one label per cell, one label per cell type

    No significance testing here — call result.add_p_values() for that.
    Rules before the final filters are in result.raw_rules.
    """
    coords = np.asarray(coords, dtype=float)
    labels = np.asarray(labels, dtype=object)
    if len(coords) != len(labels):
        raise ValueError(f"coords has {len(coords)} rows but labels has {len(labels)}")

    patches = find_patches(coords, settings)
    measured = measure_patches(patches, coords, settings)
    transactions, stats = build_transactions(measured, labels, settings)
    stats["patches_found"] = len(patches)

    rules, raw_rules = mine_rules(transactions, settings, sample_id)
    # Rare labels go first: the shuffle test after them is what the run pays for.
    rules = drop_rare_labels(rules, labels, settings)
    rules = rules.assign(rule_idx=np.arange(len(rules)))

    return Result(rules=rules, stats=stats, patches=measured, labels=labels,
                  settings=settings, raw_rules=raw_rules)
