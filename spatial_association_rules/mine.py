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
from .rules import count_candidate_rules, drop_rare_labels, empty_rules, weight_matrix
from .settings import Settings
from .validation.significance import p_values_for
from .validation.false_discovery import false_discovery_rates, minimum_shuffles_for_fdr
from .transactions import Patch, build_transactions, find_patches, measure_patches


def mine_rules(transactions, settings: Settings, sample_id: str = "") -> pd.DataFrame:
    """Transactions in, rules out: both searches, one frame."""
    if not transactions:
        return empty_rules()

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

    found = [frame for frame in found if not frame.empty]
    return pd.concat(found, ignore_index=True) if found else empty_rules()


@dataclass
class Result:
    """What one run produced, plus what it needs to test those rules later."""

    rules: pd.DataFrame
    stats: dict
    patches: List[Patch] = field(repr=False)
    labels: np.ndarray = field(repr=False)
    settings: Settings = field(repr=False)

    def add_p_values(self, n_shuffles, random_seed=None, labels_kept_fixed=(), sample_id="",
                     max_individual_fdr=None):
        """
        Return raw p-values and corrected p-values (individual_fdr) for this sample.
        Correct each rule size separately, with attraction and avoidance together.

        Rules dropped by the search count as p=1, without adding rows. With a cutoff
        set, warn if too few shuffles are planned for any rule of a size to pass.
        Those sizes still get raw p-values, but individual_fdr is NaN (missing).
        A cutoff of None skips this check and still calculates corrected values.
        """
        if max_individual_fdr is not None and not 0 < max_individual_fdr <= 1:
            raise ValueError("max_individual_fdr must be in (0, 1], or None")
        rules = self.rules.copy()
        sizes = rules["antecedents"].map(len) + rules["consequents"].map(len)
        rules["individual_fdr"] = 1.0
        families = []
        for size, group in rules.groupby(sizes):
            n_tests = count_candidate_rules(self.labels, self.settings, n_items=size)
            if max_individual_fdr is not None:
                needed = minimum_shuffles_for_fdr(n_tests, len(group), max_individual_fdr)
                if n_shuffles < needed:
                    prefix = f"[{sample_id}] " if sample_id else ""
                    logger.warning(
                        f"{prefix}Insufficient permutation resolution for {size}-item rules: "
                        f"{n_tests} candidates, {len(group)} mined, {n_shuffles} shuffles. "
                        f"At least {needed} shuffles are needed for any possibility of "
                        f"BH <= {max_individual_fdr}, even with zero shuffle successes. "
                        "Raw p-values will still be calculated; individual_fdr is NaN."
                    )
                    rules.loc[group.index, "individual_fdr"] = np.nan
                    continue
            families.append((group.index, n_tests))

        rules["p_value"] = p_values_for(
            rules, self.patches, self.labels, self.settings,
            n_shuffles, random_seed, labels_kept_fixed, sample_id,
        )
        for index, n_tests in families:
            rules.loc[index, "individual_fdr"] = false_discovery_rates(
                rules.loc[index, "p_value"].values, n_tests=n_tests,
            )
        return rules


def mine(coords, labels, settings: Settings, sample_id: str = "") -> Result:
    """
    Mine spatial association rules.

    coords:  (n_cells, 2) positions
    labels:  (n_cells,) one label per cell, one label per cell type

    No significance testing here — call result.add_p_values() for that.
    """
    coords = np.asarray(coords, dtype=float)
    labels = np.asarray(labels, dtype=object)
    if len(coords) != len(labels):
        raise ValueError(f"coords has {len(coords)} rows but labels has {len(labels)}")

    patches = find_patches(coords, settings)
    measured = measure_patches(patches, coords, settings)
    transactions, stats = build_transactions(measured, labels, settings)
    stats["patches_found"] = len(patches)

    # Rare labels go first: the shuffle test after them is what the run pays for.
    rules = drop_rare_labels(mine_rules(transactions, settings, sample_id=sample_id), labels, settings)

    return Result(rules=rules, stats=stats, patches=measured, labels=labels, settings=settings)
