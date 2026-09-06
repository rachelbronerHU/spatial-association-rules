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
from .validation.false_discovery import false_discovery_rates
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

    def add_p_values(self, n_shuffles, rules=None, random_seed=None, labels_kept_fixed=(), sample_id=""):
        """
        Test rules against shuffled labels: a raw p_value, and individual_fdr, that
        same p-value corrected across every rule tested here.

        Defaults to the rules this sample produced, but takes any subset — the
        correction is over whatever you pass in.
        """
        rules = (self.rules if rules is None else rules).copy()
        rules["p_value"] = p_values_for(
            rules, self.patches, self.labels, self.settings,
            n_shuffles, random_seed, labels_kept_fixed, sample_id,
        )
        rules["individual_fdr"] = false_discovery_rates(rules["p_value"].values)
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
