"""
From coordinates to transactions.

A patch is one center cell plus the cells around it, and becomes one transaction:
{item: weight}, where an item is a label plus its role, "CD8T_CENTER".
"""

from collections import Counter
from typing import NamedTuple

import numpy as np
from sklearn.neighbors import NearestNeighbors

from .settings import Method, Weighting

CENTER = "CENTER"
NEIGHBOR = "NEIGHBOR"


def item_of(label, role) -> str:
    """'CD8T' as a center -> 'CD8T_CENTER'."""
    return f"{label}_{role}"


def strip_role(item: str) -> str:
    """'CD8T_CENTER' -> 'CD8T'. One role, from the end only."""
    for role in (CENTER, NEIGHBOR):
        suffix = f"_{role}"
        if item.endswith(suffix):
            return item[:-len(suffix)]
    return item


def is_center(item: str) -> bool:
    return item.endswith(f"_{CENTER}")


class Patch(NamedTuple):
    center: int
    members: np.ndarray    # every cell in the patch, center included
    neighbors: np.ndarray
    weights: np.ndarray    # one weight per neighbor


def find_patches(coords, settings):
    """Group cells by position only. Labels are not looked at here."""
    coords = np.asarray(coords, dtype=float)
    if len(coords) == 0:
        return []

    if settings.method is Method.CN:
        finder = NearestNeighbors(radius=settings.radius, n_jobs=-1).fit(coords)
        return list(enumerate(finder.radius_neighbors(coords, return_distance=False)))

    # KNN_R: k+1 because a cell is its own nearest neighbor.
    finder = NearestNeighbors(n_neighbors=min(settings.k_neighbors + 1, len(coords)), n_jobs=-1).fit(coords)
    distances, members = finder.kneighbors(coords)
    return [
        (center, idxs[dists <= settings.radius])
        for center, (dists, idxs) in enumerate(zip(distances, members))
    ]


def is_crowded_by_one_type(labels, max_share) -> bool:
    """True when a single label takes more than max_share of the patch."""
    if len(labels) == 0:
        return False
    return Counter(labels).most_common(1)[0][1] / len(labels) > max_share


def measure_patches(patches, coords, settings):
    """How much each neighbor counts. Position only, so the null can reuse it."""
    coords = np.asarray(coords, dtype=float)
    measured = []

    decay = settings.decay_distance
    for center, members in patches:
        if len(members) < settings.min_cells_per_patch:
            continue
        members = np.asarray(members)
        neighbors = members[members != center]
        if len(neighbors) == 0:
            continue

        # The whole difference between the two weightings.
        if settings.weighting is Weighting.WEIGHTED:
            distances = np.linalg.norm(coords[neighbors] - coords[center], axis=1)
            weights = np.exp(-0.5 * (distances / decay) ** 2)
        else:
            weights = np.ones(len(neighbors))

        measured.append(Patch(center, members, neighbors, weights))

    return measured


def build_transactions(patches, labels, settings):
    """
    Patches to transactions.

    Neighbors of the same label add up, capped at 1.0. The center always counts 1.0.
    """
    labels = np.asarray(labels, dtype=object)
    transactions = []

    for patch in patches:
        if is_crowded_by_one_type(labels[patch.members], settings.max_one_type_share):
            continue

        transaction = {}
        for neighbor, weight in zip(patch.neighbors, patch.weights):
            item = item_of(labels[neighbor], NEIGHBOR)
            transaction[item] = transaction.get(item, 0.0) + weight
        transaction = {item: min(weight, 1.0) for item, weight in transaction.items()}
        transaction[item_of(labels[patch.center], CENTER)] = 1.0

        transactions.append(transaction)

    return transactions, {"patches_kept": len(transactions)}
