"""
Is a rule more than an accident of how common its cell types are?

Hold the tissue still, shuffle the labels between cells, and see how often the rule
still passes. See README, "add_p_values".
"""

import logging
import time
import zlib
from typing import NamedTuple

import numpy as np
from scipy import sparse

from ..attraction import attracts
from ..avoidance import avoids
from ..rules import AVOIDS, metrics, packed, support_of_many, support_of_many_packed
from ..settings import Weighting
from ..transactions import CENTER, NEIGHBOR, item_of

logger = logging.getLogger(__name__)


def seed_for(base_seed, sample_id):
    """
    A seed per sample: re-runs match, and samples stay independent.

    crc32, not hash(), which Python randomizes per process.
    """
    if base_seed is None:
        return None
    return base_seed + zlib.crc32(str(sample_id).encode())


def p_values_for(rules, patches, labels, settings, n_shuffles, random_seed, labels_kept_fixed, sample_id=""):
    """The raw p-value for each rule, in order. Nothing is corrected here (no FDR)."""

    if rules.empty or n_shuffles <= 0:
        return np.ones(len(rules))
    
    if "kind" not in rules.columns:
        raise ValueError("rules need a 'kind' column: a rule is tested against the "
                         "thresholds of the search that found it, and this cannot guess which")
    
    if (rules["kind"] == AVOIDS).any() and settings.avoidance_max_lift is None:
        raise ValueError("these rules include avoidance, but the settings have no "
                         "avoidance_max_lift to test them against. Testing them by a "
                         "different threshold than the one that found them would make "
                         "the p-values answer a different question")


    # --- 1. ENCODE LABELS ---
    # One-hot encode labels (1 row per cell, 1 column per cell type).
    #
    # Example: (Col 0=T-Cell, Col 1=B-Cell, Col 2=Macro)
    #          [ T, B, M ]
    # Cell 0:  [ 1, 0, 0 ]  (T-Cell)
    # Cell 1:  [ 0, 1, 0 ]  (B-Cell)
    # Cell 2:  [ 1, 0, 0 ]  (T-Cell)
    #
    # The shuffle test permutes these rows to randomly reassign cell types.
    labels = np.asarray(labels, dtype=object)
    names = sorted({str(label) for label in labels})
    column_of = {name: i for i, name in enumerate(names)}

    cell_labels = np.zeros((len(labels), len(names)), dtype=float)
    for cell, label in enumerate(labels):
        cell_labels[cell, column_of[str(label)]] = 1.0

    # --- 2. MAP THE TISSUE ---
    # Build sparse adjacency maps. The physical tissue layout stays fixed
    # so we avoid recalculating spatial distances for every shuffle.
    #
    # Example (centers matrix): 
    #          [ C0, C1, C2 ]
    # Patch 0: [  0,  1,  0 ]  (Patch 0's center is Cell 1)
    # Patch 1: [  0,  0,  1 ]  (Patch 1's center is Cell 2)
    # Patch 2: [  1,  0,  0 ]  (Patch 2's center is Cell 0)
    centers, neighbors, membership, patch_sizes = _adjacency(patches, len(labels))


    # --- 3. PREPARE THE RULES ---

    # Map human-readable cell names to matrix column indices.
    item_index = {}
    for name, column in column_of.items():
        item_index[item_of(name, CENTER)] = column
        item_index[item_of(name, NEIGHBOR)] = column + len(names)

    # --- 4. LOCK FIXED CELLS ---

    # Identify which cells can be randomly reassigned vs which must stay anchored.
    movable = np.arange(len(labels))[~_held_fixed(labels, labels_kept_fixed)]
    _check_enough_moves(movable.size, len(labels), labels_kept_fixed)
    
    rng = np.random.default_rng(random_seed)
    layout = _rule_columns(rules, item_index)

    prefix = f"[{sample_id}] " if sample_id else ""
    logger.info(f"{prefix}Shuffling labels {n_shuffles} times against {len(rules)} rules...")
    started = time.time()

    # --- 5. THE SHUFFLE TEST ---
    # Randomize label assignments n times to see if rules survive by chance.
    #
    # How matrix multiplication (centers @ shuffled) instantly rebuilds the patches.
    # Imagine the shuffle just randomly turned Cell 1 into a Macrophage:
    #
    #    [ centers matrix ]   @    [ shuffled matrix ]    =  [ final patch center types ]
    #       (C0, C1, C2)               (T, B, M)                    (T, B, M)
    #
    # P0: [  0,  1,  0  ]          C0: [ 0, 1, 0 ]           P0: [ 0, 0, 1 ]  <- (P0 center is now a Macro!)
    # P1: [  0,  0,  1  ]    @     C1: [ 0, 0, 1 ]    =      P1: [ 1, 0, 0 ]
    # P2: [  1,  0,  0  ]          C2: [ 1, 0, 0 ]           P2: [ 0, 1, 0 ]
    # 
    # The math instantly maps the fake labels onto the physical patches!
    survived = np.zeros(len(rules))
    for i in range(n_shuffles):
        order = np.arange(len(labels))
        if movable.size > 1:
            order[movable] = rng.permutation(movable)
        shuffled = cell_labels[order, :]

        # Multiply the fixed tissue maps by the randomized labels to rebuild patches.
        transactions = np.hstack([
            np.minimum(_dense(centers @ shuffled), 1.0),
            np.minimum(_dense(neighbors @ shuffled), 1.0),
        ])
        
        # Drop patches crowded by a single cell type, repeating the real run's procedure.
        transactions = transactions[not_crowded(membership, patch_sizes, shuffled,
                                                settings.max_one_type_share)]
                                                
        # Tally how many rules passed the statistical thresholds by pure chance.
        survived += survives_shuffle(layout, transactions, settings)

        if i == 0 or (i + 1) % 100 == 0 or i == n_shuffles - 1:
            logger.debug(f"{prefix}  shuffle {i + 1}/{n_shuffles}")

    elapsed = time.time() - started
    logger.info(f"{prefix}Shuffling took {int(elapsed // 60)}m {elapsed % 60:.1f}s" if elapsed >= 60 else f"{prefix}Shuffling took {elapsed:.2f}s")

    # --- 6. SCORE P-VALUES ---
    # (times survived by luck + 1) / (total shuffles + 1)
    p_values = (survived + 1) / (n_shuffles + 1)
    
    # If a cell type was completely absent, its rule was never tested (p = 1.0).
    p_values[~layout.usable] = 1.0
    return p_values


def _adjacency(patches, n_cells):
    """
    Maps patches to physical cell IDs so shuffle tests can run via fast sparse matrix math.

    Returns 3 sparse matrices:
    - centers: Identifies the exact single cell acting as the center of each patch.
    - neighbors: Holds the mathematical distance weights of all surrounding cells.
    - membership: Records a plain 1 for every cell inside the patch, used to quickly count raw cell totals.
    """
    center_rows, center_cols = [], []
    neighbor_rows, neighbor_cols, neighbor_weights = [], [], []
    member_rows, member_cols = [], []

    for row, patch in enumerate(patches):
        center_rows.append(row)
        center_cols.append(patch.center)
        for neighbor, weight in zip(patch.neighbors, patch.weights):
            neighbor_rows.append(row)
            neighbor_cols.append(neighbor)
            neighbor_weights.append(weight)
        for member in patch.members:
            member_rows.append(row)
            member_cols.append(member)

    shape = (len(patches), n_cells)
    membership = sparse.csr_matrix((np.ones(len(member_rows)), (member_rows, member_cols)), shape=shape)
    return (
        sparse.csr_matrix((np.ones(len(center_rows)), (center_rows, center_cols)), shape=shape),
        sparse.csr_matrix((neighbor_weights, (neighbor_rows, neighbor_cols)), shape=shape),
        membership,
        np.asarray([len(p.members) for p in patches], dtype=float),
    )


def not_crowded(membership, patch_sizes, cell_labels, max_one_type_share):
    """
    Which patches are mixed enough to keep. Same rule as build_transactions, in bulk.
    """
    counts = _dense(membership @ cell_labels)
    return counts.max(axis=1) <= max_one_type_share * patch_sizes


def _held_fixed(labels, patterns):
    """
    Cells whose label never moves.

    'Name' matches exactly. 'Name*' matches any label starting with Name.
    """
    fixed = np.zeros(len(labels), dtype=bool)
    for pattern in patterns:
        pattern = str(pattern).strip()
        if pattern == "*":
            raise ValueError("labels_kept_fixed of '*' would hold every label still, leaving nothing to shuffle")
        if not pattern:
            continue
        for cell, label in enumerate(labels):
            label = str(label)
            if label.startswith(pattern[:-1]) if pattern.endswith("*") else label == pattern:
                fixed[cell] = True

    if patterns and not fixed.any():
        logger.warning(f"No cell matched labels_kept_fixed={tuple(patterns)}")
    return fixed


ENOUGH_MOVABLE = 0.1     # below this share still free to move, warn


def _check_enough_moves(movable, n_cells, patterns):
    """Pin too many labels and the shuffled tissue is the real one, so refuse."""
    if movable <= 1:
        raise ValueError(
            f"labels_kept_fixed={tuple(patterns)} leaves {movable} of {n_cells} cells free "
            f"to move, so no shuffle changes anything and every p-value would be 1.0. "
            f"Pin fewer labels, or pass n_shuffles=0 if you meant not to test"
        )
    if movable < ENOUGH_MOVABLE * n_cells:
        logger.warning(
            f"labels_kept_fixed={tuple(patterns)} leaves only {movable} of {n_cells} cells "
            f"({movable / n_cells:.1%}) free to move. The shuffled tissue barely differs "
            f"from the real one, so the p-values will be very conservative"
        )


class _Layout(NamedTuple):
    """Where each rule finds its supports, and which judge it answers to."""

    sized: list          # (positions, columns) per group size, measured in one array
    n_groups: int
    ant_at: np.ndarray   # index into the measured supports, per rule
    con_at: np.ndarray
    joint_at: np.ndarray
    usable: np.ndarray   # False for a rule naming an item this sample does not have
    avoiding: np.ndarray


def _rule_columns(rules, item_index):
    """
    The distinct column groups to measure, and where each rule reads its own.

    Rules overlap heavily, so each group is measured once per shuffle. The layout is
    the same every shuffle, so it is worked out once.
    """
    known, groups = {}, []            # column tuple -> its place in the answers

    def place(columns):
        key = tuple(sorted(columns))
        if key not in known:
            known[key] = len(known)
            groups.append(key)
        return known[key]

    ant_at, con_at, joint_at, usable = [], [], [], []
    for rule in rules.itertuples():
        ant = [item_index.get(item) for item in rule.antecedents]
        con = [item_index.get(item) for item in rule.consequents]
        if None in ant or None in con:
            ant_at.append(0), con_at.append(0), joint_at.append(0), usable.append(False)
            continue
        ant_at.append(place(ant))
        con_at.append(place(con))
        joint_at.append(place(ant + con))
        usable.append(True)

    # Groups of the same size can be measured together in one array.
    by_size = {}
    for position, columns in enumerate(groups):
        by_size.setdefault(len(columns), []).append((position, columns))
    sized = [(np.asarray([p for p, _ in members]), np.asarray([c for _, c in members]))
             for members in by_size.values()]

    return _Layout(sized, len(groups), np.asarray(ant_at), np.asarray(con_at),
                   np.asarray(joint_at), np.asarray(usable),
                   (rules["kind"] == AVOIDS).to_numpy())


def _supports(layout, transactions, settings):
    """Support of every distinct column group, then read off per rule."""
    # Binary weights are 0 or 1, so a support is a count of bits. Weighted ones need the floats.
    binary = settings.weighting is Weighting.BINARY
    bits = packed(transactions) if binary else None

    measured = np.zeros(layout.n_groups)
    for positions, group_columns in layout.sized:
        measured[positions] = (support_of_many_packed(bits, group_columns, len(transactions))
                               if binary else support_of_many(transactions, group_columns))

    return measured[layout.ant_at], measured[layout.con_at], measured[layout.joint_at]


def survives_shuffle(layout, transactions, settings):
    """Which rules still pass their own thresholds here. Each judged by its own kind."""
    held = np.zeros(len(layout.usable))
    # Nothing usable means nothing was measured, so nothing to read off.
    if transactions.size == 0 or not layout.usable.any():
        return held

    n = transactions.shape[0]
    ant_sup, con_sup, joint = _supports(layout, transactions, settings)

    # Both sides must exist before the thresholds mean anything.
    testable = layout.usable & (ant_sup > 0) & (con_sup > 0)
    if not testable.any():
        return held

    # Both judges run over every rule; each rule takes its own answer.
    measures = metrics(joint, ant_sup, con_sup)
    held[testable] = np.where(layout.avoiding,
                              avoids(settings, joint, ant_sup, con_sup, measures, n),
                              attracts(settings, joint, ant_sup, con_sup, measures, n))[testable]
    return held



def _dense(matrix):
    return matrix.toarray() if sparse.issparse(matrix) else matrix
