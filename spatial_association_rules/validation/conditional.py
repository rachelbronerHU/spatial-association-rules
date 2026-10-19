"""Shuffle tests for complex rules, with each simpler rule's cell types held fixed."""

import json
import logging

import numpy as np
import pandas as pd

from ..complex_rules import is_one_sided_complex, rule_name, simpler_rules
from ..transactions import strip_role
from .false_discovery import check_fdr_cutoff, fdr_by_size, fdr_families
from .permutations import record_permutations
from .significance import (
    _adjacency, _encode_labels, _held_fixed, _rule_columns, _transactions,
    fill_lift_and_conviction, measure_rules, seed_for,
)

logger = logging.getLogger(__name__)

PLAN_COLUMNS = ["rule_pos", "rule", "simpler_rule", "kind", "metric", "observed",
                "fixed_types", "movable_cells", "status"]


def conditional_test_plan(rules, raw_rules, labels, labels_kept_fixed=(), metric=None):
    """List the comparisons and their fixed cell types, without shuffling.

    metric: "lift", "conviction", or None for lift on complex antecedents and
    conviction on complex consequents.
    """
    check_metric(metric)
    needed = {"antecedents", "consequents", "kind"} | ({metric} if metric else {"lift", "conviction"})
    if not needed.issubset(rules.columns):
        raise ValueError(f"rules need the columns {sorted(needed)}")
    if not rules["kind"].isin(["attracts", "avoids"]).all():
        raise ValueError("rule kind must be attracts or avoids")

    labels = np.asarray(labels, dtype=str)
    patterns = tuple(rules.attrs.get("labels_kept_fixed", ())) + tuple(labels_kept_fixed)
    configured = set(labels[_held_fixed(labels, patterns)])
    available = set(labels)
    movable_by_fixed = {}
    comparisons = []
    rows = rules.itertuples(index=False)
    for pos, (row, simpler) in enumerate(zip(rows, simpler_rules(rules, raw_rules))):
        rule_types = set(map(strip_role, tuple(row.antecedents) + tuple(row.consequents)))
        row_metric = metric or ("lift" if len(row.consequents) == 1 else "conviction")
        observed = getattr(row, row_metric)
        seen = set()
        for ants, cons in simpler:
            fixed = tuple(sorted(configured | set(ants) | set(cons)))
            if fixed in seen:
                continue  # Another simpler rule already fixes the same types.
            seen.add(fixed)
            if fixed not in movable_by_fixed:
                movable = ~np.isin(labels, fixed)
                movable_by_fixed[fixed] = (int(movable.sum()), len(set(labels[movable])) >= 2)
            movable_cells, can_shuffle = movable_by_fixed[fixed]
            status = "ready"
            if not rule_types.issubset(available):
                status = "missing_cell_type"
            elif rule_types.issubset(fixed):
                status = "all_rule_types_fixed"
            elif not can_shuffle:
                status = "no_exchangeable_labels"
            elif pd.isna(observed):
                status = "undefined_metric"
            comparisons.append(dict(
                rule_pos=pos, rule=rule_name(row.antecedents, row.consequents),
                simpler_rule=rule_name(ants, cons), kind=row.kind, metric=row_metric,
                observed=observed, fixed_types=fixed, movable_cells=movable_cells, status=status,
            ))
    plan = pd.DataFrame(comparisons, columns=PLAN_COLUMNS)
    plan.attrs["shuffle_batches"] = plan.loc[plan.status == "ready", "fixed_types"].nunique()
    return plan


def conditional_p_values(rules, raw_rules, patches, labels, settings, *, n_shuffles,
                         random_seed=None, labels_kept_fixed=(), metric=None, sample_id="",
                         calculate_fdr=True, max_conditional_individual_fdr=None,
                         permutation_dir=None):
    """How often a shuffle makes the complex rule as strong as mined. See DESIGN.md.

    Each simpler rule gives one comparison; the rule keeps their largest p-value.
    """
    if not isinstance(calculate_fdr, bool):
        raise ValueError("calculate_fdr must be a bool")
    check_fdr_cutoff(max_conditional_individual_fdr, "max_conditional_individual_fdr")
    if not calculate_fdr and max_conditional_individual_fdr is not None:
        raise ValueError("max_conditional_individual_fdr requires calculate_fdr=True")
    check_shuffle_count(n_shuffles, "n_shuffles")
    labels = np.asarray(labels, dtype=str)
    plan = conditional_test_plan(rules, raw_rules, labels, labels_kept_fixed, metric)
    plan["p_value"] = np.nan
    ids = rules["rule_idx"].to_numpy() if "rule_idx" in rules else np.arange(len(rules))
    batches = 0
    ready = plan.status == "ready"
    if calculate_fdr:
        # Only rules with every comparison ready will receive a combined p-value.
        all_ready = ready.groupby(plan.rule_pos).all()
        testable_positions = all_ready.index[all_ready.to_numpy(dtype=bool)].to_numpy(dtype=int)
        families = fdr_families(rules.iloc[testable_positions], labels, settings,
                                n_shuffles=n_shuffles, max_fdr=max_conditional_individual_fdr,
                                column="conditional_individual_fdr",
                                sample_id=sample_id)
    if ready.any():
        cell_labels, item_index = _encode_labels(labels)
        adjacency = _adjacency(patches, len(labels))
        groups = plan[ready].groupby("fixed_types", sort=True)
        batches = groups.ngroups
        prefix = f"[{sample_id}] " if sample_id else ""
        logger.info(f"{prefix}Conditional tests: {len(plan)} comparisons, "
                    f"{batches} shuffle batches, {n_shuffles} shuffles per batch")

        for batch, (fixed, group) in enumerate(groups, 1):
            movable = np.flatnonzero(~np.isin(labels, fixed))
            positions = np.unique(group.rule_pos.to_numpy(dtype=int))
            layout = _rule_columns(rules.iloc[positions], item_index)
            rng = np.random.default_rng(seed_for(random_seed, json.dumps(fixed)))
            column = np.searchsorted(positions, group.rule_pos.to_numpy(dtype=int))
            uses_conviction = (group.metric == "conviction").to_numpy()
            observed = group.observed.to_numpy(dtype=float)
            avoids = (group.kind == "avoids").to_numpy()
            hits = np.zeros(len(group), dtype=int)
            rule_idx = ids[positions] if permutation_dir is not None else None
            lift, conviction = np.empty(len(positions)), np.empty(len(positions))
            with record_permutations(permutation_dir, n_shuffles, rule_idx, sample_id=sample_id,
                                     test="conditional", fixed_types=fixed, batch=batch) as recorded:
                for i in range(n_shuffles):
                    order = np.arange(len(labels))
                    order[movable] = rng.permutation(movable)
                    transactions = _transactions(cell_labels[order], adjacency, settings)
                    fill_lift_and_conviction(measure_rules(layout, transactions, settings),
                                             (lift, conviction))
                    if recorded is not None:
                        recorded["lift"][i] = lift
                        recorded["conviction"][i] = conviction
                    shuffled = np.where(uses_conviction, conviction[column], lift[column])
                    hits += _as_strong(shuffled, observed, avoids)
            plan.loc[group.index, "p_value"] = (hits + 1) / (n_shuffles + 1)
            plan.loc[group.index, "status"] = "tested"
            logger.debug(f"{prefix}Conditional batch {batch}/{batches}: fixed {fixed}")

    result = _attach_results(rules, plan)
    if calculate_fdr:
        result["conditional_individual_fdr"] = fdr_by_size(
            result, result["conditional_p_value"], families)
    else:
        result.drop(columns="conditional_individual_fdr", errors="ignore", inplace=True)
    result.attrs["conditional_test_summary"] = dict(
        comparisons=len(plan), shuffle_batches=batches, n_shuffles=n_shuffles,
        shuffled_tissues=batches * n_shuffles,
    )
    # Report stored rule ids, so comparisons still match after rules are filtered or sorted.
    comparisons = plan.rename(columns={"rule_pos": "rule_idx"})
    comparisons["rule_idx"] = ids[plan.rule_pos.to_numpy(dtype=int)]
    return result, comparisons


def check_shuffle_count(value, name):
    """Shared by the runner, so bad input fails before any sample is mined."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def check_metric(metric):
    """Shared by the runner, so bad input fails before any sample is mined."""
    if metric not in (None, "lift", "conviction"):
        raise ValueError("metric must be 'lift', 'conviction', or None")


def _as_strong(shuffled, observed, avoids):
    """As high as observed for attraction, as low for avoidance. Ties count; NaN never does."""
    tied = np.isclose(shuffled, observed, rtol=1e-12, atol=0)
    return np.where(avoids, shuffled <= observed, shuffled >= observed) | tied


def _attach_results(rules, plan):
    """Add each rule's largest comparison p-value and a status saying why it may be missing."""
    result = rules.copy()
    by_rule = {pos: group for pos, group in plan.groupby("rule_pos")}
    p_values, statuses = [], []
    for pos, row in enumerate(rules.itertuples(index=False)):
        comparisons = by_rule.get(pos)
        p_value = np.nan
        if not is_one_sided_complex(row.antecedents, row.consequents):
            status = "not_applicable"
        elif comparisons is None:
            status = "no_simpler"
        elif comparisons["p_value"].isna().any():
            status = "untestable"
        else:
            p_value = comparisons["p_value"].max()
            status = "tested"
        p_values.append(p_value)
        statuses.append(status)
    result["conditional_p_value"] = pd.Series(p_values, index=result.index, dtype=float)
    result["conditional_status"] = pd.Series(statuses, index=result.index, dtype=object)
    return result
