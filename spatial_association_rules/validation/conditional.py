"""Exploratory shuffle tests with each simpler rule's cell types held fixed."""

import json
import logging

import numpy as np
import pandas as pd

from ..complex_rules import _name, _rule_type, simpler_positions
from ..rules import metrics
from ..transactions import strip_role
from .false_discovery import fdr_by_size, fdr_families
from .significance import (
    _adjacency, _encode_labels, _held_fixed, _rule_columns, _supports, _transactions, seed_for,
)

logger = logging.getLogger(__name__)

PLAN_COLUMNS = ["rule_pos", "simpler_pos", "rule", "simpler_rule", "kind", "metric",
                "fixed_types", "movable_cells", "status"]


def conditional_test_plan(rules, labels, max_individual_fdr, labels_kept_fixed=()):
    """List comparisons and distinct fixed-type sets without running any shuffles."""
    if max_individual_fdr is None or not 0 < max_individual_fdr <= 1:
        raise ValueError("conditional tests require max_individual_fdr in (0, 1]")
    required = {"antecedents", "consequents", "kind", "individual_fdr"}
    if not required.issubset(rules.columns):
        raise ValueError("pass rules from add_p_values(), including individual_fdr")
    if not rules["kind"].isin(["attracts", "avoids"]).all():
        raise ValueError("rule kind must be attracts or avoids")

    labels = np.asarray(labels, dtype=str)
    patterns = tuple(rules.attrs.get("labels_kept_fixed", ())) + tuple(labels_kept_fixed)
    configured = set(labels[_held_fixed(labels, patterns)])
    available = set(labels)
    rows = list(rules.itertuples(index=False))
    types = [set(map(strip_role, tuple(row.antecedents) + tuple(row.consequents))) for row in rows]
    significant = rules["individual_fdr"].le(max_individual_fdr).fillna(False).to_numpy()
    masks = {}
    comparisons = []
    for pos, parents in enumerate(simpler_positions(rules)):
        for parent in parents:
            if not significant[parent]:
                continue
            fixed = tuple(sorted(configured | types[parent]))
            if fixed not in masks:
                movable = ~np.isin(labels, fixed)
                masks[fixed] = (int(movable.sum()), len(set(labels[movable])) >= 2)
            movable_cells, can_shuffle = masks[fixed]
            status = "ready"
            if not types[pos].issubset(available):
                status = "missing_cell_type"
            elif types[pos].issubset(fixed):
                status = "all_rule_types_fixed"
            elif not can_shuffle:
                status = "no_exchangeable_labels"
            row, simpler = rows[pos], rows[parent]
            comparisons.append(dict(
                rule_pos=pos, simpler_pos=parent,
                rule=_name(row.antecedents, row.consequents),
                simpler_rule=_name(simpler.antecedents, simpler.consequents),
                kind=row.kind, metric="lift" if len(row.consequents) == 1 else "conviction",
                fixed_types=fixed, movable_cells=movable_cells, status=status,
            ))
    plan = pd.DataFrame(comparisons, columns=PLAN_COLUMNS)
    plan.attrs["shuffle_batches"] = plan.loc[plan.status == "ready", "fixed_types"].nunique()
    return plan


def conditional_p_values(rules, patches, labels, settings, *, n_shuffles,
                         max_individual_fdr, random_seed=None, labels_kept_fixed=(), sample_id="",
                         calculate_fdr=True):
    """Add comparison p-values, their maximum per rule, and optional FDR.

    These test a restricted random-label null, not Webb's conditional independence.
    Parents are selected from the same data, so FDR is not proven controlled.
    Classification is unchanged. See DESIGN.md.
    """
    if not isinstance(calculate_fdr, bool):
        raise ValueError("calculate_fdr must be a bool")
    check_shuffle_count(n_shuffles, "n_shuffles")
    labels = np.asarray(labels, dtype=str)
    plan = conditional_test_plan(rules, labels, max_individual_fdr, labels_kept_fixed)
    plan["p_value"] = np.nan
    plan["observed_gain"] = np.nan
    batches = 0
    ready = plan.status == "ready"
    if ready.any():
        cell_labels, item_index = _encode_labels(labels)
        adjacency = _adjacency(patches, len(labels))
        observed = _measure(_rule_columns(rules, item_index),
                            _transactions(cell_labels, adjacency, settings), settings)
        observed_scores = _scores(observed, plan, np.arange(len(rules)))
        plan.loc[ready, "observed_gain"] = observed_scores[ready]
        plan.loc[ready & np.isnan(observed_scores), "status"] = "undefined_metric"
        no_gain = ready & (observed_scores <= 1)
        plan.loc[no_gain, "status"] = "no_improvement"
        plan.loc[no_gain, "p_value"] = 1.0
        groups = plan[plan.status == "ready"].groupby("fixed_types", sort=True)
        batches = groups.ngroups
        prefix = f"[{sample_id}] " if sample_id else ""
        logger.info(f"{prefix}Conditional tests: {len(plan)} comparisons, "
                    f"{batches} shuffle batches, {n_shuffles} shuffles per batch")

        for batch, (fixed, group) in enumerate(groups, 1):
            movable = np.flatnonzero(~np.isin(labels, fixed))
            positions = np.union1d(group.rule_pos, group.simpler_pos).astype(int)
            layout = _rule_columns(rules.iloc[positions], item_index)
            rng = np.random.default_rng(seed_for(random_seed, json.dumps(fixed)))
            exceeded = np.zeros(len(group), dtype=int)
            scores = observed_scores[group.index]
            for _ in range(n_shuffles):
                order = np.arange(len(labels))
                order[movable] = rng.permutation(movable)
                transactions = _transactions(cell_labels[order], adjacency, settings)
                shuffled = _scores(_measure(layout, transactions, settings), group, positions)
                # An absent side cannot show improvement. Never discard a permutation.
                shuffled = np.where(np.isnan(shuffled), 0.0, shuffled)
                exceeded += (shuffled >= scores) | np.isclose(shuffled, scores, rtol=1e-12, atol=0)
            plan.loc[group.index, "p_value"] = (exceeded + 1) / (n_shuffles + 1)
            plan.loc[group.index, "status"] = "tested"
            logger.debug(f"{prefix}Conditional batch {batch}/{batches}: fixed {fixed}")

    result = _attach_results(rules, plan)
    if calculate_fdr:
        result["conditional_fdr"] = _conditional_fdr(result, labels, settings, n_shuffles=n_shuffles,
                                                     max_fdr=max_individual_fdr, sample_id=sample_id)
    else:
        result.drop(columns="conditional_fdr", errors="ignore", inplace=True)
    result.attrs["conditional_test_summary"] = dict(
        comparisons=len(plan), shuffle_batches=batches, n_shuffles=n_shuffles,
        shuffled_tissues=batches * n_shuffles,
    )
    return result


def check_shuffle_count(value, name):
    """Shared by the runner, so bad input fails before any sample is mined."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _conditional_fdr(rules, labels, settings, *, n_shuffles, max_fdr, sample_id=""):
    """Correct each size over all candidates, as individual_fdr does; untested rows stay NaN.

    Mixed rules are candidates too, although never tested, so they count as p=1.
    """
    p_values = rules["conditional_p_value"]
    families = fdr_families(rules[p_values.notna()], labels, settings, n_shuffles=n_shuffles,
                            max_fdr=max_fdr, column="conditional_fdr", sample_id=sample_id)
    return fdr_by_size(rules, p_values, families)


def _measure(layout, transactions, settings):
    """Lift and conviction; undefined metrics stay missing."""
    values = np.full((len(layout.usable), 2), np.nan)
    if not len(transactions):
        return values
    ant, con, joint = _supports(layout, transactions, settings)
    _, lift, _, conviction = metrics(joint, ant, con)
    usable = layout.usable & (ant > 0) & (con > 0)
    values[usable, 0] = lift[usable]
    # A universal consequent gives conviction 0/0, not evidence of certainty.
    usable_con = usable & (con < 1)
    values[usable_con, 1] = conviction[usable_con]
    return values


def _scores(values, comparisons, positions):
    """Gain ratios in the complex rule's direction; equal zeros or infinities tie."""
    child = np.searchsorted(positions, comparisons.rule_pos.to_numpy(dtype=int))
    parent = np.searchsorted(positions, comparisons.simpler_pos.to_numpy(dtype=int))
    metric = (comparisons.metric == "conviction").to_numpy(dtype=int)
    complex_value, simpler_value = values[child, metric], values[parent, metric]
    avoids = (comparisons.kind == "avoids").to_numpy()
    numerator = np.where(avoids, simpler_value, complex_value)
    denominator = np.where(avoids, complex_value, simpler_value)
    with np.errstate(divide="ignore", invalid="ignore"):
        scores = numerator / denominator
    return np.where(numerator == denominator, 1.0, scores)


def _attach_results(rules, plan):
    result = rules.copy()
    details = [[] for _ in range(len(rules))]
    for comparison in plan.to_dict("records"):
        details[comparison["rule_pos"]].append(comparison)
    p_values, statuses = [], []
    for pos, row in enumerate(rules.itertuples(index=False)):
        comparisons = details[pos]
        p_value = np.nan
        if _rule_type(len(row.antecedents), len(row.consequents)) in ("pairwise", "complex-mixed"):
            status = "not_applicable"
        elif not comparisons:
            status = "no_significant_simpler"
        elif any(pd.isna(test["p_value"]) for test in comparisons):
            status = "untestable"
        else:
            p_value = max(test["p_value"] for test in comparisons)
            status = "tested"
        p_values.append(p_value)
        statuses.append(status)
    result["conditional_p_value"] = pd.Series(p_values, index=result.index, dtype=float)
    result["conditional_status"] = pd.Series(statuses, index=result.index, dtype=object)
    result["conditional_tests"] = details
    return result
