"""Individual fixed cells stay fixed without freezing their entire merged type."""

from importlib import import_module

import numpy as np
import pytest

from spatial_association_rules import Weighting, run_samples
from test_conditional import sample


@pytest.mark.parametrize("weighting", list(Weighting))
def test_regular_shuffles_keep_only_selected_cells_fixed(monkeypatch, weighting):
    result, rules = sample(weighting)
    result.rules = rules
    fixed = np.arange(len(result.labels)) == 7  # Only one of the three D cells.
    module = import_module("spatial_association_rules.validation.significance")
    original = module._transactions
    names = np.array(sorted(set(result.labels)))
    shuffled = []

    def capture(encoded, adjacency, settings):
        shuffled.append(names[encoded.argmax(axis=1)])
        return original(encoded, adjacency, settings)

    monkeypatch.setattr(module, "_transactions", capture)
    result.add_p_values(19, random_seed=7, fixed_cells=fixed, labels_kept_fixed=("C",),
                        calculate_fdr=False)
    shuffled = np.array(shuffled)
    assert (shuffled[:, [4, 5, 7]] == result.labels[[4, 5, 7]]).all()
    assert (shuffled[:, [8, 9]] != "D").any()


@pytest.mark.parametrize("position,counts", [(0, [4, 6]), (7, [3, 6])])
def test_conditional_shuffles_inherit_individual_fixed_cells(monkeypatch, position, counts):
    result, rules = sample()
    result.rules = rules
    fixed = np.arange(len(result.labels)) == position
    tested = result.add_p_values(0, fixed_cells=fixed, calculate_fdr=False)
    plan = result.conditional_test_plan(tested)
    assert plan.movable_cells.tolist() == counts
    assert plan.status.tolist() == ["ready", "ready"]
    module = import_module("spatial_association_rules.validation.conditional")
    original = module._transactions
    names = np.array(sorted(set(result.labels)))
    shuffled = []

    def capture(encoded, adjacency, settings):
        shuffled.append(names[encoded.argmax(axis=1)])
        return original(encoded, adjacency, settings)

    monkeypatch.setattr(module, "_transactions", capture)
    _, comparisons = result.add_conditional_p_values(
        tested, n_shuffles=19, random_seed=7, calculate_fdr=False)
    shuffled = np.array(shuffled)
    assert (shuffled[:, position] == result.labels[position]).all()
    same_type = (result.labels == result.labels[position]) & ~fixed
    assert (shuffled[:, same_type] != result.labels[same_type]).any()
    assert comparisons.status.tolist() == ["tested", "tested"]


@pytest.mark.parametrize("mask", [[True], [1] * 10, [[False] * 10], []])
def test_invalid_fixed_masks_fail_before_mining_or_shuffling(mask):
    result, rules = sample()
    result.rules = rules
    with pytest.raises(ValueError, match="boolean mask"):
        result.add_p_values(0, fixed_cells=mask)
    with pytest.raises(ValueError, match="boolean mask"):
        run_samples([("FOV", [], result.labels)], result.settings,
                    n_shuffles=0, fixed_cells={"FOV": mask})


def test_unknown_fixed_sample_is_rejected():
    result, _ = sample()
    with pytest.raises(ValueError, match="sample IDs"):
        run_samples([("FOV", [], result.labels)], result.settings,
                    n_shuffles=0, fixed_cells={"other": np.zeros(len(result.labels), dtype=bool)})
