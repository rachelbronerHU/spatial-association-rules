"""Saved shuffle metrics checked against direct patch arithmetic and unchanged tests."""
from contextlib import nullcontext
from importlib import import_module
import json
from mmap import mmap
from urllib.parse import quote

import numpy as np
import pandas as pd
import pytest

from spatial_association_rules import Weighting, run_samples
from spatial_association_rules.transactions import build_transactions
from spatial_association_rules.validation.permutations import record_permutations
from spatial_association_rules.validation.significance import seed_for
from test_conditional import direct_pass, sample
from test_mining import base, grid_tissue


def measured(result, rules, labels):
    """Independent scalar support calculation over actual patch dictionaries."""
    transactions, _ = build_transactions(result.patches, labels, result.settings)
    values = []
    for row in rules.itertuples():
        def support(items):
            return (sum(min(t.get(item, 0) for item in items) for t in transactions)
                    / len(transactions)) if transactions else 0
        ant, con = support(row.antecedents), support(row.consequents)
        if not ant or not con:
            values.append((np.nan, np.nan))
            continue
        confidence = support(row.antecedents + row.consequents) / ant
        conviction = (np.nan if con >= 1 else np.inf if confidence >= 1
                      else (1 - con) / (1 - confidence))
        values.append((confidence / con, conviction))
    return np.array(values)


@pytest.mark.parametrize("weighting", list(Weighting))
def test_individual_recording_matches_arithmetic_and_reconstructs_p_values(tmp_path, weighting):
    result, rules = sample(weighting)
    rules["rule_idx"] = [23, 7]
    options = dict(n_shuffles=9, random_seed=7, labels_kept_fixed=("C",), calculate_fdr=False)
    expected = result.add_p_values(**options)
    path = tmp_path / "permutations/sample_FOV/individual.npz"
    recorded = result.add_p_values(**options, permutation_dir=tmp_path / "permutations", sample_id="FOV")
    pd.testing.assert_frame_equal(recorded, expected)
    labels = result.labels.copy()
    movable = np.flatnonzero(labels != "C")
    rng = np.random.default_rng(7)
    direct = []
    for _ in range(9):
        shuffled = labels.copy()
        shuffled[movable] = labels[rng.permutation(movable)]
        direct.append(measured(result, rules, shuffled))
    direct = np.array(direct)
    with np.load(path, allow_pickle=False) as data:
        assert data["rule_idx"].tolist() == [23, 7]
        assert data["fixed_types"].tolist() == ["C"]
        assert data["sample_id"].item() == "FOV"
        assert data["test"].item() == "individual"
        np.testing.assert_allclose(data["lift"], direct[:, :, 0], equal_nan=True)
        np.testing.assert_allclose(data["conviction"], direct[:, :, 1], equal_nan=True)
        assert data["passed_mining_filters"].dtype == bool
        reconstructed = (data["passed_mining_filters"].sum(axis=0) + 1) / 10
        np.testing.assert_array_equal(recorded.p_value, reconstructed)
        assert "gain" not in data and "simpler_idx" not in data
    assert list(path.parent.iterdir()) == [path]


@pytest.mark.parametrize("weighting", list(Weighting))
@pytest.mark.parametrize("consequent", [False, True])
@pytest.mark.parametrize("kind", ["attracts", "avoids"])
def test_conditional_batches_keep_metrics_and_rule_ids_without_changing_results(
        tmp_path, monkeypatch, weighting, consequent, kind):
    result, rules = sample(weighting, consequent=consequent, two_parents=not consequent)
    rules["kind"] = kind
    rules["rule_idx"] = np.arange(len(rules)) * 11 + 7
    conditional = import_module("spatial_association_rules.validation.conditional")
    original = conditional._rule_columns
    layouts = []

    def capture(rows, item_index):
        layouts.append(rows.rule_idx.tolist())
        return original(rows, item_index)

    monkeypatch.setattr(conditional, "_rule_columns", capture)
    options = dict(n_shuffles=19, random_seed=7, max_individual_fdr=.05)
    expected, expected_comparisons = result.add_conditional_p_values(rules, **options)
    batches = expected_comparisons[expected_comparisons.status == "tested"].fixed_types.nunique()
    assert layouts[0] == rules.rule_idx.tolist()  # Observed gain metadata still uses both sides.
    assert len(layouts) == batches + 1
    assert all(len(ids) == 2 for ids in layouts[1:])  # Gain needs both sides in every batch.
    layouts.clear()
    actual, comparisons = result.add_conditional_p_values(
        rules, **options, permutation_dir=tmp_path / "permutations", sample_id="FOV")
    pd.testing.assert_frame_equal(actual, expected)
    pd.testing.assert_frame_equal(comparisons, expected_comparisons)
    assert all(len(ids) == 2 for ids in layouts[1:])  # Recorded batches retain parent metrics.
    tested = comparisons[comparisons.status == "tested"]
    files = sorted((tmp_path / "permutations/sample_FOV").glob("conditional_*.npz"))
    assert len(files) == tested.fixed_types.nunique()
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            fixed = tuple(data["fixed_types"].tolist())
            group = tested[tested.fixed_types.map(lambda value: value == fixed)]
            assert not group.empty
            ids = data["rule_idx"].tolist()
            assert set(ids) == set(group.rule_idx) | set(group.simpler_idx)
            assert data["passed_mining_filters"].dtype == bool
            assert "gain" not in data and "simpler_idx" not in data
            rows = rules.set_index("rule_idx").loc[ids]
            rng = np.random.default_rng(seed_for(7, json.dumps(fixed)))
            movable = np.flatnonzero(~np.isin(result.labels, fixed))
            direct = []
            direct_passes = []
            for _ in range(19):
                labels = result.labels.copy()
                labels[movable] = result.labels[rng.permutation(movable)]
                direct.append(measured(result, rows, labels))
                direct_passes.append([direct_pass(result, row, labels) for row in rows.itertuples()])
            direct = np.array(direct)
            np.testing.assert_allclose(data["lift"], direct[:, :, 0], equal_nan=True)
            np.testing.assert_allclose(data["conviction"], direct[:, :, 1], equal_nan=True)
            np.testing.assert_array_equal(data["passed_mining_filters"], direct_passes)
            for row in group.itertuples():
                passed = data["passed_mining_filters"][:, ids.index(row.rule_idx)]
                assert (passed.sum() + 1) / 20 == row.p_value


@pytest.mark.parametrize("workers", [None, 2])
def test_runner_recording_is_optional_and_keeps_samples_separate(tmp_path, workers):
    coords, labels = grid_tissue()
    settings = base(max_items_per_rule=3)
    samples = [("FOV/1", coords, labels), ("FOV\\1", coords, labels)]
    options = dict(n_shuffles=9, random_seed=7, workers=workers)
    expected = run_samples(samples, settings, output_path=tmp_path / "off", **options)
    assert not (tmp_path / "off" / "permutations").exists()
    actual = run_samples(samples, settings, output_path=tmp_path / "on",
                         save_permutation_metrics=True, **options)
    pd.testing.assert_frame_equal(actual.rules(), expected.rules())
    config = json.loads((tmp_path / "on" / "run_config.json").read_text())
    assert config["steps"]["save_permutation_metrics"] is True
    for result in actual.results:
        path = tmp_path / "on" / "permutations" / ("sample_" + quote(result.sample_id, safe=""))
        with np.load(path / "individual.npz", allow_pickle=False) as data:
            assert data["sample_id"].item() == result.sample_id
            np.testing.assert_array_equal(data["rule_idx"], result.rules.rule_idx)
            np.testing.assert_array_equal((data["passed_mining_filters"].sum(axis=0) + 1) / 10,
                                          result.rules.p_value)


def test_empty_rules_and_no_shuffles_do_not_create_archives(tmp_path):
    result, rules = sample()
    result.rules = rules.iloc[:0]
    result.add_p_values(9, permutation_dir=tmp_path)
    result.rules = rules
    result.add_p_values(0, permutation_dir=tmp_path)
    result.add_conditional_p_values(rules.assign(individual_fdr=1), n_shuffles=9,
                                    permutation_dir=tmp_path)
    assert not list(tmp_path.iterdir())


def test_recording_preserves_special_values_and_saves_nothing_on_failure(tmp_path):
    path = tmp_path / "permutations/sample_FOV/conditional_0001.npz"
    with record_permutations(tmp_path / "permutations", 1, [2, 9], sample_id="FOV", test="conditional", batch=1) as arrays:
        arrays["lift"][:] = [[np.nan, np.inf]]
        arrays["conviction"][:] = [[np.inf, 0]]
    with np.load(path, allow_pickle=False) as data:
        np.testing.assert_array_equal(data["lift"], [[np.nan, np.inf]])
        np.testing.assert_array_equal(data["conviction"], [[np.inf, 0]])
    with pytest.raises(RuntimeError, match="failed batch"):
        with record_permutations(tmp_path / "permutations", 1, [2], sample_id="FOV", test="conditional", batch=2):
            raise RuntimeError("failed batch")
    assert list(path.parent.iterdir()) == [path]


@pytest.mark.parametrize("options", [dict(save_permutation_metrics=True),
                                    dict(save_permutation_metrics="yes", output_path="unused")])
def test_runner_validates_recording_before_mining(options):
    with pytest.raises(ValueError, match="save_permutation_metrics"):
        run_samples([], None, n_shuffles=9, **options)


def test_recording_refuses_to_overwrite_an_archive(tmp_path):
    with record_permutations(tmp_path, 1, [2], sample_id="FOV", test="individual"):
        pass
    with pytest.raises(FileExistsError, match="individual.npz"):
        with record_permutations(tmp_path, 1, [2], sample_id="FOV", test="individual"):
            pass


def test_runner_refuses_a_non_empty_permutation_folder(tmp_path):
    (tmp_path / "permutations" / "sample_old").mkdir(parents=True)
    coords, labels = grid_tissue()
    with pytest.raises(ValueError, match="not empty"):
        run_samples([("FOV", coords, labels)], base(), n_shuffles=9,
                    output_path=tmp_path, save_permutation_metrics=True)
    assert not (tmp_path / "run_config.json").exists()


def test_runner_refuses_sample_ids_sharing_a_folder(tmp_path):
    coords, labels = grid_tissue()
    samples = [("FOV1", coords, labels), ("fov1", coords, labels)]
    with pytest.raises(ValueError, match="share a permutation folder"):
        run_samples(samples, base(), n_shuffles=9, output_path=tmp_path, save_permutation_metrics=True)


def test_recording_refuses_an_archive_created_during_the_batch(tmp_path):
    with pytest.raises(FileExistsError, match="individual.npz"):
        with record_permutations(tmp_path, 1, [2], sample_id="FOV", test="individual") as first:
            first["lift"][:] = 1
            with record_permutations(tmp_path, 1, [2], sample_id="FOV", test="individual") as second:
                second["lift"][:] = 2
    path = tmp_path / "sample_FOV/individual.npz"
    with np.load(path, allow_pickle=False) as data:
        np.testing.assert_array_equal(data["lift"], [[2]])
    assert list(path.parent.iterdir()) == [path]


@pytest.mark.parametrize("test", ["individual", "conditional"])
@pytest.mark.parametrize("failure", [False, True])
def test_recording_uses_disk_buffers_and_closes_them(tmp_path, test, failure):
    expected = pytest.raises(RuntimeError, match="failed batch") if failure else nullcontext()
    with expected:
        with record_permutations(tmp_path, 2, [2, 9], sample_id="FOV", test=test, batch=1) as arrays:
            buffers = [array.base for array in arrays.values()]
            assert all(isinstance(buffer, mmap) and not buffer.closed for buffer in buffers)
            assert not (tmp_path / "sample_FOV").exists()
            for array in arrays.values():
                array[:] = 0
            if failure:
                raise RuntimeError("failed batch")
    assert all(buffer.closed for buffer in buffers)
    files = list(tmp_path.rglob("*.npz"))
    assert len(files) == (0 if failure else 1)
    if failure:
        assert not (tmp_path / "sample_FOV").exists()


def test_individual_recording_reuses_owned_row_buffers(tmp_path, monkeypatch):
    result, rules = sample()
    module = import_module("spatial_association_rules.validation.significance")
    original = module.survives_shuffle
    rows = []

    def measure(*args, metrics_out=None):
        assert all(row.flags.owndata and row.base is None for row in metrics_out)
        rows.append(metrics_out)
        return original(*args, metrics_out=metrics_out)

    monkeypatch.setattr(module, "survives_shuffle", measure)
    result.add_p_values(9, random_seed=7, permutation_dir=tmp_path, sample_id="FOV")
    assert len(rows) == 9 and all(row is rows[0] for row in rows)
    # The row buffers remain safe to read after the mapped files are closed.
    with np.load(tmp_path / "sample_FOV/individual.npz", allow_pickle=False) as data:
        np.testing.assert_array_equal(rows[0][0], data["lift"][-1])
        np.testing.assert_array_equal(rows[0][1], data["conviction"][-1])


def test_failed_save_removes_only_its_partial_archive_and_allows_retry(tmp_path, monkeypatch):
    module = import_module("spatial_association_rules.validation.permutations")
    files = []

    def fail(archive, **data):
        files.append(archive)
        archive.write(b"incomplete archive")
        raise OSError("save failed")

    with monkeypatch.context() as patch:
        patch.setattr(module.np, "savez_compressed", fail)
        with pytest.raises(OSError, match="save failed"):
            with record_permutations(tmp_path, 1, [2], sample_id="FOV", test="individual") as arrays:
                arrays["lift"][:] = 2
    assert files[0].closed
    assert not (tmp_path / "sample_FOV/individual.npz").exists()
    with record_permutations(tmp_path, 1, [2], sample_id="FOV", test="individual") as arrays:
        arrays["lift"][:] = 3
    with np.load(tmp_path / "sample_FOV/individual.npz", allow_pickle=False) as data:
        np.testing.assert_array_equal(data["lift"], [[3]])
