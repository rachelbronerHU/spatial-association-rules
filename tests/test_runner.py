"""Check that multi-sample runs forward and record their settings."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from spatial_association_rules import Method, Settings, Weighting
from spatial_association_rules import runner


def test_runner_exposes_raw_rules(monkeypatch):
    rules = pd.DataFrame([{"antecedents": ("A_CENTER",),
                           "consequents": ("B_NEIGHBOR",), "kind": "attracts",
                           "lift": 2.0}])
    raw_rules = rules.copy()
    result = SimpleNamespace(rules=rules, raw_rules=raw_rules,
                             stats={"patches_kept": 1}, add_p_values=lambda **kwargs: rules)
    monkeypatch.setattr(runner, "mine", lambda *args, **kwargs: result)
    settings = Settings(weighting=Weighting.BINARY, method=Method.CN, radius=1,
                        min_support=0.1, min_lift=1.2, max_items_per_rule=2,
                        include_avoidance_rules=False)
    samples = [("sample", [], [])]
    report = runner.run_samples(samples, settings, n_shuffles=0,
                                calculate_individual_fdr=False)
    assert report.raw_rules().sample_id.tolist() == ["sample"]
    assert report.raw_rules().lift.tolist() == [2.0]


def test_runner_returns_tested_rules_and_records_steps(monkeypatch, tmp_path):
    rules = pd.DataFrame([(("A_CENTER",), ("B_NEIGHBOR",), "attracts", 2.0)],
                         columns=["antecedents", "consequents", "kind", "lift"])
    result = SimpleNamespace(rules=rules, raw_rules=pd.DataFrame(), stats={"patches_kept": 100},
                             add_p_values=lambda **kwargs: rules.assign(regular_p_value=0.1))
    monkeypatch.setattr(runner, "mine", lambda *args, **kwargs: result)
    settings = Settings(weighting=Weighting.BINARY, method=Method.CN, radius=1,
                        min_support=0.1, min_lift=1.2, max_items_per_rule=4,
                        include_avoidance_rules=False, one_sided_complex_rules=False)
    report = runner.run_samples([("sample", [], [])], settings, n_shuffles=0, output_path=tmp_path)
    assert not report.failures
    assert report.rules().regular_p_value.tolist() == [0.1]
    assert not {"rule_type", "complex_class", "adds_information"} & set(report.rules())
    config = json.loads((tmp_path / "run_config.json").read_text())
    assert "min_lift_gain" not in config["steps"]
    assert config["settings"]["one_sided_complex_rules"] is False


def test_runner_can_add_conditional_tests_and_records_the_budget(monkeypatch, tmp_path):
    rules = pd.DataFrame([
        (("A_CENTER",), ("C_NEIGHBOR",), "attracts", 2.0, 0.01),
        (("A_CENTER", "B_NEIGHBOR"), ("C_NEIGHBOR",), "attracts", 3.0, 0.01),
    ], columns=["antecedents", "consequents", "kind", "lift", "individual_fdr"])
    seen = {}

    def conditional(tested, **options):
        seen.update(options)
        assert tested is rules
        return (tested.assign(conditional_p_value=[float("nan"), 0.02],
                              conditional_individual_fdr=[float("nan"), 0.12]),
                pd.DataFrame([{"rule_idx": 1, "simpler_rule": "A -> C", "p_value": 0.02}]))

    result = SimpleNamespace(rules=rules, raw_rules=pd.DataFrame(), stats={"patches_kept": 100},
                             add_p_values=lambda **kwargs: rules,
                             add_conditional_p_values=conditional)
    monkeypatch.setattr(runner, "mine", lambda *args, **kwargs: result)
    settings = Settings(weighting=Weighting.BINARY, method=Method.CN, radius=1,
                        min_support=0.1, min_lift=1.2, max_items_per_rule=3,
                        include_avoidance_rules=False)
    report = runner.run_samples([("sample", [], [])], settings, n_shuffles=np.int64(100),
                                n_conditional_shuffles=np.int64(29), random_seed=7,
                                max_individual_fdr=0.05, labels_kept_fixed=("D",),
                                output_path=tmp_path)
    assert report.rules().conditional_p_value.iloc[1] == 0.02
    assert report.rules().conditional_individual_fdr.iloc[1] == 0.12
    assert "conditional_tests" not in report.rules()
    assert report.results[0].comparisons.to_dict("records") == [
        {"rule_idx": 1, "simpler_rule": "A -> C", "p_value": 0.02}
    ]
    assert report.comparisons().to_dict("records") == [
        {"rule_idx": 1, "simpler_rule": "A -> C", "p_value": 0.02, "sample_id": "sample"}
    ]
    assert seen == dict(n_shuffles=29, random_seed=runner.seed_for(7, "sample"),
                        labels_kept_fixed=("D",), sample_id="sample",
                        metric=None, calculate_fdr=True, max_conditional_individual_fdr=None,
                        permutation_dir=None)
    config = json.loads((tmp_path / "run_config.json").read_text())
    assert config["steps"]["n_shuffles"] == 100
    assert isinstance(config["steps"]["n_shuffles"], int)
    assert config["steps"]["n_conditional_shuffles"] == 29
    assert isinstance(config["steps"]["n_conditional_shuffles"], int)


@pytest.mark.parametrize("options", [
    dict(n_conditional_shuffles=0), dict(n_conditional_shuffles=True),
    dict(n_conditional_shuffles=2.5), dict(conditional_metric="confidence"),
    dict(max_conditional_individual_fdr=0), dict(max_conditional_individual_fdr=1.1),
    dict(max_conditional_individual_fdr=float("nan")),
    dict(max_conditional_individual_fdr=float("inf")),
    dict(max_conditional_individual_fdr=0.05, calculate_conditional_individual_fdr=False),
])
def test_invalid_conditional_settings_fail_before_starting_samples(options):
    with pytest.raises(ValueError, match="conditional|metric"):
        runner.run_samples([], None, n_shuffles=9, **options)


@pytest.mark.parametrize("cutoff", [0, -0.1, 1.1, float("nan"), float("inf")])
def test_invalid_individual_cutoff_fails_before_starting_samples(cutoff):
    with pytest.raises(ValueError, match="max_individual_fdr"):
        runner.run_samples([], None, n_shuffles=9, max_individual_fdr=cutoff)


def test_runner_passes_independent_fdr_choices(monkeypatch, tmp_path):
    rules = pd.DataFrame([(("A_CENTER",), ("B_NEIGHBOR",), "attracts", 2.0, 0.01),
                          (("A_CENTER", "C_NEIGHBOR"), ("B_NEIGHBOR",), "attracts", 3.0, 0.01)],
                         columns=["antecedents", "consequents", "kind", "lift", "individual_fdr"])
    seen = {}

    def original(**options):
        seen["original"] = options["calculate_fdr"]
        return rules

    def conditional(tested, **options):
        seen["conditional"] = options["calculate_fdr"]
        return (tested.assign(conditional_p_value=[float("nan"), 0.2]),
                pd.DataFrame([{"rule_idx": 1, "simpler_rule": "A -> C", "p_value": 0.2}]))

    result = SimpleNamespace(rules=rules, raw_rules=pd.DataFrame(), stats={"patches_kept": 1},
                             add_p_values=original, add_conditional_p_values=conditional)
    monkeypatch.setattr(runner, "mine", lambda *args, **kwargs: result)
    settings = Settings(weighting=Weighting.BINARY, method=Method.CN, radius=1,
                        min_support=0.1, min_lift=1.2, max_items_per_rule=3,
                        include_avoidance_rules=False)
    report = runner.run_samples([("sample", [], [])], settings, n_shuffles=9,
                                n_conditional_shuffles=9, max_individual_fdr=0.05,
                                calculate_conditional_individual_fdr=False, output_path=tmp_path)
    assert seen == {"original": True, "conditional": False}
    assert "conditional_individual_fdr" not in report.rules()
    config = json.loads((tmp_path / "run_config.json").read_text())
    assert config["steps"]["calculate_individual_fdr"] is True
    assert config["steps"]["calculate_conditional_individual_fdr"] is False


@pytest.mark.parametrize("options", [
    dict(calculate_individual_fdr=False, max_individual_fdr=0.05),
    dict(calculate_individual_fdr=False, n_conditional_shuffles=9, max_individual_fdr=0.05),
])
def test_runner_rejects_a_cutoff_without_individual_fdr(options):
    with pytest.raises(ValueError, match="max_individual_fdr requires calculate_individual_fdr"):
        runner.run_samples([], None, n_shuffles=9, **options)


def test_runner_can_return_raw_p_values_without_fdr(monkeypatch):
    rules = pd.DataFrame([(("A_CENTER",), ("B_NEIGHBOR",), "attracts", 2.0)],
                         columns=["antecedents", "consequents", "kind", "lift"])

    def original(**options):
        assert options["calculate_fdr"] is False
        return rules.assign(regular_p_value=0.2)

    result = SimpleNamespace(rules=rules, raw_rules=pd.DataFrame(),
                             stats={"patches_kept": 1}, add_p_values=original)
    monkeypatch.setattr(runner, "mine", lambda *args, **kwargs: result)
    settings = Settings(weighting=Weighting.BINARY, method=Method.CN, radius=1,
                        min_support=0.1, min_lift=1.2, max_items_per_rule=3,
                        include_avoidance_rules=False)
    report = runner.run_samples([("sample", [], [])], settings, n_shuffles=9,
                                calculate_individual_fdr=False)
    output = report.rules()
    assert output.regular_p_value.tolist() == [0.2]
    assert "individual_fdr" not in output
    assert report.results[0].comparisons.empty
