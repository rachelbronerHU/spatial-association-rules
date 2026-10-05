"""Check that multi-sample runs forward and record classification settings."""

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


@pytest.mark.parametrize("gain,expected", [
    ("default", True), (None, True), (0, True), (1.1, True), (1.5, False),
])
def test_runner_uses_and_records_consequent_gain(monkeypatch, tmp_path, gain, expected):
    rules = pd.DataFrame([
        (("A_CENTER",), ("B_NEIGHBOR",), 2.0),
        (("A_CENTER",), ("B_NEIGHBOR", "C_NEIGHBOR"), 2.4),
        (("A_CENTER", "D_NEIGHBOR"), ("B_NEIGHBOR", "C_NEIGHBOR"), 3.0),
        (("A_CENTER", "D_NEIGHBOR"), ("B_NEIGHBOR",), 3.0),
        (("A_CENTER",), ("B_NEIGHBOR", "E_NEIGHBOR"), 2.1),
    ], columns=["antecedents", "consequents", "conviction"])
    rules["kind"] = "attracts"
    rules["lift"] = [2.0, 2.0, 2.0, 2.1, 2.0]
    result = SimpleNamespace(rules=rules, raw_rules=pd.DataFrame(), stats={"patches_kept": 100},
                             add_p_values=lambda **kwargs: rules)
    monkeypatch.setattr(runner, "mine", lambda *args, **kwargs: result)
    settings = Settings(weighting=Weighting.BINARY, method=Method.CN, radius=1,
                        min_support=0.1, min_lift=1.2, max_items_per_rule=4,
                        include_avoidance_rules=False, one_sided_complex_rules=False)
    options = {} if gain == "default" else {"min_consequent_conviction_gain": gain}
    report = runner.run_samples([("sample", [], [])], settings, n_shuffles=0,
                                output_path=tmp_path, **options)
    assert not report.failures
    assert report.rules().adds_information.iloc[1] == expected
    assert pd.isna(report.rules().adds_information.iloc[2])
    assert not report.rules().adds_information.iloc[3]  # A 5% lift gain fails the default.
    assert report.rules().adds_information.iloc[4] == (gain in (None, 0))
    config = json.loads((tmp_path / "run_config.json").read_text())
    assert config["steps"]["min_consequent_conviction_gain"] == (1.1 if gain == "default" else gain)
    assert config["steps"]["min_lift_gain"] == 1.1
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
                              conditional_fdr=[float("nan"), 0.12]),
                pd.DataFrame([{"rule_idx": 1, "simpler_idx": 0, "p_value": 0.02}]))

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
    assert report.rules().conditional_fdr.iloc[1] == 0.12
    assert "conditional_tests" not in report.rules()
    assert report.results[0].comparisons.to_dict("records") == [
        {"rule_idx": 1, "simpler_idx": 0, "p_value": 0.02}
    ]
    assert report.comparisons().to_dict("records") == [
        {"rule_idx": 1, "simpler_idx": 0, "p_value": 0.02, "sample_id": "sample"}
    ]
    assert report.rules().complex_class.iloc[1] == "stronger_than_simpler"
    assert seen == dict(n_shuffles=29, random_seed=runner.seed_for(7, "sample"),
                        max_individual_fdr=0.05, labels_kept_fixed=("D",), sample_id="sample",
                        calculate_fdr=True)
    config = json.loads((tmp_path / "run_config.json").read_text())
    assert config["steps"]["n_shuffles"] == 100
    assert isinstance(config["steps"]["n_shuffles"], int)
    assert config["steps"]["n_conditional_shuffles"] == 29
    assert isinstance(config["steps"]["n_conditional_shuffles"], int)


@pytest.mark.parametrize("budget,cutoff", [(0, 0.05), (True, 0.05), (2.5, 0.05), (9, None)])
def test_invalid_conditional_settings_fail_before_starting_samples(budget, cutoff):
    with pytest.raises(ValueError, match="conditional"):
        runner.run_samples([], None, n_shuffles=9, n_conditional_shuffles=budget,
                           max_individual_fdr=cutoff)


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
                pd.DataFrame([{"rule_idx": 1, "simpler_idx": 0, "p_value": 0.2}]))

    result = SimpleNamespace(rules=rules, raw_rules=pd.DataFrame(), stats={"patches_kept": 1},
                             add_p_values=original, add_conditional_p_values=conditional)
    monkeypatch.setattr(runner, "mine", lambda *args, **kwargs: result)
    settings = Settings(weighting=Weighting.BINARY, method=Method.CN, radius=1,
                        min_support=0.1, min_lift=1.2, max_items_per_rule=3,
                        include_avoidance_rules=False)
    report = runner.run_samples([("sample", [], [])], settings, n_shuffles=9,
                                n_conditional_shuffles=9, max_individual_fdr=0.05,
                                calculate_conditional_fdr=False, output_path=tmp_path)
    assert seen == {"original": True, "conditional": False}
    assert "conditional_fdr" not in report.rules()
    config = json.loads((tmp_path / "run_config.json").read_text())
    assert config["steps"]["calculate_individual_fdr"] is True
    assert config["steps"]["calculate_conditional_fdr"] is False


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
        return rules.assign(p_value=0.2)

    result = SimpleNamespace(rules=rules, raw_rules=pd.DataFrame(),
                             stats={"patches_kept": 1}, add_p_values=original)
    monkeypatch.setattr(runner, "mine", lambda *args, **kwargs: result)
    settings = Settings(weighting=Weighting.BINARY, method=Method.CN, radius=1,
                        min_support=0.1, min_lift=1.2, max_items_per_rule=3,
                        include_avoidance_rules=False)
    report = runner.run_samples([("sample", [], [])], settings, n_shuffles=9,
                                calculate_individual_fdr=False)
    output = report.rules()
    assert output.p_value.tolist() == [0.2]
    assert "individual_fdr" not in output
    assert output.adds_information.tolist() == [True]
    assert report.results[0].comparisons.empty
