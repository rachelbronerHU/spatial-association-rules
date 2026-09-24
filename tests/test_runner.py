"""Check that multi-sample runs forward and record classification settings."""

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from spatial_association_rules import Method, Settings, Weighting
from spatial_association_rules import runner


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
    result = SimpleNamespace(rules=rules, stats={"patches_kept": 100},
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
