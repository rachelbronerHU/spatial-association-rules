"""
Run the same mining over many samples: mine everything, test everything, then filter.
See README, "run_samples".
"""

import json
import logging
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import List, Tuple

import numpy as np
import pandas as pd

from .mine import mine
from .complex_rules import DEFAULT_IMPROVEMENT_GAIN
from .rules import classify_rules
from .validation.conditional import check_shuffle_count
from .settings import Settings
from .validation.significance import seed_for

logger = logging.getLogger(__name__)


@dataclass
class SampleResult:
    sample_id: object
    rules: pd.DataFrame     # every rule, with raw p-values and classification columns
    stats: dict
    comparisons: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)
    raw_rules: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)


@dataclass
class RunReport:
    """Every sample that worked, and every one that did not."""

    results: List[SampleResult] = field(default_factory=list)
    failures: List[Tuple[object, str]] = field(default_factory=list)

    def rules(self) -> pd.DataFrame:
        """Every sample's rules in one frame, with a sample_id column."""
        frames = [r.rules.assign(sample_id=r.sample_id)
                  for r in self.results if not r.rules.empty]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def raw_rules(self) -> pd.DataFrame:
        """Rules before final filters, with sample_id."""
        frames = [r.raw_rules.assign(sample_id=r.sample_id)
                  for r in self.results if not r.raw_rules.empty]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def comparisons(self) -> pd.DataFrame:
        """Every sample's conditional comparisons, with sample_id."""
        frames = [r.comparisons.assign(sample_id=r.sample_id)
                  for r in self.results if not r.comparisons.empty]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


@dataclass(frozen=True)
class _SampleTask:
    sample_id: object
    coords: object
    labels: object
    settings: Settings
    n_shuffles: int
    seed: object
    kept_fixed: tuple
    min_lift_gain: object
    max_individual_fdr: object
    min_consequent_conviction_gain: object
    n_conditional_shuffles: object
    calculate_individual_fdr: bool
    calculate_conditional_fdr: bool


def run_samples(samples, settings: Settings, *, n_shuffles, random_seed=None,
                labels_kept_fixed=(), min_lift_gain=DEFAULT_IMPROVEMENT_GAIN, max_individual_fdr=None,
                min_consequent_conviction_gain=DEFAULT_IMPROVEMENT_GAIN,
                workers=None, output_path=None, n_conditional_shuffles=None,
                calculate_individual_fdr=True, calculate_conditional_fdr=True) -> RunReport:
    """
    Mine every sample and report what came back, with raw p-values and nothing cut.

    samples:      iterable of (sample_id, coords, labels)
    workers:      An integer runs that many processes — on Windows,
                  guard the caller with `if __name__ == "__main__"`.
                  `None` makes it to run here. 
    output_path:  where to write run_config.json. `None` writes nothing.
    n_conditional_shuffles: optional extra testing; requires max_individual_fdr.
    calculate_individual_fdr, calculate_conditional_fdr: whether to correct each test.

    A sample that raises lands in report.failures. If every sample fails, this raises.
    """
    if not isinstance(calculate_individual_fdr, bool) or not isinstance(calculate_conditional_fdr, bool):
        raise ValueError("calculate_individual_fdr and calculate_conditional_fdr must be bools")
    if not calculate_individual_fdr and max_individual_fdr is not None:
        raise ValueError("max_individual_fdr requires calculate_individual_fdr=True")
    if isinstance(n_shuffles, np.integer):
        n_shuffles = int(n_shuffles)
    if n_conditional_shuffles is not None:
        check_shuffle_count(n_conditional_shuffles, "n_conditional_shuffles")
        n_conditional_shuffles = int(n_conditional_shuffles)
        if max_individual_fdr is None or not 0 < max_individual_fdr <= 1:
            raise ValueError("conditional tests require max_individual_fdr in (0, 1]")
    setup_console_logging()
    tasks = [
        _SampleTask(
            sample_id=sample_id, coords=coords, labels=labels, settings=settings,
            n_shuffles=n_shuffles, seed=seed_for(random_seed, sample_id),
            kept_fixed=tuple(labels_kept_fixed), min_lift_gain=min_lift_gain,
            max_individual_fdr=max_individual_fdr,
            min_consequent_conviction_gain=min_consequent_conviction_gain,
            n_conditional_shuffles=n_conditional_shuffles,
            calculate_individual_fdr=calculate_individual_fdr,
            calculate_conditional_fdr=calculate_conditional_fdr,
        )
        for sample_id, coords, labels in samples
    ]
    if not tasks:
        raise ValueError("no samples were given")

    if output_path is not None:
        _save_config(output_path, settings, dict(
            n_shuffles=n_shuffles, random_seed=random_seed, labels_kept_fixed=list(labels_kept_fixed),
            min_lift_gain=min_lift_gain, max_individual_fdr=max_individual_fdr, workers=workers,
            min_consequent_conviction_gain=min_consequent_conviction_gain,
            n_conditional_shuffles=n_conditional_shuffles,
            calculate_individual_fdr=calculate_individual_fdr,
            calculate_conditional_fdr=calculate_conditional_fdr,
        ))

    logger.info(f"Mining {len(tasks)} samples" + (f" across {workers} processes" if workers else ""))
    if workers:
        with ProcessPoolExecutor(max_workers=workers, initializer=setup_console_logging) as pool:
            outcomes = list(pool.map(_run_one, tasks))
    else:
        outcomes = [_run_one(task) for task in tasks]

    report = RunReport()
    for result, failure in outcomes:
        if result is not None:
            report.results.append(result)
        else:
            report.failures.append(failure)

    if report.failures:
        logger.warning(f"{len(report.failures)} of {len(tasks)} samples failed:")
        for sample_id, message in report.failures:
            logger.warning(f"  {sample_id}: {message.splitlines()[-1]}")
    if not report.results:
        raise RuntimeError(f"every one of the {len(tasks)} samples failed. First: {report.failures[0][1]}")

    logger.info(f"Done. {len(report.results)} samples, {len(report.rules())} rules")
    return report


def _run_one(task):
    """One sample. Returns (result, None) or (None, (sample_id, traceback))."""
    try:
        result = mine(task.coords, task.labels, task.settings, sample_id=task.sample_id)
        tested = result.add_p_values(
            n_shuffles=task.n_shuffles, random_seed=task.seed,
            labels_kept_fixed=task.kept_fixed, sample_id=task.sample_id,
            max_individual_fdr=task.max_individual_fdr,
            calculate_fdr=task.calculate_individual_fdr,
        )
        comparisons = pd.DataFrame()
        if task.n_conditional_shuffles is not None:
            tested, comparisons = result.add_conditional_p_values(
                tested, n_shuffles=task.n_conditional_shuffles,
                max_individual_fdr=task.max_individual_fdr, random_seed=task.seed,
                labels_kept_fixed=task.kept_fixed, sample_id=task.sample_id,
                calculate_fdr=task.calculate_conditional_fdr,
            )
        classified = classify_rules(
            tested, min_lift_gain=task.min_lift_gain,
            max_individual_fdr=task.max_individual_fdr,
            min_consequent_conviction_gain=task.min_consequent_conviction_gain,
        )

        logger.info(f"[{task.sample_id}] {result.stats['patches_kept']} transactions, "
                    f"{len(result.rules)} mined, "
                    f"{int(classified['adds_information'].sum())} add information")
        return SampleResult(task.sample_id, classified, result.stats, comparisons,
                            result.raw_rules), None
    except Exception:
        return None, (task.sample_id, traceback.format_exc())


def setup_console_logging(level=logging.INFO):
    """Show progress on the console. Does nothing if logging is already configured."""
    package = logging.getLogger(__package__)
    if package.handlers or logging.getLogger().handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
    package.addHandler(handler)
    package.setLevel(level)


def _save_config(output_path, settings, steps):
    """Record what this run was asked to do. The only file the library writes."""
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, "run_config.json"), "w") as f:
        record = {"settings": asdict(settings), "steps": steps}
        # bandwidth may be empty in the settings; record what the run actually used.
        record["settings"]["decay_distance"] = settings.decay_distance
        json.dump(record, f, indent=2, default=str)
