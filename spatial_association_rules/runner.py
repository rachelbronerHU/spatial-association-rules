"""
Run the same mining over many samples: mine everything, then test everything.
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
from .validation.conditional import check_metric, check_shuffle_count
from .validation.false_discovery import check_fdr_cutoff
from .validation.permutations import sample_folder
from .settings import Settings
from .validation.significance import seed_for

logger = logging.getLogger(__name__)


@dataclass
class SampleResult:
    sample_id: object
    rules: pd.DataFrame     # every rule, with raw p-values
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
    max_individual_fdr: object
    n_conditional_shuffles: object
    conditional_metric: object
    max_conditional_individual_fdr: object
    calculate_individual_fdr: bool
    calculate_conditional_individual_fdr: bool
    permutation_dir: object = None


def run_samples(samples, settings: Settings, *, n_shuffles, random_seed=None,
                labels_kept_fixed=(), max_individual_fdr=None, workers=None, output_path=None,
                n_conditional_shuffles=None, conditional_metric=None,
                max_conditional_individual_fdr=None, calculate_individual_fdr=True,
                calculate_conditional_individual_fdr=True,
                save_permutation_metrics=False) -> RunReport:
    """
    Mine every sample and report what came back, with raw p-values and nothing cut.

    samples:      iterable of (sample_id, coords, labels)
    workers:      An integer runs that many processes — on Windows,
                  guard the caller with `if __name__ == "__main__"`.
                  `None` makes it to run here. 
    output_path:  where to write run_config.json. `None` writes nothing.
    max_individual_fdr, max_conditional_individual_fdr: warn when too few shuffles can
                  reach these FDRs.
    n_conditional_shuffles: shuffles for the conditional test of complex rules; None skips it.
    conditional_metric: "lift", "conviction", or None to choose by the rule's complex side.
    calculate_individual_fdr, calculate_conditional_individual_fdr: whether to correct
                  each test.
    save_permutation_metrics: save per-sample shuffle arrays; requires output_path.

    A sample that raises lands in report.failures. If every sample fails, this raises.
    """
    if not all(isinstance(value, bool)
               for value in (calculate_individual_fdr, calculate_conditional_individual_fdr)):
        raise ValueError("calculate_individual_fdr and calculate_conditional_individual_fdr "
                         "must be bools")
    check_fdr_cutoff(max_individual_fdr, "max_individual_fdr")
    check_fdr_cutoff(max_conditional_individual_fdr, "max_conditional_individual_fdr")
    if not calculate_individual_fdr and max_individual_fdr is not None:
        raise ValueError("max_individual_fdr requires calculate_individual_fdr=True")
    if not calculate_conditional_individual_fdr and max_conditional_individual_fdr is not None:
        raise ValueError("max_conditional_individual_fdr requires "
                         "calculate_conditional_individual_fdr=True")
    check_metric(conditional_metric)
    if not isinstance(save_permutation_metrics, bool):
        raise ValueError("save_permutation_metrics must be a bool")
    if save_permutation_metrics and output_path is None:
        raise ValueError("save_permutation_metrics requires output_path")
    if isinstance(n_shuffles, np.integer):
        n_shuffles = int(n_shuffles)
    if n_conditional_shuffles is not None:
        check_shuffle_count(n_conditional_shuffles, "n_conditional_shuffles")
        n_conditional_shuffles = int(n_conditional_shuffles)
    setup_console_logging()
    permutation_dir = os.path.join(output_path, "permutations") if save_permutation_metrics else None
    tasks = [
        _SampleTask(
            sample_id=sample_id, coords=coords, labels=labels, settings=settings,
            n_shuffles=n_shuffles, seed=seed_for(random_seed, sample_id),
            kept_fixed=tuple(labels_kept_fixed), max_individual_fdr=max_individual_fdr,
            n_conditional_shuffles=n_conditional_shuffles, conditional_metric=conditional_metric,
            max_conditional_individual_fdr=max_conditional_individual_fdr,
            calculate_individual_fdr=calculate_individual_fdr,
            calculate_conditional_individual_fdr=calculate_conditional_individual_fdr,
            permutation_dir=permutation_dir,
        )
        for sample_id, coords, labels in samples
    ]
    if not tasks:
        raise ValueError("no samples were given")
    if permutation_dir is not None:
        _check_permutation_dir(permutation_dir, [task.sample_id for task in tasks])

    if output_path is not None:
        _save_config(output_path, settings, dict(
            n_shuffles=n_shuffles, random_seed=random_seed, labels_kept_fixed=list(labels_kept_fixed),
            max_individual_fdr=max_individual_fdr, workers=workers,
            n_conditional_shuffles=n_conditional_shuffles, conditional_metric=conditional_metric,
            max_conditional_individual_fdr=max_conditional_individual_fdr,
            calculate_individual_fdr=calculate_individual_fdr,
            calculate_conditional_individual_fdr=calculate_conditional_individual_fdr,
            save_permutation_metrics=save_permutation_metrics,
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
            permutation_dir=task.permutation_dir,
        )
        comparisons = pd.DataFrame()
        if task.n_conditional_shuffles is not None:
            tested, comparisons = result.add_conditional_p_values(
                tested, n_shuffles=task.n_conditional_shuffles, random_seed=task.seed,
                labels_kept_fixed=task.kept_fixed, metric=task.conditional_metric,
                sample_id=task.sample_id, calculate_fdr=task.calculate_conditional_individual_fdr,
                max_conditional_individual_fdr=task.max_conditional_individual_fdr,
                permutation_dir=task.permutation_dir,
            )

        logger.info(f"[{task.sample_id}] {result.stats['patches_kept']} transactions, "
                    f"{len(result.rules)} mined")
        return SampleResult(task.sample_id, tested, result.stats, comparisons,
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


def _check_permutation_dir(permutation_dir, sample_ids):
    """Refuse to mix runs, or two samples, in one permutation folder. The caller clears it."""
    if os.path.isdir(permutation_dir) and os.listdir(permutation_dir):
        raise ValueError(f"{permutation_dir} is not empty; clear it before saving permutation metrics")
    # Case-folded, because Windows and macOS file names ignore case.
    folders = {}
    for sample_id in sample_ids:
        folder = sample_folder(permutation_dir, sample_id).name.casefold()
        if folder in folders:
            raise ValueError(f"samples {folders[folder]!r} and {sample_id!r} share a permutation folder")
        folders[folder] = sample_id


def _save_config(output_path, settings, steps):
    """Record what this run was asked to do, including optional metric recording."""
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, "run_config.json"), "w") as f:
        record = {"settings": asdict(settings), "steps": steps}
        # bandwidth may be empty in the settings; record what the run actually used.
        record["settings"]["decay_distance"] = settings.decay_distance
        json.dump(record, f, indent=2, default=str)
