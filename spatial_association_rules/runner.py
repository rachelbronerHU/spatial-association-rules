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

import pandas as pd

from .mine import mine
from .rules import filter_rules
from .settings import Settings
from .validation.significance import seed_for

logger = logging.getLogger(__name__)


@dataclass
class SampleResult:
    sample_id: object
    rules: pd.DataFrame     # every rule, with raw p-values and classification columns
    stats: dict


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


def run_samples(samples, settings: Settings, *, n_shuffles, random_seed=None,
                labels_kept_fixed=(), min_lift_gain=None, max_individual_fdr=None,
                workers=None, output_path=None) -> RunReport:
    """
    Mine every sample and report what came back, with raw p-values and nothing cut.

    samples:      iterable of (sample_id, coords, labels)
    workers:      An integer runs that many processes — on Windows,
                  guard the caller with `if __name__ == "__main__"`.
                  `None` makes it to run here. 
    output_path:  where to write run_config.json. `None` writes nothing.

    A sample that raises lands in report.failures. If every sample fails, this raises.
    """
    setup_console_logging()
    tasks = [
        (sample_id, coords, labels, settings, n_shuffles, seed_for(random_seed, sample_id),
         tuple(labels_kept_fixed), min_lift_gain, max_individual_fdr)
        for sample_id, coords, labels in samples
    ]
    if not tasks:
        raise ValueError("no samples were given")

    if output_path is not None:
        _save_config(output_path, settings, dict(
            n_shuffles=n_shuffles, random_seed=random_seed, labels_kept_fixed=list(labels_kept_fixed),
            min_lift_gain=min_lift_gain, workers=workers,
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
    sample_id, coords, labels, settings, n_shuffles, seed, kept_fixed, min_lift_gain, max_individual_fdr = task
    try:
        result = mine(coords, labels, settings, sample_id=sample_id)
        tested = result.add_p_values(
            n_shuffles=n_shuffles, random_seed=seed, labels_kept_fixed=kept_fixed, sample_id=sample_id,
        )
        classified = filter_rules(tested, min_lift_gain=min_lift_gain,
                                  max_individual_fdr=max_individual_fdr)

        logger.info(f"[{sample_id}] {result.stats['patches_kept']} transactions, "
                    f"{len(result.rules)} mined, "
                    f"{int(classified['adds_information'].sum())} add information")
        return SampleResult(sample_id, classified, result.stats), None
    except Exception:
        return None, (sample_id, traceback.format_exc())


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
