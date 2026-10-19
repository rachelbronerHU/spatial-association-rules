"""Optional per-batch shuffle metrics, buffered on disk and saved once the batch finishes."""

from contextlib import ExitStack, contextmanager
from mmap import ACCESS_WRITE, mmap
from pathlib import Path
from tempfile import TemporaryFile
from urllib.parse import quote

import numpy as np


def sample_folder(permutation_dir, sample_id):
    """Where one sample's archives go. Sample IDs are URL-encoded to stay valid file names."""
    return Path(permutation_dir) / ("sample_" + quote(str(sample_id), safe=""))


@contextmanager
def record_permutations(permutation_dir, n_shuffles, rule_idx, *, sample_id, test, fixed_types=(), batch=None):
    """Yield arrays valid inside this context, then save an archive if recording is enabled.

    Refuses to overwrite an archive, so files from two runs or two samples never mix.
    """
    if permutation_dir is None:
        yield None
        return

    filename = "regular_p_value.npz" if test == "regular_p_value" else f"conditional_{batch:04d}.npz"
    path = sample_folder(permutation_dir, sample_id) / filename
    if path.exists():
        raise FileExistsError(f"{path} already exists; clear the permutation folder before re-running")
    shape = (n_shuffles, len(rule_idx))
    dtypes = {"lift": float, "conviction": float}
    
    if test == "regular_p_value": 
        # Only the regular p-value test counts threshold passes, and not for conditional_p_value (complex rules)
        dtypes["passed_mining_filters"] = bool
    with ExitStack() as buffers:
        arrays = {}
        try:
            for name, dtype in dtypes.items():
                size = n_shuffles * len(rule_idx) * np.dtype(dtype).itemsize
                mapping = None
                if size:
                    temporary = buffers.enter_context(TemporaryFile())
                    temporary.truncate(size)
                    mapping = buffers.enter_context(mmap(temporary.fileno(), size, access=ACCESS_WRITE))
                arrays[name] = np.ndarray(shape, dtype=dtype, buffer=mapping)
            yield arrays
            path.parent.mkdir(parents=True, exist_ok=True)
            # Open before the cleanup block: an existing archive must never be deleted.
            archive = path.open("xb")
            try:
                with archive:
                    np.savez_compressed(
                        archive, rule_idx=np.asarray(rule_idx), sample_id=np.asarray(str(sample_id)),
                        test=np.asarray(test), fixed_types=np.asarray(fixed_types, dtype=str), **arrays,
                    )
            except BaseException:
                path.unlink(missing_ok=True)
                raise
        finally:
            # Release batch arrays before closing their buffers.
            arrays.clear()
