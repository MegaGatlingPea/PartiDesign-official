"""Atomic fixed-schema long-table output."""
import csv
import os
from pathlib import Path
import tempfile

CSV_FIELDS = ("input_file", "scope", "partner_a", "partner_b", "score", "mean_log_density",
              "density_sum", "pair_count", "status", "error", "checkpoint", "weights")

def result_rows(results):
    for result in results:
        common = dict(input_file=result.identity, checkpoint=result.checkpoint, weights=result.weights)
        metrics = {key: getattr(result, key) for key in
                   ("score", "mean_log_density", "density_sum", "pair_count", "status", "error")}
        yield dict(common, scope=result.scope, partner_a="", partner_b="", **metrics)
        for interface in result.interfaces:
            yield dict(common, scope="interface", **interface.as_dict())


def write_csv_atomic(results, path, *, overwrite=False):
    requested = Path(path).expanduser()
    # Resolve the parent only: never follow the final output symlink.
    path = requested.parent.resolve() / requested.name
    if path.is_symlink():
        raise FileExistsError(f"output must not be a symbolic link: {path}")
    if path.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(result_rows(results))
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
            os.unlink(temporary)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path
