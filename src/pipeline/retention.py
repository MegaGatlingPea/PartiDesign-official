"""Discard completed-run prediction caches without removing published evidence."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def _unlink(path):
    try:
        path.unlink()
    except FileNotFoundError:
        return 0
    return 1


def prune_prediction_cache(cache):
    """Call only after all owners close, while the run lock is still held."""
    cache = Path(cache)
    raw = list(cache.glob("request_*/raw_predictions.json"))
    if raw:
        with ThreadPoolExecutor(max_workers=min(8, len(raw))) as pool:
            removed = sum(pool.map(_unlink, raw))
    else:
        removed = 0
    # Remove the registry last: if interrupted, remaining raw requests retain
    # their original identities rather than colliding with reused request IDs.
    for name in ("requests.sqlite3", "requests.sqlite3-wal", "requests.sqlite3-shm"):
        _unlink(cache / name)
    return removed
