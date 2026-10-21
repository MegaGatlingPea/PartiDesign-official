"""CPU structure preparation with a reusable spawn process pool.

The worker entry points in this module intentionally carry no model or device
state. A caller that creates a StructurePreprocessor from a script must
protect top-level execution with if __name__ == "__main__": because the pool
uses Python's spawn start method.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator, Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any


_MISSING = object()


def _initialize_worker() -> None:
    """Initialize one worker with a single CPU thread for each torch pool."""

    # Set these before importing torch so its CPU libraries observe the
    # one-thread policy during initialization as well as during inference.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


@dataclass(frozen=True)
class _TensorWire:
    """Pickle-safe representation of one tensor crossing the process boundary."""

    array: Any


def _wire_value(value: Any) -> Any:
    """Convert tensors to ordinary NumPy payloads before process transport."""

    import torch

    if torch.is_tensor(value):
        # A NumPy copy avoids torch's multiprocessing storage-FD protocol,
        # which can exhaust the parent process's descriptor limit on batches
        # containing many canonical tensors.
        return _TensorWire(value.detach().cpu().numpy().copy())
    if isinstance(value, Mapping):
        return {key: _wire_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_wire_value(item) for item in value)
    if isinstance(value, list):
        return [_wire_value(item) for item in value]
    return value


def _restore_value(value: Any) -> Any:
    """Restore process-wire arrays as CPU tensors in the parent process."""

    if isinstance(value, _TensorWire):
        import torch

        return torch.from_numpy(value.array)
    if isinstance(value, Mapping):
        return {key: _restore_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_restore_value(item) for item in value)
    if isinstance(value, list):
        return [_restore_value(item) for item in value]
    return value


def _restore_result(result: tuple[int, Any | None, str | None]) -> tuple[int, Any | None, str | None]:
    index, prepared, error = result
    if prepared is None:
        return index, None, error
    from src.scoring.features import PreparedStructure

    return index, PreparedStructure(
        canonical=_restore_value(prepared.canonical),
        partner_labels=prepared.partner_labels,
        design_source_ids=prepared.design_source_ids,
        design_group_ids=prepared.design_group_ids,
        design_group_labels=prepared.design_group_labels,
    ), error


def _prepare_worker(task: tuple[int, Any, str, Any, bool]) -> tuple[int, Any | None, str | None]:
    """Prepare one input and convert declared input failures to a row error."""

    from src.design.core.records import DesignInputError
    from .records import ScoreInputError

    index, value, identity, design_chains, is_file, *options = task
    scope = options[0] if options else None
    try:
        if is_file:
            from src.design.utils.reader import read_structure

            atoms = read_structure(Path(value))
        else:
            atoms = value
        from src.scoring.features import PreparedStructure, prepare_structure

        prepared = prepare_structure(
            atoms,
            identity=str(identity),
            design_chains=design_chains,
            scope=scope,
        )
        # Transport ordinary NumPy payloads, not torch shared-storage handles.
        # The parent restores the exact CPU tensor schema before yielding.
        prepared = PreparedStructure(
            canonical=_wire_value(prepared.canonical),
            partner_labels=prepared.partner_labels,
            design_source_ids=prepared.design_source_ids,
            design_group_ids=prepared.design_group_ids,
            design_group_labels=prepared.design_group_labels,
        )
        return index, prepared, None
    except (DesignInputError, ScoreInputError) as exc:
        return index, None, f"{type(exc).__name__}: {exc}"


class StructurePreprocessor:
    """Prepare structures serially or through a bounded CPU spawn pool."""

    def __init__(self, num_workers: int = 16) -> None:
        if (
            isinstance(num_workers, bool)
            or not isinstance(num_workers, int)
            or num_workers < 0
        ):
            raise ValueError("num_workers must be a nonnegative integer")
        self.num_workers = int(num_workers)
        self._executor: ProcessPoolExecutor | None = None
        self._closed = False

    def _get_executor(self) -> ProcessPoolExecutor:
        if self._closed:
            raise RuntimeError("StructurePreprocessor is closed")
        if self._executor is None:
            context = mp.get_context("spawn")
            self._executor = ProcessPoolExecutor(
                max_workers=self.num_workers,
                mp_context=context,
                initializer=_initialize_worker,
            )
        return self._executor

    def prepare(
        self,
        tasks: Iterable[tuple[int, Any, str, Any, bool]],
    ) -> Iterator[tuple[int, Any | None, str | None]]:
        """Yield input-ordered preparation results.

        At most 2 * num_workers futures are submitted at once. The executor
        is created only after the first task is observed, so an empty input
        never starts a pool.
        """

        if self._closed:
            raise RuntimeError("StructurePreprocessor is closed")
        iterator = iter(tasks)
        return self._prepare_iter(iterator)

    def _prepare_iter(
        self,
        iterator: Iterator[tuple[int, Any, str, Any, bool]],
    ) -> Iterator[tuple[int, Any | None, str | None]]:
        if self.num_workers == 0:
            for task in iterator:
                yield _restore_result(_prepare_worker(task))
            return

        first = next(iterator, _MISSING)
        if first is _MISSING:
            return
        pending = deque()
        executor = self._get_executor()
        limit = 2 * self.num_workers
        pending.append(executor.submit(_prepare_worker, first))
        while len(pending) < limit:
            task = next(iterator, _MISSING)
            if task is _MISSING:
                break
            pending.append(executor.submit(_prepare_worker, task))

        while pending:
            future = pending.popleft()
            # Deliberately do not catch future exceptions: worker/runtime
            # failures are fatal and must not trigger a serial fallback.
            yield _restore_result(future.result())
            task = next(iterator, _MISSING)
            if task is not _MISSING:
                pending.append(executor.submit(_prepare_worker, task))

    def close(self) -> None:
        """Shut down the pool; repeated calls are safe."""

        if self._closed:
            return
        self._closed = True
        executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    def __enter__(self) -> "StructurePreprocessor":
        if self._closed:
            raise RuntimeError("StructurePreprocessor is closed")
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
