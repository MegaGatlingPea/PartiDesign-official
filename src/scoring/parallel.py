"""Reusable, bounded ordered CPU process mapping.

The pool in this module is intentionally generic: callbacks receive one item,
carry no model or device state, and return a value that can cross a process
boundary. Tensor values are converted to NumPy payloads in the worker and
restored to CPU tensors in the parent using the scoring preprocessing helpers.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
import inspect
import multiprocessing as mp
import os
import pickle
from typing import Any

from . import preprocessing as _preprocessing


_MISSING = object()


@dataclass(frozen=True)
class TaskResult:
    """Result for one input item.

    A callback or process/executor failure is kept on the result so callers
    can aggregate failures while preserving successful items that follow it.
    """

    value: Any = None
    error: BaseException | None = None

    def get(self) -> Any:
        """Return ``value`` or raise the per-item ``error``."""

        if self.error is not None:
            raise self.error
        return self.value


@dataclass(frozen=True)
class _WorkerOutcome:
    """Pickle-safe envelope returned by the child process."""

    ok: bool
    value: Any = None
    error: BaseException | None = None


def _safe_exception_text(error: BaseException) -> str:
    try:
        return str(error)
    except BaseException:
        return f"<{type(error).__name__} could not be rendered>"


def _unpicklable_exception(error: BaseException) -> RuntimeError:
    """Represent an exception whose object cannot cross the process boundary."""

    return RuntimeError(f"{type(error).__name__}: {_safe_exception_text(error)}")


def _pickle_roundtrip(value: Any) -> None:
    """Check both serialization and deserialization before returning a result."""

    payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.loads(payload)


def _safe_error(error: BaseException) -> BaseException:
    """Keep a normal exception type, with a RuntimeError fallback if needed."""

    try:
        _pickle_roundtrip(error)
    except BaseException:
        return _unpicklable_exception(error)
    return error


def _worker_call(fn: Any, item: Any) -> _WorkerOutcome:
    """Invoke one callback and make every outcome safe for process transport."""

    try:
        value = fn(item)
    except BaseException as error:
        return _WorkerOutcome(ok=False, error=_safe_error(error))

    try:
        # This avoids torch's multiprocessing storage-FD protocol for nested
        # canonical mappings while preserving ordinary dataclasses and NumPy
        # fields used by benchmark records. The executor performs the final
        # result serialization; failures from that step are converted by the
        # parent iterator into an item-level TaskResult error.
        wired = _preprocessing._wire_value(value)
    except BaseException as error:
        transport_error = RuntimeError(
            "callback result could not be transported: "
            f"{type(error).__name__}: {_safe_exception_text(error)}"
        )
        return _WorkerOutcome(ok=False, error=_safe_error(transport_error))
    return _WorkerOutcome(ok=True, value=wired)


def _decode_outcome(outcome: Any) -> TaskResult:
    """Restore a worker envelope; tolerate an unexpected raw future value."""

    if not isinstance(outcome, _WorkerOutcome):
        return TaskResult(value=_preprocessing._restore_value(outcome))
    if not outcome.ok:
        error = outcome.error
        if not isinstance(error, BaseException):
            error = RuntimeError("worker returned an invalid error result")
        return TaskResult(error=error)
    return TaskResult(value=_preprocessing._restore_value(outcome.value))


def _completed_future(error: BaseException) -> Future:
    future = Future()
    future.set_exception(error)
    return future


def _validate_callback(fn: Any) -> None:
    if not callable(fn):
        raise TypeError("fn must be a module-level callable")
    # Bound methods and callable instances can capture a model/device. A
    # module-level function (or builtin) is explicit and spawn-pickleable.
    if inspect.ismethod(fn) or inspect.isclass(fn):
        raise TypeError("fn must be a module-level callable")
    if not (inspect.isfunction(fn) or inspect.isbuiltin(fn)):
        raise TypeError("fn must be a module-level callable")
    qualname = getattr(fn, "__qualname__", "")
    if "<locals>" in qualname:
        raise TypeError("fn must be a module-level callable")
    try:
        _pickle_roundtrip(fn)
    except BaseException as error:
        raise TypeError("fn must be a picklable module-level callable") from error


class OrderedCPUMap:
    """Map module-level callbacks in input order through a spawn pool.

    The process executor is created on first iteration, and no more than
    ``2 * num_workers`` tasks are pending for one map iterator. A zero-worker
    pool executes through the same wire/restore path in the parent.
    """

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
        self._iterators: set[_OrderedMapIterator] = set()

    def _get_executor(self) -> ProcessPoolExecutor:
        if self._closed:
            raise RuntimeError("OrderedCPUMap is closed")
        if self._executor is None:
            self._executor = ProcessPoolExecutor(
                max_workers=self.num_workers,
                mp_context=mp.get_context("spawn"),
                initializer=_preprocessing._initialize_worker,
            )
        return self._executor

    def _submit(self, fn: Any, item: Any) -> Future:
        # Spawned workers inherit the environment present when Popen starts.
        # Cap BLAS threads for that short launch window, then restore the
        # parent's exact environment so a surrounding training process is
        # unaffected. The initializer applies the same policy after import.
        previous = {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        }
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        try:
            return self._get_executor().submit(_worker_call, fn, item)
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def map(self, fn: Any, items: Iterable[Any]) -> Iterator[TaskResult]:
        """Return an input-ordered iterator of :class:`TaskResult` objects."""

        if self._closed:
            raise RuntimeError("OrderedCPUMap is closed")
        _validate_callback(fn)
        return _OrderedMapIterator(self, fn, iter(items))

    def _register(self, iterator: _OrderedMapIterator) -> None:
        self._iterators.add(iterator)

    def _unregister(self, iterator: _OrderedMapIterator) -> None:
        self._iterators.discard(iterator)

    def close(self) -> None:
        """Cancel active map iterators and shut down the process pool."""

        if self._closed:
            return
        self._closed = True
        for iterator in tuple(self._iterators):
            iterator._cancel(close_source=True)
        executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    def __enter__(self) -> "OrderedCPUMap":
        if self._closed:
            raise RuntimeError("OrderedCPUMap is closed")
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


class _OrderedMapIterator:
    """Iterator holding one bounded submission window."""

    def __init__(self, pool: OrderedCPUMap, fn: Any, items: Iterator[Any]) -> None:
        self._pool = pool
        self._fn = fn
        self._items = items
        self._pending: deque[Future | TaskResult] = deque()
        self._input_exhausted = False
        self._started = False
        self._closed = False
        pool._register(self)

    def __iter__(self) -> "_OrderedMapIterator":
        return self

    def _next_item(self) -> Any:
        if self._input_exhausted:
            return _MISSING
        try:
            return next(self._items)
        except StopIteration:
            self._input_exhausted = True
            return _MISSING

    def _entry_for(self, item: Any) -> Future | TaskResult:
        if self._pool.num_workers == 0:
            try:
                value = self._fn(item)
                wired = _preprocessing._wire_value(value)
                return _decode_outcome(_WorkerOutcome(ok=True, value=wired))
            except KeyboardInterrupt:
                raise
            except Exception as error:
                return TaskResult(error=_safe_error(error))
        try:
            return self._pool._submit(self._fn, item)
        except KeyboardInterrupt:
            raise
        except Exception as error:
            # A constructor, submit, or broken-executor error belongs to this
            # input item and must remain visible to caller-side accounting.
            return _completed_future(error)

    def _fill_initial(self) -> None:
        limit = 2 * self._pool.num_workers
        while len(self._pending) < limit:
            item = self._next_item()
            if item is _MISSING:
                return
            self._pending.append(self._entry_for(item))

    def _fill_one(self) -> None:
        item = self._next_item()
        if item is not _MISSING:
            self._pending.append(self._entry_for(item))

    @staticmethod
    def _consume(entry: Future | TaskResult) -> TaskResult:
        if isinstance(entry, TaskResult):
            return entry
        try:
            outcome = entry.result()
        except KeyboardInterrupt:
            raise
        except Exception as error:
            # Includes BrokenProcessPool and result-pickling failures.
            return TaskResult(error=error)
        try:
            return _decode_outcome(outcome)
        except KeyboardInterrupt:
            raise
        except Exception as error:
            return TaskResult(error=error)

    def __next__(self) -> TaskResult:
        if self._closed:
            raise StopIteration
        if self._pool._closed:
            self._cancel(close_source=True)
            raise RuntimeError("OrderedCPUMap is closed")
        try:
            if self._pool.num_workers == 0:
                item = self._next_item()
                if item is _MISSING:
                    self.close()
                    raise StopIteration
                return self._entry_for(item)  # type: ignore[return-value]

            if not self._started:
                self._started = True
                self._fill_initial()
            if not self._pending:
                self.close()
                raise StopIteration
            entry = self._pending.popleft()
            result = self._consume(entry)
            self._fill_one()
            if self._input_exhausted and not self._pending:
                self.close()
            return result
        except KeyboardInterrupt:
            self.close()
            raise
        except StopIteration:
            self.close()
            raise
        except BaseException:
            self.close()
            raise

    def _cancel(self, *, close_source: bool) -> None:
        if self._closed:
            return
        self._closed = True
        for entry in self._pending:
            cancel = getattr(entry, "cancel", None)
            if callable(cancel):
                cancel()
        self._pending.clear()
        if close_source:
            close = getattr(self._items, "close", None)
            if callable(close):
                close()
        self._pool._unregister(self)

    def close(self) -> None:
        """Cancel this iterator's pending futures while keeping the pool reusable."""

        self._cancel(close_source=True)

    def __del__(self) -> None:
        try:
            self._cancel(close_source=True)
        except BaseException:
            pass
