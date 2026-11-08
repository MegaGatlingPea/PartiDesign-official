"""Opt-in phase observations; never changes sampling or loader settings."""
from contextlib import contextmanager
from pathlib import Path
import datetime
import json
import os

ENV = "PARTIDESIGN_MEMORY_TRACE"


def snapshot(phase, **facts):
    directory = os.environ.get(ENV)
    if not directory:
        return
    record = dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  pid=os.getpid(), phase=phase, **facts)
    try:
        import psutil
        process = psutil.Process()
        records = []
        for child in [process, *process.children(recursive=True)]:
            try:
                memory = child.memory_full_info()
                records.append(dict(pid=child.pid, ppid=child.ppid(), name=child.name(),
                                    rss=memory.rss, pss=getattr(memory, "pss", None)))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        record["processes"] = records
        for key in ("memory.current", "memory.max"):
            path = Path("/sys/fs/cgroup") / key
            if path.is_file():
                record["cgroup_"+key] = path.read_text().strip()
        # Node/cgroup totals include other processes; they are not driver RSS.
        import sys
        ray = sys.modules.get("ray")
        if ray is not None and ray.is_initialized():
            try:
                from ray._private.internal_api import memory_summary
                record["ray_node_memory_summary"] = memory_summary(stats_only=True)
            except Exception as exc:
                record["ray_capture_error"] = f"{type(exc).__name__}: {exc}"
        import subprocess
        gpu = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                              "--format=csv,noheader,nounits"], capture_output=True,
                             text=True, timeout=2)
        ids = {entry["pid"] for entry in records}
        record["owned_gpu_memory_mib"] = {
            parts[0].strip(): parts[1].strip()
            for line in gpu.stdout.splitlines() if len(parts := line.split(",")) == 2
            and parts[0].strip().isdigit() and int(parts[0]) in ids}
    except Exception as exc:
        record["capture_error"] = f"{type(exc).__name__}: {exc}"
    try:
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        with (root/f"pid_{os.getpid()}.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False)+"\n")
    except OSError:
        pass


@contextmanager
def generation_memory_trace():
    if not os.environ.get(ENV):
        yield
        return
    import lightning.pytorch as pl
    original = pl.Trainer.predict

    def loader_state(trainer):
        loaders = trainer.predict_dataloaders
        loaders = loaders if isinstance(loaders, (list, tuple)) else [loaders]
        fetcher = getattr(trainer.predict_loop, "_data_fetcher", None)
        iterator = getattr(fetcher, "iterator", None)
        mode = getattr(iterator, "_iterator", iterator)
        return dict(loaders=[{
            **{name: getattr(loader, name, None) for name in
               ("batch_size", "num_workers", "prefetch_factor", "pin_memory")},
            "batch_sampler_batch_size": getattr(getattr(loader, "batch_sampler", None), "batch_size", None)}
            for loader in loaders],
            data_iterators=[dict(type=type(item).__name__,
                tasks_outstanding=getattr(item, "_tasks_outstanding", None),
                tasks_sent=getattr(item, "_send_idx", None),
                tasks_received=getattr(item, "_rcvd_idx", None))
                for item in getattr(mode, "iterators", ())])

    class Trace(pl.Callback):
        def on_predict_start(self, trainer, pl_module):
            snapshot("generation_inputs_ready", **loader_state(trainer))

        def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
            if batch_idx == 0 and dataloader_idx == 0:
                snapshot("generation_first_batch_finished", batch_index=batch_idx,
                         **loader_state(trainer))

    def predict(trainer, *args, **kwargs):
        callback = Trace()
        trainer.callbacks.append(callback)
        try:
            return original(trainer, *args, **kwargs)
        finally:
            if callback in trainer.callbacks:
                trainer.callbacks.remove(callback)

    pl.Trainer.predict = predict
    try:
        yield
    finally:
        pl.Trainer.predict = original
