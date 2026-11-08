"""Keep Lightning's generation ranks out of pipeline orchestration."""
from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path

_CONFIG_ENV = "PARTIDESIGN_PIPELINE_GENERATION_CONFIG"


def run_generation_worker():
    """Resume the parent's native generation when Lightning replays the CLI."""
    config_path = os.environ.get(_CONFIG_ENV)
    if not config_path or int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))) == 0:
        return None

    from omegaconf import OmegaConf
    from src.design.core.api import DesignRunner

    snapshot = OmegaConf.load(config_path)
    result = DesignRunner(snapshot.config, invocation_dir=snapshot.invocation_dir).run()
    return 2 if result.failed or result.postprocess_failed or result.score_failed else 0


@contextmanager
def generation_launch(config, invocation_dir, workdir):
    """Pass the exact effective native config to re-executed child ranks."""
    from omegaconf import OmegaConf

    path = Path(workdir) / "generation_worker.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create({
        "config": OmegaConf.to_container(config, resolve=True),
        "invocation_dir": str(Path(invocation_dir).resolve()),
    }), path)
    previous = os.environ.get(_CONFIG_ENV)
    os.environ[_CONFIG_ENV] = str(path.resolve())
    try:
        from .memory_trace import generation_memory_trace
        with generation_memory_trace():
            yield
    finally:
        if previous is None:
            os.environ.pop(_CONFIG_ENV, None)
        else:
            os.environ[_CONFIG_ENV] = previous
        path.unlink(missing_ok=True)
