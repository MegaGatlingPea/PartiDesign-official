from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from omegaconf import DictConfig, OmegaConf, open_dict

from .config import LoadedDesignConfig, compose_design_config
from .ligand_length import resolve_length_distributions
from .records import DesignRunResult, ExpandedRequest, TaskSpec
from .tasks import expand_requests, select_and_parse_tasks


class DesignRunner:
    """Reusable standalone inference runner."""

    def __init__(
        self,
        config: str | Path | Mapping[str, Any] | DictConfig | None = None,
        *,
        overrides: Sequence[str] = (),
        invocation_dir: str | Path | None = None,
    ) -> None:
        self.loaded: LoadedDesignConfig = compose_design_config(
            config, overrides=overrides, invocation_dir=invocation_dir
        )
        task_names = self.loaded.data.get("task_names")
        task_payload = OmegaConf.to_container(self.loaded.data.tasks, resolve=True)
        assert isinstance(task_payload, dict)
        self.tasks: tuple[TaskSpec, ...] = select_and_parse_tasks(
            task_payload,
            task_names=None if task_names is None else list(task_names),
            invocation_dir=self.loaded.invocation_dir,
            global_rm_token_type=(
                None if self.loaded.data.get("rm_token_type") is None
                else list(self.loaded.data.rm_token_type)
            ),
        )
        # Report the effective selected task policy, including the public default.
        for task in self.tasks:
            with open_dict(self.loaded.data.tasks[task.name]):
                self.loaded.data.tasks[task.name].rm_token_type = list(task.rm_token_type or ())
        expanded = expand_requests(self.tasks)
        ccd_path = (
            None
            if self.loaded.data.ccd_path is None
            else Path(str(self.loaded.data.ccd_path))
        )
        # Resolve distribution lengths after base-request expansion and before
        # runtime/Best-of-N setup. Replica expansion therefore shares one
        # sampled integer plan per base request.
        self.requests: tuple[ExpandedRequest, ...] = resolve_length_distributions(
            expanded,
            global_seed=int(self.loaded.data.seed),
            ccd_path=ccd_path,
        )

    @property
    def config(self) -> DictConfig:
        return self.loaded.data

    def run(self) -> DesignRunResult:
        # Import lazily so config/task inspection remains lightweight and so the
        # public API does not create a second runtime registry.
        from .runtime import run_runtime

        return run_runtime(
            self.loaded,
            self.tasks,
            self.requests,
        )


def run_design(
    config: str | Path | Mapping[str, Any] | DictConfig | None = None,
    *,
    overrides: Sequence[str] = (),
    invocation_dir: str | Path | None = None,
) -> DesignRunResult:
    return DesignRunner(
        config, overrides=overrides, invocation_dir=invocation_dir
    ).run()
