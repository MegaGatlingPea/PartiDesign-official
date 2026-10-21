from __future__ import annotations

import sys

from omegaconf import OmegaConf

from partidesign._lightning_notices import configure_lightning_notices

from .core.api import DesignRunner
from .core.records import DesignError


def main(argv: list[str] | None = None) -> int:
    configure_lightning_notices()
    arguments = list(sys.argv[1:] if argv is None else argv)
    show_config = False
    if "--show-config" in arguments:
        arguments.remove("--show-config")
        show_config = True
    try:
        runner = DesignRunner(overrides=arguments)
        if show_config:
            sys.stdout.write(OmegaConf.to_yaml(runner.config, resolve=True))
            return 0
        result = runner.run()
    except DesignError as exc:
        sys.stderr.write(f"{type(exc).__name__}: {exc}\n")
        return 1
    return 2 if result.failed or result.postprocess_failed or result.score_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
