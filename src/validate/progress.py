"""Timestamped validation progress, mirrored to task-local log files."""
from datetime import datetime
import json
from pathlib import Path
import sys


def emit(payload: dict, path: Path | None = None, *, error: bool = False) -> None:
    record = {"timestamp": datetime.now().astimezone().isoformat(timespec="seconds"), **payload}
    line = json.dumps(record, ensure_ascii=False, default=str)
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
    print(line, file=sys.stderr if error else sys.stdout, flush=True)
