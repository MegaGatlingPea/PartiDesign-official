"""Read-only validation for the native/uv pair created by install.sh."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def fail(message: str) -> None:
    raise ValueError(message)


def validate_native(prefix: Path, python_version: str, cuda: str, *, allow_incomplete: bool = False) -> dict:
    prefix = prefix.resolve()
    if Path(sys.prefix).resolve() != prefix or Path(sys.base_prefix).resolve() != prefix:
        fail(f"The native Python does not belong to the requested prefix: {prefix}")
    actual = ".".join(map(str, sys.version_info[:3]))
    if actual != python_version:
        fail(f"Native Python is {actual}; the selected profile requires {python_version}")
    metadata = prefix / "conda-meta"
    if not allow_incomplete and not (metadata / "history").is_file():
        fail(f"The native environment has no completed Conda metadata: {prefix}")
    records = list(metadata.glob("cuda-version-*.json"))
    if not records and not allow_incomplete:
        fail("The native CUDA version record is missing; finish the native stage with --update")
    for record in records:
        version = str(json.loads(record.read_text())["version"])
        if version.split(".", 1)[0] != cuda:
            fail(f"Native CUDA {version} does not match the selected CUDA {cuda} profile")
    return {"native_prefix": str(prefix), "python": actual, "cuda": cuda}


def validate_venv(prefix: Path, venv: Path, python_version: str, *, allow_incomplete: bool = False) -> dict:
    prefix, venv = prefix.resolve(), venv.resolve()
    if not venv.exists():
        if allow_incomplete:
            return {"venv": str(venv), "state": "missing"}
        fail(f"The virtual environment is missing: {venv}")
    if not venv.is_dir():
        fail(f"The virtual environment target is not a directory: {venv}")
    config = venv / "pyvenv.cfg"
    if not config.is_file():
        if allow_incomplete:
            return {"venv": str(venv), "state": "incomplete"}
        fail(f"The virtual environment configuration is missing: {config}")
    settings = {}
    for line in config.read_text().splitlines():
        key, separator, value = line.partition("=")
        if separator:
            key = key.strip().lower()
            if key in settings:
                fail(f"Duplicate virtual environment setting: {key}")
            settings[key] = value.strip()
    if settings.get("include-system-site-packages", "").lower() != "true":
        fail("The virtual environment must inherit its paired native packages (--system-site-packages)")
    if not settings.get("home") or Path(settings["home"]).resolve() != (prefix / "bin").resolve():
        fail(f"The virtual environment belongs to a different native Python: {settings.get('home')!r}")
    python = venv / "bin" / "python"
    if not python.is_file() or not os.access(python, os.X_OK):
        fail(f"The virtual environment Python is missing or not executable: {python}")
    probe = "import json,sys; print(json.dumps({'prefix':sys.prefix,'base_prefix':sys.base_prefix,'version':'.'.join(map(str,sys.version_info[:3]))}))"
    result = subprocess.run([str(python), "-B", "-I", "-c", probe], capture_output=True, text=True, timeout=15)
    if result.returncode:
        fail(f"The virtual environment Python could not start: {result.stderr.strip()}")
    observed = json.loads(result.stdout)
    if Path(observed["prefix"]).resolve() != venv or Path(observed["base_prefix"]).resolve() != prefix:
        fail("The virtual environment interpreter does not match the recorded native/venv pair")
    if observed["version"] != python_version:
        fail(f"Virtual environment Python {observed['version']} does not match {python_version}")
    return {"venv": str(venv), "native_prefix": str(prefix), "state": "ready"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("native", "venv"))
    parser.add_argument("--native-prefix", type=Path, required=True)
    parser.add_argument("--venv", type=Path)
    parser.add_argument("--python-version", required=True)
    parser.add_argument("--cuda", choices=("12", "13"))
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    try:
        if args.kind == "native":
            if args.cuda is None:
                parser.error("native validation requires --cuda")
            result = validate_native(args.native_prefix, args.python_version, args.cuda, allow_incomplete=args.allow_incomplete)
        else:
            if args.venv is None:
                parser.error("venv validation requires --venv")
            result = validate_venv(args.native_prefix, args.venv, args.python_version, allow_incomplete=args.allow_incomplete)
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print(f"Installation target check failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
