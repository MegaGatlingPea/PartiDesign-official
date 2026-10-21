"""Persistent ESMFold2 child used by standalone validation.

The model-owned Python environment is launched once per validation worker.  A
small JSON-lines protocol keeps the parent responsive between sequence
variants while reusing PipeTools' native model bundle and request execution
helpers. Source confidence is calculated by the shared request-local native hook.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping


_RECEPTOR_SEGMENT_LEDGER = "partidesign_receptor_segments"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("ESMFold2 validation startup payload must be an object")
    return payload


def _request(raw: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "request_id", "input_json", "output_dir",
        "expected_prefix", "expected_samples",
    }
    optional = {"sample_index", "seed"}
    if set(raw).difference(required | optional) or not required.issubset(raw):
        raise ValueError("ESMFold2 validation request fields are invalid")
    request = dict(raw)
    request["input_json"] = Path(str(request["input_json"])).resolve()
    request["output_dir"] = Path(str(request["output_dir"])).resolve()
    sample_index = request.get("sample_index", 0)
    if isinstance(sample_index, bool) or not isinstance(sample_index, int) or sample_index < 0:
        raise ValueError("ESMFold2 validation sample_index is invalid")
    request["sample_index"] = sample_index
    if "seed" in request:
        seed = request["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("ESMFold2 validation seed is invalid")
    return request


def _emit(payload: Mapping[str, Any]) -> None:
    sys.stdout.write(json.dumps(dict(payload), sort_keys=True, allow_nan=False) + "\n")
    sys.stdout.flush()


def _error_payload(request: Mapping[str, Any], exc: BaseException) -> dict[str, Any]:
    return {
        "type": "result",
        "request_id": request.get("request_id"),
        "status": "error",
        "error": f"{type(exc).__name__}: {exc}",
        "traceback": traceback.format_exc()[-16_000:],
        "artifacts": [],
    }


def _segment_map(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    from src.design.rewards.esmfold2_mapping import source_mapping

    return source_mapping(payload)


def _native_binder_confidence(*args: Any, **kwargs: Any) -> dict[str, Any]:
    from src.design.rewards.esmfold2_mapping import native_binder_confidence

    return native_binder_confidence(*args, **kwargs)


def run(startup_path: Path) -> None:
    startup = _load_json(startup_path)
    if startup.get("schema_version") != 1:
        raise ValueError("unsupported ESMFold2 validation startup schema")
    mode = str(startup["mode"])
    paths = {
        str(key): Path(str(value)).resolve()
        for key, value in dict(startup["paths"]).items()
    }
    config = dict(startup["config"])
    # The subprocess environment puts the vendored source ahead of all host
    # packages.  Importing the worker module here preserves its model-owned
    # ESM/transformers environment and avoids duplicating numerical code.
    from pipetools.workers import esmfold2 as native

    bundle = native._load_model_bundle(
        mode,
        paths,
        backend=str(config["esmc_attention_backend"]),
        optimization_config=config.get("optimization", {"mode": "auto", "disable": []}),
    )
    _emit({"type": "ready", "schema_version": 1})
    try:
        for line in sys.stdin:
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError("ESMFold2 validation request must be an object")
            if raw.get("type") == "close":
                break
            request = _request(raw)
            request_config = dict(config)
            request_config["samples"] = 1
            request_config["seed"] = request.get("seed", config["seed"])
            fatal = False
            try:
                artifacts = native._execute_request(bundle, request, request_config, mode)
                result = {
                    "type": "result", "request_id": request["request_id"],
                    "status": "ok", "error": None, "artifacts": artifacts,
                }
            except Exception as exc:
                # Keep the child alive for candidate-local failures.  A later
                # request can still complete with the resident model bundle.
                output_dir = request["output_dir"]
                if not output_dir.exists() and not output_dir.is_symlink():
                    output_dir.mkdir(parents=True, exist_ok=False)
                if output_dir.is_dir() and not output_dir.is_symlink():
                    native._atomic_json(
                        output_dir / "error.json",
                        {"request_id": request["request_id"], "error": str(exc)},
                    )
                result = _error_payload(request, exc)
                fatal = isinstance(exc, native._SystemicWorkerError)
                if fatal:
                    result["fatal"] = True
            finally:
                if not fatal:
                    native._release_request_memory(bundle)
            _emit(result)
            if fatal:
                break
    finally:
        native._close_model_bundle(bundle)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="partidesign.validate.esmfold2_worker")
    parser.add_argument("--startup-json", type=Path, required=True)
    args = parser.parse_args(argv)
    run(args.startup_json)


if __name__ == "__main__":
    main()
