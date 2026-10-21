"""Hydra CLI for direct-child PDB/CIF directory scoring."""
from pathlib import Path
import sys
import hydra
from omegaconf import OmegaConf
from partidesign.paths import resource_root
from .api import PartiScorer
from .output import write_csv_atomic


_STRUCTURE_SUFFIXES = frozenset({".pdb", ".cif"})
_SCORE_FILENAME = "score.csv"


def discover_input_groups(input_dirs):
    """Return distinct resolved input directories and their direct-child files.

    Directory identity is resolved before deduplication.  Files are resolved and
    deduplicated within each group, while a file shared by two directories stays
    present in both groups so both output CSVs retain their directory's inputs.
    """
    if isinstance(input_dirs, (str, Path)):
        input_dirs = [input_dirs]
    if not input_dirs:
        raise ValueError("input_dirs must name at least one directory")
    groups, seen_dirs = [], set()
    for value in input_dirs:
        directory = Path(value).expanduser().resolve()
        if directory in seen_dirs:
            continue
        if not directory.is_dir():
            raise ValueError(f"input directory does not exist: {directory}")
        children = sorted(
            p for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() in _STRUCTURE_SUFFIXES
        )
        if not children:
            raise ValueError(f"input directory contains no PDB/CIF files: {directory}")
        if len({p.suffix.lower() for p in children}) != 1:
            raise ValueError(f"input directory mixes PDB and CIF files: {directory}")
        files, seen_files = [], set()
        for child in children:
            resolved = child.resolve()
            if resolved not in seen_files:
                files.append(resolved)
                seen_files.add(resolved)
        groups.append((directory, files))
        seen_dirs.add(directory)
    return groups


def discover_inputs(input_dirs):
    """Return the legacy flat, globally canonicalized file list."""
    files, seen = [], set()
    for _, group_files in discover_input_groups(input_dirs):
        for resolved in group_files:
            if resolved not in seen:
                files.append(resolved)
                seen.add(resolved)
    return files


def _preflight_destinations(groups, *, overwrite):
    destinations = [(directory, directory / _SCORE_FILENAME) for directory, _ in groups]
    directories = [path for _, path in destinations if path.is_dir()]
    if directories:
        rendered = ", ".join(str(path) for path in directories)
        raise IsADirectoryError(f"output destination is a directory: {rendered}")
    symlinks = [path for _, path in destinations if path.is_symlink()]
    if symlinks:
        rendered = ", ".join(str(path) for path in symlinks)
        raise FileExistsError(f"output must not be a symbolic link: {rendered}")
    if not overwrite:
        existing = [path for _, path in destinations if path.exists() or path.is_symlink()]
        if existing:
            rendered = ", ".join(str(path) for path in existing)
            raise FileExistsError(f"output already exists: {rendered}")
    return destinations


def run(config):
    config = OmegaConf.to_container(config, resolve=True) if OmegaConf.is_config(config) else config
    if set(config) != {"input_dirs", "overwrite", "scorer"}:
        raise ValueError("unknown or missing scoring configuration fields")
    if not isinstance(config["overwrite"], bool):
        raise ValueError("overwrite must be boolean")
    groups = discover_input_groups(config["input_dirs"])
    destinations = _preflight_destinations(groups, overwrite=config["overwrite"])
    paths, seen = [], set()
    for _, group_files in groups:
        for path in group_files:
            if path not in seen:
                paths.append(path)
                seen.add(path)
    options = dict(config["scorer"])
    checkpoint = options.pop("checkpoint")
    scorer = PartiScorer.from_checkpoint(checkpoint, **options)
    try:
        results = scorer.score_files(paths)
        if len(results) != len(paths):
            raise RuntimeError("scorer returned an unexpected number of results")
        by_path = dict(zip(paths, results))
        for (directory, group_files), (_, output) in zip(groups, destinations):
            write_csv_atomic(
                [by_path[path] for path in group_files],
                output,
                overwrite=config["overwrite"],
            )
        return 2 if any(result.status == "error" for result in results) else 0
    finally:
        scorer.close()


@hydra.main(version_base=None, config_path=str(resource_root() / "configs"), config_name="score")
def _hydra_main(config):
    try:
        code = run(config)
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        code = 1
    raise SystemExit(code)


def main():
    from partidesign.cache import configure_runtime_caches
    cache = configure_runtime_caches()
    arguments = list(sys.argv)
    if not any(value.startswith("hydra.run.dir=") for value in arguments[1:]):
        sys.argv.append(f"hydra.run.dir={cache}/hydra/scoring")
    try:
        _hydra_main()
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
