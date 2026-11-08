def main(argv=None):
    import argparse
    from .cli import add_arguments, run
    parser = argparse.ArgumentParser(prog="python -m src.screen",description="Screen sequence candidates with resident co-folding models")
    add_arguments(parser)
    try:
        return run(parser.parse_args(argv))
    except (ValueError,OSError,RuntimeError) as exc:
        parser.exit(2,f"{type(exc).__name__}: {exc}\n")

if __name__ == "__main__":
    raise SystemExit(main())
