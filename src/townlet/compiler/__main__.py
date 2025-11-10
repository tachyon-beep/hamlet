"""CLI entry point for the Universe compiler (python -m townlet.compiler)."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from collections.abc import Iterable, Mapping
from pathlib import Path

from townlet.universe.compiled import CompiledUniverse
from townlet.universe.compiler import UniverseCompiler
from townlet.universe.errors import CompilationError


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m townlet.compiler",
        description="Utility commands for the UniverseCompiler (TASK-004A CLI).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    compile_parser = subparsers.add_parser("compile", help="Compile a config pack and optionally cache the artifact.")
    compile_parser.add_argument("config_dir", help="Path to config directory (contains training.yaml, bars.yaml, etc.)")
    compile_parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Skip cache reads/writes (always rebuild).",
    )

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Inspect a compiled universe artifact (MessagePack file).",
    )
    inspect_parser.add_argument("artifact", help="Path to config directory or .compiled/universe.msgpack artifact")
    inspect_parser.add_argument(
        "--format",
        choices=("table", "json"),
        default="table",
        help="Output format for inspection (default: table)",
    )

    validate_parser = subparsers.add_parser("validate", help="Run compilation without touching the cache (lint-style check).")
    validate_parser.add_argument("config_dir", help="Path to config directory to validate")

    return parser


def _format_metadata_lines(metadata) -> list[str]:
    rows: list[tuple[str, str]] = [
        ("Universe", metadata.universe_name),
        ("Substrate", metadata.substrate_type),
        ("Meters", f"{metadata.meter_count}"),
        ("Affordances", f"{metadata.affordance_count}"),
        ("Actions", f"{metadata.action_count}"),
        ("Observation Dim", f"{metadata.observation_dim}"),
        ("Grid Cells", metadata.grid_cells if metadata.grid_cells is not None else "N/A"),
        ("Config Hash", metadata.config_hash[:16] if metadata.config_hash else ""),
        ("Compiled At", metadata.compiled_at),
    ]
    width = max(len(label) for label, _ in rows)
    return [f"  {label.ljust(width)} : {value}" for label, value in rows]


def _print_summary(metadata) -> None:
    print("Summary:")
    for line in _format_metadata_lines(metadata):
        print(line)


def _cmd_compile(args: argparse.Namespace) -> int:
    config_dir = Path(args.config_dir).resolve()
    if not config_dir.exists():
        raise FileNotFoundError(f"Config directory not found: {config_dir}")

    compiler = UniverseCompiler()
    start = time.perf_counter()
    compiled = compiler.compile(config_dir, use_cache=not args.no_cache)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    _print_summary(compiled.metadata)
    print(f"Compilation succeeded in {elapsed_ms:.1f} ms")

    if not args.no_cache:
        cache_path = config_dir / ".compiled" / "universe.msgpack"
        if cache_path.exists():
            print(f"Cache artifact written to: {cache_path}")

    return 0


def _convert_for_json(value):
    if isinstance(value, Mapping):
        return {k: _convert_for_json(v) for k, v in dict(value).items()}
    if isinstance(value, list | tuple):
        return [_convert_for_json(v) for v in value]
    return value


def _metadata_to_dict(metadata) -> dict:
    payload = {}
    for field in dataclasses.fields(metadata):
        payload[field.name] = _convert_for_json(getattr(metadata, field.name))
    return payload


def _cmd_inspect(args: argparse.Namespace) -> int:
    artifact_path = Path(args.artifact).resolve()

    # Auto-resolve config directory to artifact path for better UX
    if artifact_path.is_dir():
        artifact_path = artifact_path / ".compiled" / "universe.msgpack"

    if not artifact_path.exists():
        raise FileNotFoundError(f"Artifact not found: {artifact_path}")

    compiled = CompiledUniverse.load_from_cache(artifact_path)
    if args.format == "json":
        payload = {
            "artifact": str(artifact_path),
            "metadata": _metadata_to_dict(compiled.metadata),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _print_summary(compiled.metadata)
        print(f"Artifact path: {artifact_path}")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    config_dir = Path(args.config_dir).resolve()
    if not config_dir.exists():
        raise FileNotFoundError(f"Config directory not found: {config_dir}")

    compiler = UniverseCompiler()
    start = time.perf_counter()
    compiler.compile(config_dir, use_cache=False)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    print(f"Validation succeeded in {elapsed_ms:.1f} ms (no cache artifacts written)")
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        if args.command == "compile":
            return _cmd_compile(args)
        if args.command == "inspect":
            return _cmd_inspect(args)
        if args.command == "validate":
            return _cmd_validate(args)
    except CompilationError as exc:  # pragma: no cover - exercised via tests indirectly
        print(f"Compilation failed: {exc}", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
