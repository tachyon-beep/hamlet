#!/usr/bin/env python3
"""Validate config packs by invoking the CLI compiler."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_ROOT = REPO_ROOT / "configs"
# `aspatial_test` is a trimmed-down pack used by substrate unit tests; it intentionally violates
# several Stage 4 assumptions (no spatial layout, partial schema), so the CLI validator would fail
# every run. We skip it here to keep CI signal clean while still covering all real packs.
EXCLUDED_DIRS = {"templates", "aspatial_test"}


def iter_config_dirs(base: Path) -> list[Path]:
    dirs: list[Path] = []
    for entry in sorted(base.iterdir()):
        if not entry.is_dir() or entry.name in EXCLUDED_DIRS:
            continue
        if (entry / "training.yaml").exists():
            dirs.append(entry)
    return dirs


def run_cli_validate(config_dir: Path) -> None:
    cmd = [sys.executable, "-m", "townlet.compiler", "validate", str(config_dir)]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate config packs via python -m townlet.compiler validate")
    parser.add_argument("config", nargs="?", help="Optional single config directory to validate")
    args = parser.parse_args()

    if args.config:
        config_dirs = [Path(args.config).resolve()]
    else:
        config_dirs = iter_config_dirs(CONFIGS_ROOT)

    if not config_dirs:
        print("No config packs found for validation", file=sys.stderr)
        return 1

    for config_dir in config_dirs:
        try:
            display_path = config_dir.relative_to(REPO_ROOT)
        except ValueError:
            display_path = config_dir
        print(f"🔧 Validating {display_path} via CLI ...")
        run_cli_validate(config_dir)

    print("✅ Universe compiler CLI validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
