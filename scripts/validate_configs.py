#!/usr/bin/env python3
"""Validate all config packs using HamletConfig DTOs.

This script validates that all configuration packs in configs/ directory
conform to the HamletConfig DTO schema, enforcing the no-defaults principle.

Exit codes:
    0: All configs valid
    1: One or more configs failed validation
    2: Script error (missing dependencies, etc.)

Usage:
    python scripts/validate_configs.py
    python scripts/validate_configs.py --verbose
    python scripts/validate_configs.py --config configs/L0_0_minimal
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from townlet.config import HamletConfig


def find_config_packs(configs_dir: Path) -> list[Path]:
    """Find all config pack directories.

    A config pack is a directory containing training.yaml.

    Args:
        configs_dir: Base configs directory

    Returns:
        List of config pack directory paths
    """
    config_packs = []
    for item in configs_dir.iterdir():
        if item.is_dir() and (item / "training.yaml").exists():
            # Skip templates directory (not a real config pack)
            if item.name == "templates":
                continue
            config_packs.append(item)
    return sorted(config_packs)


def validate_config(config_dir: Path) -> tuple[bool, str]:
    """Validate a single config pack.

    Args:
        config_dir: Config pack directory

    Returns:
        (success, message) tuple
    """
    try:
        config = HamletConfig.load(config_dir)
        return True, f"Valid ({config.training.device}, {config.environment.grid_size}×{config.environment.grid_size})"
    except Exception as e:
        # Extract first line of error for concise reporting
        error_msg = str(e).split("\n")[0]
        return False, f"FAILED: {error_msg[:100]}"


def main():
    """Main validation entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate HAMLET config packs")
    parser.add_argument(
        "--config",
        type=Path,
        help="Validate single config pack (default: validate all in configs/)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed validation errors",
    )
    args = parser.parse_args()

    # Determine which configs to validate
    if args.config:
        if not args.config.exists():
            print(f"❌ Config directory not found: {args.config}", file=sys.stderr)
            return 2
        config_packs = [args.config]
    else:
        configs_dir = Path(__file__).parent.parent / "configs"
        config_packs = find_config_packs(configs_dir)
        if not config_packs:
            print(f"❌ No config packs found in {configs_dir}", file=sys.stderr)
            return 2

    print(f"🔍 Validating {len(config_packs)} config pack(s)...\n")

    # Validate each config
    results = []
    for config_dir in config_packs:
        success, message = validate_config(config_dir)
        results.append((config_dir.name, success, message))

        # Print result
        icon = "✅" if success else "❌"
        print(f"{icon} {config_dir.name:<35} {message}")

        # Print detailed error if verbose and failed
        if not success and args.verbose:
            try:
                HamletConfig.load(config_dir)
            except Exception as e:
                print(f"    {str(e)[:500]}\n")

    # Summary
    passed = sum(1 for _, success, _ in results if success)
    failed = len(results) - passed
    print(f"\n📊 Summary: {passed} passed, {failed} failed")

    # Exit code
    if failed > 0:
        print(f"\n❌ Validation FAILED ({failed} configs with errors)")
        return 1
    else:
        print("\n✅ All configs valid!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
