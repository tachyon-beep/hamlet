#!/usr/bin/env python3
"""Remove redundant 'meters' section from affordances.yaml files.

The meters are already defined in bars.yaml and the compiler uses that as the
single source of truth. The meters section in affordances.yaml is ignored and
creates unnecessary duplication.
"""

import sys
from pathlib import Path

import yaml


def cleanup_affordances_file(yaml_path: Path) -> bool:
    """Remove meters section from affordances.yaml.

    Args:
        yaml_path: Path to affordances.yaml file

    Returns:
        True if file was modified, False otherwise
    """
    if not yaml_path.exists():
        print(f"⚠️  Skipped: {yaml_path} (not found)")
        return False

    # Load YAML
    with yaml_path.open() as f:
        data = yaml.safe_load(f)

    # Check if meters section exists
    if "meters" not in data:
        print(f"⏭️  Skipped: {yaml_path} (no meters section)")
        return False

    # Remove meters section
    meters_count = len(data["meters"]) if isinstance(data["meters"], list) else 0
    del data["meters"]

    # Write back with proper formatting
    with yaml_path.open("w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False, width=120)

    print(f"✅ Cleaned: {yaml_path} (removed {meters_count} meter definitions)")
    return True


def main():
    """Clean all affordances.yaml files in configs/."""

    if len(sys.argv) > 1:
        # Clean specific file
        yaml_path = Path(sys.argv[1])
        cleanup_affordances_file(yaml_path)
    else:
        # Clean all configs
        configs_dir = Path(__file__).parent.parent / "configs"
        modified_count = 0

        for affordances_file in configs_dir.glob("*/affordances.yaml"):
            if cleanup_affordances_file(affordances_file):
                modified_count += 1

        print(f"\n📊 Summary: Modified {modified_count} files")


if __name__ == "__main__":
    main()
