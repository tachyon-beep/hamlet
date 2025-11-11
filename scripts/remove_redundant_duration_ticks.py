#!/usr/bin/env python3
"""Remove redundant duration_ticks from capabilities array (legacy migration artifact).

The runtime only uses root-level duration_ticks. The one inside capabilities[].duration_ticks
is ignored and causes validation errors with strict schema enforcement.

Usage:
    python scripts/remove_redundant_duration_ticks.py configs/L0_5_dual_resource/affordances.yaml
    python scripts/remove_redundant_duration_ticks.py configs/*/affordances.yaml  # Fix all
"""

import sys
from pathlib import Path

import yaml


def clean_affordance(aff):
    """Remove duration_ticks from capabilities array if present."""

    if "capabilities" not in aff:
        return aff

    capabilities = aff["capabilities"]
    if not capabilities:
        return aff

    # Remove duration_ticks from all multi_tick capabilities
    for cap in capabilities:
        if isinstance(cap, dict) and cap.get("type") == "multi_tick":
            if "duration_ticks" in cap:
                del cap["duration_ticks"]
                print(f"  - Removed duration_ticks from {aff['name']} capabilities")

    return aff


def main():
    if len(sys.argv) < 2:
        print("Usage: remove_redundant_duration_ticks.py <affordances.yaml> [...]")
        print("Example: python scripts/remove_redundant_duration_ticks.py configs/*/affordances.yaml")
        sys.exit(1)

    yaml_paths = [Path(p) for p in sys.argv[1:]]

    for yaml_path in yaml_paths:
        if not yaml_path.exists():
            print(f"⚠️  Skipping {yaml_path} (not found)")
            continue

        print(f"\n🔧 Processing {yaml_path}")

        # Load YAML
        with yaml_path.open() as f:
            data = yaml.safe_load(f)

        # Clean each affordance
        if "affordances" in data:
            data["affordances"] = [clean_affordance(aff) for aff in data["affordances"]]

        # Write back with proper formatting
        with yaml_path.open("w") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False, width=120)

        print(f"✅ Cleaned {yaml_path}")

    print(f"\n✅ Done! Processed {len(yaml_paths)} file(s)")


if __name__ == "__main__":
    main()
