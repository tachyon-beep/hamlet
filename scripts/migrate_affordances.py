#!/usr/bin/env python3
"""Migrate affordances.yaml from legacy format to effect_pipeline format."""

import sys
from pathlib import Path

import yaml


def migrate_affordance(aff):
    """Migrate a single affordance from legacy to modern format."""

    # Check if this affordance uses legacy format
    has_legacy_format = "effects_per_tick" in aff or "completion_bonus" in aff or "required_ticks" in aff

    if not has_legacy_format:
        return aff  # Already modern format

    # Extract legacy fields
    required_ticks = aff.pop("required_ticks", None)
    effects_per_tick = aff.pop("effects_per_tick", [])
    completion_bonus = aff.pop("completion_bonus", [])

    # Add multi_tick capability if required_ticks exists
    if required_ticks is not None:
        capabilities = aff.get("capabilities", [])

        # Add multi_tick capability
        multi_tick_cap = {"type": "multi_tick", "required_ticks": required_ticks}
        capabilities.append(multi_tick_cap)
        aff["capabilities"] = capabilities

    # Create effect_pipeline if we have effects
    if effects_per_tick or completion_bonus:
        pipeline = {}

        if effects_per_tick:
            pipeline["per_tick"] = effects_per_tick

        if completion_bonus:
            pipeline["on_completion"] = completion_bonus

        aff["effect_pipeline"] = pipeline

    return aff


def main():
    if len(sys.argv) != 2:
        print("Usage: migrate_affordances.py <affordances.yaml>")
        sys.exit(1)

    yaml_path = Path(sys.argv[1])

    if not yaml_path.exists():
        print(f"Error: {yaml_path} not found")
        sys.exit(1)

    # Load YAML
    with yaml_path.open() as f:
        data = yaml.safe_load(f)

    # Migrate each affordance
    if "affordances" in data:
        data["affordances"] = [migrate_affordance(aff) for aff in data["affordances"]]

    # Write back with proper formatting
    with yaml_path.open("w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False, width=120)

    print(f"✅ Migrated {yaml_path}")


if __name__ == "__main__":
    main()
