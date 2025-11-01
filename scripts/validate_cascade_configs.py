#!/usr/bin/env python3
"""
Test script to validate teaching example cascade configs.
"""

from pathlib import Path
import yaml
from pydantic import ValidationError

from townlet.environment.cascade_config import (
    BarsConfig,
    CascadesConfig,
    EnvironmentConfig,
    load_bars_config,
)


def validate_cascade_config(cascade_path: Path) -> bool:
    """Validate a cascade YAML file."""
    try:
        with open(cascade_path) as f:
            data = yaml.safe_load(f)

        # Validate with Pydantic
        config = CascadesConfig(**data)

        print(f"✅ {cascade_path.name} is VALID")
        print(f"   Version: {config.version}")
        print(f"   Description: {config.description}")
        print(f"   Modulations: {len(config.modulations)}")
        print(f"   Cascades: {len(config.cascades)}")

        if config.modulations:
            mod = config.modulations[0]
            print(f"   Fitness modulation: base={mod.base_multiplier}, range={mod.range}")

        return True
    except ValidationError as e:
        print(f"❌ {cascade_path.name} VALIDATION ERROR:")
        print(e)
        return False
    except Exception as e:
        print(f"❌ {cascade_path.name} ERROR: {e}")
        return False


def main():
    configs_dir = Path("configs")

    print("Validating teaching example cascade configs...\n")

    # Validate all cascade configs
    cascade_files = [
        "cascades.yaml",
        "cascades_weak.yaml",
        "cascades_strong.yaml",
    ]

    results = []
    for filename in cascade_files:
        path = configs_dir / filename
        if path.exists():
            valid = validate_cascade_config(path)
            results.append((filename, valid))
            print()
        else:
            print(f"⚠️  {filename} not found")
            results.append((filename, False))
            print()

    # Summary
    print("=" * 60)
    print("SUMMARY:")
    print("=" * 60)

    all_valid = all(valid for _, valid in results)
    for filename, valid in results:
        status = "✅ VALID" if valid else "❌ INVALID"
        print(f"{status:12} {filename}")

    if all_valid:
        print("\n🎉 All teaching configs are valid!")
        return 0
    else:
        print("\n❌ Some configs have errors")
        return 1


if __name__ == "__main__":
    exit(main())
