#!/usr/bin/env python3
"""
Quick validation script for bars.yaml and cascades.yaml

Tests:
1. YAML files load without errors
2. All required fields are present
3. Values match current meter_dynamics.py implementation
4. No schema violations
"""

from pathlib import Path

import yaml


def load_yaml(filepath: Path) -> dict:
    """Load YAML file and return parsed dict."""
    with open(filepath) as f:
        return yaml.safe_load(f)


def validate_bars_yaml(config: dict) -> list[str]:
    """Validate bars.yaml structure and values."""
    errors = []

    # Check required top-level keys
    required_keys = ["version", "description", "bars", "terminal_conditions"]
    for key in required_keys:
        if key not in config:
            errors.append(f"Missing required key: {key}")

    # Check meters
    bars = config.get("bars", [])
    if len(bars) != 8:
        errors.append(f"Expected 8 bars, found {len(bars)}")

    # Validate each bar
    expected_indices = {0, 1, 2, 3, 4, 5, 6, 7}
    found_indices = set()

    for bar in bars:
        name = bar.get("name", "UNKNOWN")
        idx = bar.get("index")
        found_indices.add(idx)

        # Check required fields
        required_bar_keys = [
            "name",
            "index",
            "tier",
            "range",
            "initial",
            "base_depletion",
            "description",
        ]
        for key in required_bar_keys:
            if key not in bar:
                errors.append(f"Bar {name}: Missing required key '{key}'")

        # Validate range
        if "range" in bar:
            r = bar["range"]
            if r != [0.0, 1.0]:
                errors.append(f"Bar {name}: range should be [0.0, 1.0], got {r}")

    # Check all indices present
    if found_indices != expected_indices:
        missing = expected_indices - found_indices
        errors.append(f"Missing bar indices: {missing}")

    # Check terminal conditions
    terminal = config.get("terminal_conditions", [])
    if len(terminal) != 2:
        errors.append(f"Expected 2 terminal conditions, found {len(terminal)}")

    return errors


def validate_cascades_yaml(config: dict) -> list[str]:
    """Validate cascades.yaml structure and values."""
    errors = []

    # Check required top-level keys
    required_keys = [
        "version",
        "description",
        "math_type",
        "modulations",
        "cascades",
        "execution_order",
    ]
    for key in required_keys:
        if key not in config:
            errors.append(f"Missing required key: {key}")

    # Check math type
    if config.get("math_type") != "gradient_penalty":
        errors.append(f"Expected math_type='gradient_penalty', got {config.get('math_type')}")

    # Check modulations
    modulations = config.get("modulations", [])
    if len(modulations) != 1:
        errors.append(f"Expected 1 modulation (fitness_health), found {len(modulations)}")

    # Check cascades
    cascades = config.get("cascades", [])
    if len(cascades) != 10:
        errors.append(f"Expected 10 cascades, found {len(cascades)}")

    # Validate each cascade
    for cascade in cascades:
        name = cascade.get("name", "UNKNOWN")

        required_cascade_keys = [
            "name",
            "description",
            "category",
            "source",
            "source_index",
            "target",
            "target_index",
            "threshold",
            "strength",
        ]
        for key in required_cascade_keys:
            if key not in cascade:
                errors.append(f"Cascade {name}: Missing required key '{key}'")

        # Check threshold
        if "threshold" in cascade and cascade["threshold"] != 0.3:
            errors.append(f"Cascade {name}: Expected threshold=0.3, got {cascade['threshold']}")

    return errors


def validate_values_match_code(bars_config: dict, cascades_config: dict) -> list[str]:
    """Validate that YAML values match meter_dynamics.py."""
    errors = []

    # Expected base depletion rates from meter_dynamics.py line 43-52
    expected_depletions = {
        "energy": 0.005,
        "hygiene": 0.003,
        "satiation": 0.004,
        "money": 0.0,
        "mood": 0.001,
        "social": 0.006,
        "health": 0.001,  # baseline before modulation
        "fitness": 0.002,
    }

    # Check bars
    bars = {bar["name"]: bar for bar in bars_config.get("bars", [])}
    for name, expected_depletion in expected_depletions.items():
        if name in bars:
            actual = bars[name]["base_depletion"]
            if actual != expected_depletion:
                errors.append(f"Bar {name}: Expected base_depletion={expected_depletion}, got {actual}")

    # Check cascade strengths from meter_dynamics.py
    expected_strengths = {
        "satiation_to_health": 0.004,
        "satiation_to_energy": 0.005,
        "mood_to_energy": 0.005,
        "hygiene_to_satiation": 0.002,
        "hygiene_to_fitness": 0.002,
        "hygiene_to_mood": 0.003,
        "social_to_mood": 0.004,
        "hygiene_to_health": 0.0005,
        "hygiene_to_energy": 0.0005,
        "social_to_energy": 0.0008,
    }

    cascades = {c["name"]: c for c in cascades_config.get("cascades", [])}
    for name, expected_strength in expected_strengths.items():
        if name in cascades:
            actual = cascades[name]["strength"]
            if actual != expected_strength:
                errors.append(f"Cascade {name}: Expected strength={expected_strength}, got {actual}")

    return errors


def main():
    """Run all validations."""
    print("🔍 Validating YAML configuration files...")
    print()

    config_dir = Path(__file__).parent.parent / "configs"
    bars_path = config_dir / "bars.yaml"
    cascades_path = config_dir / "cascades.yaml"

    # Check files exist
    if not bars_path.exists():
        print(f"❌ ERROR: {bars_path} not found")
        return 1
    if not cascades_path.exists():
        print(f"❌ ERROR: {cascades_path} not found")
        return 1

    # Load YAML files
    try:
        bars_config = load_yaml(bars_path)
        print("✅ bars.yaml loaded successfully")
    except Exception as e:
        print(f"❌ ERROR loading bars.yaml: {e}")
        return 1

    try:
        cascades_config = load_yaml(cascades_path)
        print("✅ cascades.yaml loaded successfully")
    except Exception as e:
        print(f"❌ ERROR loading cascades.yaml: {e}")
        return 1

    print()

    # Validate structure
    print("🔍 Validating bars.yaml structure...")
    bars_errors = validate_bars_yaml(bars_config)
    if bars_errors:
        print(f"❌ Found {len(bars_errors)} error(s):")
        for error in bars_errors:
            print(f"   - {error}")
    else:
        print("✅ bars.yaml structure valid")

    print()

    print("🔍 Validating cascades.yaml structure...")
    cascades_errors = validate_cascades_yaml(cascades_config)
    if cascades_errors:
        print(f"❌ Found {len(cascades_errors)} error(s):")
        for error in cascades_errors:
            print(f"   - {error}")
    else:
        print("✅ cascades.yaml structure valid")

    print()

    print("🔍 Validating values match meter_dynamics.py...")
    value_errors = validate_values_match_code(bars_config, cascades_config)
    if value_errors:
        print(f"❌ Found {len(value_errors)} mismatch(es):")
        for error in value_errors:
            print(f"   - {error}")
    else:
        print("✅ All values match meter_dynamics.py implementation")

    print()

    # Summary
    total_errors = len(bars_errors) + len(cascades_errors) + len(value_errors)
    if total_errors == 0:
        print("🎉 SUCCESS! All validations passed.")
        print()
        print("Next steps:")
        print("  1. Create Pydantic models for type-safe loading")
        print("  2. Build cascade_config.py loader")
        print("  3. Write comprehensive tests")
        return 0
    else:
        print(f"❌ FAILED: {total_errors} total error(s) found")
        return 1


if __name__ == "__main__":
    exit(main())
