#!/usr/bin/env python3
"""Validate substrate.yaml configs for all config packs.

Pre-training validation tool for operators. Run before starting experiments
to catch configuration errors early.

Usage:
    python scripts/validate_substrate_configs.py
    python scripts/validate_substrate_configs.py --config-pack L1_full_observability
    python scripts/validate_substrate_configs.py --verbose

Exit codes:
    0: All configs valid
    1: Validation errors found
"""

import sys
from pathlib import Path

import torch

from townlet.substrate.config import load_substrate_config
from townlet.substrate.factory import SubstrateFactory


def validate_config_pack(config_pack_path: Path, verbose: bool = False) -> tuple[bool, list[str]]:
    """Validate substrate.yaml for a single config pack.

    Args:
        config_pack_path: Path to config pack directory
        verbose: Print detailed validation steps

    Returns:
        (is_valid, errors): True if valid, list of error messages
    """
    errors = []
    config_name = config_pack_path.name

    if verbose:
        print(f"\n{'='*60}")
        print(f"Validating: {config_name}")
        print(f"{'='*60}")

    # Check substrate.yaml exists
    substrate_path = config_pack_path / "substrate.yaml"
    if not substrate_path.exists():
        errors.append(f"Missing substrate.yaml in {config_name}")
        return False, errors

    if verbose:
        print("✓ Found substrate.yaml")

    # Load and validate schema
    try:
        config = load_substrate_config(substrate_path)
        if verbose:
            print("✓ Schema validation passed")
            print(f"  Type: {config.type}")
            if config.type == "grid":
                print(f"  Grid: {config.grid.width}×{config.grid.height}")
                print(f"  Boundary: {config.grid.boundary}")
                print(f"  Distance: {config.grid.distance_metric}")
    except Exception as e:
        errors.append(f"Schema validation failed for {config_name}: {e}")
        return False, errors

    # Verify factory can build substrate
    try:
        substrate = SubstrateFactory.build(config, device=torch.device("cpu"))
        if verbose:
            print("✓ Factory build successful")
            print(f"  Position dim: {substrate.position_dim}")
            print(f"  Observation dim: {substrate.get_observation_dim()}")
    except Exception as e:
        errors.append(f"Factory build failed for {config_name}: {e}")
        return False, errors

    # Verify substrate operations work
    try:
        if substrate.position_dim > 0:
            # Test position initialization
            positions = substrate.initialize_positions(num_agents=1, device=torch.device("cpu"))
            assert positions.shape[0] == 1
            assert positions.shape[1] == substrate.position_dim

            # Test movement
            deltas = torch.zeros(1, substrate.position_dim, dtype=torch.long)
            new_positions = substrate.apply_movement(positions, deltas)
            assert new_positions.shape == positions.shape

            # Test distance
            distance = substrate.compute_distance(positions, positions[0])
            assert distance.shape[0] == 1

            if verbose:
                print("✓ Substrate operations verified")
        else:
            if verbose:
                print("✓ Aspatial substrate (no position operations)")
    except Exception as e:
        errors.append(f"Substrate operations failed for {config_name}: {e}")
        return False, errors

    if verbose:
        print(f"\n✅ {config_name}: VALID")

    return True, []


def main():
    """Main validation logic."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate substrate.yaml configs")
    parser.add_argument(
        "--config-pack",
        type=str,
        help="Validate specific config pack (default: all)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed validation steps",
    )
    args = parser.parse_args()

    # Determine which config packs to validate
    project_root = Path(__file__).parent.parent
    configs_dir = project_root / "configs"

    if args.config_pack:
        config_packs = [configs_dir / args.config_pack]
        if not config_packs[0].exists():
            print(f"❌ Config pack not found: {args.config_pack}")
            return 1
    else:
        # Validate all production config packs
        config_packs = [
            configs_dir / "L0_0_minimal",
            configs_dir / "L0_5_dual_resource",
            configs_dir / "L1_full_observability",
            configs_dir / "L2_partial_observability",
            configs_dir / "L3_temporal_mechanics",
            configs_dir / "test",
        ]

    print(f"Validating {len(config_packs)} config pack(s)...\n")

    # Validate each config pack
    all_valid = True
    all_errors = []

    for config_pack_path in config_packs:
        is_valid, errors = validate_config_pack(config_pack_path, verbose=args.verbose)

        if not is_valid:
            all_valid = False
            all_errors.extend(errors)

        if not args.verbose:
            status = "✅ VALID" if is_valid else "❌ INVALID"
            print(f"{config_pack_path.name:30s} {status}")

    # Print summary
    print(f"\n{'='*60}")
    if all_valid:
        print("✅ All configs valid!")
        print(f"{'='*60}")
        return 0
    else:
        print("❌ Validation errors found:")
        for error in all_errors:
            print(f"  - {error}")
        print(f"{'='*60}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
