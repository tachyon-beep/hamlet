#!/usr/bin/env python3
"""Runtime validation script for substrate integration (TASK-002A Opportunity 1).

This script creates environments with each substrate configuration and verifies:
1. Observation dimensions match expected values
2. Position tensors have correct shape
3. Substrate methods work correctly
4. No runtime errors occur

Usage:
    python scripts/validate_substrate_runtime.py
    python scripts/validate_substrate_runtime.py --config configs/L1_full_observability
    python scripts/validate_substrate_runtime.py --verbose

Purpose:
    Catch substrate integration bugs BEFORE training starts, preventing wasted
    training time. Complements YAML validation (validate_substrate_configs.py)
    with runtime checks.
"""

import argparse
import sys
from pathlib import Path

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from townlet.environment.vectorized_env import VectorizedHamletEnv
from townlet.universe.compiler import UniverseCompiler


class SubstrateRuntimeValidator:
    """Runtime validator for substrate integration."""

    def log(self, message: str, level: str = "INFO"):
        """Log message if verbose mode enabled."""
        if self.verbose:
            prefix = "✓" if level == "INFO" else "⚠" if level == "WARN" else "✗"
            print(f"{prefix} {message}")

    def __init__(self, *, verbose: bool = False):
        """Initialize validator.

        Args:
            verbose: If True, print detailed validation steps
        """
        self.verbose = verbose
        self.errors = []
        self.warnings = []
        self.compiler = UniverseCompiler()

    def validate_config_pack(self, config_path: Path) -> bool:
        """Validate a single config pack.

        Args:
            config_path: Path to config pack directory

        Returns:
            True if validation passes, False otherwise
        """
        config_name = config_path.name
        self.log(f"Validating {config_name}...", "INFO")

        try:
            # Create environment
            universe = self.compiler.compile(config_path)
            env = VectorizedHamletEnv.from_universe(
                universe,
                num_agents=2,  # Test multi-agent
                device=torch.device("cpu"),  # Use CPU for validation
            )

            # Check 1: Substrate loaded correctly
            if not hasattr(env, "substrate"):
                self.errors.append(f"{config_name}: No substrate attribute")
                return False

            self.log(f"  Substrate type: {env.substrate.type}", "INFO")
            self.log(f"  Position dim: {env.substrate.position_dim}", "INFO")

            # Check 2: Position tensor shape
            env.reset()
            expected_pos_shape = (2, env.substrate.position_dim)
            actual_pos_shape = env.positions.shape

            if actual_pos_shape != expected_pos_shape:
                self.errors.append(f"{config_name}: Position shape mismatch. " f"Expected {expected_pos_shape}, got {actual_pos_shape}")
                return False

            self.log(f"  Position shape: {actual_pos_shape} ✓", "INFO")

            # Check 3: Observation dimension
            obs = env.reset()
            actual_obs_dim = obs.shape[1]

            # Expected dimension calculation
            if env.substrate.type == "grid2d":
                grid_dim = env.substrate.width * env.substrate.height
            elif env.substrate.type == "aspatial":
                grid_dim = 0
            else:
                grid_dim = 0  # Unknown substrate, skip check
                self.warnings.append(f"{config_name}: Unknown substrate type {env.substrate.type}")

            meter_dim = env.meter_count
            affordance_dim = 15  # 14 affordances + "none"
            temporal_dim = 4  # time_of_day, retirement_age, interaction_progress, interaction_ticks

            expected_obs_dim = grid_dim + meter_dim + affordance_dim + temporal_dim

            if actual_obs_dim != expected_obs_dim:
                self.errors.append(
                    f"{config_name}: Observation dim mismatch. "
                    f"Expected {expected_obs_dim} (grid={grid_dim} + meters={meter_dim} "
                    f"+ affordances={affordance_dim} + temporal={temporal_dim}), "
                    f"got {actual_obs_dim}"
                )
                return False

            self.log(f"  Observation dim: {actual_obs_dim} ✓", "INFO")

            # Check 4: Substrate methods work
            try:
                # Test initialize_positions
                positions = env.substrate.initialize_positions(num_agents=5, device=torch.device("cpu"))
                if positions.shape != (5, env.substrate.position_dim):
                    self.errors.append(f"{config_name}: initialize_positions() returned wrong shape: {positions.shape}")
                    return False

                self.log("  initialize_positions() ✓", "INFO")

                # Test apply_movement (if spatial)
                if env.substrate.position_dim > 0:
                    deltas = torch.zeros((5, env.substrate.position_dim), dtype=torch.long)
                    new_positions = env.substrate.apply_movement(positions, deltas)
                    if new_positions.shape != positions.shape:
                        self.errors.append(f"{config_name}: apply_movement() returned wrong shape: {new_positions.shape}")
                        return False

                    self.log("  apply_movement() ✓", "INFO")

                # Test get_all_positions
                all_positions = env.substrate.get_all_positions()
                if env.substrate.type == "grid2d":
                    expected_count = env.substrate.width * env.substrate.height
                    if len(all_positions) != expected_count:
                        self.errors.append(
                            f"{config_name}: get_all_positions() returned {len(all_positions)} positions, " f"expected {expected_count}"
                        )
                        return False

                self.log("  get_all_positions() ✓", "INFO")

            except Exception as e:
                self.errors.append(f"{config_name}: Substrate method error: {e}")
                return False

            # Check 5: Episode execution (smoke test)
            try:
                for _ in range(10):
                    actions = torch.randint(0, 5, (2,), dtype=torch.long)
                    obs, rewards, dones, info = env.step(actions, depletion_multiplier=1.0)

                    if obs.shape[0] != 2 or obs.shape[1] != actual_obs_dim:
                        self.errors.append(f"{config_name}: Step observation shape changed: {obs.shape}")
                        return False

                self.log("  Episode execution (10 steps) ✓", "INFO")

            except Exception as e:
                self.errors.append(f"{config_name}: Episode execution error: {e}")
                return False

            print(f"✓ {config_name}: PASS")
            return True

        except Exception as e:
            self.errors.append(f"{config_name}: Unexpected error: {e}")
            self.log(f"  ERROR: {e}", "ERROR")
            return False

    def validate_all_configs(self, config_dir: Path) -> bool:
        """Validate all config packs in directory.

        Args:
            config_dir: Path to configs directory

        Returns:
            True if all validations pass, False otherwise
        """
        # Find all config packs (directories with substrate.yaml)
        config_packs = []
        for path in config_dir.iterdir():
            if path.is_dir() and (path / "substrate.yaml").exists():
                # Skip template and aspatial_test (may not be complete)
                if path.name not in ["templates", "aspatial_test"]:
                    config_packs.append(path)

        if not config_packs:
            print(f"ERROR: No config packs found in {config_dir}")
            return False

        print(f"\nValidating {len(config_packs)} config packs...")
        print("=" * 60)

        success_count = 0
        for config_path in sorted(config_packs):
            if self.validate_config_pack(config_path):
                success_count += 1
            print()

        # Print summary
        print("=" * 60)
        print(f"Results: {success_count}/{len(config_packs)} config packs passed")

        if self.warnings:
            print(f"\nWarnings ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  ⚠ {warning}")

        if self.errors:
            print(f"\nErrors ({len(self.errors)}):")
            for error in self.errors:
                print(f"  ✗ {error}")

        return success_count == len(config_packs)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Runtime validation for substrate integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Validate all config packs
    python scripts/validate_substrate_runtime.py

    # Validate specific config pack
    python scripts/validate_substrate_runtime.py --config configs/L1_full_observability

    # Verbose output
    python scripts/validate_substrate_runtime.py --verbose
        """,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to specific config pack (default: validate all in configs/)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    validator = SubstrateRuntimeValidator(verbose=args.verbose)

    if args.config:
        # Validate single config pack
        success = validator.validate_config_pack(args.config)
    else:
        # Validate all config packs
        config_dir = Path(__file__).parent.parent / "configs"
        success = validator.validate_all_configs(config_dir)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
