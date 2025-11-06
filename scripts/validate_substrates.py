#!/usr/bin/env python3
"""Substrate configuration validator - lint tool for substrate.yaml files.

Validates substrate configs are well-formed and instantiable. Designed as a
fast pre-commit lint tool and foundation for general config validation.

What this validates (config-level):
  ✓ substrate.yaml exists
  ✓ YAML schema is valid (Pydantic)
  ✓ Metadata extraction is implemented for substrate type
  ✓ Substrate can be instantiated (factory build)
  ✓ Basic substrate operations work (position init, movement, distance)

What this does NOT validate (belongs in test suite):
  ✗ Environment integration (see tests/test_townlet/unit/test_env_substrate_loading.py)
  ✗ Episode execution (see tests/test_townlet/integration/test_substrate_migration.py)
  ✗ Observation dimensions (see tests/test_townlet/integration/test_data_flows.py)

Adding a new substrate type (e.g., graph, hexgrid):
  1. Update src/townlet/substrate/config.py - add GraphSubstrateConfig and update SubstrateConfig
  2. Update src/townlet/substrate/factory.py - add build logic for new type
  3. Update scripts/validate_substrates.py (THIS FILE) - add metadata extraction (line ~167)
     ⚠️  CI will FAIL if you forget step 3 - this is intentional!

Vision: Foundation for general config linter that validates all YAML configs
        (substrate.yaml, bars.yaml, cascades.yaml, affordances.yaml, etc.)

Usage:
    # Validate all production configs (default)
    python scripts/validate_substrates.py

    # Validate specific config
    python scripts/validate_substrates.py --config configs/L1_full_observability

    # CI mode (JSON output, no colors, fail fast)
    python scripts/validate_substrates.py --ci --json results.json

    # Parallel execution (faster for multiple configs)
    python scripts/validate_substrates.py --parallel 4

Exit codes:
    0: All validations passed
    1: Validation errors found
    2: Invalid arguments or setup error
"""

import argparse
import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from townlet.substrate.config import load_substrate_config
from townlet.substrate.factory import SubstrateFactory


class CheckStatus(Enum):
    """Individual check status."""

    PASS = "✓"
    FAIL = "✗"
    SKIP = "○"


@dataclass
class ValidationResult:
    """Result of validating a single config pack."""

    config_name: str
    success: bool
    duration_ms: float
    checks: dict[str, tuple[CheckStatus, str]] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_check(self, name: str, status: CheckStatus, message: str = ""):
        """Add a check result."""
        self.checks[name] = (status, message)

    def add_error(self, message: str):
        """Add an error."""
        self.errors.append(message)
        self.success = False

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "config_name": self.config_name,
            "success": self.success,
            "duration_ms": self.duration_ms,
            "checks": {name: {"status": status.value, "message": msg} for name, (status, msg) in self.checks.items()},
            "errors": self.errors,
            "metadata": self.metadata,
        }


class SubstrateValidator:
    """Substrate configuration validator (lint tool)."""

    def __init__(self, verbose: bool = False, use_colors: bool = True):
        """Initialize validator.

        Args:
            verbose: Print detailed progress
            use_colors: Use ANSI colors in output
        """
        self.verbose = verbose
        self.use_colors = use_colors

    def log(self, message: str, level: str = "INFO"):
        """Log message if verbose mode enabled."""
        if not self.verbose:
            return

        if self.use_colors:
            colors = {"INFO": "\033[36m", "WARN": "\033[33m", "ERROR": "\033[31m", "SUCCESS": "\033[32m", "RESET": "\033[0m"}
            color = colors.get(level, "")
            reset = colors["RESET"]
        else:
            color = reset = ""

        prefix = {"INFO": "ℹ", "WARN": "⚠", "ERROR": "✗", "SUCCESS": "✓"}.get(level, "•")
        print(f"{color}{prefix} {message}{reset}")

    def validate_config_pack(self, config_path: Path) -> ValidationResult:
        """Validate substrate.yaml for a config pack.

        Progressive validation pipeline (fail-fast):
        1. Check substrate.yaml exists
        2. Validate YAML schema (Pydantic)
        3. Build substrate via factory
        4. Test basic operations (position init, movement, distance)

        Args:
            config_path: Path to config pack directory

        Returns:
            ValidationResult with check results and metadata
        """
        start_time = time.time()
        config_name = config_path.name
        result = ValidationResult(config_name=config_name, success=True, duration_ms=0.0)

        self.log(f"Validating {config_name}...", "INFO")

        try:
            # Check 1: substrate.yaml exists
            substrate_path = config_path / "substrate.yaml"
            if not substrate_path.exists():
                result.add_check("substrate_yaml_exists", CheckStatus.FAIL, "File not found")
                result.add_error(f"Missing substrate.yaml in {config_name}")
                result.duration_ms = (time.time() - start_time) * 1000
                return result

            result.add_check("substrate_yaml_exists", CheckStatus.PASS)
            self.log("  Found substrate.yaml", "INFO")

            # Check 2: YAML schema validation (Pydantic)
            try:
                config = load_substrate_config(substrate_path)
                result.add_check("schema_validation", CheckStatus.PASS)
                result.metadata["substrate_type"] = config.type

                # Extract type-specific metadata (ADD NEW TYPES HERE)
                if config.type == "grid":
                    result.metadata["grid_size"] = f"{config.grid.width}×{config.grid.height}"
                    result.metadata["boundary"] = config.grid.boundary
                    result.metadata["distance_metric"] = config.grid.distance_metric
                elif config.type == "aspatial":
                    # Aspatial has no specific config to extract
                    pass
                else:
                    # Unknown substrate type - developer must add metadata extraction
                    result.add_check("metadata_extraction", CheckStatus.FAIL, f"Unknown type: {config.type}")
                    result.add_error(
                        f"Unknown substrate type '{config.type}' - please update scripts/validate_substrates.py "
                        f"to extract metadata for this type (see line ~167)"
                    )
                    result.duration_ms = (time.time() - start_time) * 1000
                    return result

                self.log(f"  Schema valid (type: {config.type})", "SUCCESS")
            except Exception as e:
                result.add_check("schema_validation", CheckStatus.FAIL, str(e))
                result.add_error(f"Schema validation failed: {e}")
                result.duration_ms = (time.time() - start_time) * 1000
                return result

            # Check 3: Factory can build substrate
            try:
                substrate = SubstrateFactory.build(config, device=torch.device("cpu"))
                result.add_check("factory_build", CheckStatus.PASS)
                result.metadata["position_dim"] = substrate.position_dim
                self.log(f"  Factory build successful (pos_dim={substrate.position_dim})", "SUCCESS")
            except Exception as e:
                result.add_check("factory_build", CheckStatus.FAIL, str(e))
                result.add_error(f"Factory build failed: {e}")
                result.duration_ms = (time.time() - start_time) * 1000
                return result

            # Check 4: Basic substrate operations (if spatial)
            try:
                if substrate.position_dim > 0:
                    # Test position initialization
                    positions = substrate.initialize_positions(num_agents=2, device=torch.device("cpu"))
                    assert positions.shape == (2, substrate.position_dim), f"Position shape mismatch: {positions.shape}"

                    # Test movement
                    deltas = torch.zeros(2, substrate.position_dim, dtype=torch.long)
                    new_positions = substrate.apply_movement(positions, deltas)
                    assert new_positions.shape == positions.shape, f"Movement shape mismatch: {new_positions.shape}"

                    # Test distance
                    distance = substrate.compute_distance(positions, positions[0])
                    assert distance.shape[0] == 2, f"Distance shape mismatch: {distance.shape}"

                    result.add_check("substrate_operations", CheckStatus.PASS)
                    self.log("  Substrate operations verified", "SUCCESS")
                else:
                    result.add_check("substrate_operations", CheckStatus.SKIP, "Aspatial (no position ops)")
                    self.log("  Aspatial substrate (position operations skipped)", "INFO")
            except Exception as e:
                result.add_check("substrate_operations", CheckStatus.FAIL, str(e))
                result.add_error(f"Substrate operations failed: {e}")
                result.duration_ms = (time.time() - start_time) * 1000
                return result

            # All checks passed
            result.duration_ms = (time.time() - start_time) * 1000
            self.log(f"  ✅ All checks passed ({result.duration_ms:.0f}ms)", "SUCCESS")
            return result

        except Exception as e:
            # Unexpected error (should not happen with proper check structure)
            result.add_check("unexpected_error", CheckStatus.FAIL, str(e))
            result.add_error(f"Unexpected error: {e}")
            result.duration_ms = (time.time() - start_time) * 1000
            return result


def validate_single_config(config_path: Path, verbose: bool, use_colors: bool) -> ValidationResult:
    """Validate a single config (for parallel execution).

    Args:
        config_path: Path to config pack
        verbose: Verbose output
        use_colors: Use ANSI colors

    Returns:
        ValidationResult
    """
    validator = SubstrateValidator(verbose=verbose, use_colors=use_colors)
    return validator.validate_config_pack(config_path)


def print_summary(results: list[ValidationResult], use_colors: bool):
    """Print validation summary.

    Args:
        results: List of validation results
        use_colors: Use ANSI colors
    """
    if use_colors:
        green = "\033[32m"
        red = "\033[31m"
        reset = "\033[0m"
    else:
        green = red = reset = ""

    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    passed = sum(1 for r in results if r.success)
    failed = sum(1 for r in results if not r.success)
    total = len(results)
    total_duration = sum(r.duration_ms for r in results)
    avg_duration = total_duration / total if total > 0 else 0

    # Print individual results
    for result in results:
        status_symbol = f"{green}✓{reset}" if result.success else f"{red}✗{reset}"
        duration_str = f"{result.duration_ms:.0f}ms"
        print(f"{status_symbol} {result.config_name:30s} {duration_str:>8s}")

        # Show errors if any
        if result.errors:
            for error in result.errors:
                print(f"    {red}ERROR:{reset} {error}")

    # Print summary statistics
    print("=" * 70)
    print(f"Total:    {total}")
    print(f"Passed:   {green}{passed}{reset}")
    print(f"Failed:   {red}{failed}{reset}")
    print(f"Duration: {total_duration:.0f}ms (avg: {avg_duration:.0f}ms)")

    # Print check breakdown
    check_names = set()
    for result in results:
        check_names.update(result.checks.keys())

    if check_names:
        print("\nCheck Results:")
        for check_name in sorted(check_names):
            pass_count = sum(1 for r in results if check_name in r.checks and r.checks[check_name][0] == CheckStatus.PASS)
            fail_count = sum(1 for r in results if check_name in r.checks and r.checks[check_name][0] == CheckStatus.FAIL)
            skip_count = sum(1 for r in results if check_name in r.checks and r.checks[check_name][0] == CheckStatus.SKIP)

            status_str = f"{green}{pass_count}✓{reset}" if pass_count == total else f"{pass_count}✓"
            if fail_count > 0:
                status_str += f" {red}{fail_count}✗{reset}"
            if skip_count > 0:
                status_str += f" {skip_count}○"

            print(f"  {check_name:30s} {status_str}")

    print("=" * 70)

    if passed == total:
        print(f"{green}✅ ALL VALIDATIONS PASSED{reset}")
    else:
        print(f"{red}❌ {failed} VALIDATION(S) FAILED{reset}")

    print("=" * 70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Substrate configuration validator (lint tool)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
What this validates:
  ✓ substrate.yaml exists
  ✓ YAML schema is valid (Pydantic)
  ✓ Substrate can be instantiated (factory build)
  ✓ Basic substrate operations work (position init, movement, distance)

What this does NOT validate (use test suite):
  ✗ Environment integration → pytest tests/test_townlet/unit/test_env_substrate_loading.py
  ✗ Episode execution → pytest tests/test_townlet/integration/test_substrate_migration.py
  ✗ Observation dimensions → pytest tests/test_townlet/integration/test_data_flows.py

Examples:
  # Validate all configs
  python scripts/validate_substrates.py

  # Validate specific config
  python scripts/validate_substrates.py --config configs/L1_full_observability

  # CI mode (JSON output, no colors)
  python scripts/validate_substrates.py --ci --json results.json

  # Parallel execution (4 workers)
  python scripts/validate_substrates.py --parallel 4
        """,
    )

    parser.add_argument("--config", type=Path, default=None, help="Validate specific config pack (default: all production configs)")

    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output (show all checks)")

    parser.add_argument("--ci", action="store_true", help="CI mode (no colors, machine-readable)")

    parser.add_argument("--json", type=Path, default=None, help="Write results to JSON file")

    parser.add_argument("--parallel", type=int, default=1, help="Number of parallel workers (default: 1, use 0 for auto)")

    parser.add_argument("--no-colors", action="store_true", help="Disable ANSI colors")

    args = parser.parse_args()

    # Determine color usage
    use_colors = not args.no_colors and not args.ci and sys.stdout.isatty()

    # Find config packs to validate
    project_root = Path(__file__).parent.parent
    configs_dir = project_root / "configs"

    if args.config:
        if not args.config.exists():
            print(f"❌ Config pack not found: {args.config}", file=sys.stderr)
            return 2
        config_packs = [args.config]
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

        # Filter to only existing packs with substrate.yaml
        config_packs = [p for p in config_packs if p.exists() and (p / "substrate.yaml").exists()]

    if not config_packs:
        print("❌ No config packs found to validate", file=sys.stderr)
        return 2

    print(f"Validating {len(config_packs)} config pack(s)...")
    if args.parallel > 1 or args.parallel == 0:
        workers = args.parallel if args.parallel > 0 else mp.cpu_count()
        print(f"Using {workers} parallel workers")

    # Validate configs (parallel or sequential)
    results = []

    if args.parallel > 1 or args.parallel == 0:
        # Parallel execution
        workers = args.parallel if args.parallel > 0 else mp.cpu_count()

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(validate_single_config, config_path, args.verbose, use_colors): config_path for config_path in config_packs
            }

            for future in as_completed(futures):
                config_path = futures[future]
                try:
                    result = future.result(timeout=30)  # 30 second timeout (faster than before)
                    results.append(result)
                    if not args.verbose and not args.ci:
                        status = "✓ PASS" if result.success else "✗ FAIL"
                        print(f"{config_path.name:30s} {status}")
                except Exception as e:
                    error_result = ValidationResult(config_name=config_path.name, success=False, duration_ms=0.0)
                    error_result.add_error(f"Validation failed: {e}")
                    results.append(error_result)
                    print(f"{config_path.name:30s} ✗ FAIL ({e})")
    else:
        # Sequential execution
        for config_path in config_packs:
            result = validate_single_config(config_path, args.verbose, use_colors)
            results.append(result)

            if not args.verbose and not args.ci:
                status = "✓ PASS" if result.success else "✗ FAIL"
                print(f"{config_path.name:30s} {status}")

    # Print summary
    print_summary(results, use_colors)

    # Write JSON output if requested
    if args.json:
        output = {
            "total": len(results),
            "passed": sum(1 for r in results if r.success),
            "failed": sum(1 for r in results if not r.success),
            "results": [r.to_dict() for r in results],
        }

        with open(args.json, "w") as f:
            json.dump(output, f, indent=2)

        print(f"\nJSON results written to: {args.json}")

    # Exit with appropriate code
    all_passed = all(r.success for r in results)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
