"""Shared test fixtures for configuration DTO tests.

This module provides config-specific fixtures WITHOUT torch dependencies.
This allows config tests to run independently of the full environment stack.

Fixtures:
    - temp_config_dir: Temporary directory for config file tests
    - production_configs: Paths to all production config packs
"""

from pathlib import Path

import pytest


@pytest.fixture
def temp_config_dir(tmp_path: Path) -> Path:
    """Provide temporary directory for config file tests.

    Args:
        tmp_path: pytest's built-in temporary directory fixture

    Returns:
        Path to temporary directory (cleaned up after test)

    Usage:
        def test_something(temp_config_dir):
            yaml_path = temp_config_dir / "training.yaml"
            # Test logic
    """
    return tmp_path


@pytest.fixture(scope="session")
def production_configs() -> dict[str, Path]:
    """Return paths to all production config packs.

    Returns:
        Dict mapping config pack name to Path

    Usage:
        def test_something(production_configs):
            for name, path in production_configs.items():
                # Test each production config
    """
    repo_root = Path(__file__).parent.parent.parent.parent.parent
    configs_dir = repo_root / "configs"

    return {
        "L0_0_minimal": configs_dir / "L0_0_minimal",
        "L0_5_dual_resource": configs_dir / "L0_5_dual_resource",
        "L1_full_observability": configs_dir / "L1_full_observability",
        "L2_partial_observability": configs_dir / "L2_partial_observability",
        "L3_temporal_mechanics": configs_dir / "L3_temporal_mechanics",
    }
