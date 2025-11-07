"""Shared test utilities for VFS (Variable & Feature System) tests.

This module provides common helper functions used across VFS unit and integration
tests to avoid code duplication and ensure consistent testing patterns.
"""

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import yaml

from townlet.vfs.observation_builder import VFSObservationSpecBuilder
from townlet.vfs.schema import VariableDef

# Expected observation dimensions for all curriculum levels
# These values MUST match the current production environment to ensure
# checkpoint compatibility. If these change, checkpoints will be incompatible!
EXPECTED_DIMENSIONS = {
    "L0_0_minimal": 38,
    "L0_5_dual_resource": 78,
    "L1_full_observability": 93,
    "L2_partial_observability": 54,
    "L3_temporal_mechanics": 54,  # POMDP mode (local_window instead of grid_encoding)
}

# Config pack paths (relative to project root)
CONFIG_PATHS = {
    "L0_0_minimal": Path("configs/L0_0_minimal"),
    "L0_5_dual_resource": Path("configs/L0_5_dual_resource"),
    "L1_full_observability": Path("configs/L1_full_observability"),
    "L2_partial_observability": Path("configs/L2_partial_observability"),
    "L3_temporal_mechanics": Path("configs/L3_temporal_mechanics"),
}


def load_variables_from_config(config_path: Path) -> list[VariableDef]:
    """Load VFS variables from config pack's variables_reference.yaml.

    Args:
        config_path: Path to config pack directory

    Returns:
        List of VariableDef objects

    Raises:
        FileNotFoundError: If variables_reference.yaml not found
    """
    variables_file = config_path / "variables_reference.yaml"

    if not variables_file.exists():
        pytest.skip(f"variables_reference.yaml not found: {variables_file}")

    with open(variables_file) as f:
        data = yaml.safe_load(f)

    return [VariableDef(**var_data) for var_data in data["variables"]]


def load_exposures_from_config(config_path: Path) -> list[dict[str, Any]]:
    """Load observation exposure configuration from variables_reference.yaml.

    Args:
        config_path: Path to config pack directory

    Returns:
        List of exposure entries (each mirrors YAML exposed_observations schema)
    """
    variables_file = config_path / "variables_reference.yaml"

    if not variables_file.exists():
        pytest.skip(f"variables_reference.yaml not found: {variables_file}")

    with open(variables_file) as f:
        data = yaml.safe_load(f)

    exposures: list[dict[str, Any]]

    if "exposed_observations" in data:
        # Use explicit exposure configuration (deep copy to avoid caller mutation)
        exposures = [deepcopy(obs) for obs in data["exposed_observations"]]
    else:
        # Fallback: expose all agent-readable variables
        variables = [VariableDef(**var_data) for var_data in data["variables"]]
        exposures = [
            {
                "id": f"obs_{var.id}",
                "source_variable": var.id,
                "exposed_to": ["agent"],
                "normalization": None,
            }
            for var in variables
            if "agent" in var.readable_by
        ]

    return exposures


def calculate_vfs_observation_dim(
    variables: list[VariableDef],
    exposures: list[dict[str, Any]] | dict[str, dict[str, Any]],
    partial_observability: bool = False,
) -> int:
    """Calculate total observation dimension from VFS specification.

    Args:
        variables: List of variable definitions
        exposures: Exposure configuration (list or legacy dict form)
        partial_observability: If True, filter for POMDP mode (exclude grid_encoding)

    Returns:
        Total observation dimension (sum of all field dimensions)
    """
    builder = VFSObservationSpecBuilder()
    obs_spec = builder.build_observation_spec(variables, exposures)

    # Filter observation spec based on observability mode (matches environment filtering)
    if partial_observability:
        # POMDP: Exclude grid_encoding, include local_window
        obs_spec = [field for field in obs_spec if field.source_variable != "grid_encoding"]
    else:
        # Full observability: Exclude local_window, include grid_encoding
        obs_spec = [field for field in obs_spec if field.source_variable != "local_window"]

    total_dims = 0
    for field in obs_spec:
        if field.shape:
            total_dims += field.shape[0]
        else:
            total_dims += 1

    return total_dims


def assert_dimension_equivalence(config_name: str, vfs_dim: int, expected_dim: int) -> None:
    """Assert VFS dimension matches expected dimension with helpful error message.

    Args:
        config_name: Name of curriculum level (e.g., "L1_full_observability")
        vfs_dim: Calculated VFS dimension
        expected_dim: Expected dimension from legacy system

    Raises:
        AssertionError: If dimensions don't match (checkpoint incompatibility)
    """
    assert vfs_dim == expected_dim, (
        f"{config_name}: VFS dimension {vfs_dim} != expected {expected_dim}. "
        f"CHECKPOINT INCOMPATIBILITY! "
        f"This will break existing checkpoints. "
        f"Check variables_reference.yaml in configs/{config_name}/"
    )
