"""CRITICAL: Regression tests for observation dimension compatibility.

These tests ensure VFS-generated observation_dim matches the current
hardcoded calculation. If these fail, checkpoints will be incompatible!

Reference Dimensions (from Cycle 0.4 manual validation):
- L0_0_minimal: 38 dims (3x3 grid)
- L0_5_dual_resource: 78 dims (7x7 grid)
- L1_full_observability: 93 dims (8x8 grid)
- L2_partial_observability: 54 dims (POMDP 5x5 window)
- L3_temporal_mechanics: 93 dims (8x8 grid)
"""

from pathlib import Path

import pytest
import yaml

from townlet.vfs.observation_builder import VFSObservationSpecBuilder
from townlet.vfs.schema import VariableDef


def load_reference_variables(config_path: Path) -> list[VariableDef]:
    """Load reference variables.yaml from config pack.

    Args:
        config_path: Path to config pack directory

    Returns:
        List of VariableDef objects
    """
    variables_file = config_path / "variables_reference.yaml"
    if not variables_file.exists():
        pytest.skip(f"Reference variables not found: {variables_file}")

    with open(variables_file) as f:
        data = yaml.safe_load(f)

    variables = []
    for var_data in data["variables"]:
        variables.append(VariableDef(**var_data))

    return variables


# Expected dimensions from Cycle 0 manual validation
EXPECTED_DIMENSIONS = {
    "L0_0_minimal": 38,
    "L0_5_dual_resource": 78,
    "L1_full_observability": 93,
    "L2_partial_observability": 54,
    "L3_temporal_mechanics": 93,
}


def compute_vfs_observation_dim(variables: list[VariableDef]) -> int:
    """Compute observation_dim using VFS observation spec builder.

    This is how BAC compiler will calculate dimensions.

    Args:
        variables: List of variable definitions

    Returns:
        VFS-calculated observation dimension
    """
    # Build exposures from all variables marked as readable by agent
    exposures = {}
    for var in variables:
        if "agent" in var.readable_by:
            exposures[var.id] = {"normalization": None}

    builder = VFSObservationSpecBuilder()
    spec = builder.build_observation_spec(variables, exposures)

    # Calculate total dimensions
    total_dims = 0
    for field in spec:
        if field.shape:
            # Vector: sum all dimensions (for multi-dim vectors)
            total_dims += field.shape[0]
        else:
            # Scalar: 1 dimension
            total_dims += 1

    return total_dims


class TestObservationDimensionRegressionL0:
    """Test L0 config observation dimension compatibility."""

    def test_l0_0_minimal_dimension(self):
        """L0_0_minimal: VFS dims must match current implementation (38 dims)."""
        config_path = Path("configs/L0_0_minimal")
        expected_dim = EXPECTED_DIMENSIONS["L0_0_minimal"]

        # VFS calculation from reference variables
        variables = load_reference_variables(config_path)
        vfs_dim = compute_vfs_observation_dim(variables)

        # CRITICAL: Must match exactly
        assert vfs_dim == expected_dim, (
            f"L0_0_minimal: VFS dim {vfs_dim} != expected dim {expected_dim}. " f"CHECKPOINT INCOMPATIBILITY! Check reference variables."
        )

    def test_l0_5_dual_resource_dimension(self):
        """L0_5_dual_resource: VFS dims must match current implementation (78 dims)."""
        config_path = Path("configs/L0_5_dual_resource")
        expected_dim = EXPECTED_DIMENSIONS["L0_5_dual_resource"]

        # VFS calculation from reference variables
        variables = load_reference_variables(config_path)
        vfs_dim = compute_vfs_observation_dim(variables)

        # CRITICAL: Must match exactly
        assert vfs_dim == expected_dim, (
            f"L0_5_dual_resource: VFS dim {vfs_dim} != expected dim {expected_dim}. "
            f"CHECKPOINT INCOMPATIBILITY! Check reference variables."
        )


class TestObservationDimensionRegressionL1:
    """Test L1 config observation dimension compatibility."""

    def test_l1_full_observability_dimension(self):
        """L1_full_observability: VFS dims must match current implementation (93 dims)."""
        config_path = Path("configs/L1_full_observability")
        expected_dim = EXPECTED_DIMENSIONS["L1_full_observability"]

        # VFS calculation from reference variables
        variables = load_reference_variables(config_path)
        vfs_dim = compute_vfs_observation_dim(variables)

        # CRITICAL: Must match exactly
        assert vfs_dim == expected_dim, (
            f"L1_full_observability: VFS dim {vfs_dim} != expected dim {expected_dim}. "
            f"CHECKPOINT INCOMPATIBILITY! Check reference variables."
        )


class TestObservationDimensionRegressionL2:
    """Test L2 POMDP config observation dimension compatibility."""

    def test_l2_partial_observability_dimension(self):
        """L2_partial_observability: VFS dims must match current implementation (54 dims)."""
        config_path = Path("configs/L2_partial_observability")
        expected_dim = EXPECTED_DIMENSIONS["L2_partial_observability"]

        # VFS calculation from reference variables
        variables = load_reference_variables(config_path)
        vfs_dim = compute_vfs_observation_dim(variables)

        # CRITICAL: Must match exactly
        assert vfs_dim == expected_dim, (
            f"L2_partial_observability: VFS dim {vfs_dim} != expected dim {expected_dim}. "
            f"CHECKPOINT INCOMPATIBILITY! Check reference variables."
        )


class TestObservationDimensionRegressionL3:
    """Test L3 temporal mechanics config observation dimension compatibility."""

    def test_l3_temporal_mechanics_dimension(self):
        """L3_temporal_mechanics: VFS dims must match current implementation (93 dims)."""
        config_path = Path("configs/L3_temporal_mechanics")
        expected_dim = EXPECTED_DIMENSIONS["L3_temporal_mechanics"]

        # VFS calculation from reference variables
        variables = load_reference_variables(config_path)
        vfs_dim = compute_vfs_observation_dim(variables)

        # CRITICAL: Must match exactly
        assert vfs_dim == expected_dim, (
            f"L3_temporal_mechanics: VFS dim {vfs_dim} != expected dim {expected_dim}. "
            f"CHECKPOINT INCOMPATIBILITY! Check reference variables."
        )


class TestObservationDimensionBreakdown:
    """Test individual component dimensions for debugging.

    These tests validate the dimension breakdown formula to help debug
    any future dimension mismatches. They decompose observations into
    substrate, meter, affordance, and temporal components.
    """

    def test_l1_dimension_breakdown(self):
        """L1_full_observability: Verify dimension breakdown matches formula.

        Formula: substrate_dim + meter_count + (num_affordances + 1) + 4
        Expected: 66 + 8 + 15 + 4 = 93

        Breakdown:
        - Substrate: 64 grid cells + 2 position = 66 dims
        - Meters: 8 scalars (energy, health, satiation, money, mood, social, fitness, hygiene)
        - Affordance: 15 one-hot (14 affordance types + none)
        - Temporal: 4 scalars (time_sin, time_cos, interaction_progress, lifetime_progress)
        """
        config_path = Path("configs/L1_full_observability")
        variables = load_reference_variables(config_path)

        # Group variables by category
        substrate_vars = [v for v in variables if v.id.startswith("grid_encoding") or v.id == "position"]
        meter_vars = [v for v in variables if v.id in ["energy", "health", "satiation", "money", "mood", "social", "fitness", "hygiene"]]
        affordance_vars = [v for v in variables if v.id == "affordance_at_position"]
        temporal_vars = [v for v in variables if v.id in ["time_sin", "time_cos", "interaction_progress", "lifetime_progress"]]

        # Calculate dimensions per category
        builder = VFSObservationSpecBuilder()

        substrate_dim = sum(builder._infer_shape(v)[0] if builder._infer_shape(v) else 1 for v in substrate_vars)
        meter_dim = len(meter_vars)  # All scalars
        affordance_dim = sum(builder._infer_shape(v)[0] if builder._infer_shape(v) else 1 for v in affordance_vars)
        temporal_dim = len(temporal_vars)  # All scalars

        total = substrate_dim + meter_dim + affordance_dim + temporal_dim

        # Expected breakdown
        assert substrate_dim == 66, f"Substrate dims: expected 66, got {substrate_dim}"
        assert meter_dim == 8, f"Meter dims: expected 8, got {meter_dim}"
        assert affordance_dim == 15, f"Affordance dims: expected 15, got {affordance_dim}"
        assert temporal_dim == 4, f"Temporal dims: expected 4, got {temporal_dim}"
        assert total == 93, f"Total dims: expected 93, got {total}"


# ========================================
# TEST COVERAGE SUMMARY
# ========================================
# This module validates VFS observation dimension calculations against
# the current hardcoded implementation for checkpoint compatibility.
#
# Configs Validated:
# ✓ L0_0_minimal: 38 dims (3x3 grid)
# ✓ L0_5_dual_resource: 78 dims (7x7 grid)
# ✓ L1_full_observability: 93 dims (8x8 grid, full breakdown tested)
# ✓ L2_partial_observability: 54 dims (POMDP 5x5 window)
# ✓ L3_temporal_mechanics: 93 dims (8x8 grid, temporal features)
#
# Critical: If any test fails, ALL checkpoints become incompatible!
# These tests must ALWAYS pass to maintain checkpoint compatibility.
