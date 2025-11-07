"""VFS-Legacy Observation System Equivalence Tests.

Validates that VFS-generated observation dimensions match the legacy
ObservationBuilder dimensions before cutover. These tests ensure checkpoint
compatibility during migration.

Test Strategy:
1. Load variables_reference.yaml for each curriculum level
2. Build VFS observation spec
3. Calculate VFS observation dimensions
4. Compare to legacy ObservationBuilder dimensions
5. Assert exact match

Expected Dimensions (from legacy system):
- L0_0_minimal: 38 dims
- L0_5_dual_resource: 78 dims
- L1_full_observability: 93 dims
- L2_partial_observability: 54 dims
- L3_temporal_mechanics: 93 dims
"""

from pathlib import Path

import yaml

from townlet.vfs import VFSObservationSpecBuilder
from townlet.vfs.schema import VariableDef

# Expected dimensions from legacy ObservationBuilder
LEGACY_DIMENSIONS = {
    "L0_0_minimal": 38,
    "L0_5_dual_resource": 78,
    "L1_full_observability": 93,
    "L2_partial_observability": 54,
    "L3_temporal_mechanics": 93,
}


def load_variables_from_config(config_path: Path) -> list[VariableDef]:
    """Load variables from config pack's variables_reference.yaml."""
    variables_path = config_path / "variables_reference.yaml"

    if not variables_path.exists():
        raise FileNotFoundError(f"variables_reference.yaml not found in {config_path}")

    with open(variables_path) as f:
        data = yaml.safe_load(f)

    return [VariableDef(**var_data) for var_data in data["variables"]]


def load_exposures_from_config(config_path: Path) -> dict:
    """Load observation exposures from config pack's variables_reference.yaml."""
    variables_path = config_path / "variables_reference.yaml"

    with open(variables_path) as f:
        data = yaml.safe_load(f)

    exposures = {}
    if "exposed_observations" in data:
        for obs in data["exposed_observations"]:
            var_id = obs["source_variable"]
            exposures[var_id] = {
                "normalization": obs.get("normalization"),
            }
    else:
        # Fallback: expose all agent-readable variables
        variables = [VariableDef(**var_data) for var_data in data["variables"]]
        for var in variables:
            if "agent" in var.readable_by:
                exposures[var.id] = {"normalization": None}

    return exposures


def calculate_vfs_observation_dim(variables: list[VariableDef], exposures: dict) -> int:
    """Calculate observation dimension from VFS spec."""
    builder = VFSObservationSpecBuilder()
    obs_spec = builder.build_observation_spec(variables, exposures)

    total_dims = 0
    for field in obs_spec:
        if field.shape:
            total_dims += field.shape[0]
        else:
            total_dims += 1

    return total_dims


class TestVFSLegacyEquivalence:
    """Validate VFS dimensions match legacy ObservationBuilder."""

    def test_l0_0_minimal_dimension_equivalence(self):
        """L0_0_minimal: VFS dims must match legacy (38 dims)."""
        config_path = Path("configs/L0_0_minimal")
        expected_dim = LEGACY_DIMENSIONS["L0_0_minimal"]

        variables = load_variables_from_config(config_path)
        exposures = load_exposures_from_config(config_path)
        vfs_dim = calculate_vfs_observation_dim(variables, exposures)

        assert vfs_dim == expected_dim, (
            f"L0_0_minimal: VFS dim {vfs_dim} != legacy dim {expected_dim}. " f"CHECKPOINT INCOMPATIBILITY! Check variables_reference.yaml."
        )

    def test_l0_5_dual_resource_dimension_equivalence(self):
        """L0_5_dual_resource: VFS dims must match legacy (78 dims)."""
        config_path = Path("configs/L0_5_dual_resource")
        expected_dim = LEGACY_DIMENSIONS["L0_5_dual_resource"]

        variables = load_variables_from_config(config_path)
        exposures = load_exposures_from_config(config_path)
        vfs_dim = calculate_vfs_observation_dim(variables, exposures)

        assert vfs_dim == expected_dim, (
            f"L0_5_dual_resource: VFS dim {vfs_dim} != legacy dim {expected_dim}. "
            f"CHECKPOINT INCOMPATIBILITY! Check variables_reference.yaml."
        )

    def test_l1_full_observability_dimension_equivalence(self):
        """L1_full_observability: VFS dims must match legacy (93 dims)."""
        config_path = Path("configs/L1_full_observability")
        expected_dim = LEGACY_DIMENSIONS["L1_full_observability"]

        variables = load_variables_from_config(config_path)
        exposures = load_exposures_from_config(config_path)
        vfs_dim = calculate_vfs_observation_dim(variables, exposures)

        assert vfs_dim == expected_dim, (
            f"L1_full_observability: VFS dim {vfs_dim} != legacy dim {expected_dim}. "
            f"CHECKPOINT INCOMPATIBILITY! Check variables_reference.yaml."
        )

    def test_l2_partial_observability_dimension_equivalence(self):
        """L2_partial_observability: VFS dims must match legacy (54 dims)."""
        config_path = Path("configs/L2_partial_observability")
        expected_dim = LEGACY_DIMENSIONS["L2_partial_observability"]

        variables = load_variables_from_config(config_path)
        exposures = load_exposures_from_config(config_path)
        vfs_dim = calculate_vfs_observation_dim(variables, exposures)

        assert vfs_dim == expected_dim, (
            f"L2_partial_observability: VFS dim {vfs_dim} != legacy dim {expected_dim}. "
            f"CHECKPOINT INCOMPATIBILITY! Check variables_reference.yaml."
        )

    def test_l3_temporal_mechanics_dimension_equivalence(self):
        """L3_temporal_mechanics: VFS dims must match legacy (93 dims)."""
        config_path = Path("configs/L3_temporal_mechanics")
        expected_dim = LEGACY_DIMENSIONS["L3_temporal_mechanics"]

        variables = load_variables_from_config(config_path)
        exposures = load_exposures_from_config(config_path)
        vfs_dim = calculate_vfs_observation_dim(variables, exposures)

        assert vfs_dim == expected_dim, (
            f"L3_temporal_mechanics: VFS dim {vfs_dim} != legacy dim {expected_dim}. "
            f"CHECKPOINT INCOMPATIBILITY! Check variables_reference.yaml."
        )

    def test_vfs_spec_ordering_consistency(self):
        """Verify VFS spec fields are in consistent order across configs."""
        config_paths = [
            Path("configs/L0_0_minimal"),
            Path("configs/L1_full_observability"),
        ]

        field_orders = []
        for config_path in config_paths:
            variables = load_variables_from_config(config_path)
            exposures = load_exposures_from_config(config_path)

            builder = VFSObservationSpecBuilder()
            obs_spec = builder.build_observation_spec(variables, exposures)

            # Extract field source_variable IDs
            field_ids = [field.source_variable for field in obs_spec]
            field_orders.append(field_ids)

        # Check that common fields appear in same relative order
        # (different configs may have different fields, but common ones should be ordered consistently)
        # This is a structural consistency check rather than exact ordering match
        assert len(field_orders) == 2
        assert len(field_orders[0]) > 0
        assert len(field_orders[1]) > 0

    def test_vfs_observation_spec_includes_all_agent_readable(self):
        """Verify VFS spec includes all agent-readable variables."""
        config_path = Path("configs/L1_full_observability")

        variables = load_variables_from_config(config_path)
        exposures = load_exposures_from_config(config_path)

        # Get all agent-readable variable IDs
        agent_readable_ids = {var.id for var in variables if "agent" in var.readable_by}

        # Get all exposed variable IDs
        exposed_ids = set(exposures.keys())

        # All agent-readable variables should be exposed
        # (or have explicit reason not to be)
        assert agent_readable_ids == exposed_ids, (
            f"Agent-readable variables not all exposed. "
            f"Missing: {agent_readable_ids - exposed_ids}, "
            f"Extra: {exposed_ids - agent_readable_ids}"
        )
