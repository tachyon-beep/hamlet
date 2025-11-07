"""Integration tests for Variable & Feature System (VFS).

These tests validate that VFS components work together correctly in
end-to-end scenarios, including loading from YAML, building observation
specs, and integrating with the training environment.

Test Coverage:
- Loading variable definitions from YAML config files
- Building observation specs from variables and exposures
- Calculating observation dimensions for all curriculum levels
- Applying normalization specifications
- Access control validation in realistic scenarios
- Registry initialization with config-loaded variables
"""

from pathlib import Path

import pytest
import torch
import yaml

from townlet.vfs import (
    NormalizationSpec,
    ObservationField,
    VariableDef,
    VariableRegistry,
    VFSObservationSpecBuilder,
    WriteSpec,
)


class TestVFSYAMLLoading:
    """Test loading VFS configurations from YAML files."""

    def test_load_variables_from_l1_config(self):
        """Load variable definitions from L1_full_observability config."""
        config_path = Path("configs/L1_full_observability/variables_reference.yaml")

        if not config_path.exists():
            pytest.skip(f"Reference variables not found: {config_path}")

        with open(config_path) as f:
            data = yaml.safe_load(f)

        # Validate YAML structure
        assert "variables" in data, "YAML must contain 'variables' key"
        assert isinstance(data["variables"], list), "Variables must be a list"
        assert len(data["variables"]) > 0, "Variables list cannot be empty"

        # Load variables as VariableDef objects
        variables = []
        for var_data in data["variables"]:
            var_def = VariableDef(**var_data)
            variables.append(var_def)

        # Validate loaded variables
        assert len(variables) > 0, "Should load at least one variable"

        # Check for expected meter variables
        meter_ids = {v.id for v in variables}
        expected_meters = {"energy", "health", "satiation", "money", "mood", "social", "fitness", "hygiene"}
        assert expected_meters.issubset(meter_ids), f"Missing expected meters: {expected_meters - meter_ids}"

    def test_load_variables_from_all_curriculum_levels(self):
        """Load variables from all curriculum levels successfully."""
        configs = [
            "L0_0_minimal",
            "L0_5_dual_resource",
            "L1_full_observability",
            "L2_partial_observability",
            "L3_temporal_mechanics",
        ]

        for config_name in configs:
            config_path = Path(f"configs/{config_name}/variables_reference.yaml")

            if not config_path.exists():
                pytest.skip(f"Reference variables not found: {config_path}")

            with open(config_path) as f:
                data = yaml.safe_load(f)

            variables = []
            for var_data in data["variables"]:
                var_def = VariableDef(**var_data)
                variables.append(var_def)

            assert len(variables) > 0, f"Config {config_name} should have variables"


class TestVFSObservationPipeline:
    """Test end-to-end observation spec building pipeline."""

    def test_build_observation_spec_from_yaml(self):
        """Build observation spec from YAML-loaded variables."""
        config_path = Path("configs/L1_full_observability/variables_reference.yaml")

        if not config_path.exists():
            pytest.skip(f"Reference variables not found: {config_path}")

        # Load variables from YAML
        with open(config_path) as f:
            data = yaml.safe_load(f)

        variables = []
        for var_data in data["variables"]:
            variables.append(VariableDef(**var_data))

        # Build exposures (expose all agent-readable variables)
        exposures = {}
        for var in variables:
            if "agent" in var.readable_by:
                exposures[var.id] = {"normalization": None}

        # Build observation spec
        builder = VFSObservationSpecBuilder()
        obs_spec = builder.build_observation_spec(variables, exposures)

        # Validate observation spec
        assert isinstance(obs_spec, list), "Observation spec must be a list"
        assert len(obs_spec) > 0, "Observation spec cannot be empty"

        # All observation fields should have valid IDs
        for field in obs_spec:
            assert isinstance(field, ObservationField), "Each field must be ObservationField"
            assert field.id.startswith("obs_"), "Observation field IDs should start with 'obs_'"
            assert field.source_variable in [v.id for v in variables], "Source variable must exist"

    def test_observation_dimension_calculation_integration(self):
        """Calculate total observation dimension from loaded config."""
        config_path = Path("configs/L1_full_observability/variables_reference.yaml")

        if not config_path.exists():
            pytest.skip(f"Reference variables not found: {config_path}")

        # Load and build observation spec
        with open(config_path) as f:
            data = yaml.safe_load(f)

        variables = [VariableDef(**var_data) for var_data in data["variables"]]

        exposures = {}
        for var in variables:
            if "agent" in var.readable_by:
                exposures[var.id] = {"normalization": None}

        builder = VFSObservationSpecBuilder()
        obs_spec = builder.build_observation_spec(variables, exposures)

        # Calculate total dimension
        total_dim = sum(field.shape[0] if field.shape else 1 for field in obs_spec)

        # L1_full_observability should be 93 dims
        assert total_dim == 93, f"L1 should have 93 dims, got {total_dim}"

    def test_normalization_specs_loaded_from_yaml(self):
        """Normalization specs from YAML are correctly parsed."""
        config_path = Path("configs/L1_full_observability/variables_reference.yaml")

        if not config_path.exists():
            pytest.skip(f"Reference variables not found: {config_path}")

        # Load variables
        with open(config_path) as f:
            data = yaml.safe_load(f)

        # Check if exposed_observations are defined
        if "exposed_observations" not in data:
            pytest.skip("Config does not define exposed_observations")

        # Parse exposed observations
        exposed_obs = data["exposed_observations"]

        # Find observations with normalization
        normalized_obs = [obs for obs in exposed_obs if obs.get("normalization")]

        assert len(normalized_obs) > 0, "Should have at least one normalized observation"

        # Validate normalization specs can be parsed
        for obs in normalized_obs:
            norm_data = obs["normalization"]
            norm_spec = NormalizationSpec(**norm_data)

            assert norm_spec.kind in ["minmax", "zscore"], "Normalization kind must be valid"


class TestVFSRegistryIntegration:
    """Test VariableRegistry integration with loaded configurations."""

    def test_registry_initialization_from_yaml(self):
        """Initialize registry with variables loaded from YAML."""
        config_path = Path("configs/L1_full_observability/variables_reference.yaml")

        if not config_path.exists():
            pytest.skip(f"Reference variables not found: {config_path}")

        # Load variables
        with open(config_path) as f:
            data = yaml.safe_load(f)

        variables = [VariableDef(**var_data) for var_data in data["variables"]]

        # Initialize registry with num_agents=4
        num_agents = 4
        registry = VariableRegistry(variables, num_agents, device=torch.device("cpu"))

        # Registry should contain all variables
        for var in variables:
            assert var.id in registry.variables, f"Variable {var.id} should be in registry"

    def test_registry_get_set_with_loaded_variables(self):
        """Get and set values in registry initialized from YAML."""
        config_path = Path("configs/L1_full_observability/variables_reference.yaml")

        if not config_path.exists():
            pytest.skip(f"Reference variables not found: {config_path}")

        # Load variables
        with open(config_path) as f:
            data = yaml.safe_load(f)

        variables = [VariableDef(**var_data) for var_data in data["variables"]]

        # Initialize registry
        num_agents = 4
        registry = VariableRegistry(variables, num_agents, device=torch.device("cpu"))

        # Test energy meter (should be scalar, agent-scoped)
        energy_var = next((v for v in variables if v.id == "energy"), None)
        if energy_var:
            # Get initial value
            initial_energy = registry.get("energy", reader="agent")
            assert initial_energy.shape == (num_agents,), f"Energy shape should be ({num_agents},)"

            # Set new value
            new_energy = torch.tensor([0.8, 0.9, 0.7, 0.6])
            registry.set("energy", new_energy, writer="engine")

            # Verify value was set
            updated_energy = registry.get("energy", reader="agent")
            assert torch.allclose(updated_energy, new_energy), "Energy should be updated"

    def test_registry_access_control_integration(self):
        """Access control works with YAML-loaded variables."""
        config_path = Path("configs/L1_full_observability/variables_reference.yaml")

        if not config_path.exists():
            pytest.skip(f"Reference variables not found: {config_path}")

        # Load variables
        with open(config_path) as f:
            data = yaml.safe_load(f)

        variables = [VariableDef(**var_data) for var_data in data["variables"]]

        # Initialize registry
        registry = VariableRegistry(variables, num_agents=4, device=torch.device("cpu"))

        # Find a variable readable by agent but not by acs
        agent_only_vars = [v for v in variables if "agent" in v.readable_by and "acs" not in v.readable_by]

        if len(agent_only_vars) > 0:
            var_id = agent_only_vars[0].id

            # Agent should be able to read
            value = registry.get(var_id, reader="agent")
            assert value is not None, f"Agent should be able to read {var_id}"

            # ACS should be denied
            with pytest.raises(PermissionError):
                registry.get(var_id, reader="acs")


class TestVFSEndToEndPipeline:
    """Test complete VFS pipeline from YAML to observations."""

    def test_complete_pipeline_l1_config(self):
        """Complete pipeline: Load YAML → Build spec → Calculate dims → Initialize registry."""
        config_path = Path("configs/L1_full_observability/variables_reference.yaml")

        if not config_path.exists():
            pytest.skip(f"Reference variables not found: {config_path}")

        # Step 1: Load variables from YAML
        with open(config_path) as f:
            data = yaml.safe_load(f)

        variables = [VariableDef(**var_data) for var_data in data["variables"]]

        # Step 2: Build observation spec
        exposures = {}
        for var in variables:
            if "agent" in var.readable_by:
                exposures[var.id] = {"normalization": None}

        builder = VFSObservationSpecBuilder()
        obs_spec = builder.build_observation_spec(variables, exposures)

        # Step 3: Calculate observation dimension
        obs_dim = sum(field.shape[0] if field.shape else 1 for field in obs_spec)

        # Step 4: Initialize registry
        num_agents = 8
        registry = VariableRegistry(variables, num_agents, device=torch.device("cpu"))

        # Step 5: Validate pipeline results
        assert obs_dim == 93, f"L1 should have 93 observation dims, got {obs_dim}"
        assert len(obs_spec) > 0, "Observation spec should have fields"

        # Verify all observation sources exist in registry
        for field in obs_spec:
            assert field.source_variable in registry.variables, f"Source variable {field.source_variable} not in registry"

        # Verify we can read all observation sources
        for field in obs_spec:
            value = registry.get(field.source_variable, reader="agent")
            assert value is not None, f"Should be able to read {field.source_variable}"

    def test_pipeline_all_curriculum_levels(self):
        """Complete pipeline works for all curriculum levels."""
        expected_dims = {
            "L0_0_minimal": 38,
            "L0_5_dual_resource": 78,
            "L1_full_observability": 93,
            "L2_partial_observability": 54,
            "L3_temporal_mechanics": 93,
        }

        for config_name, expected_dim in expected_dims.items():
            config_path = Path(f"configs/{config_name}/variables_reference.yaml")

            if not config_path.exists():
                pytest.skip(f"Reference variables not found: {config_path}")

            # Load variables
            with open(config_path) as f:
                data = yaml.safe_load(f)

            variables = [VariableDef(**var_data) for var_data in data["variables"]]

            # Build observation spec
            exposures = {}
            for var in variables:
                if "agent" in var.readable_by:
                    exposures[var.id] = {"normalization": None}

            builder = VFSObservationSpecBuilder()
            obs_spec = builder.build_observation_spec(variables, exposures)

            # Calculate dimension
            obs_dim = sum(field.shape[0] if field.shape else 1 for field in obs_spec)

            # Validate
            assert obs_dim == expected_dim, f"{config_name}: Expected {expected_dim} dims, got {obs_dim}"

            # Initialize registry (verify no crashes)
            registry = VariableRegistry(variables, num_agents=4, device=torch.device("cpu"))
            assert len(registry.variables) == len(variables), f"{config_name}: Registry should contain all variables"


class TestVFSActionConfigIntegration:
    """Test ActionConfig integration with VFS (reads/writes fields)."""

    def test_action_config_reads_field_references_valid_variables(self):
        """ActionConfig.reads field references variables that exist in registry."""
        from townlet.environment.action_config import ActionConfig

        config_path = Path("configs/L1_full_observability/variables_reference.yaml")

        if not config_path.exists():
            pytest.skip(f"Reference variables not found: {config_path}")

        # Load variables
        with open(config_path) as f:
            data = yaml.safe_load(f)

        variables = [VariableDef(**var_data) for var_data in data["variables"]]
        var_ids = {v.id for v in variables}

        # Create action that reads position and energy
        action = ActionConfig(
            id=0,
            name="MOVE",
            type="movement",
            costs={"energy": 0.005},
            effects={},
            delta=[1, 0],
            teleport_to=None,
            enabled=True,
            description="Move right",
            icon=None,
            source="substrate",
            source_affordance=None,
            reads=["position", "energy"],  # VFS integration
        )

        # Validate reads references exist
        for var_id in action.reads:
            assert var_id in var_ids, f"Action reads variable '{var_id}' which doesn't exist in config"

    def test_action_config_writes_field_references_valid_variables(self):
        """ActionConfig.writes field references variables that exist in registry."""
        from townlet.environment.action_config import ActionConfig

        config_path = Path("configs/L1_full_observability/variables_reference.yaml")

        if not config_path.exists():
            pytest.skip(f"Reference variables not found: {config_path}")

        # Load variables
        with open(config_path) as f:
            data = yaml.safe_load(f)

        variables = [VariableDef(**var_data) for var_data in data["variables"]]
        var_ids = {v.id for v in variables}

        # Create action that writes position
        action = ActionConfig(
            id=0,
            name="TELEPORT",
            type="movement",
            costs={"energy": 0.01},
            effects={},
            delta=None,
            teleport_to=[0, 0],
            enabled=True,
            description="Teleport home",
            icon=None,
            source="custom",
            source_affordance=None,
            reads=["position"],
            writes=[
                WriteSpec(variable_id="position", expression="home_position"),
            ],
        )

        # Validate writes references exist
        for write_spec in action.writes:
            assert write_spec.variable_id in var_ids, f"Action writes variable '{write_spec.variable_id}' which doesn't exist in config"


# ========================================
# INTEGRATION TEST SUMMARY
# ========================================
# This module validates VFS components work together in realistic scenarios:
#
# Test Categories:
# ✓ YAML Loading: Load variable definitions from config files
# ✓ Observation Pipeline: Build specs and calculate dimensions
# ✓ Registry Integration: Initialize registry with loaded variables
# ✓ End-to-End: Complete pipeline from YAML to observations
# ✓ ActionConfig Integration: Validate reads/writes reference valid variables
#
# Coverage:
# - All 5 curriculum levels (L0_0, L0_5, L1, L2, L3)
# - Variable loading, validation, and schema enforcement
# - Observation spec building and dimension calculation
# - Registry initialization and access control
# - ActionConfig VFS field validation
