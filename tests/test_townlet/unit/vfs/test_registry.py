"""Test VFS VariableRegistry (Cycle 2 - TDD RED phase).

This module tests the Variable Registry which manages runtime storage of VFS variables
with access control and scope semantics.

Scope patterns:
- global: Single value (shape [] or [dims])
- agent: Per-agent values (shape [num_agents] or [num_agents, dims])
- agent_private: Per-agent private values (shape [num_agents] or [num_agents, dims])
"""

import pytest
import torch


class TestRegistryInitialization:
    """Test VariableRegistry initialization with different scopes."""

    def test_registry_creation_empty(self):
        """Create empty registry with no variables."""
        from townlet.vfs.registry import VariableRegistry

        registry = VariableRegistry(
            variables=[],
            num_agents=4,
            device=torch.device("cpu"),
        )

        assert registry.num_agents == 4
        assert registry.device == torch.device("cpu")

    def test_registry_with_global_scalar(self):
        """Initialize registry with global scalar variable."""
        from townlet.vfs.registry import VariableRegistry
        from townlet.vfs.schema import VariableDef

        variables = [
            VariableDef(
                id="time_sin",
                scope="global",
                type="scalar",
                lifetime="tick",
                readable_by=["agent", "engine"],
                writable_by=["engine"],
                default=0.0,
            )
        ]

        registry = VariableRegistry(
            variables=variables,
            num_agents=4,
            device=torch.device("cpu"),
        )

        # Global scalar: shape []
        value = registry.get("time_sin", reader="engine")
        assert value.shape == torch.Size([])
        assert value.item() == 0.0

    def test_registry_with_agent_scalar(self):
        """Initialize registry with agent-scoped scalar variable."""
        from townlet.vfs.registry import VariableRegistry
        from townlet.vfs.schema import VariableDef

        variables = [
            VariableDef(
                id="energy",
                scope="agent",
                type="scalar",
                lifetime="episode",
                readable_by=["agent", "engine"],
                writable_by=["engine"],
                default=1.0,
            )
        ]

        registry = VariableRegistry(
            variables=variables,
            num_agents=4,
            device=torch.device("cpu"),
        )

        # Agent scalar: shape [num_agents]
        value = registry.get("energy", reader="engine")
        assert value.shape == torch.Size([4])
        assert torch.all(value == 1.0)

    def test_registry_with_agent_vector(self):
        """Initialize registry with agent-scoped vector variable."""
        from townlet.vfs.registry import VariableRegistry
        from townlet.vfs.schema import VariableDef

        variables = [
            VariableDef(
                id="position",
                scope="agent",
                type="vecNf",
                dims=2,
                lifetime="episode",
                readable_by=["agent"],
                writable_by=["engine"],
                default=[0.0, 0.0],
            )
        ]

        registry = VariableRegistry(
            variables=variables,
            num_agents=4,
            device=torch.device("cpu"),
        )

        # Agent vector: shape [num_agents, dims]
        value = registry.get("position", reader="agent")
        assert value.shape == torch.Size([4, 2])
        assert torch.all(value == 0.0)

    def test_registry_with_agent_private_vector(self):
        """Initialize registry with agent_private scoped vector variable."""
        from townlet.vfs.registry import VariableRegistry
        from townlet.vfs.schema import VariableDef

        variables = [
            VariableDef(
                id="home_position",
                scope="agent_private",
                type="vecNf",
                dims=2,
                lifetime="episode",
                readable_by=["agent", "engine"],
                writable_by=["engine"],
                default=[5.0, 5.0],
            )
        ]

        registry = VariableRegistry(
            variables=variables,
            num_agents=4,
            device=torch.device("cpu"),
        )

        # Agent_private vector: shape [num_agents, dims]
        value = registry.get("home_position", reader="engine")
        assert value.shape == torch.Size([4, 2])
        assert torch.all(value == 5.0)

    def test_registry_with_mixed_scopes(self):
        """Initialize registry with multiple variables of different scopes."""
        from townlet.vfs.registry import VariableRegistry
        from townlet.vfs.schema import VariableDef

        variables = [
            VariableDef(
                id="time_sin",
                scope="global",
                type="scalar",
                lifetime="tick",
                readable_by=["agent", "engine"],
                writable_by=["engine"],
                default=0.0,
            ),
            VariableDef(
                id="energy",
                scope="agent",
                type="scalar",
                lifetime="episode",
                readable_by=["agent", "engine"],
                writable_by=["engine"],
                default=1.0,
            ),
            VariableDef(
                id="home_pos",
                scope="agent_private",
                type="vecNf",
                dims=2,
                lifetime="episode",
                readable_by=["agent", "engine"],
                writable_by=["engine"],
                default=[0.0, 0.0],
            ),
        ]

        registry = VariableRegistry(
            variables=variables,
            num_agents=4,
            device=torch.device("cpu"),
        )

        # Verify all three scopes initialized correctly
        time_sin = registry.get("time_sin", reader="engine")
        assert time_sin.shape == torch.Size([])

        energy = registry.get("energy", reader="engine")
        assert energy.shape == torch.Size([4])

        home_pos = registry.get("home_pos", reader="engine")
        assert home_pos.shape == torch.Size([4, 2])


class TestRegistryAccessControl:
    """Test access control enforcement (readable_by/writable_by)."""

    def test_read_allowed(self):
        """Read variable when reader is in readable_by list."""
        from townlet.vfs.registry import VariableRegistry
        from townlet.vfs.schema import VariableDef

        variables = [
            VariableDef(
                id="energy",
                scope="agent",
                type="scalar",
                lifetime="episode",
                readable_by=["agent", "engine"],
                writable_by=["engine"],
                default=1.0,
            )
        ]

        registry = VariableRegistry(variables=variables, num_agents=4, device=torch.device("cpu"))

        # Both agent and engine can read
        value = registry.get("energy", reader="agent")
        assert value is not None

        value = registry.get("energy", reader="engine")
        assert value is not None

    def test_read_denied(self):
        """Read variable when reader is NOT in readable_by list."""
        from townlet.vfs.registry import VariableRegistry
        from townlet.vfs.schema import VariableDef

        variables = [
            VariableDef(
                id="energy",
                scope="agent",
                type="scalar",
                lifetime="episode",
                readable_by=["agent"],  # Only agent can read
                writable_by=["engine"],
                default=1.0,
            )
        ]

        registry = VariableRegistry(variables=variables, num_agents=4, device=torch.device("cpu"))

        # acs cannot read (not in readable_by)
        with pytest.raises(PermissionError, match="acs.*not allowed to read.*energy"):
            registry.get("energy", reader="acs")

    def test_agent_cannot_read_agent_private(self):
        """Agents should not directly read agent_private variables."""
        from townlet.vfs.registry import VariableRegistry
        from townlet.vfs.schema import VariableDef

        variables = [
            VariableDef(
                id="secret",
                scope="agent_private",
                type="scalar",
                lifetime="episode",
                readable_by=["agent", "engine"],
                writable_by=["engine"],
                default=0.5,
            )
        ]

        registry = VariableRegistry(variables=variables, num_agents=3, device=torch.device("cpu"))

        with pytest.raises(PermissionError, match="agent_private"):
            registry.get("secret", reader="agent")

    def test_engine_can_read_agent_private(self):
        """Privileged readers (engine) can access full agent_private tensors."""
        from townlet.vfs.registry import VariableRegistry
        from townlet.vfs.schema import VariableDef

        variables = [
            VariableDef(
                id="secret",
                scope="agent_private",
                type="scalar",
                lifetime="episode",
                readable_by=["agent", "engine"],
                writable_by=["engine"],
                default=0.5,
            )
        ]

        registry = VariableRegistry(variables=variables, num_agents=3, device=torch.device("cpu"))

        value = registry.get("secret", reader="engine")
        assert torch.all(value == 0.5)

    def test_write_allowed(self):
        """Write variable when writer is in writable_by list."""
        from townlet.vfs.registry import VariableRegistry
        from townlet.vfs.schema import VariableDef

        variables = [
            VariableDef(
                id="energy",
                scope="agent",
                type="scalar",
                lifetime="episode",
                readable_by=["agent"],
                writable_by=["engine"],
                default=1.0,
            )
        ]

        registry = VariableRegistry(variables=variables, num_agents=4, device=torch.device("cpu"))

        # Engine can write
        new_value = torch.full((4,), 0.5, device=torch.device("cpu"))
        registry.set("energy", new_value, writer="engine")

        # Verify written
        value = registry.get("energy", reader="agent")
        assert torch.all(value == 0.5)

    def test_write_denied(self):
        """Write variable when writer is NOT in writable_by list."""
        from townlet.vfs.registry import VariableRegistry
        from townlet.vfs.schema import VariableDef

        variables = [
            VariableDef(
                id="energy",
                scope="agent",
                type="scalar",
                lifetime="episode",
                readable_by=["agent"],
                writable_by=["engine"],  # Only engine can write
                default=1.0,
            )
        ]

        registry = VariableRegistry(variables=variables, num_agents=4, device=torch.device("cpu"))

        # agent cannot write (not in writable_by)
        new_value = torch.full((4,), 0.5, device=torch.device("cpu"))
        with pytest.raises(PermissionError, match="agent.*not allowed to write.*energy"):
            registry.set("energy", new_value, writer="agent")


class TestRegistryGetSet:
    """Test get/set operations."""

    def test_get_nonexistent_variable(self):
        """Get non-existent variable should raise KeyError."""
        from townlet.vfs.registry import VariableRegistry

        registry = VariableRegistry(variables=[], num_agents=4, device=torch.device("cpu"))

        with pytest.raises(KeyError, match="nonexistent"):
            registry.get("nonexistent", reader="engine")

    def test_set_nonexistent_variable(self):
        """Set non-existent variable should raise KeyError."""
        from townlet.vfs.registry import VariableRegistry

        registry = VariableRegistry(variables=[], num_agents=4, device=torch.device("cpu"))

        with pytest.raises(KeyError, match="nonexistent"):
            registry.set("nonexistent", torch.tensor([1.0]), writer="engine")

    def test_set_scalar_updates_value(self):
        """Set scalar variable updates its value."""
        from townlet.vfs.registry import VariableRegistry
        from townlet.vfs.schema import VariableDef

        variables = [
            VariableDef(
                id="energy",
                scope="agent",
                type="scalar",
                lifetime="episode",
                readable_by=["agent"],
                writable_by=["engine"],
                default=1.0,
            )
        ]

        registry = VariableRegistry(variables=variables, num_agents=4, device=torch.device("cpu"))

        # Initial value
        value = registry.get("energy", reader="agent")
        assert torch.all(value == 1.0)

        # Update
        new_value = torch.tensor([0.9, 0.8, 0.7, 0.6], device=torch.device("cpu"))
        registry.set("energy", new_value, writer="engine")

        # Verify update
        value = registry.get("energy", reader="agent")
        assert torch.allclose(value, new_value)

    def test_set_vector_updates_value(self):
        """Set vector variable updates its value."""
        from townlet.vfs.registry import VariableRegistry
        from townlet.vfs.schema import VariableDef

        variables = [
            VariableDef(
                id="position",
                scope="agent",
                type="vecNf",
                dims=2,
                lifetime="episode",
                readable_by=["agent"],
                writable_by=["engine"],
                default=[0.0, 0.0],
            )
        ]

        registry = VariableRegistry(variables=variables, num_agents=4, device=torch.device("cpu"))

        # Update positions
        new_positions = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
            device=torch.device("cpu"),
        )
        registry.set("position", new_positions, writer="engine")

        # Verify update
        value = registry.get("position", reader="agent")
        assert torch.allclose(value, new_positions)

    def test_get_returns_clone_for_readers(self):
        """Readers should receive a clone they cannot mutate in-place."""
        from townlet.vfs.registry import VariableRegistry
        from townlet.vfs.schema import VariableDef

        variables = [
            VariableDef(
                id="energy",
                scope="agent",
                type="scalar",
                lifetime="episode",
                readable_by=["agent", "engine"],
                writable_by=["engine"],
                default=1.0,
            )
        ]

        registry = VariableRegistry(variables=variables, num_agents=2, device=torch.device("cpu"))

        view = registry.get("energy", reader="agent")
        view.fill_(0.0)

        fresh = registry.get("energy", reader="agent")
        assert torch.all(fresh == 1.0)

    def test_set_validates_shape(self):
        """Setting wrong-shaped tensors should raise ValueError."""
        from townlet.vfs.registry import VariableRegistry
        from townlet.vfs.schema import VariableDef

        variables = [
            VariableDef(
                id="position",
                scope="agent",
                type="vecNf",
                dims=2,
                lifetime="episode",
                readable_by=["agent"],
                writable_by=["engine"],
                default=[0.0, 0.0],
            )
        ]

        registry = VariableRegistry(variables=variables, num_agents=2, device=torch.device("cpu"))

        wrong_shape = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        with pytest.raises(ValueError, match="shape"):
            registry.set("position", wrong_shape, writer="engine")

    def test_set_validates_dtype(self):
        """Setting tensors with wrong dtype should raise ValueError."""
        from townlet.vfs.registry import VariableRegistry
        from townlet.vfs.schema import VariableDef

        variables = [
            VariableDef(
                id="energy",
                scope="agent",
                type="scalar",
                lifetime="episode",
                readable_by=["agent"],
                writable_by=["engine"],
                default=1.0,
            )
        ]

        registry = VariableRegistry(variables=variables, num_agents=2, device=torch.device("cpu"))

        wrong_dtype = torch.ones(2, dtype=torch.int64)
        with pytest.raises(ValueError, match="dtype"):
            registry.set("energy", wrong_dtype, writer="engine")

    def test_set_global_scalar_updates_single_value(self):
        """Set global scalar variable (single value)."""
        from townlet.vfs.registry import VariableRegistry
        from townlet.vfs.schema import VariableDef

        variables = [
            VariableDef(
                id="time_sin",
                scope="global",
                type="scalar",
                lifetime="tick",
                readable_by=["agent"],
                writable_by=["engine"],
                default=0.0,
            )
        ]

        registry = VariableRegistry(variables=variables, num_agents=4, device=torch.device("cpu"))

        # Update global scalar
        new_value = torch.tensor(0.707, device=torch.device("cpu"))
        registry.set("time_sin", new_value, writer="engine")

        # Verify update
        value = registry.get("time_sin", reader="agent")
        assert value.item() == pytest.approx(0.707)


class TestRegistryScopeSemantics:
    """Test scope-specific tensor shape semantics."""

    def test_global_scalar_shape(self):
        """Global scalar has shape []."""
        from townlet.vfs.registry import VariableRegistry
        from townlet.vfs.schema import VariableDef

        variables = [
            VariableDef(
                id="global_var",
                scope="global",
                type="scalar",
                lifetime="tick",
                readable_by=["agent"],
                writable_by=["engine"],
                default=1.0,
            )
        ]

        registry = VariableRegistry(variables=variables, num_agents=10, device=torch.device("cpu"))

        value = registry.get("global_var", reader="agent")
        assert value.shape == torch.Size([])  # Single value, no agent dimension

    def test_global_vector_shape(self):
        """Global vector has shape [dims]."""
        from townlet.vfs.registry import VariableRegistry
        from townlet.vfs.schema import VariableDef

        variables = [
            VariableDef(
                id="global_vec",
                scope="global",
                type="vecNf",
                dims=3,
                lifetime="tick",
                readable_by=["agent"],
                writable_by=["engine"],
                default=[1.0, 2.0, 3.0],
            )
        ]

        registry = VariableRegistry(variables=variables, num_agents=10, device=torch.device("cpu"))

        value = registry.get("global_vec", reader="agent")
        assert value.shape == torch.Size([3])  # [dims], no agent dimension

    def test_agent_scalar_shape(self):
        """Agent scalar has shape [num_agents]."""
        from townlet.vfs.registry import VariableRegistry
        from townlet.vfs.schema import VariableDef

        variables = [
            VariableDef(
                id="agent_var",
                scope="agent",
                type="scalar",
                lifetime="episode",
                readable_by=["agent"],
                writable_by=["engine"],
                default=1.0,
            )
        ]

        registry = VariableRegistry(variables=variables, num_agents=10, device=torch.device("cpu"))

        value = registry.get("agent_var", reader="agent")
        assert value.shape == torch.Size([10])  # [num_agents]

    def test_agent_vector_shape(self):
        """Agent vector has shape [num_agents, dims]."""
        from townlet.vfs.registry import VariableRegistry
        from townlet.vfs.schema import VariableDef

        variables = [
            VariableDef(
                id="agent_vec",
                scope="agent",
                type="vecNf",
                dims=2,
                lifetime="episode",
                readable_by=["agent"],
                writable_by=["engine"],
                default=[0.0, 0.0],
            )
        ]

        registry = VariableRegistry(variables=variables, num_agents=10, device=torch.device("cpu"))

        value = registry.get("agent_vec", reader="agent")
        assert value.shape == torch.Size([10, 2])  # [num_agents, dims]

    def test_agent_private_scalar_shape(self):
        """Agent_private scalar has shape [num_agents]."""
        from townlet.vfs.registry import VariableRegistry
        from townlet.vfs.schema import VariableDef

        variables = [
            VariableDef(
                id="private_var",
                scope="agent_private",
                type="scalar",
                lifetime="episode",
                readable_by=["agent", "engine"],
                writable_by=["engine"],
                default=1.0,
            )
        ]

        registry = VariableRegistry(variables=variables, num_agents=10, device=torch.device("cpu"))

        value = registry.get("private_var", reader="engine")
        assert value.shape == torch.Size([10])  # [num_agents]

    def test_agent_private_vector_shape(self):
        """Agent_private vector has shape [num_agents, dims]."""
        from townlet.vfs.registry import VariableRegistry
        from townlet.vfs.schema import VariableDef

        variables = [
            VariableDef(
                id="private_vec",
                scope="agent_private",
                type="vecNf",
                dims=2,
                lifetime="episode",
                readable_by=["agent", "engine"],
                writable_by=["engine"],
                default=[0.0, 0.0],
            )
        ]

        registry = VariableRegistry(variables=variables, num_agents=10, device=torch.device("cpu"))

        value = registry.get("private_vec", reader="engine")
        assert value.shape == torch.Size([10, 2])  # [num_agents, dims]


class TestRegistryVariablesProperty:
    """Test the public variables property for introspection."""

    def test_variables_property_exposes_definitions(self):
        """Registry.variables property exposes variable definitions."""
        from townlet.vfs.registry import VariableRegistry
        from townlet.vfs.schema import VariableDef

        variables = [
            VariableDef(
                id="energy",
                scope="agent",
                type="scalar",
                lifetime="episode",
                readable_by=["agent"],
                writable_by=["engine"],
                default=1.0,
            ),
            VariableDef(
                id="position",
                scope="agent",
                type="vecNf",
                dims=2,
                lifetime="episode",
                readable_by=["agent"],
                writable_by=["engine"],
                default=[0.0, 0.0],
            ),
        ]

        registry = VariableRegistry(variables=variables, num_agents=4, device=torch.device("cpu"))

        # Variables property should expose definitions dict
        assert "energy" in registry.variables
        assert "position" in registry.variables
        assert len(registry.variables) == 2

    def test_variables_property_returns_dict(self):
        """Registry.variables returns dictionary mapping IDs to definitions."""
        from townlet.vfs.registry import VariableRegistry
        from townlet.vfs.schema import VariableDef

        variables = [
            VariableDef(
                id="energy",
                scope="agent",
                type="scalar",
                lifetime="episode",
                readable_by=["agent"],
                writable_by=["engine"],
                default=1.0,
            ),
        ]

        registry = VariableRegistry(variables=variables, num_agents=4, device=torch.device("cpu"))

        # Check type and structure
        assert isinstance(registry.variables, dict)
        assert registry.variables["energy"].id == "energy"
        assert registry.variables["energy"].scope == "agent"
        assert registry.variables["energy"].type == "scalar"

    def test_variables_property_empty_registry(self):
        """Registry.variables works with empty registry."""
        from townlet.vfs.registry import VariableRegistry

        registry = VariableRegistry(variables=[], num_agents=4, device=torch.device("cpu"))

        assert isinstance(registry.variables, dict)
        assert len(registry.variables) == 0
