"""Test ActionConfig extension with reads/writes fields for VFS integration."""

import pytest

from townlet.environment.action_config import ActionConfig
from townlet.vfs.schema import WriteSpec


class TestActionConfigReadsField:
    """Test ActionConfig with reads field (variable dependencies)."""

    def test_action_config_with_reads(self):
        """ActionConfig with reads field specifies variable dependencies."""
        action = ActionConfig(
            id=0,
            name="TELEPORT_HOME",
            type="movement",
            costs={"energy": 0.05},
            effects={},
            delta=None,
            teleport_to=[0, 0],  # Teleport to home position
            enabled=True,
            description="Teleport to home position",
            icon=None,
            source="custom",
            source_affordance=None,
            reads=["home_pos", "position"],  # NEW FIELD
        )

        assert action.reads == ["home_pos", "position"]

    def test_action_config_with_empty_reads(self):
        """ActionConfig with empty reads list."""
        action = ActionConfig(
            id=0,
            name="REST",
            type="passive",
            costs={},
            effects={"energy": 0.02},
            delta=None,
            teleport_to=None,
            enabled=True,
            description=None,
            icon=None,
            source="custom",
            source_affordance=None,
            reads=[],  # Explicitly empty
        )

        assert action.reads == []

    def test_action_config_with_single_read(self):
        """ActionConfig with single variable read."""
        action = ActionConfig(
            id=0,
            name="CONSUME_ENERGY",
            type="passive",
            costs={},
            effects={},
            delta=None,
            teleport_to=None,
            enabled=True,
            description=None,
            icon=None,
            source="custom",
            source_affordance=None,
            reads=["energy"],
        )

        assert action.reads == ["energy"]
        assert len(action.reads) == 1


class TestActionConfigWritesField:
    """Test ActionConfig with writes field (VFS WriteSpec)."""

    def test_action_config_with_writes(self):
        """ActionConfig with writes field specifies variable updates."""
        write_spec = WriteSpec(variable_id="position", expression="home_pos")

        action = ActionConfig(
            id=0,
            name="TELEPORT_HOME",
            type="movement",
            costs={"energy": 0.05},
            effects={},
            delta=None,
            teleport_to=[0, 0],  # Teleport to home position
            enabled=True,
            description=None,
            icon=None,
            source="custom",
            source_affordance=None,
            writes=[write_spec],  # NEW FIELD
        )

        assert len(action.writes) == 1
        assert action.writes[0].variable_id == "position"
        assert action.writes[0].expression == "home_pos"

    def test_action_config_with_multiple_writes(self):
        """ActionConfig with multiple write specs."""
        write_specs = [
            WriteSpec(variable_id="energy", expression="energy - 0.1"),
            WriteSpec(variable_id="mood", expression="mood + 0.05"),
        ]

        action = ActionConfig(
            id=0,
            name="REST",
            type="passive",
            costs={},
            effects={},
            delta=None,
            teleport_to=None,
            enabled=True,
            description=None,
            icon=None,
            source="custom",
            source_affordance=None,
            writes=write_specs,
        )

        assert len(action.writes) == 2
        assert action.writes[0].variable_id == "energy"
        assert action.writes[1].variable_id == "mood"

    def test_action_config_with_empty_writes(self):
        """ActionConfig with empty writes list."""
        action = ActionConfig(
            id=0,
            name="WAIT",
            type="passive",
            costs={"energy": 0.001},
            effects={},
            delta=None,
            teleport_to=None,
            enabled=True,
            description=None,
            icon=None,
            source="substrate",
            source_affordance=None,
            writes=[],  # Explicitly empty
        )

        assert action.writes == []


class TestActionConfigBackwardCompatibility:
    """Test ActionConfig backward compatibility (reads/writes optional)."""

    def test_action_config_without_reads_writes(self):
        """ActionConfig without reads/writes uses defaults (backward compatible)."""
        action = ActionConfig(
            id=0,
            name="UP",
            type="movement",
            costs={"energy": 0.005},
            effects={},
            delta=[0, -1],
            teleport_to=None,
            enabled=True,
            description=None,
            icon=None,
            source="substrate",
            source_affordance=None,
            # reads and writes omitted - should use defaults
        )

        assert action.reads == []  # Default empty list
        assert action.writes == []  # Default empty list

    def test_action_config_substrate_movement_no_vfs(self):
        """Standard substrate movement action without VFS integration."""
        action = ActionConfig(
            id=1,
            name="DOWN",
            type="movement",
            costs={"energy": 0.005},
            effects={},
            delta=[0, 1],
            teleport_to=None,
            enabled=True,
            description=None,
            icon=None,
            source="substrate",
            source_affordance=None,
        )

        assert action.reads == []
        assert action.writes == []

    def test_action_config_custom_action_no_vfs(self):
        """Custom action without VFS integration."""
        action = ActionConfig(
            id=6,
            name="REST",
            type="passive",
            costs={},
            effects={"energy": 0.02},
            delta=None,
            teleport_to=None,
            enabled=True,
            description=None,
            icon=None,
            source="custom",
            source_affordance=None,
        )

        assert action.reads == []
        assert action.writes == []


class TestActionConfigSerialization:
    """Test ActionConfig serialization with reads/writes."""

    def test_action_config_serialization_with_reads_writes(self):
        """ActionConfig serializes/deserializes with reads/writes."""
        write_spec = WriteSpec(variable_id="energy", expression="energy - 0.1")

        action = ActionConfig(
            id=0,
            name="REST",
            type="passive",
            costs={},
            effects={"energy": 0.02},
            delta=None,
            teleport_to=None,
            enabled=True,
            description=None,
            icon=None,
            source="custom",
            source_affordance=None,
            reads=["energy"],
            writes=[write_spec],
        )

        # Serialize
        data = action.model_dump()
        assert "reads" in data
        assert "writes" in data
        assert data["reads"] == ["energy"]
        assert len(data["writes"]) == 1

        # Deserialize
        action2 = ActionConfig.model_validate(data)
        assert action2.reads == ["energy"]
        assert len(action2.writes) == 1
        assert action2.writes[0].variable_id == "energy"

    def test_action_config_serialization_empty_reads_writes(self):
        """ActionConfig serializes empty reads/writes correctly."""
        action = ActionConfig(
            id=0,
            name="WAIT",
            type="passive",
            costs={"energy": 0.001},
            effects={},
            delta=None,
            teleport_to=None,
            enabled=True,
            description=None,
            icon=None,
            source="substrate",
            source_affordance=None,
            reads=[],
            writes=[],
        )

        data = action.model_dump()
        assert data["reads"] == []
        assert data["writes"] == []

    def test_action_config_serialization_omitted_reads_writes(self):
        """ActionConfig serializes with defaults when reads/writes omitted."""
        action = ActionConfig(
            id=0,
            name="UP",
            type="movement",
            costs={"energy": 0.005},
            effects={},
            delta=[0, -1],
            teleport_to=None,
            enabled=True,
            description=None,
            icon=None,
            source="substrate",
            source_affordance=None,
        )

        data = action.model_dump()
        assert data["reads"] == []
        assert data["writes"] == []


class TestActionConfigValidation:
    """Test ActionConfig validation with reads/writes."""

    def test_action_config_reads_must_be_list(self):
        """reads field must be a list."""
        # This should be caught by Pydantic type validation
        with pytest.raises(Exception):  # Pydantic ValidationError
            ActionConfig(
                id=0,
                name="TEST",
                type="passive",
                costs={},
                effects={},
                delta=None,
                teleport_to=None,
                enabled=True,
                description=None,
                icon=None,
                source="custom",
                source_affordance=None,
                reads="not_a_list",  # Invalid type
            )

    def test_action_config_writes_must_be_list(self):
        """writes field must be a list."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            ActionConfig(
                id=0,
                name="TEST",
                type="passive",
                costs={},
                effects={},
                delta=None,
                teleport_to=None,
                enabled=True,
                description=None,
                icon=None,
                source="custom",
                source_affordance=None,
                writes="not_a_list",  # Invalid type
            )
