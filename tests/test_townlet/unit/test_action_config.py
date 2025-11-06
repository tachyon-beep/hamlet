"""Tests for action configuration schemas."""

import pytest

from townlet.environment.action_config import ActionConfig


def test_action_config_basic():
    """Basic action config should parse successfully."""
    action = ActionConfig(
        id=0,
        name="UP",
        type="movement",
        delta=[0, -1],
        teleport_to=None,
        costs={"energy": 0.005},
        effects={},
        enabled=True,
        description="Move up",
        icon=None,
        source="substrate",
        source_affordance=None,
    )

    assert action.id == 0
    assert action.name == "UP"
    assert action.type == "movement"
    assert action.delta == [0, -1]
    assert action.costs["energy"] == 0.005
    assert action.enabled is True


def test_action_config_disabled():
    """Disabled action should have enabled=False."""
    action = ActionConfig(
        id=1,
        name="REST",
        type="passive",
        delta=None,
        teleport_to=None,
        costs={"energy": -0.002},
        effects={},
        enabled=False,  # Disabled in L0, enabled in L1
        description="Rest",
        icon=None,
        source="custom",
        source_affordance=None,
    )

    assert action.enabled is False


def test_action_config_default_enabled():
    """Action with enabled=True should be enabled."""
    action = ActionConfig(
        id=0,
        name="UP",
        type="movement",
        delta=[0, -1],
        teleport_to=None,
        costs={},
        effects={},
        enabled=True,
        description="Move up",
        icon=None,
        source="substrate",
        source_affordance=None,
    )

    assert action.enabled is True


def test_movement_action_requires_delta_or_teleport():
    """Movement action must have delta or teleport_to."""
    # Valid: has delta
    ActionConfig(
        id=0,
        name="UP",
        type="movement",
        delta=[0, -1],
        teleport_to=None,
        costs={},
        effects={},
        enabled=True,
        description="Move up",
        icon=None,
        source="substrate",
        source_affordance=None,
    )

    # Valid: has teleport_to
    ActionConfig(
        id=1,
        name="TELEPORT",
        type="movement",
        delta=None,
        teleport_to=[0, 0],
        costs={},
        effects={},
        enabled=True,
        description="Teleport",
        icon=None,
        source="custom",
        source_affordance=None,
    )

    # Invalid: has neither
    with pytest.raises(ValueError, match="must define delta or teleport_to"):
        ActionConfig(
            id=2,
            name="INVALID",
            type="movement",
            delta=None,
            teleport_to=None,
            costs={},
            effects={},
            enabled=True,
            description="Invalid",
            icon=None,
            source="substrate",
            source_affordance=None,
        )


def test_non_movement_cannot_have_delta():
    """Non-movement actions cannot have delta."""
    with pytest.raises(ValueError, match="cannot have delta"):
        ActionConfig(
            id=0,
            name="INTERACT",
            type="interaction",
            delta=[0, 0],
            teleport_to=None,
            costs={},
            effects={},
            enabled=True,
            description="Interact",
            icon=None,
            source="substrate",
            source_affordance=None,
        )


def test_negative_costs_allowed():
    """Negative costs (restoration) should be allowed."""
    action = ActionConfig(
        id=0,
        name="REST",
        type="passive",
        delta=None,
        teleport_to=None,
        costs={"energy": -0.002, "mood": -0.01},  # Restores meters
        effects={},
        enabled=True,
        description="Rest",
        icon=None,
        source="custom",
        source_affordance=None,
    )

    assert action.costs["energy"] == -0.002
    assert action.costs["mood"] == -0.01
