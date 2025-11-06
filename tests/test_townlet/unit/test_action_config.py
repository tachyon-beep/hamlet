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
        costs={"energy": 0.005},
        enabled=True,
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
        costs={"energy": -0.002},
        enabled=False,  # Disabled in L0, enabled in L1
    )

    assert action.enabled is False


def test_action_config_default_enabled():
    """Action should default to enabled=True if not specified."""
    action = ActionConfig(
        id=0,
        name="UP",
        type="movement",
        delta=[0, -1],
        costs={},
    )

    assert action.enabled is True


def test_movement_action_requires_delta_or_teleport():
    """Movement action must have delta or teleport_to."""
    # Valid: has delta
    ActionConfig(id=0, name="UP", type="movement", delta=[0, -1], costs={})

    # Valid: has teleport_to
    ActionConfig(id=1, name="TELEPORT", type="movement", teleport_to=[0, 0], costs={})

    # Invalid: has neither
    with pytest.raises(ValueError, match="must define delta or teleport_to"):
        ActionConfig(id=2, name="INVALID", type="movement", costs={})


def test_non_movement_cannot_have_delta():
    """Non-movement actions cannot have delta."""
    with pytest.raises(ValueError, match="cannot have delta"):
        ActionConfig(
            id=0,
            name="INTERACT",
            type="interaction",
            delta=[0, 0],
            costs={},
        )


def test_negative_costs_allowed():
    """Negative costs (restoration) should be allowed."""
    action = ActionConfig(
        id=0,
        name="REST",
        type="passive",
        costs={"energy": -0.002, "mood": -0.01},  # Restores meters
    )

    assert action.costs["energy"] == -0.002
    assert action.costs["mood"] == -0.01
