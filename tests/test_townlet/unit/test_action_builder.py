"""Tests for ActionSpaceBuilder and ComposedActionSpace."""

import torch

from townlet.environment.action_builder import ComposedActionSpace
from townlet.environment.action_config import ActionConfig


def test_composed_action_space_basic():
    """ComposedActionSpace should track actions and metadata."""
    actions = [
        ActionConfig(id=0, name="UP", type="movement", delta=[0, -1], costs={}, source="substrate"),
        ActionConfig(id=1, name="DOWN", type="movement", delta=[0, 1], costs={}, source="substrate"),
        ActionConfig(id=2, name="REST", type="passive", costs={}, source="custom"),
    ]

    space = ComposedActionSpace(
        actions=actions,
        substrate_action_count=2,
        custom_action_count=1,
        affordance_action_count=0,
    )

    assert space.action_dim == 3
    assert space.substrate_action_count == 2
    assert space.custom_action_count == 1


def test_composed_action_space_get_by_id():
    """Should retrieve action by ID."""
    actions = [
        ActionConfig(id=0, name="UP", type="movement", delta=[0, -1], costs={}, source="substrate"),
        ActionConfig(id=1, name="REST", type="passive", costs={}, source="custom"),
    ]

    space = ComposedActionSpace(actions=actions, substrate_action_count=1, custom_action_count=1, affordance_action_count=0)

    assert space.get_action_by_id(0).name == "UP"
    assert space.get_action_by_id(1).name == "REST"


def test_composed_action_space_get_by_name():
    """Should retrieve action by name."""
    actions = [
        ActionConfig(id=0, name="UP", type="movement", delta=[0, -1], costs={}, source="substrate"),
        ActionConfig(id=1, name="REST", type="passive", costs={}, source="custom"),
    ]

    space = ComposedActionSpace(actions=actions, substrate_action_count=1, custom_action_count=1, affordance_action_count=0)

    assert space.get_action_by_name("UP").id == 0
    assert space.get_action_by_name("REST").id == 1


def test_composed_action_space_enabled_count():
    """Should count enabled vs disabled actions."""
    actions = [
        ActionConfig(id=0, name="UP", type="movement", delta=[0, -1], costs={}, source="substrate", enabled=True),
        ActionConfig(id=1, name="DOWN", type="movement", delta=[0, 1], costs={}, source="substrate", enabled=True),
        ActionConfig(id=2, name="REST", type="passive", costs={}, source="custom", enabled=True),
        ActionConfig(id=3, name="MEDITATE", type="passive", costs={}, source="custom", enabled=False),  # Disabled
    ]

    space = ComposedActionSpace(
        actions=actions,
        substrate_action_count=2,
        custom_action_count=2,
        affordance_action_count=0,
        enabled_action_names={"UP", "DOWN", "REST"},  # MEDITATE not enabled
    )

    assert space.action_dim == 4  # Total actions (including disabled)
    assert space.enabled_action_count == 3  # Only enabled ones


def test_composed_action_space_get_base_mask():
    """Should generate action mask with disabled actions masked out."""
    actions = [
        ActionConfig(id=0, name="UP", type="movement", delta=[0, -1], costs={}, source="substrate", enabled=True),
        ActionConfig(id=1, name="DOWN", type="movement", delta=[0, 1], costs={}, source="substrate", enabled=True),
        ActionConfig(id=2, name="REST", type="passive", costs={}, source="custom", enabled=True),
        ActionConfig(id=3, name="MEDITATE", type="passive", costs={}, source="custom", enabled=False),  # Disabled
    ]

    space = ComposedActionSpace(
        actions=actions,
        substrate_action_count=2,
        custom_action_count=2,
        affordance_action_count=0,
    )

    mask = space.get_base_action_mask(num_agents=2, device=torch.device("cpu"))

    # Shape: [2 agents, 4 actions]
    assert mask.shape == (2, 4)

    # Actions 0, 1, 2 enabled (True)
    assert mask[0, 0]
    assert mask[0, 1]
    assert mask[0, 2]

    # Action 3 disabled (False)
    assert not mask[0, 3]
    assert not mask[1, 3]  # Disabled for all agents
