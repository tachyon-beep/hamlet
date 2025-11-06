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


def test_action_space_builder_substrate_only():
    """Builder with no custom actions should return substrate actions only."""
    from pathlib import Path

    from townlet.environment.action_builder import ActionSpaceBuilder
    from townlet.substrate.grid2d import Grid2DSubstrate

    substrate = Grid2DSubstrate(width=8, height=8, boundary="clamp", distance_metric="manhattan")

    # No global_actions.yaml
    builder = ActionSpaceBuilder(
        substrate=substrate,
        global_actions_path=Path("/nonexistent/global_actions.yaml"),
    )

    space = builder.build()

    assert space.action_dim == 6  # Grid2D substrate actions only
    assert space.substrate_action_count == 6
    assert space.custom_action_count == 0


def test_action_space_builder_with_custom_actions(tmp_path):
    """Builder with custom actions should compose substrate + custom."""
    from townlet.environment.action_builder import ActionSpaceBuilder
    from townlet.substrate.grid2d import Grid2DSubstrate

    # Create temporary global_actions.yaml
    global_actions_yaml = tmp_path / "global_actions.yaml"
    global_actions_yaml.write_text(
        """
version: "1.0"
description: "Global custom actions"

custom_actions:
  - name: "REST"
    type: "passive"
    costs: {energy: -0.002}
  - name: "MEDITATE"
    type: "passive"
    costs: {mood: 0.02}
"""
    )

    substrate = Grid2DSubstrate(width=8, height=8, boundary="clamp", distance_metric="manhattan")

    builder = ActionSpaceBuilder(
        substrate=substrate,
        global_actions_path=global_actions_yaml,
    )

    space = builder.build()

    # 6 substrate + 2 custom = 8 actions
    assert space.action_dim == 8
    assert space.substrate_action_count == 6
    assert space.custom_action_count == 2

    # Substrate actions come first (IDs 0-5)
    assert space.get_action_by_id(0).name == "UP"
    assert space.get_action_by_id(0).source == "substrate"

    # Custom actions come after (IDs 6-7)
    assert space.get_action_by_id(6).name == "REST"
    assert space.get_action_by_id(6).source == "custom"
    assert space.get_action_by_id(7).name == "MEDITATE"
    assert space.get_action_by_id(7).source == "custom"


def test_action_space_builder_with_enabled_actions(tmp_path):
    """Builder should mark disabled actions as enabled=False."""
    from townlet.environment.action_builder import ActionSpaceBuilder
    from townlet.substrate.grid2d import Grid2DSubstrate

    global_actions_yaml = tmp_path / "global_actions.yaml"
    global_actions_yaml.write_text(
        """
version: "1.0"
custom_actions:
  - name: "REST"
    type: "passive"
    costs: {}
  - name: "MEDITATE"
    type: "passive"
    costs: {}
"""
    )

    substrate = Grid2DSubstrate(width=8, height=8, boundary="clamp", distance_metric="manhattan")

    # Only enable UP, DOWN, REST (disable others)
    builder = ActionSpaceBuilder(
        substrate=substrate,
        global_actions_path=global_actions_yaml,
        enabled_action_names=["UP", "DOWN", "REST"],
    )

    space = builder.build()

    # All 8 actions defined
    assert space.action_dim == 8

    # UP, DOWN, REST enabled
    assert space.get_action_by_name("UP").enabled
    assert space.get_action_by_name("DOWN").enabled
    assert space.get_action_by_name("REST").enabled

    # LEFT, RIGHT, INTERACT, WAIT, MEDITATE disabled
    assert not space.get_action_by_name("LEFT").enabled
    assert not space.get_action_by_name("RIGHT").enabled
    assert not space.get_action_by_name("INTERACT").enabled
    assert not space.get_action_by_name("WAIT").enabled
    assert not space.get_action_by_name("MEDITATE").enabled

    # Enabled count = 3
    assert space.enabled_action_count == 3


def test_empty_enabled_list_disables_all():
    """Empty enabled_action_names list should disable all actions."""
    from pathlib import Path

    from townlet.environment.action_builder import ActionSpaceBuilder
    from townlet.substrate.grid2d import Grid2DSubstrate

    substrate = Grid2DSubstrate(width=8, height=8, boundary="clamp", distance_metric="manhattan")

    # Empty list = explicitly disable all actions
    builder = ActionSpaceBuilder(
        substrate=substrate,
        global_actions_path=Path("/nonexistent/global_actions.yaml"),
        enabled_action_names=[],  # Empty list (disable all)
    )

    space = builder.build()

    # All 6 substrate actions exist
    assert space.action_dim == 6

    # But ZERO actions enabled
    assert space.enabled_action_count == 0

    # Verify all actions disabled
    for action in space.actions:
        assert not action.enabled
