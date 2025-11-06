"""Tests for substrate default action generation."""

from townlet.environment.action_config import ActionConfig
from townlet.substrate.grid2d import Grid2DSubstrate


def test_grid2d_generates_6_default_actions():
    """Grid2D should provide 6 default actions."""
    substrate = Grid2DSubstrate(width=8, height=8, boundary="clamp", distance_metric="manhattan")

    actions = substrate.get_default_actions()

    assert len(actions) == 6  # UP, DOWN, LEFT, RIGHT, INTERACT, WAIT
    assert all(isinstance(a, ActionConfig) for a in actions)


def test_grid2d_action_names():
    """Grid2D actions should have correct names."""
    substrate = Grid2DSubstrate(width=8, height=8, boundary="clamp", distance_metric="manhattan")

    actions = substrate.get_default_actions()
    names = [a.name for a in actions]

    assert names == ["UP", "DOWN", "LEFT", "RIGHT", "INTERACT", "WAIT"]


def test_grid2d_movement_actions_have_deltas():
    """Grid2D movement actions should have correct deltas."""
    substrate = Grid2DSubstrate(width=8, height=8, boundary="clamp", distance_metric="manhattan")

    actions = substrate.get_default_actions()
    actions_by_name = {a.name: a for a in actions}

    assert actions_by_name["UP"].delta == [0, -1]
    assert actions_by_name["DOWN"].delta == [0, 1]
    assert actions_by_name["LEFT"].delta == [-1, 0]
    assert actions_by_name["RIGHT"].delta == [1, 0]


def test_grid2d_action_costs():
    """Grid2D actions should have default costs."""
    substrate = Grid2DSubstrate(width=8, height=8, boundary="clamp", distance_metric="manhattan")

    actions = substrate.get_default_actions()
    actions_by_name = {a.name: a for a in actions}

    # Movement costs energy + hygiene + satiation
    up = actions_by_name["UP"]
    assert up.costs["energy"] == 0.005
    assert up.costs["hygiene"] == 0.003
    assert up.costs["satiation"] == 0.004

    # INTERACT costs energy only
    interact = actions_by_name["INTERACT"]
    assert interact.costs["energy"] == 0.003

    # WAIT costs energy only
    wait = actions_by_name["WAIT"]
    assert wait.costs["energy"] == 0.004


def test_grid2d_all_actions_enabled_by_default():
    """Grid2D actions should default to enabled=True."""
    substrate = Grid2DSubstrate(width=8, height=8, boundary="clamp", distance_metric="manhattan")

    actions = substrate.get_default_actions()

    assert all(a.enabled for a in actions)


def test_grid2d_all_actions_marked_as_substrate():
    """Grid2D actions should have source='substrate'."""
    substrate = Grid2DSubstrate(width=8, height=8, boundary="clamp", distance_metric="manhattan")

    actions = substrate.get_default_actions()

    assert all(a.source == "substrate" for a in actions)
