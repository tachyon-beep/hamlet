"""Tests for substrate default action generation."""

from townlet.environment.action_config import ActionConfig
from townlet.substrate.aspatial import AspatialSubstrate
from townlet.substrate.continuous import Continuous1DSubstrate, Continuous2DSubstrate, Continuous3DSubstrate
from townlet.substrate.continuousnd import ContinuousNDSubstrate
from townlet.substrate.grid2d import Grid2DSubstrate
from townlet.substrate.grid3d import Grid3DSubstrate
from townlet.substrate.gridnd import GridNDSubstrate


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


def test_grid3d_generates_8_default_actions():
    """Grid3D should provide 8 default actions (adds UP_Z, DOWN_Z)."""
    substrate = Grid3DSubstrate(width=8, height=8, depth=3, boundary="clamp", distance_metric="manhattan")

    actions = substrate.get_default_actions()

    assert len(actions) == 8  # UP, DOWN, LEFT, RIGHT, UP_Z, DOWN_Z, INTERACT, WAIT


def test_grid3d_action_names():
    """Grid3D actions should include Z-axis movement."""
    substrate = Grid3DSubstrate(width=8, height=8, depth=3, boundary="clamp", distance_metric="manhattan")

    actions = substrate.get_default_actions()
    names = [a.name for a in actions]

    assert names == ["UP", "DOWN", "LEFT", "RIGHT", "UP_Z", "DOWN_Z", "INTERACT", "WAIT"]


def test_grid3d_z_axis_deltas():
    """Grid3D Z-axis actions should have correct deltas."""
    substrate = Grid3DSubstrate(width=8, height=8, depth=3, boundary="clamp", distance_metric="manhattan")

    actions = substrate.get_default_actions()
    actions_by_name = {a.name: a for a in actions}

    assert actions_by_name["UP_Z"].delta == [0, 0, -1]  # Decrease Z (up floor)
    assert actions_by_name["DOWN_Z"].delta == [0, 0, 1]  # Increase Z (down floor)


def test_gridnd_generates_2n_plus_2_actions():
    """GridND should provide 2N+2 actions (±1 per dimension + INTERACT + WAIT)."""
    # 7D hypercube
    substrate = GridNDSubstrate(
        dimension_sizes=[5, 5, 5, 5, 5, 5, 5],
        boundary="clamp",
        distance_metric="manhattan",
    )

    actions = substrate.get_default_actions()

    # 7 dimensions × 2 directions + INTERACT + WAIT = 16 actions
    assert len(actions) == 16


def test_gridnd_action_naming_pattern():
    """GridND actions should follow DIM{N}_{NEG|POS} naming pattern."""
    substrate = GridNDSubstrate(
        dimension_sizes=[3, 3, 3, 3],  # 4D
        boundary="clamp",
        distance_metric="manhattan",
    )

    actions = substrate.get_default_actions()
    names = [a.name for a in actions]

    expected = [
        "DIM0_NEG",
        "DIM0_POS",  # Dimension 0
        "DIM1_NEG",
        "DIM1_POS",  # Dimension 1
        "DIM2_NEG",
        "DIM2_POS",  # Dimension 2
        "DIM3_NEG",
        "DIM3_POS",  # Dimension 3
        "INTERACT",
        "WAIT",
    ]
    assert names == expected


def test_gridnd_movement_deltas():
    """GridND actions should have correct deltas."""
    substrate = GridNDSubstrate(
        dimension_sizes=[5, 5, 5, 5],  # 4D
        boundary="clamp",
        distance_metric="manhattan",
    )

    actions = substrate.get_default_actions()
    actions_by_name = {a.name: a for a in actions}

    # DIM0_NEG: [-1, 0, 0, 0]
    assert actions_by_name["DIM0_NEG"].delta == [-1, 0, 0, 0]
    # DIM0_POS: [+1, 0, 0, 0]
    assert actions_by_name["DIM0_POS"].delta == [1, 0, 0, 0]
    # DIM3_NEG: [0, 0, 0, -1]
    assert actions_by_name["DIM3_NEG"].delta == [0, 0, 0, -1]
    # DIM3_POS: [0, 0, 0, +1]
    assert actions_by_name["DIM3_POS"].delta == [0, 0, 0, 1]


# ============================================================================
# Continuous Substrate Tests
# ============================================================================


def test_continuous1d_generates_4_actions():
    """Continuous1D should provide 4 actions (LEFT/RIGHT/INTERACT/WAIT)."""
    substrate = Continuous1DSubstrate(
        min_x=0.0,
        max_x=10.0,
        boundary="clamp",
        movement_delta=0.5,
        interaction_radius=0.8,
    )

    actions = substrate.get_default_actions()

    assert len(actions) == 4
    names = [a.name for a in actions]
    assert names == ["LEFT", "RIGHT", "INTERACT", "WAIT"]


def test_continuous1d_uses_integer_deltas():
    """Continuous1D deltas are integers (scaled by movement_delta in apply_movement)."""
    substrate = Continuous1DSubstrate(
        min_x=0.0,
        max_x=10.0,
        boundary="clamp",
        movement_delta=0.5,
        interaction_radius=0.8,
    )

    actions = substrate.get_default_actions()
    actions_by_name = {a.name: a for a in actions}

    # Deltas are integers that get scaled by movement_delta
    assert actions_by_name["LEFT"].delta == [-1]
    assert actions_by_name["RIGHT"].delta == [1]


def test_continuous2d_generates_6_actions():
    """Continuous2D should provide 6 actions (same as Grid2D)."""
    substrate = Continuous2DSubstrate(
        min_x=0.0,
        max_x=10.0,
        min_y=0.0,
        max_y=10.0,
        boundary="clamp",
        movement_delta=0.5,
        interaction_radius=0.8,
    )

    actions = substrate.get_default_actions()

    assert len(actions) == 6
    names = [a.name for a in actions]
    assert names == ["UP", "DOWN", "LEFT", "RIGHT", "INTERACT", "WAIT"]


def test_continuous2d_uses_integer_deltas():
    """Continuous2D deltas are integers (scaled by movement_delta in apply_movement)."""
    substrate = Continuous2DSubstrate(
        min_x=0.0,
        max_x=10.0,
        min_y=0.0,
        max_y=10.0,
        boundary="clamp",
        movement_delta=0.5,
        interaction_radius=0.8,
    )

    actions = substrate.get_default_actions()
    actions_by_name = {a.name: a for a in actions}

    # Deltas are integers that get scaled by movement_delta
    assert actions_by_name["UP"].delta == [0, -1]
    assert actions_by_name["DOWN"].delta == [0, 1]
    assert actions_by_name["LEFT"].delta == [-1, 0]
    assert actions_by_name["RIGHT"].delta == [1, 0]


def test_continuous3d_generates_8_actions():
    """Continuous3D should provide 8 actions (same as Grid3D)."""
    substrate = Continuous3DSubstrate(
        min_x=0.0,
        max_x=10.0,
        min_y=0.0,
        max_y=10.0,
        min_z=0.0,
        max_z=10.0,
        boundary="clamp",
        movement_delta=0.5,
        interaction_radius=0.8,
    )

    actions = substrate.get_default_actions()

    assert len(actions) == 8
    names = [a.name for a in actions]
    assert names == ["UP", "DOWN", "LEFT", "RIGHT", "UP_Z", "DOWN_Z", "INTERACT", "WAIT"]


def test_continuous3d_uses_integer_deltas():
    """Continuous3D deltas are integers (scaled by movement_delta in apply_movement)."""
    substrate = Continuous3DSubstrate(
        min_x=0.0,
        max_x=10.0,
        min_y=0.0,
        max_y=10.0,
        min_z=0.0,
        max_z=10.0,
        boundary="clamp",
        movement_delta=0.5,
        interaction_radius=0.8,
    )

    actions = substrate.get_default_actions()
    actions_by_name = {a.name: a for a in actions}

    # Deltas are integers that get scaled by movement_delta
    assert actions_by_name["UP"].delta == [0, -1, 0]
    assert actions_by_name["DOWN"].delta == [0, 1, 0]
    assert actions_by_name["LEFT"].delta == [-1, 0, 0]
    assert actions_by_name["RIGHT"].delta == [1, 0, 0]
    assert actions_by_name["UP_Z"].delta == [0, 0, -1]
    assert actions_by_name["DOWN_Z"].delta == [0, 0, 1]


def test_continuousnd_generates_2n_plus_2_actions():
    """ContinuousND should provide 2N+2 actions (same pattern as GridND)."""
    substrate = ContinuousNDSubstrate(
        bounds=[(0.0, 10.0)] * 5,  # 5D
        boundary="clamp",
        movement_delta=0.5,
        interaction_radius=0.8,
    )

    actions = substrate.get_default_actions()

    assert len(actions) == 12  # 5D × 2 + INTERACT + WAIT


def test_continuousnd_action_naming_pattern():
    """ContinuousND actions should follow DIM{N}_{NEG|POS} naming pattern."""
    substrate = ContinuousNDSubstrate(
        bounds=[(0.0, 10.0)] * 4,  # 4D
        boundary="clamp",
        movement_delta=0.5,
        interaction_radius=0.8,
    )

    actions = substrate.get_default_actions()
    names = [a.name for a in actions]

    expected = [
        "DIM0_NEG",
        "DIM0_POS",  # Dimension 0
        "DIM1_NEG",
        "DIM1_POS",  # Dimension 1
        "DIM2_NEG",
        "DIM2_POS",  # Dimension 2
        "DIM3_NEG",
        "DIM3_POS",  # Dimension 3
        "INTERACT",
        "WAIT",
    ]
    assert names == expected


def test_continuousnd_uses_integer_deltas():
    """ContinuousND deltas are integers (scaled by movement_delta in apply_movement)."""
    substrate = ContinuousNDSubstrate(
        bounds=[(0.0, 10.0)] * 4,  # 4D
        boundary="clamp",
        movement_delta=0.5,
        interaction_radius=0.8,
    )

    actions = substrate.get_default_actions()
    actions_by_name = {a.name: a for a in actions}

    # Deltas are integers that get scaled by movement_delta
    # DIM0_NEG: [-1, 0, 0, 0]
    assert actions_by_name["DIM0_NEG"].delta == [-1, 0, 0, 0]
    # DIM0_POS: [+1, 0, 0, 0]
    assert actions_by_name["DIM0_POS"].delta == [1, 0, 0, 0]
    # DIM3_NEG: [0, 0, 0, -1]
    assert actions_by_name["DIM3_NEG"].delta == [0, 0, 0, -1]
    # DIM3_POS: [0, 0, 0, +1]
    assert actions_by_name["DIM3_POS"].delta == [0, 0, 0, 1]


# ============================================================================
# Aspatial Substrate Tests
# ============================================================================


def test_aspatial_generates_2_actions():
    """Aspatial should provide only 2 actions (no movement)."""
    substrate = AspatialSubstrate()

    actions = substrate.get_default_actions()

    assert len(actions) == 2  # INTERACT + WAIT only
    names = [a.name for a in actions]
    assert names == ["INTERACT", "WAIT"]


def test_aspatial_no_movement_actions():
    """Aspatial should have zero movement actions."""
    substrate = AspatialSubstrate()

    actions = substrate.get_default_actions()

    movement_actions = [a for a in actions if a.type == "movement"]
    assert len(movement_actions) == 0


# ============================================================================
# Meta-Action Ordering Tests (Canonical Order Contract)
# ============================================================================


def test_grid2d_meta_actions_at_end():
    """Grid2D meta-actions (INTERACT, WAIT) should be last two actions."""
    substrate = Grid2DSubstrate(width=8, height=8, boundary="clamp", distance_metric="manhattan")
    actions = substrate.get_default_actions()

    # Last two actions should be INTERACT, WAIT (in that order)
    assert len(actions) == 6
    assert actions[-2].name == "INTERACT"
    assert actions[-1].name == "WAIT"

    # First 4 should be movement actions (non-zero deltas)
    for i in range(4):
        assert actions[i].type == "movement"
        assert actions[i].delta is not None
        assert any(d != 0 for d in actions[i].delta)


def test_grid3d_meta_actions_at_end():
    """Grid3D meta-actions (INTERACT, WAIT) should be last two actions."""
    substrate = Grid3DSubstrate(width=8, height=8, depth=3, boundary="clamp", distance_metric="manhattan")
    actions = substrate.get_default_actions()

    # Last two actions should be INTERACT, WAIT (in that order)
    assert len(actions) == 8
    assert actions[-2].name == "INTERACT"
    assert actions[-1].name == "WAIT"

    # First 6 should be movement actions (non-zero deltas)
    for i in range(6):
        assert actions[i].type == "movement"
        assert actions[i].delta is not None
        assert any(d != 0 for d in actions[i].delta)


def test_gridnd_meta_actions_at_end():
    """GridND meta-actions (INTERACT, WAIT) should be last two actions."""
    substrate = GridNDSubstrate(
        dimension_sizes=[5, 5, 5, 5],  # 4D
        boundary="clamp",
        distance_metric="manhattan",
    )
    actions = substrate.get_default_actions()

    # Last two actions should be INTERACT, WAIT (in that order)
    assert len(actions) == 10  # 4D × 2 + 2 = 10
    assert actions[-2].name == "INTERACT"
    assert actions[-1].name == "WAIT"

    # First 8 should be movement actions (non-zero deltas)
    for i in range(8):
        assert actions[i].type == "movement"
        assert actions[i].delta is not None
        assert any(d != 0 for d in actions[i].delta)


def test_continuous1d_meta_actions_at_end():
    """Continuous1D meta-actions (INTERACT, WAIT) should be last two actions."""
    substrate = Continuous1DSubstrate(
        min_x=0.0,
        max_x=10.0,
        boundary="clamp",
        movement_delta=0.5,
        interaction_radius=0.8,
    )
    actions = substrate.get_default_actions()

    # Last two actions should be INTERACT, WAIT (in that order)
    assert len(actions) == 4
    assert actions[-2].name == "INTERACT"
    assert actions[-1].name == "WAIT"

    # First 2 should be movement actions (non-zero deltas)
    for i in range(2):
        assert actions[i].type == "movement"
        assert actions[i].delta is not None
        assert any(d != 0 for d in actions[i].delta)


def test_continuous2d_meta_actions_at_end():
    """Continuous2D meta-actions (INTERACT, WAIT) should be last two actions."""
    substrate = Continuous2DSubstrate(
        min_x=0.0,
        max_x=10.0,
        min_y=0.0,
        max_y=10.0,
        boundary="clamp",
        movement_delta=0.5,
        interaction_radius=0.8,
    )
    actions = substrate.get_default_actions()

    # Last two actions should be INTERACT, WAIT (in that order)
    assert len(actions) == 6
    assert actions[-2].name == "INTERACT"
    assert actions[-1].name == "WAIT"

    # First 4 should be movement actions (non-zero deltas)
    for i in range(4):
        assert actions[i].type == "movement"
        assert actions[i].delta is not None
        assert any(d != 0 for d in actions[i].delta)


def test_continuous3d_meta_actions_at_end():
    """Continuous3D meta-actions (INTERACT, WAIT) should be last two actions."""
    substrate = Continuous3DSubstrate(
        min_x=0.0,
        max_x=10.0,
        min_y=0.0,
        max_y=10.0,
        min_z=0.0,
        max_z=10.0,
        boundary="clamp",
        movement_delta=0.5,
        interaction_radius=0.8,
    )
    actions = substrate.get_default_actions()

    # Last two actions should be INTERACT, WAIT (in that order)
    assert len(actions) == 8
    assert actions[-2].name == "INTERACT"
    assert actions[-1].name == "WAIT"

    # First 6 should be movement actions (non-zero deltas)
    for i in range(6):
        assert actions[i].type == "movement"
        assert actions[i].delta is not None
        assert any(d != 0 for d in actions[i].delta)


def test_continuousnd_meta_actions_at_end():
    """ContinuousND meta-actions (INTERACT, WAIT) should be last two actions."""
    substrate = ContinuousNDSubstrate(
        bounds=[(0.0, 10.0)] * 5,  # 5D
        boundary="clamp",
        movement_delta=0.5,
        interaction_radius=0.8,
    )
    actions = substrate.get_default_actions()

    # Last two actions should be INTERACT, WAIT (in that order)
    assert len(actions) == 12  # 5D × 2 + 2 = 12
    assert actions[-2].name == "INTERACT"
    assert actions[-1].name == "WAIT"

    # First 10 should be movement actions (non-zero deltas)
    for i in range(10):
        assert actions[i].type == "movement"
        assert actions[i].delta is not None
        assert any(d != 0 for d in actions[i].delta)


def test_aspatial_only_meta_actions():
    """Aspatial substrate should ONLY have INTERACT and WAIT (no movement)."""
    substrate = AspatialSubstrate()
    actions = substrate.get_default_actions()

    # Should only have INTERACT and WAIT
    assert len(actions) == 2
    assert actions[0].name == "INTERACT"
    assert actions[1].name == "WAIT"

    # Neither should be movement actions
    assert actions[0].type != "movement"
    assert actions[1].type != "movement"
    assert actions[0].delta is None
    assert actions[1].delta is None
