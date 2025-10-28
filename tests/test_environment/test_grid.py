"""
Tests for Grid class.
"""

import pytest
from hamlet.environment.grid import Grid
from hamlet.environment.entities import Agent, Bed


def test_grid_initialization():
    """Test that grid initializes with correct dimensions."""
    grid = Grid(width=8, height=8)
    assert grid.width == 8
    assert grid.height == 8
    assert grid.cells == {}  # Empty on init


def test_grid_position_validation():
    """Test that position validation works correctly."""
    grid = Grid(width=8, height=8)
    # Valid positions
    assert grid.is_valid_position(0, 0) is True
    assert grid.is_valid_position(7, 7) is True
    assert grid.is_valid_position(3, 4) is True

    # Invalid positions
    assert grid.is_valid_position(-1, 0) is False
    assert grid.is_valid_position(0, -1) is False
    assert grid.is_valid_position(8, 0) is False
    assert grid.is_valid_position(0, 8) is False
    assert grid.is_valid_position(10, 10) is False


def test_grid_add_entity():
    """Test adding entities to grid."""
    grid = Grid(width=8, height=8)
    agent = Agent("agent_0", 2, 3)

    grid.add_entity(agent, 2, 3)

    contents = grid.get_cell_contents(2, 3)
    assert agent in contents
    assert len(contents) == 1


def test_grid_multiple_entities_same_cell():
    """Test multiple entities can occupy same cell."""
    grid = Grid(width=8, height=8)
    agent = Agent("agent_0", 2, 3)
    bed = Bed(2, 3)

    grid.add_entity(agent, 2, 3)
    grid.add_entity(bed, 2, 3)

    contents = grid.get_cell_contents(2, 3)
    assert agent in contents
    assert bed in contents
    assert len(contents) == 2


def test_grid_get_empty_cell():
    """Test getting contents of empty cell."""
    grid = Grid(width=8, height=8)
    contents = grid.get_cell_contents(5, 5)
    assert contents == []


def test_grid_remove_entity():
    """Test removing entities from grid."""
    grid = Grid(width=8, height=8)
    agent = Agent("agent_0", 4, 4)

    grid.add_entity(agent, 4, 4)
    assert agent in grid.get_cell_contents(4, 4)

    grid.remove_entity(agent, 4, 4)
    assert agent not in grid.get_cell_contents(4, 4)


def test_grid_move_entity():
    """Test moving entity on grid."""
    grid = Grid(width=8, height=8)
    agent = Agent("agent_0", 2, 2)
    agent.x = 2
    agent.y = 2

    grid.add_entity(agent, 2, 2)

    # Move right (dx=1, dy=0)
    grid.move_entity(agent, 1, 0)

    assert agent.x == 3
    assert agent.y == 2
    assert agent not in grid.get_cell_contents(2, 2)
    assert agent in grid.get_cell_contents(3, 2)


def test_grid_move_entity_clamped_to_bounds():
    """Test that movement is clamped to grid bounds."""
    grid = Grid(width=8, height=8)
    agent = Agent("agent_0", 7, 7)
    agent.x = 7
    agent.y = 7

    grid.add_entity(agent, 7, 7)

    # Try to move out of bounds
    grid.move_entity(agent, 5, 5)

    # Should clamp to edge
    assert agent.x == 7
    assert agent.y == 7


def test_grid_move_to_negative():
    """Test that negative movement is clamped."""
    grid = Grid(width=8, height=8)
    agent = Agent("agent_0", 0, 0)
    agent.x = 0
    agent.y = 0

    grid.add_entity(agent, 0, 0)

    # Try to move negative
    grid.move_entity(agent, -1, -1)

    # Should stay at 0,0
    assert agent.x == 0
    assert agent.y == 0
