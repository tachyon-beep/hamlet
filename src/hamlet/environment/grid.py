"""
Grid world implementation for Hamlet.

Manages the 2D spatial grid, cell contents, and movement logic.
"""

from typing import List, Tuple
from collections import defaultdict


class Grid:
    """
    2D grid world for Hamlet simulation.

    Manages spatial positioning of agents and affordances.
    Initially 8x8, designed to scale to larger cityscapes.
    """

    def __init__(self, width: int = 8, height: int = 8):
        """
        Initialize the grid.

        Args:
            width: Grid width (default 8)
            height: Grid height (default 8)
        """
        self.width = width
        self.height = height
        self.cells = {}  # Dict[(x, y): List[entities]]

    def is_valid_position(self, x: int, y: int) -> bool:
        """Check if position is within grid bounds."""
        return 0 <= x < self.width and 0 <= y < self.height

    def get_cell_contents(self, x: int, y: int) -> List:
        """Get all entities at the specified position."""
        return self.cells.get((x, y), [])

    def add_entity(self, entity, x: int, y: int):
        """
        Add an entity to a cell.

        Args:
            entity: Entity to add
            x: X position
            y: Y position
        """
        if not self.is_valid_position(x, y):
            return

        if (x, y) not in self.cells:
            self.cells[(x, y)] = []

        if entity not in self.cells[(x, y)]:
            self.cells[(x, y)].append(entity)

    def remove_entity(self, entity, x: int, y: int):
        """
        Remove an entity from a cell.

        Args:
            entity: Entity to remove
            x: X position
            y: Y position
        """
        if (x, y) in self.cells and entity in self.cells[(x, y)]:
            self.cells[(x, y)].remove(entity)
            # Clean up empty cells
            if not self.cells[(x, y)]:
                del self.cells[(x, y)]

    def move_entity(self, entity, dx: int, dy: int):
        """
        Move an entity by the specified offset.

        Args:
            entity: Entity to move
            dx: X offset
            dy: Y offset
        """
        old_x, old_y = entity.x, entity.y
        new_x = old_x + dx
        new_y = old_y + dy

        # Clamp to grid bounds
        new_x = max(0, min(self.width - 1, new_x))
        new_y = max(0, min(self.height - 1, new_y))

        # Only move if position changed and is valid
        if (new_x, new_y) != (old_x, old_y):
            self.remove_entity(entity, old_x, old_y)
            entity.x = new_x
            entity.y = new_y
            self.add_entity(entity, new_x, new_y)
