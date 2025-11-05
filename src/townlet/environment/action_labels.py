"""Configurable action label system for domain-specific terminology.

This module separates canonical action semantics (what substrates interpret)
from user-facing labels (what students/practitioners see), enabling domain-appropriate
terminology without changing the underlying action space.

Key concepts:
- **Canonical actions**: Substrate-interpreted movement semantics (MOVE_X_NEGATIVE, etc.)
- **User labels**: Domain-specific terminology (LEFT, SWAY_LEFT, WEST, X_NEG, etc.)
- **Presets**: Pre-configured label sets for common domains (gaming, 6-DoF, cardinal, math)
- **Custom labels**: User-defined terminology for specialized domains

Pedagogical value:
- Demonstrates that labels are arbitrary, semantics matter
- Enables domain-appropriate learning (robotics, marine, aviation, gaming)
- Reveals how different communities label identical mathematical transformations
"""

from dataclasses import dataclass
from enum import IntEnum


class CanonicalAction(IntEnum):
    """Canonical action semantics (substrate-interpreted).

    These are the **mathematical** actions that substrates understand.
    The indices map directly to Q-network outputs.

    Naming convention:
    - MOVE_<AXIS>_<DIRECTION>: Movement actions
    - INTERACT: Affordance interaction (no movement)
    - WAIT: No-op (still incurs energy cost)

    Action space varies by substrate dimensionality:
    - Aspatial (0D): INTERACT, WAIT (2 actions)
    - 1D: MOVE_X_NEGATIVE, MOVE_X_POSITIVE, INTERACT, WAIT (4 actions)
    - 2D: +MOVE_Y_NEGATIVE, +MOVE_Y_POSITIVE (6 actions)
    - 3D: +MOVE_Z_POSITIVE, +MOVE_Z_NEGATIVE (8 actions)
    """

    # 1D actions (X-axis)
    MOVE_X_NEGATIVE = 0  # Move left (-X direction)
    MOVE_X_POSITIVE = 1  # Move right (+X direction)

    # 2D actions (Y-axis) - added when substrate.position_dim >= 2
    MOVE_Y_NEGATIVE = 2  # Move down (-Y direction)
    MOVE_Y_POSITIVE = 3  # Move up (+Y direction)

    # Meta actions (always present)
    INTERACT = 4  # Interact with affordance (no movement)
    WAIT = 5  # No-op (still costs energy)

    # 3D actions (Z-axis) - added when substrate.position_dim == 3
    MOVE_Z_POSITIVE = 6  # Move forward/ascend (+Z direction)
    MOVE_Z_NEGATIVE = 7  # Move backward/descend (-Z direction)


@dataclass(frozen=True)
class ActionLabels:
    """Domain-specific action labels mapped to canonical actions.

    Immutable container for user-facing action labels. Each label maps to
    a canonical action index for substrate interpretation.

    Attributes:
        labels: Dictionary mapping canonical action indices to user-facing labels
        description: Human-readable description of this label set
        domain: Domain name (e.g., "gaming", "robotics", "marine")
    """

    labels: dict[int, str]
    description: str
    domain: str

    def get_label(self, action_index: int) -> str:
        """Get user-facing label for action index.

        Args:
            action_index: Canonical action index (0-7)

        Returns:
            User-facing label string

        Raises:
            KeyError: If action_index not in labels (invalid for substrate)
        """
        return self.labels[action_index]

    def get_all_labels(self) -> dict[int, str]:
        """Get all labels as dictionary."""
        return self.labels.copy()

    def get_action_count(self) -> int:
        """Get number of actions in this label set."""
        return len(self.labels)


# Preset label configurations

PRESET_LABELS: dict[str, ActionLabels] = {
    # Gaming preset: Standard gaming terminology (LEFT, RIGHT, UP, DOWN)
    # Use for: Gaming contexts, general education, intuitive controls
    "gaming": ActionLabels(
        labels={
            CanonicalAction.MOVE_X_NEGATIVE: "LEFT",
            CanonicalAction.MOVE_X_POSITIVE: "RIGHT",
            CanonicalAction.MOVE_Y_NEGATIVE: "DOWN",
            CanonicalAction.MOVE_Y_POSITIVE: "UP",
            CanonicalAction.INTERACT: "INTERACT",
            CanonicalAction.WAIT: "WAIT",
            CanonicalAction.MOVE_Z_POSITIVE: "FORWARD",
            CanonicalAction.MOVE_Z_NEGATIVE: "BACKWARD",
        },
        description="Standard gaming controls (LEFT/RIGHT/UP/DOWN/FORWARD/BACKWARD)",
        domain="gaming",
    ),
    # 6-DoF preset: Robotics six degrees of freedom (SWAY, HEAVE, SURGE)
    # Use for: Robotics, aerospace, underwater vehicles
    # Reference: https://en.wikipedia.org/wiki/Six_degrees_of_freedom
    "6dof": ActionLabels(
        labels={
            CanonicalAction.MOVE_X_NEGATIVE: "SWAY_LEFT",
            CanonicalAction.MOVE_X_POSITIVE: "SWAY_RIGHT",
            CanonicalAction.MOVE_Y_NEGATIVE: "HEAVE_DOWN",
            CanonicalAction.MOVE_Y_POSITIVE: "HEAVE_UP",
            CanonicalAction.INTERACT: "INTERACT",
            CanonicalAction.WAIT: "WAIT",
            CanonicalAction.MOVE_Z_POSITIVE: "SURGE_FORWARD",
            CanonicalAction.MOVE_Z_NEGATIVE: "SURGE_BACKWARD",
        },
        description="Robotics 6-DoF terminology (SWAY/HEAVE/SURGE for translation)",
        domain="robotics",
    ),
    # Cardinal preset: Compass directions (NORTH, SOUTH, EAST, WEST)
    # Use for: Navigation, geography, outdoor contexts
    "cardinal": ActionLabels(
        labels={
            CanonicalAction.MOVE_X_NEGATIVE: "WEST",
            CanonicalAction.MOVE_X_POSITIVE: "EAST",
            CanonicalAction.MOVE_Y_NEGATIVE: "SOUTH",
            CanonicalAction.MOVE_Y_POSITIVE: "NORTH",
            CanonicalAction.INTERACT: "INTERACT",
            CanonicalAction.WAIT: "WAIT",
            CanonicalAction.MOVE_Z_POSITIVE: "ASCEND",
            CanonicalAction.MOVE_Z_NEGATIVE: "DESCEND",
        },
        description="Cardinal directions (NORTH/SOUTH/EAST/WEST/ASCEND/DESCEND)",
        domain="navigation",
    ),
    # Math preset: Explicit axis notation (X_NEG, X_POS, Y_NEG, Y_POS, Z_POS, Z_NEG)
    # Use for: Mathematical contexts, explicit coordinate systems
    "math": ActionLabels(
        labels={
            CanonicalAction.MOVE_X_NEGATIVE: "X_NEG",
            CanonicalAction.MOVE_X_POSITIVE: "X_POS",
            CanonicalAction.MOVE_Y_NEGATIVE: "Y_NEG",
            CanonicalAction.MOVE_Y_POSITIVE: "Y_POS",
            CanonicalAction.INTERACT: "INTERACT",
            CanonicalAction.WAIT: "WAIT",
            CanonicalAction.MOVE_Z_POSITIVE: "Z_POS",
            CanonicalAction.MOVE_Z_NEGATIVE: "Z_NEG",
        },
        description="Mathematical axis notation (X_NEG/X_POS/Y_NEG/Y_POS/Z_POS/Z_NEG)",
        domain="mathematics",
    ),
}


def get_labels(
    preset: str | None = None,
    custom_labels: dict[int, str] | None = None,
    substrate_position_dim: int = 2,
) -> ActionLabels:
    """Get action labels for substrate.

    Args:
        preset: Preset name ("gaming", "6dof", "cardinal", "math") or None for custom
        custom_labels: Custom label dictionary (required if preset=None)
        substrate_position_dim: Substrate dimensionality (0, 1, 2, 3)

    Returns:
        ActionLabels instance filtered to substrate's action space

    Raises:
        ValueError: If preset unknown or custom_labels invalid

    Examples:
        >>> # Gaming labels for 2D substrate
        >>> labels = get_labels(preset="gaming", substrate_position_dim=2)
        >>> labels.get_label(CanonicalAction.MOVE_X_NEGATIVE)
        'LEFT'

        >>> # Custom submarine labels for 3D substrate
        >>> labels = get_labels(
        ...     custom_labels={
        ...         0: "PORT", 1: "STARBOARD", 2: "AFT", 3: "FORE",
        ...         4: "INTERACT", 5: "WAIT", 6: "SURFACE", 7: "DIVE"
        ...     },
        ...     substrate_position_dim=3
        ... )
        >>> labels.get_label(CanonicalAction.MOVE_Z_POSITIVE)
        'SURFACE'
    """
    # Validate inputs
    if preset is None and custom_labels is None:
        raise ValueError("Must provide either preset or custom_labels")

    if preset is not None and preset not in PRESET_LABELS:
        raise ValueError(f"Unknown preset '{preset}'. Available: {list(PRESET_LABELS.keys())}")

    # Get base labels (preset or custom)
    if preset is not None:
        base_labels = PRESET_LABELS[preset]
        all_labels = base_labels.labels.copy()
        description = base_labels.description
        domain = base_labels.domain
    else:
        all_labels = custom_labels.copy()
        description = "Custom action labels"
        domain = "custom"

    # Filter to substrate's action space
    filtered_labels = _filter_labels_for_substrate(all_labels, substrate_position_dim)

    return ActionLabels(labels=filtered_labels, description=description, domain=domain)


def _filter_labels_for_substrate(labels: dict[int, str], position_dim: int) -> dict[int, str]:
    """Filter labels to match substrate's action space.

    Args:
        labels: Full label dictionary (all 8 actions)
        position_dim: Substrate dimensionality (0, 1, 2, 3)

    Returns:
        Filtered label dictionary matching substrate's action count

    Action space mapping:
    - 0D (Aspatial): INTERACT (0), WAIT (1) → 2 actions
    - 1D: MOVE_X_NEGATIVE (0), MOVE_X_POSITIVE (1), INTERACT (2), WAIT (3) → 4 actions
    - 2D: + MOVE_Y_NEGATIVE, MOVE_Y_POSITIVE → 6 actions
    - 3D: + MOVE_Z_POSITIVE, MOVE_Z_NEGATIVE → 8 actions

    Note: Action indices are remapped for aspatial and 1D substrates.
    """
    if position_dim == 0:
        # Aspatial: INTERACT=0, WAIT=1
        return {
            0: labels.get(CanonicalAction.INTERACT, "INTERACT"),
            1: labels.get(CanonicalAction.WAIT, "WAIT"),
        }
    elif position_dim == 1:
        # 1D: MOVE_X_NEGATIVE=0, MOVE_X_POSITIVE=1, INTERACT=2, WAIT=3
        return {
            0: labels.get(CanonicalAction.MOVE_X_NEGATIVE, "MOVE_X_NEGATIVE"),
            1: labels.get(CanonicalAction.MOVE_X_POSITIVE, "MOVE_X_POSITIVE"),
            2: labels.get(CanonicalAction.INTERACT, "INTERACT"),
            3: labels.get(CanonicalAction.WAIT, "WAIT"),
        }
    elif position_dim == 2:
        # 2D: UP=0, DOWN=1, LEFT=2, RIGHT=3, INTERACT=4, WAIT=5
        return {
            0: labels.get(CanonicalAction.MOVE_Y_POSITIVE, "MOVE_Y_POSITIVE"),
            1: labels.get(CanonicalAction.MOVE_Y_NEGATIVE, "MOVE_Y_NEGATIVE"),
            2: labels.get(CanonicalAction.MOVE_X_NEGATIVE, "MOVE_X_NEGATIVE"),
            3: labels.get(CanonicalAction.MOVE_X_POSITIVE, "MOVE_X_POSITIVE"),
            4: labels.get(CanonicalAction.INTERACT, "INTERACT"),
            5: labels.get(CanonicalAction.WAIT, "WAIT"),
        }
    elif position_dim == 3:
        # 3D: UP=0, DOWN=1, LEFT=2, RIGHT=3, INTERACT=4, WAIT=5, UP_Z=6, DOWN_Z=7
        return {
            0: labels.get(CanonicalAction.MOVE_Y_POSITIVE, "MOVE_Y_POSITIVE"),
            1: labels.get(CanonicalAction.MOVE_Y_NEGATIVE, "MOVE_Y_NEGATIVE"),
            2: labels.get(CanonicalAction.MOVE_X_NEGATIVE, "MOVE_X_NEGATIVE"),
            3: labels.get(CanonicalAction.MOVE_X_POSITIVE, "MOVE_X_POSITIVE"),
            4: labels.get(CanonicalAction.INTERACT, "INTERACT"),
            5: labels.get(CanonicalAction.WAIT, "WAIT"),
            6: labels.get(CanonicalAction.MOVE_Z_POSITIVE, "MOVE_Z_POSITIVE"),
            7: labels.get(CanonicalAction.MOVE_Z_NEGATIVE, "MOVE_Z_NEGATIVE"),
        }
    else:
        raise ValueError(f"Invalid position_dim: {position_dim}. Must be 0, 1, 2, or 3.")
