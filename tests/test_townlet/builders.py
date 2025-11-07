"""Centralized test data builders and factories.

Provides single source of truth for test entity construction.
Eliminates magic numbers and boilerplate Pydantic instantiation.

Usage:
    from tests.test_townlet.builders import (
        TestDimensions,
        make_test_meters,
        make_test_bars_config,
    )

    # Use canonical dimensions
    obs_dim = TestDimensions.GRID2D_OBS_DIM  # 29

    # Use standard test meters
    meters = make_test_meters()  # (1.0, 0.9, 0.8, ...)

    # Create minimal config
    bars = make_test_bars_config(num_meters=8)
"""

from dataclasses import dataclass
from typing import Literal

from townlet.environment.affordance_config import (
    AffordanceConfig,
    AffordanceEffect,
)
from townlet.environment.cascade_config import (
    BarConfig,
    BarsConfig,
    TerminalCondition,
)
from townlet.recording.data_structures import (
    EpisodeMetadata,
    RecordedStep,
)


@dataclass
class TestDimensions:
    """Canonical dimension calculations for all substrates.

    These are SINGLE SOURCE OF TRUTH for test dimension expectations.
    Any changes to substrate dimensions should update these values.
    """

    # Standard test grid
    GRID_SIZE: int = 8
    NUM_METERS: int = 8
    NUM_AFFORDANCES: int = 14  # Vocabulary size

    # Grid2D substrate (relative encoding)
    GRID2D_POSITION_DIM: int = 2  # x, y normalized
    GRID2D_METER_DIM: int = 8
    GRID2D_AFFORDANCE_DIM: int = 15  # 14 affordances + 1 "none"
    GRID2D_TEMPORAL_DIM: int = 4  # sin, cos, progress, lifetime
    GRID2D_OBS_DIM: int = 29  # 2 + 8 + 15 + 4
    GRID2D_ACTION_DIM: int = 8  # 6 substrate + 2 custom

    # POMDP (5x5 window)
    POMDP_VISION_RANGE: int = 2
    POMDP_WINDOW_SIZE: int = 5  # 2 * 2 + 1
    POMDP_WINDOW_CELLS: int = 25  # 5 * 5
    POMDP_OBS_DIM: int = 54  # 25 + 2 + 8 + 15 + 4

    # Grid3D substrate
    GRID3D_POSITION_DIM: int = 3  # x, y, z
    GRID3D_ACTION_DIM: int = 10  # 8 substrate + 2 custom

    # GridND (7D example)
    GRIDND_7D_POSITION_DIM: int = 7
    GRIDND_7D_ACTION_DIM: int = 16  # 14 substrate + 2 custom


def make_test_meters() -> tuple[float, ...]:
    """Create standard 8-meter test values.

    Returns normalized meter values for:
    (energy, health, satiation, money, mood, social, fitness, hygiene)

    Each value chosen to be distinct for debugging.
    """
    return (1.0, 0.9, 0.8, 0.5, 0.7, 0.6, 0.95, 0.85)


def make_test_bar(
    name: str = "energy",
    index: int = 0,
    tier: Literal["pivotal", "primary", "secondary", "resource"] = "pivotal",
    initial: float = 1.0,
    base_depletion: float = 0.01,
    description: str | None = None,
) -> BarConfig:
    """Create minimal BarConfig for testing.

    Args:
        name: Meter name
        index: Meter index in tensor
        tier: Cascade tier
        initial: Initial normalized value [0, 1]
        base_depletion: Passive decay per step
        description: Human-readable description

    Returns:
        Valid BarConfig ready for testing
    """
    if description is None:
        description = f"{name.capitalize()} meter"

    return BarConfig(
        name=name,
        index=index,
        tier=tier,
        initial=initial,
        base_depletion=base_depletion,
        description=description,
    )


def make_test_terminal_condition(
    meter: str = "energy",
    operator: Literal["<=", ">=", "<", ">", "=="] = "<=",
    value: float = 0.0,
    description: str | None = None,
) -> TerminalCondition:
    """Create minimal TerminalCondition for testing."""
    if description is None:
        description = f"Death by {meter} {operator} {value}"

    return TerminalCondition(
        meter=meter,
        operator=operator,
        value=value,
        description=description,
    )


def make_test_bars_config(
    num_meters: int = 8,
    include_terminal: bool = True,
) -> BarsConfig:
    """Create minimal BarsConfig for testing.

    Args:
        num_meters: Number of meters to include (1-8)
        include_terminal: Include energy depletion terminal condition

    Returns:
        Valid BarsConfig with standard test meters
    """
    meter_names = ["energy", "health", "satiation", "money", "mood", "social", "fitness", "hygiene"]
    tiers: list[Literal["pivotal", "primary", "secondary", "resource"]] = [
        "pivotal",
        "pivotal",
        "primary",
        "resource",
        "secondary",
        "secondary",
        "secondary",
        "secondary",
    ]

    if num_meters > 8:
        raise ValueError(f"make_test_bars_config supports up to 8 meters, got {num_meters}")

    bars = [
        make_test_bar(
            name=meter_names[i],
            index=i,
            tier=tiers[i],
            initial=1.0,
            base_depletion=0.01 if i < 2 else 0.005,
        )
        for i in range(num_meters)
    ]

    terminal_conditions = []
    if include_terminal:
        terminal_conditions.append(
            make_test_terminal_condition(
                meter="energy",
                operator="<=",
                value=0.0,
            )
        )

    return BarsConfig(
        version="1.0",
        description="Test bars configuration",
        bars=bars,
        terminal_conditions=terminal_conditions,
    )


def make_test_affordance(
    id: str = "Bed",
    name: str | None = None,
    category: str = "energy_restoration",
    interaction_type: Literal["instant", "multi_tick", "continuous", "dual"] = "instant",
    required_ticks: int | None = None,
    effects: list[tuple[str, float]] | None = None,
    operating_hours: tuple[int, int] = (0, 24),
) -> AffordanceConfig:
    """Create minimal AffordanceConfig for testing.

    Args:
        id: Affordance ID
        name: Human-readable name (defaults to id)
        category: Affordance category
        interaction_type: Type of interaction
        required_ticks: Required ticks (for multi_tick/dual)
        effects: List of (meter, amount) tuples
        operating_hours: (open, close) tuple

    Returns:
        Valid AffordanceConfig ready for testing
    """
    if name is None:
        name = id

    # Auto-set required_ticks for multi_tick/dual
    if interaction_type in ["multi_tick", "dual"] and required_ticks is None:
        required_ticks = 5

    # Default effects
    effect_list = []
    if effects:
        effect_list = [AffordanceEffect(meter=meter, amount=amount) for meter, amount in effects]

    return AffordanceConfig(
        id=id,
        name=name,
        category=category,
        interaction_type=interaction_type,
        required_ticks=required_ticks,
        effects=effect_list,
        operating_hours=list(operating_hours),
    )


def make_test_episode_metadata(
    episode_id: int = 100,
    survival_steps: int = 10,
    total_reward: float = 10.0,
    curriculum_stage: int = 1,
) -> EpisodeMetadata:
    """Create minimal EpisodeMetadata for testing."""
    return EpisodeMetadata(
        episode_id=episode_id,
        survival_steps=survival_steps,
        total_reward=total_reward,
        extrinsic_reward=total_reward,
        intrinsic_reward=0.0,
        curriculum_stage=curriculum_stage,
        epsilon=0.5,
        intrinsic_weight=0.0,
        timestamp=1234567890.0,
        affordance_layout={"Bed": (2, 3)},
        affordance_visits={"Bed": 1},
    )


def make_test_recorded_step(
    step: int = 0,
    position: tuple[int, ...] = (3, 5),
    meters: tuple[float, ...] | None = None,
    action: int = 2,
    reward: float = 1.0,
    done: bool = False,
) -> RecordedStep:
    """Create minimal RecordedStep for testing.

    Args:
        step: Step number
        position: Agent position tuple
        meters: Meter values (defaults to standard 8 meters)
        action: Action index
        reward: Extrinsic reward
        done: Episode done flag

    Returns:
        Valid RecordedStep ready for testing
    """
    if meters is None:
        meters = make_test_meters()

    return RecordedStep(
        step=step,
        position=position,
        meters=meters,
        action=action,
        reward=reward,
        intrinsic_reward=0.1,
        done=done,
        q_values=None,
    )
