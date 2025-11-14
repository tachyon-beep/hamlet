"""Builder factories for common test objects.

This module provides factory functions and data-only helpers (formerly split
between ``tests.test_townlet.builders`` and ``tests.test_townlet.utils.builders``)
so contributors have a single canonical import path.

Design principles:
- All parameters have sensible defaults
- Keyword-only arguments for clarity
- Comprehensive docstrings with examples
- Type hints for IDE autocomplete

Example:
    from tests.test_townlet.utils.builders import make_grid2d_substrate, make_test_bars_config

    def test_something():
        substrate = make_grid2d_substrate(width=3, height=3)
        bars = make_test_bars_config()
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

from townlet.environment.affordance_config import AffordanceConfig, AffordanceEffect
from townlet.environment.cascade_config import BarConfig, BarsConfig, TerminalCondition
from townlet.environment.vectorized_env import VectorizedHamletEnv
from townlet.recording.data_structures import EpisodeMetadata, RecordedStep
from townlet.substrate.grid2d import Grid2DSubstrate
from townlet.substrate.grid3d import Grid3DSubstrate
from townlet.universe.compiled import CompiledUniverse
from townlet.universe.compiler import UniverseCompiler

# =============================================================================
# INTERNAL HELPERS
# =============================================================================

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONFIGS_ROOT = (_REPO_ROOT / "configs").resolve()
_COMPILER = UniverseCompiler()
_UNIVERSE_CACHE: dict[Path, CompiledUniverse] = {}


def _resolve_config_path(config_dir: Path | str) -> Path:
    return Path(config_dir).resolve()


def _is_repo_config(path: Path) -> bool:
    try:
        path.relative_to(_CONFIGS_ROOT)
        return True
    except ValueError:
        return False


def _compile_universe(config_dir: Path | str) -> CompiledUniverse:
    path = _resolve_config_path(config_dir)
    cacheable = _is_repo_config(path)

    if cacheable and path in _UNIVERSE_CACHE:
        return _UNIVERSE_CACHE[path]

    compiled = _COMPILER.compile(path)
    if cacheable:
        _UNIVERSE_CACHE[path] = compiled
    return compiled


@dataclass
class TestDimensions:
    """Canonical dimension calculations used throughout the test suite."""

    GRID_SIZE: int = 8
    NUM_METERS: int = 8
    NUM_AFFORDANCES: int = 14

    GRID2D_POSITION_DIM: int = 2
    GRID2D_METER_DIM: int = 8
    GRID2D_AFFORDANCE_DIM: int = 15
    GRID2D_TEMPORAL_DIM: int = 4
    GRID2D_OBS_DIM: int = 29
    GRID2D_ACTION_DIM: int = 8

    POMDP_VISION_RANGE: int = 2
    POMDP_WINDOW_SIZE: int = 5
    POMDP_WINDOW_CELLS: int = 25
    POMDP_OBS_DIM: int = 54

    GRID3D_POSITION_DIM: int = 3
    GRID3D_ACTION_DIM: int = 10

    GRIDND_7D_POSITION_DIM: int = 7
    GRIDND_7D_ACTION_DIM: int = 16


# =============================================================================
# SUBSTRATE BUILDERS
# =============================================================================


def make_grid2d_substrate(
    *,
    width: int = 8,
    height: int = 8,
    boundary: Literal["clamp", "wrap", "bounce", "sticky"] = "clamp",
    distance_metric: Literal["manhattan", "euclidean", "chebyshev"] = "manhattan",
    observation_encoding: Literal["relative", "scaled", "absolute"] = "relative",
) -> Grid2DSubstrate:
    """Create Grid2D substrate with sensible defaults.

    Args:
        width: Grid width (default: 8)
        height: Grid height (default: 8)
        boundary: Boundary mode (default: "clamp")
        distance_metric: Distance metric (default: "manhattan")
        observation_encoding: Position encoding (default: "relative")

    Returns:
        Configured Grid2DSubstrate

    Example:
        >>> substrate = make_grid2d_substrate(width=3, height=3)
        >>> substrate.action_space_size
        6
        >>> substrate.position_dim
        2
    """
    return Grid2DSubstrate(
        width=width,
        height=height,
        boundary=boundary,
        distance_metric=distance_metric,
        observation_encoding=observation_encoding,
    )


def make_grid3d_substrate(
    *,
    width: int = 5,
    height: int = 5,
    depth: int = 3,
    boundary: Literal["clamp", "wrap", "bounce", "sticky"] = "clamp",
    distance_metric: Literal["manhattan", "euclidean", "chebyshev"] = "manhattan",
    observation_encoding: Literal["relative", "scaled", "absolute"] = "relative",
) -> Grid3DSubstrate:
    """Create Grid3D substrate with sensible defaults.

    Args:
        width: Grid width (default: 5)
        height: Grid height (default: 5)
        depth: Grid depth (default: 3)
        boundary: Boundary mode (default: "clamp")
        distance_metric: Distance metric (default: "manhattan")
        observation_encoding: Position encoding (default: "relative")

    Returns:
        Configured Grid3DSubstrate

    Example:
        >>> substrate = make_grid3d_substrate(width=5, height=5, depth=3)
        >>> substrate.action_space_size
        8
        >>> substrate.position_dim
        3
    """
    return Grid3DSubstrate(
        width=width,
        height=height,
        depth=depth,
        boundary=boundary,
        distance_metric=distance_metric,
        observation_encoding=observation_encoding,
    )


# =============================================================================
# BARS CONFIG BUILDERS
# =============================================================================


def make_bars_config(
    *,
    meter_count: int = 8,
    version: str = "2.0",
    description: str = "Test universe",
) -> BarsConfig:
    """Create BarsConfig with variable meter count.

    Creates a minimal bars config with the specified number of meters.
    All meters are named meter_0, meter_1, etc. for simplicity.

    Args:
        meter_count: Number of meters (1-32, default: 8)
        version: Config version string (default: "2.0")
        description: Config description (default: "Test universe")

    Returns:
        BarsConfig with specified meter count

    Raises:
        ValueError: If meter_count < 1 or meter_count > 32

    Example:
        >>> config = make_bars_config(meter_count=4)
        >>> config.meter_count
        4
        >>> config.meter_names
        ['meter_0', 'meter_1', 'meter_2', 'meter_3']
    """
    if meter_count < 1:
        raise ValueError(f"meter_count must be >= 1, got {meter_count}")
    if meter_count > 32:
        raise ValueError(f"meter_count must be <= 32, got {meter_count}")

    # Create minimal bar configs
    bars = []
    for i in range(meter_count):
        # First two meters are pivotal (energy, health)
        tier = "pivotal" if i < 2 else "secondary"

        bars.append(
            BarConfig(
                name=f"meter_{i}",
                index=i,
                tier=tier,
                range=(0.0, 1.0),
                initial=1.0,
                base_depletion=0.001,
                base_move_depletion=0.0,
                base_interaction_cost=0.0,
                description=f"Test meter {i}",
            )
        )

    # Terminal conditions for pivotal meters
    terminal_conditions = [
        TerminalCondition(
            meter="meter_0",
            operator="<=",
            value=0.0,
            description="Death by meter_0 depletion",
        )
    ]

    if meter_count >= 2:
        terminal_conditions.append(
            TerminalCondition(
                meter="meter_1",
                operator="<=",
                value=0.0,
                description="Death by meter_1 depletion",
            )
        )

    return BarsConfig(
        version=version,
        description=description,
        bars=bars,
        terminal_conditions=terminal_conditions,
    )


def make_standard_8meter_config() -> BarsConfig:
    """Create standard 8-meter config matching test config pack.

    This creates the canonical 8-meter configuration used in most tests:
    energy, health, satiation, money, mood, social, fitness, hygiene.

    Returns:
        BarsConfig with standard 8 meters

    Example:
        >>> config = make_standard_8meter_config()
        >>> config.meter_count
        8
        >>> config.meter_names[0]
        'energy'
    """
    return BarsConfig(
        version="2.0",
        description="Standard 8-meter test universe",
        bars=[
            BarConfig(
                name="energy",
                index=0,
                tier="pivotal",
                range=(0.0, 1.0),
                initial=1.0,
                base_depletion=0.005,
                base_move_depletion=0.0,
                base_interaction_cost=0.0,
                description="Energy level",
            ),
            BarConfig(
                name="health",
                index=1,
                tier="pivotal",
                range=(0.0, 1.0),
                initial=1.0,
                base_depletion=0.0,
                base_move_depletion=0.0,
                base_interaction_cost=0.0,
                description="Health status",
            ),
            BarConfig(
                name="satiation",
                index=2,
                tier="primary",
                range=(0.0, 1.0),
                initial=0.8,
                base_depletion=0.004,
                base_move_depletion=0.0,
                base_interaction_cost=0.0,
                description="Hunger level",
            ),
            BarConfig(
                name="money",
                index=3,
                tier="resource",
                range=(0.0, 1.0),
                initial=0.5,
                base_depletion=0.0,
                base_move_depletion=0.0,
                base_interaction_cost=0.0,
                description="Financial resources",
            ),
            BarConfig(
                name="mood",
                index=4,
                tier="secondary",
                range=(0.0, 1.0),
                initial=0.7,
                base_depletion=0.001,
                base_move_depletion=0.0,
                base_interaction_cost=0.0,
                description="Mood state",
            ),
            BarConfig(
                name="social",
                index=5,
                tier="secondary",
                range=(0.0, 1.0),
                initial=0.6,
                base_depletion=0.002,
                base_move_depletion=0.0,
                base_interaction_cost=0.0,
                description="Social wellbeing",
            ),
            BarConfig(
                name="fitness",
                index=6,
                tier="secondary",
                range=(0.0, 1.0),
                initial=0.5,
                base_depletion=0.003,
                base_move_depletion=0.0,
                base_interaction_cost=0.0,
                description="Physical fitness",
            ),
            BarConfig(
                name="hygiene",
                index=7,
                tier="secondary",
                range=(0.0, 1.0),
                initial=0.9,
                base_depletion=0.006,
                base_move_depletion=0.0,
                base_interaction_cost=0.0,
                description="Hygiene level",
            ),
        ],
        terminal_conditions=[
            TerminalCondition(
                meter="energy",
                operator="<=",
                value=0.0,
                description="Death by energy depletion",
            ),
            TerminalCondition(
                meter="health",
                operator="<=",
                value=0.0,
                description="Death by health failure",
            ),
        ],
    )


# =============================================================================
# POSITION BUILDERS
# =============================================================================


def make_positions(
    *,
    num_agents: int,
    position_dim: int,
    device: torch.device | None = None,
    value: int = 0,
) -> torch.Tensor:
    """Create position tensor with uniform values.

    Args:
        num_agents: Number of agents
        position_dim: Position dimensionality (2 for Grid2D, 3 for Grid3D)
        device: Device to place tensor on (default: CPU)
        value: Position value for all agents and dimensions (default: 0)

    Returns:
        Position tensor of shape (num_agents, position_dim)

    Example:
        >>> positions = make_positions(num_agents=4, position_dim=2, value=0)
        >>> positions.shape
        torch.Size([4, 2])
        >>> positions[0].tolist()
        [0, 0]
    """
    if device is None:
        device = torch.device("cpu")

    return torch.full(
        (num_agents, position_dim),
        value,
        dtype=torch.long,
        device=device,
    )


# =============================================================================
# ENVIRONMENT BUILDERS
# =============================================================================


def make_vectorized_env_from_pack(
    config_dir: Path | str,
    *,
    num_agents: int = 1,
    device: torch.device | str = "cpu",
) -> VectorizedHamletEnv:
    """Instantiate VectorizedHamletEnv from a compiled config pack."""

    universe = _compile_universe(config_dir)
    return VectorizedHamletEnv.from_universe(
        universe,
        num_agents=num_agents,
        device=device,
    )


# =============================================================================
# LEGACY TEST DATA BUILDERS (from tests.test_townlet.builders)
# =============================================================================


def make_test_meters() -> tuple[float, ...]:
    """Return canonical 8-meter tuple used across recording/config tests."""

    return (1.0, 0.9, 0.8, 0.5, 0.7, 0.6, 0.95, 0.85)


def make_test_bar(
    name: str = "energy",
    index: int = 0,
    tier: Literal["pivotal", "primary", "secondary", "resource"] = "pivotal",
    initial: float = 1.0,
    base_depletion: float = 0.01,
    base_move_depletion: float = 0.0,
    base_interaction_cost: float = 0.0,
    description: str | None = None,
) -> BarConfig:
    """Construct a single BarConfig with sensible defaults."""

    if description is None:
        description = f"{name.capitalize()} meter"

    return BarConfig(
        name=name,
        index=index,
        tier=tier,
        initial=initial,
        base_depletion=base_depletion,
        base_move_depletion=base_move_depletion,
        base_interaction_cost=base_interaction_cost,
        description=description,
    )


def make_test_terminal_condition(
    meter: str = "energy",
    operator: Literal["<=", ">=", "<", ">", "=="] = "<=",
    value: float = 0.0,
    description: str | None = None,
) -> TerminalCondition:
    """Construct a minimal terminal condition for meter depletion tests."""

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
    """Canonical BarsConfig (up to 8 meters) used in unit tests."""

    meter_names = [
        "energy",
        "health",
        "satiation",
        "money",
        "mood",
        "social",
        "fitness",
        "hygiene",
    ]
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

    if num_meters > len(meter_names):
        raise ValueError("make_test_bars_config supports up to 8 meters")

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

    terminal_conditions: list[TerminalCondition] = []
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
    """Construct a lightweight AffordanceConfig for tests."""

    if name is None:
        name = id

    if interaction_type in {"multi_tick", "dual"} and required_ticks is None:
        required_ticks = 5

    effect_list = [AffordanceEffect(meter=meter, amount=amount) for (meter, amount) in (effects or [("energy", 0.5)])]

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
    """Create a deterministic EpisodeMetadata payload."""

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
        custom_action_uses={},
    )


def make_test_recorded_step(
    step: int = 0,
    position: tuple[int, ...] = (3, 5),
    meters: tuple[float, ...] | None = None,
    action: int = 2,
    reward: float = 1.0,
    done: bool = False,
) -> RecordedStep:
    """Create a RecordedStep with canonical defaults."""

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
