"""Substrate configuration schema (Pydantic DTOs).

Defines the YAML schema for substrate.yaml files, enforcing structure and
validating configuration at load time.

Design Principles (from TASK-001):
- No-Defaults Principle: All behavioral parameters must be explicit
- Conceptual Agnosticism: Don't assume 2D, grid-based, or Euclidean
- Structural Enforcement: Validate dimensions, boundary modes, metrics
- Permissive Semantics: Allow 3D, hex, continuous, graph, aspatial
"""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class GridConfig(BaseModel):
    """Configuration for grid-based substrates (2D square or 3D cubic).

    No-Defaults Principle: All fields required (no implicit defaults).
    """

    topology: Literal["square", "cubic"] = Field(description="Grid topology (square=2D, cubic=3D)")
    width: int = Field(gt=0, description="Grid width (X dimension)")
    height: int = Field(gt=0, description="Grid height (Y dimension)")
    depth: int | None = Field(None, gt=0, description="Grid depth (Z dimension) - required for cubic topology")
    boundary: Literal["clamp", "wrap", "bounce", "sticky"] = Field(description="Boundary handling mode")
    distance_metric: Literal["manhattan", "euclidean", "chebyshev"] = Field(default="manhattan", description="Distance calculation method")

    @model_validator(mode="after")
    def validate_cubic_requires_depth(self) -> "GridConfig":
        """Cubic topology requires depth parameter."""
        if self.topology == "cubic" and self.depth is None:
            raise ValueError(
                "Cubic topology requires 'depth' parameter.\n"
                "Example:\n"
                "  topology: cubic\n"
                "  width: 8\n"
                "  height: 8\n"
                "  depth: 3  # Required for 3D\n"
            )
        if self.topology == "square" and self.depth is not None:
            raise ValueError("Square topology does not use 'depth' parameter. Remove 'depth' or use topology: cubic")
        return self


# Backward compatibility alias
Grid2DSubstrateConfig = GridConfig


class ContinuousConfig(BaseModel):
    """Configuration for continuous substrates.

    Continuous substrates use float-based positions in bounded space.
    Supports 1D (line), 2D (plane), or 3D (volume) continuous space.

    No-Defaults Principle: All fields required (no implicit defaults).
    """

    dimensions: int = Field(ge=1, le=3, description="Number of dimensions (1, 2, or 3)")

    bounds: list[tuple[float, float]] = Field(description="Bounds for each dimension [(min, max), ...]")

    boundary: Literal["clamp", "wrap", "bounce", "sticky"] = Field(description="Boundary handling mode")

    movement_delta: float = Field(gt=0, description="Distance discrete actions move agent")

    interaction_radius: float = Field(gt=0, description="Distance threshold for affordance interaction")

    distance_metric: Literal["euclidean", "manhattan"] = Field(default="euclidean", description="Distance calculation method")

    @model_validator(mode="after")
    def validate_bounds_match_dimensions(self) -> "ContinuousConfig":
        """Validate bounds and interaction parameters."""
        if len(self.bounds) != self.dimensions:
            raise ValueError(
                f"Number of bounds ({len(self.bounds)}) must match dimensions ({self.dimensions}). "
                f"Example for 2D: bounds=[(0.0, 10.0), (0.0, 10.0)]"
            )

        for i, (min_val, max_val) in enumerate(self.bounds):
            if min_val >= max_val:
                raise ValueError(f"Bound {i} invalid: min ({min_val}) must be < max ({max_val})")

            # Check space is large enough for interaction
            range_size = max_val - min_val
            if range_size < self.interaction_radius:
                raise ValueError(
                    f"Dimension {i} range ({range_size}) < interaction_radius ({self.interaction_radius}). "
                    f"Space too small for affordance interaction."
                )

        # Warn if interaction_radius < movement_delta
        if self.interaction_radius < self.movement_delta:
            import warnings

            warnings.warn(
                f"interaction_radius ({self.interaction_radius}) < movement_delta ({self.movement_delta}). "
                f"Agent may step over affordances without interaction. "
                f"This may be intentional for challenge, but verify configuration.",
                UserWarning,
            )

        return self


class AspatialSubstrateConfig(BaseModel):
    """Configuration for aspatial substrate (no positioning).

    This is the simplest config - presence indicates aspatial mode is enabled.
    No additional configuration needed for aspatial substrates.
    """

    pass  # No configuration fields needed


class ActionLabelConfig(BaseModel):
    """Configuration for action label system (domain-specific terminology).

    Enables configurable action labels to support domain-appropriate terminology
    (gaming, robotics, navigation, mathematics, custom domains).

    Action labels separate canonical action semantics (what substrates interpret)
    from user-facing labels (what students/practitioners see).

    Examples:
        # Gaming preset
        action_labels:
          preset: "gaming"  # Uses LEFT/RIGHT/UP/DOWN/FORWARD/BACKWARD

        # Custom submarine labels
        action_labels:
          custom:
            0: "PORT"
            1: "STARBOARD"
            2: "AFT"
            3: "FORE"
            4: "INTERACT"
            5: "WAIT"
            6: "SURFACE"
            7: "DIVE"

        # Robotics 6-DoF preset
        action_labels:
          preset: "6dof"  # Uses SWAY_LEFT/RIGHT, HEAVE_UP/DOWN, SURGE_FORWARD/BACKWARD
    """

    preset: str | None = Field(None, description="Preset label set (gaming, 6dof, cardinal, math)")
    custom: dict[int, str] | None = Field(None, description="Custom label mapping (action_index → label)")

    @model_validator(mode="after")
    def validate_preset_or_custom(self) -> "ActionLabelConfig":
        """Validate that exactly one of preset or custom is provided."""
        if self.preset is None and self.custom is None:
            raise ValueError("Must specify either 'preset' or 'custom' for action labels")
        if self.preset is not None and self.custom is not None:
            raise ValueError("Cannot specify both 'preset' and 'custom' - choose one")

        # Validate preset name
        if self.preset is not None:
            valid_presets = ["gaming", "6dof", "cardinal", "math"]
            if self.preset not in valid_presets:
                raise ValueError(f"Invalid preset '{self.preset}'. Valid presets: {valid_presets}")

        # Validate custom labels (if provided)
        if self.custom is not None:
            # Check all keys are integers 0-7
            for key in self.custom.keys():
                if not isinstance(key, int) or key < 0 or key > 7:
                    raise ValueError(f"Custom label keys must be integers 0-7, got: {key}")

            # Check all values are non-empty strings
            for value in self.custom.values():
                if not isinstance(value, str) or len(value) == 0:
                    raise ValueError(f"Custom label values must be non-empty strings, got: {value}")

        return self


class SubstrateConfig(BaseModel):
    """Complete substrate configuration.

    Exactly one substrate type must be specified via the 'type' field,
    and the corresponding config must be provided.

    No-Defaults Principle: version, description, type are all required.
    """

    version: str = Field(description="Config version (e.g., '1.0')")
    description: str = Field(description="Human-readable description")
    type: Literal["grid", "continuous", "aspatial"] = Field(description="Substrate type selection")

    # Substrate-specific configs (only one should be populated)
    grid: GridConfig | None = Field(
        None,
        description="Grid substrate configuration (required if type='grid')",
    )
    continuous: ContinuousConfig | None = Field(
        None,
        description="Continuous substrate configuration (required if type='continuous')",
    )
    aspatial: AspatialSubstrateConfig | None = Field(
        None,
        description="Aspatial substrate configuration (required if type='aspatial')",
    )

    @model_validator(mode="after")
    def validate_substrate_type_match(self) -> "SubstrateConfig":
        """Ensure substrate config matches declared type."""
        if self.type == "grid" and self.grid is None:
            raise ValueError("type='grid' requires grid configuration. Add grid: { topology: 'square', width: 8, height: 8, ... }")

        if self.type == "continuous" and self.continuous is None:
            raise ValueError(
                "type='continuous' requires continuous configuration. "
                "Add continuous: { dimensions: 2, bounds: [(0.0, 10.0), (0.0, 10.0)], ... }"
            )

        if self.type == "aspatial" and self.aspatial is None:
            raise ValueError("type='aspatial' requires aspatial configuration. Add aspatial: {}")

        # Ensure only one config is provided
        configs_provided = sum([self.grid is not None, self.continuous is not None, self.aspatial is not None])

        if configs_provided > 1:
            raise ValueError("Only one substrate configuration should be provided")

        if self.type == "grid" and (self.continuous is not None or self.aspatial is not None):
            raise ValueError("type='grid' should not have continuous or aspatial configuration")

        if self.type == "continuous" and (self.grid is not None or self.aspatial is not None):
            raise ValueError("type='continuous' should not have grid or aspatial configuration")

        if self.type == "aspatial" and (self.grid is not None or self.continuous is not None):
            raise ValueError("type='aspatial' should not have grid or continuous configuration")

        return self


def load_substrate_config(config_path: Path) -> SubstrateConfig:
    """Load and validate substrate configuration from YAML.

    Args:
        config_path: Path to substrate.yaml file

    Returns:
        SubstrateConfig: Validated substrate configuration

    Raises:
        FileNotFoundError: If config file does not exist
        ValueError: If config validation fails

    Example:
        >>> config = load_substrate_config(Path("configs/L1/substrate.yaml"))
        >>> print(f"Substrate type: {config.type}")
        >>> print(f"Grid size: {config.grid.width}×{config.grid.height}")
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Substrate config not found: {config_path}")

    with open(config_path) as f:
        data = yaml.safe_load(f)

    try:
        return SubstrateConfig(**data)
    except Exception as e:
        raise ValueError(f"Invalid substrate config at {config_path}: {e}") from e
