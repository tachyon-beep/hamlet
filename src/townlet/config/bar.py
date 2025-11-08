"""Bar (meter) configuration DTO with no-defaults enforcement.

Philosophy: All behavioral parameters must be explicitly specified.
No implicit defaults. Operator accountability.

Design: Validates meter structure from bars.yaml (basic structural validation only).
Cross-file validation (meter references) deferred to TASK-004A.
"""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from townlet.config.base import format_validation_error, load_yaml_section


class BarConfig(BaseModel):
    """Meter (bar) configuration - structural validation only.

    ALL REQUIRED FIELDS must be specified (no defaults) - operator accountability.

    Example:
        >>> bar = BarConfig(
        ...     name="energy",
        ...     index=0,
        ...     tier="pivotal",
        ...     range=[0.0, 1.0],
        ...     initial=1.0,
        ...     base_depletion=0.005,
        ... )
    """

    # Allow extra fields (for metadata like key_insight, cascade_pattern, etc.)
    model_config = ConfigDict(extra="allow")

    # Meter identity (REQUIRED)
    name: str = Field(min_length=1, description="Meter name (e.g., 'energy')")
    index: int = Field(ge=0, description="Index in meters tensor")

    # Tier classification (REQUIRED)
    tier: str = Field(min_length=1, description="Tier (pivotal, primary, secondary, resource)")

    # Value bounds (REQUIRED)
    range: list[float] = Field(min_length=2, max_length=2, description="[min, max] bounds")

    # Initial state (REQUIRED)
    initial: float = Field(description="Starting value (must be within range)")

    # Decay rate (REQUIRED)
    base_depletion: float = Field(description="Passive depletion per step")

    # Metadata (OPTIONAL)
    description: str | None = None

    @field_validator("range")
    @classmethod
    def validate_range_order(cls, v: list[float]) -> list[float]:
        """Ensure range[0] < range[1]."""
        if len(v) == 2 and v[0] >= v[1]:
            raise ValueError(f"range min ({v[0]}) must be < max ({v[1]}). " f"Got reversed or equal bounds: {v}")
        return v

    @model_validator(mode="after")
    def validate_initial_in_range(self) -> "BarConfig":
        """Ensure initial value is within [min, max]."""
        min_val, max_val = self.range
        if not (min_val <= self.initial <= max_val):
            raise ValueError(
                f"initial value ({self.initial}) must be within range [{min_val}, {max_val}]. "
                f"Got: initial={self.initial}, range={self.range}"
            )
        return self


def load_bars_config(config_dir: Path) -> list[BarConfig]:
    """Load and validate bar (meter) configurations.

    Args:
        config_dir: Directory containing bars.yaml

    Returns:
        List of validated BarConfig objects

    Raises:
        FileNotFoundError: If bars.yaml not found
        ValueError: If validation fails (with helpful error message)

    Example:
        >>> bars = load_bars_config(Path("configs/L0_0_minimal"))
        >>> print(f"Loaded {len(bars)} meters")
        Loaded 8 meters
    """
    try:
        data = load_yaml_section(config_dir, "bars.yaml", "bars")
        return [BarConfig(**bar_data) for bar_data in data]
    except ValidationError as e:
        raise ValueError(format_validation_error(e, "bars.yaml")) from e
