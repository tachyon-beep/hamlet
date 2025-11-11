"""Drive As Code configuration DTOs.

Declarative reward function specification system that extracts reward logic
from Python into composable YAML configurations.

Philosophy: Reward functions are compositions of extrinsic structure, modifiers,
shaping bonuses, and intrinsic computation. All components configurable via YAML.
"""

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RangeConfig(BaseModel):
    """Single range in a modifier.

    Ranges define multiplier values for different value ranges.
    Used for context-sensitive reward adjustment (crisis suppression, etc).

    Example:
        >>> crisis_range = RangeConfig(
        ...     name="crisis",
        ...     min=0.0,
        ...     max=0.2,
        ...     multiplier=0.0,
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, description="Human-readable range name")
    min: float = Field(description="Range minimum (inclusive)")
    max: float = Field(description="Range maximum (exclusive for all but last)")
    multiplier: float = Field(description="Multiplier to apply when value in this range")

    @model_validator(mode="after")
    def validate_range_bounds(self) -> "RangeConfig":
        """Ensure min < max."""
        if self.min >= self.max:
            raise ValueError(f"min ({self.min}) must be < max ({self.max})")
        return self


class ModifierConfig(BaseModel):
    """Range-based modifier for contextual reward adjustment.

    Modifiers apply multipliers based on the current value of a bar or VFS variable.
    Used for crisis suppression, temporal decay, boredom boost, etc.

    Example:
        >>> energy_crisis = ModifierConfig(
        ...     bar="energy",
        ...     ranges=[
        ...         RangeConfig(name="crisis", min=0.0, max=0.2, multiplier=0.0),
        ...         RangeConfig(name="low", min=0.2, max=0.4, multiplier=0.3),
        ...         RangeConfig(name="normal", min=0.4, max=1.0, multiplier=1.0),
        ...     ],
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    # Source (exactly one required)
    bar: str | None = Field(default=None, description="Bar name to monitor")
    variable: str | None = Field(default=None, description="VFS variable name to monitor")

    # Range definitions
    ranges: list[RangeConfig] = Field(min_length=1, description="Range definitions with multipliers")

    @model_validator(mode="after")
    def validate_source(self) -> "ModifierConfig":
        """Ensure exactly one source specified."""
        if self.bar is None and self.variable is None:
            raise ValueError("Must specify either 'bar' or 'variable'")
        if self.bar is not None and self.variable is not None:
            raise ValueError("Cannot specify both 'bar' and 'variable'")
        return self

    @model_validator(mode="after")
    def validate_ranges_coverage(self) -> "ModifierConfig":
        """Ensure ranges cover [0.0, 1.0] without gaps or overlaps."""
        sorted_ranges = sorted(self.ranges, key=lambda r: r.min)

        # Check coverage starts at 0.0
        if sorted_ranges[0].min != 0.0:
            raise ValueError(f"Ranges must start at 0.0, got {sorted_ranges[0].min}")

        # Check no gaps or overlaps
        for i in range(len(sorted_ranges) - 1):
            current_max = sorted_ranges[i].max
            next_min = sorted_ranges[i + 1].min
            if current_max != next_min:
                raise ValueError(
                    f"Gap or overlap between ranges: "
                    f"{sorted_ranges[i].name} (max={current_max}) and "
                    f"{sorted_ranges[i+1].name} (min={next_min})"
                )

        # Check coverage ends at 1.0
        if sorted_ranges[-1].max != 1.0:
            raise ValueError(f"Ranges must end at 1.0, got {sorted_ranges[-1].max}")

        return self
