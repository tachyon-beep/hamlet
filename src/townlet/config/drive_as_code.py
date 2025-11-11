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
