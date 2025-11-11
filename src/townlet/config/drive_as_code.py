"""Drive As Code configuration DTOs.

Declarative reward function specification system that extracts reward logic
from Python into composable YAML configurations.

Philosophy: Reward functions are compositions of extrinsic structure, modifiers,
shaping bonuses, and intrinsic computation. All components configurable via YAML.
"""

from typing import Literal

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


class BarBonusConfig(BaseModel):
    """Bar-based bonus for constant_base_with_shaped_bonus strategy.

    Applies a bonus/penalty based on deviation from center point.
    Formula: bonus = scale * (bar_value - center)

    Example:
        >>> energy_bonus = BarBonusConfig(
        ...     bar="energy",
        ...     center=0.5,  # Neutral point
        ...     scale=0.5,   # Magnitude
        ... )
        # energy=1.0 → bonus = 0.5 * (1.0 - 0.5) = +0.25
        # energy=0.0 → bonus = 0.5 * (0.0 - 0.5) = -0.25
    """

    model_config = ConfigDict(extra="forbid")

    bar: str = Field(description="Bar name")
    center: float = Field(ge=0.0, le=1.0, description="Neutral point (no bonus/penalty)")
    scale: float = Field(gt=0.0, description="Magnitude of bonus/penalty")


class VariableBonusConfig(BaseModel):
    """VFS variable-based bonus.

    Applies a weighted bonus from a VFS-computed variable.
    Formula: bonus = weight * variable_value

    Example:
        >>> urgency_bonus = VariableBonusConfig(
        ...     variable="energy_urgency",
        ...     weight=0.5,
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    variable: str = Field(description="VFS variable name")
    weight: float = Field(description="Weight (can be negative for penalties)")


class ExtrinsicStrategyConfig(BaseModel):
    """Configuration for extrinsic reward strategies.

    Supports 9 strategy types:
    - multiplicative: reward = base * bar1 * bar2 * ...
    - constant_base_with_shaped_bonus: reward = base + sum(bar_bonuses) + sum(variable_bonuses)
    - additive_unweighted: reward = base + sum(bars)
    - weighted_sum: reward = sum(weight_i * source_i)
    - polynomial: reward = sum(weight_i * source_i^exponent_i)
    - threshold_based: reward = sum(threshold bonuses/penalties)
    - aggregation: reward = base + op(bars) where op ∈ {min, max, mean, product}
    - vfs_variable: reward = variable (delegate to VFS)
    - hybrid: reward = weighted combination of multiple strategies

    Example:
        >>> strategy = ExtrinsicStrategyConfig(
        ...     type="constant_base_with_shaped_bonus",
        ...     base_reward=1.0,
        ...     bar_bonuses=[
        ...         BarBonusConfig(bar="energy", center=0.5, scale=0.5),
        ...     ],
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal[
        "multiplicative",
        "constant_base_with_shaped_bonus",
        "additive_unweighted",
        "weighted_sum",
        "polynomial",
        "threshold_based",
        "aggregation",
        "vfs_variable",
        "hybrid",
    ] = Field(description="Strategy type")

    # constant_base_with_shaped_bonus fields
    base_reward: float | None = Field(default=None, description="Base survival reward")
    bar_bonuses: list[BarBonusConfig] = Field(default_factory=list, description="Bar-based bonuses")
    variable_bonuses: list[VariableBonusConfig] = Field(default_factory=list, description="VFS variable bonuses")

    # multiplicative / additive_unweighted fields
    base: float | None = Field(default=None, description="Base value")
    bars: list[str] = Field(default_factory=list, description="Bar names")

    # vfs_variable fields
    variable: str | None = Field(default=None, description="VFS variable name")

    # TODO: Add fields for other strategy types (weighted_sum, polynomial, threshold_based, aggregation, hybrid)
    # These will be added incrementally as needed

    # Modifier application (optional)
    apply_modifiers: list[str] = Field(default_factory=list, description="Modifier names to apply to extrinsic rewards")
