"""
Drive As Code (DAC) configuration DTOs.

This module defines Pydantic schemas for declarative reward function specifications.
DAC extracts all reward logic from Python code into composable YAML configurations.

Task 1.1: RangeConfig DTO - Range definitions for modifiers.
Task 1.2: ModifierConfig DTO - Range-based multipliers for contextual reward adjustment.
Task 1.3: Extrinsic Strategy DTOs - BarBonusConfig, VariableBonusConfig, ExtrinsicStrategyConfig.
Task 1.4: IntrinsicStrategyConfig DTO - Configuration for intrinsic curiosity strategies.
Task 1.5: Shaping Bonus DTOs - TriggerCondition, ApproachRewardConfig, CompletionBonusConfig, VFSVariableBonusConfig.
Task 1.6: Top-level DAC Config - CompositionConfig, DriveAsCodeConfig, load_drive_as_code_config.
"""

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


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

    # Modifier application (optional)
    apply_modifiers: list[str] = Field(default_factory=list, description="Modifier names to apply to extrinsic rewards")


class IntrinsicStrategyConfig(BaseModel):
    """Configuration for intrinsic curiosity rewards.

    Supports five strategy types:
    - rnd: Random Network Distillation (novelty-seeking)
    - icm: Intrinsic Curiosity Module (prediction error)
    - count_based: State visitation counts
    - adaptive_rnd: RND with performance-based annealing
    - none: No intrinsic rewards (pure extrinsic)

    Modifiers can be applied to adjust intrinsic weight contextually
    (e.g., crisis suppression when resources are low).

    Example:
        ```yaml
        intrinsic:
          strategy: rnd
          base_weight: 0.1
          apply_modifiers: [energy_crisis]
          rnd_config:
            feature_dim: 128
            learning_rate: 0.001
        ```
    """

    model_config = ConfigDict(extra="forbid")

    # Required fields
    strategy: Literal["rnd", "icm", "count_based", "adaptive_rnd", "none"]
    base_weight: float = Field(ge=0.0, le=1.0)
    apply_modifiers: list[str] = Field(default_factory=list)

    # Optional strategy-specific configurations
    rnd_config: dict[str, Any] | None = None
    icm_config: dict[str, Any] | None = None
    count_config: dict[str, Any] | None = None
    adaptive_config: dict[str, Any] | None = None


class TriggerCondition(BaseModel):
    """Condition for triggering a shaping bonus.

    Evaluates to true when source value crosses threshold.

    Example:
        >>> low_energy = TriggerCondition(
        ...     source="bar",
        ...     name="energy",
        ...     below=0.3,
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    source: Literal["bar", "variable"] = Field(description="Source type")
    name: str = Field(description="Source name (bar name or variable name)")
    above: float | None = Field(default=None, description="Trigger when value > threshold")
    below: float | None = Field(default=None, description="Trigger when value < threshold")

    @model_validator(mode="after")
    def validate_threshold(self) -> "TriggerCondition":
        """Ensure at least one threshold specified."""
        if self.above is None and self.below is None:
            raise ValueError("Must specify 'above' or 'below'")
        return self


class ApproachRewardConfig(BaseModel):
    """Encourage moving toward goals when needed.

    Gives bonus reward for moving closer to target affordance when trigger condition is met.

    Example:
        >>> approach_bed = ApproachRewardConfig(
        ...     type="approach_reward",
        ...     target_affordance="Bed",
        ...     trigger=TriggerCondition(source="bar", name="energy", below=0.3),
        ...     bonus=1.0,
        ...     decay_with_distance=True,
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["approach_reward"]
    target_affordance: str = Field(description="Affordance to approach")
    trigger: TriggerCondition = Field(description="When to apply this bonus")
    bonus: float = Field(description="Bonus magnitude")
    decay_with_distance: bool = Field(default=True, description="Reduce bonus with distance")


class CompletionBonusConfig(BaseModel):
    """Reward for completing affordances.

    Gives bonus when agent finishes interacting with affordance.

    Example:
        >>> completion = CompletionBonusConfig(
        ...     type="completion_bonus",
        ...     affordances="all",
        ...     bonus=1.0,
        ...     scale_with_duration=True,
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["completion_bonus"]
    affordances: Literal["all"] | list[str] = Field(description="Which affordances to reward")
    bonus: float = Field(description="Bonus magnitude")
    scale_with_duration: bool = Field(default=True, description="Scale bonus by interaction duration")


class VFSVariableBonusConfig(BaseModel):
    """Shaping bonus from VFS variable (escape hatch for custom logic).

    Allows arbitrary shaping logic to be computed in VFS and referenced here.

    Example:
        >>> custom_bonus = VFSVariableBonusConfig(
        ...     type="vfs_variable",
        ...     variable="custom_shaping_signal",
        ...     weight=1.0,
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["vfs_variable"]
    variable: str = Field(description="VFS variable name")
    weight: float = Field(description="Weight to apply")


# Union type for all shaping bonuses (expand as more types are added)
ShapingBonusConfig = ApproachRewardConfig | CompletionBonusConfig | VFSVariableBonusConfig


class CompositionConfig(BaseModel):
    """Reward composition settings.

    Controls how components are combined into total reward.

    Example:
        >>> composition = CompositionConfig(
        ...     normalize=False,
        ...     clip={"min": -10.0, "max": 100.0},
        ...     log_components=True,
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    normalize: bool = Field(default=False, description="Normalize total reward to [-1, 1] with tanh")
    clip: dict[str, float] | None = Field(default=None, description="Clip total reward (keys: min, max)")
    log_components: bool = Field(default=True, description="Log extrinsic/intrinsic/shaping separately")
    log_modifiers: bool = Field(default=True, description="Log modifier values each step")


class DriveAsCodeConfig(BaseModel):
    """Complete DAC configuration.

    Declarative reward function specification.

    Formula:
        total_reward = extrinsic + (intrinsic * effective_intrinsic_weight) + shaping

    Where:
        effective_intrinsic_weight = base_weight * modifier1 * modifier2 * ...

    Example:
        >>> dac = DriveAsCodeConfig(
        ...     version="1.0",
        ...     modifiers={
        ...         "energy_crisis": ModifierConfig(...),
        ...     },
        ...     extrinsic=ExtrinsicStrategyConfig(...),
        ...     intrinsic=IntrinsicStrategyConfig(...),
        ...     shaping=[...],
        ...     composition=CompositionConfig(),
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    version: str = Field(default="1.0", description="DAC schema version")
    modifiers: dict[str, ModifierConfig] = Field(default_factory=dict, description="Named modifier definitions")
    extrinsic: ExtrinsicStrategyConfig = Field(description="Extrinsic reward strategy")
    intrinsic: IntrinsicStrategyConfig = Field(description="Intrinsic reward strategy")
    shaping: list[ShapingBonusConfig] = Field(default_factory=list, description="Shaping bonuses")
    composition: CompositionConfig = Field(default_factory=CompositionConfig, description="Composition settings")

    @model_validator(mode="after")
    def validate_modifier_references(self) -> "DriveAsCodeConfig":
        """Ensure all referenced modifiers exist."""
        defined = set(self.modifiers.keys())

        # Check extrinsic modifiers
        for mod in self.extrinsic.apply_modifiers:
            if mod not in defined:
                raise ValueError(f"Extrinsic references undefined modifier: {mod}")

        # Check intrinsic modifiers
        for mod in self.intrinsic.apply_modifiers:
            if mod not in defined:
                raise ValueError(f"Intrinsic references undefined modifier: {mod}")

        return self


def load_drive_as_code_config(config_dir: Path) -> DriveAsCodeConfig:
    """Load and validate DAC configuration.

    Args:
        config_dir: Config pack directory (e.g., configs/L0_0_minimal)

    Returns:
        Validated DriveAsCodeConfig

    Raises:
        FileNotFoundError: If drive_as_code.yaml not found
        ValueError: If validation fails

    Example:
        >>> dac = load_drive_as_code_config(Path("configs/L0_5_dual_resource"))
    """
    from townlet.config.base import format_validation_error, load_yaml_section

    try:
        data = load_yaml_section(config_dir, "drive_as_code.yaml", "drive_as_code")
        return DriveAsCodeConfig(**data)
    except ValidationError as e:
        raise ValueError(format_validation_error(e, "drive_as_code.yaml")) from e
