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
    """Reward agents for moving closer to target affordance.

    Gives distance-based bonus for proximity to target affordance.

    Example:
        >>> approach_bed = ApproachRewardConfig(
        ...     type="approach_reward",
        ...     weight=0.5,
        ...     target_affordance="Bed",
        ...     max_distance=10.0,
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["approach_reward"]
    weight: float = Field(gt=0.0, description="Bonus weight/magnitude")
    target_affordance: str = Field(description="Affordance to approach")
    max_distance: float = Field(gt=0.0, description="Maximum distance for bonus (beyond this, bonus=0)")


class CompletionBonusConfig(BaseModel):
    """Fixed bonus when agent completes interaction with affordance.

    Gives bonus when agent finishes interacting with specific affordance.

    Example:
        >>> completion = CompletionBonusConfig(
        ...     type="completion_bonus",
        ...     weight=1.0,
        ...     affordance="Bed",
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["completion_bonus"]
    weight: float = Field(gt=0.0, description="Bonus magnitude")
    affordance: str = Field(description="Affordance to reward completion for")


class EfficiencyBonusConfig(BaseModel):
    """Bonus for maintaining bar above threshold.

    Encourages agents to keep resources in healthy ranges.

    Example:
        >>> efficiency = EfficiencyBonusConfig(
        ...     type="efficiency_bonus",
        ...     weight=0.5,
        ...     bar="energy",
        ...     threshold=0.7,
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["efficiency_bonus"]
    weight: float = Field(gt=0.0, description="Bonus magnitude")
    bar: str = Field(description="Bar to monitor")
    threshold: float = Field(ge=0.0, le=1.0, description="Minimum value for bonus")


class BarCondition(BaseModel):
    """Condition on a bar value for state_achievement.

    Example:
        >>> cond = BarCondition(bar="energy", min_value=0.8)
    """

    model_config = ConfigDict(extra="forbid")

    bar: str = Field(description="Bar name")
    min_value: float = Field(ge=0.0, le=1.0, description="Minimum required value")


class StateAchievementConfig(BaseModel):
    """Bonus when ALL specified bar conditions are met.

    Rewards agents for achieving target state (all bars above thresholds).

    Example:
        >>> state_goal = StateAchievementConfig(
        ...     type="state_achievement",
        ...     weight=2.0,
        ...     conditions=[
        ...         BarCondition(bar="energy", min_value=0.8),
        ...         BarCondition(bar="health", min_value=0.8),
        ...     ],
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["state_achievement"]
    weight: float = Field(gt=0.0, description="Bonus magnitude")
    conditions: list[BarCondition] = Field(min_length=1, description="All conditions must be met")


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


class StreakBonusConfig(BaseModel):
    """Bonus for consecutive uses of affordance.

    Rewards agents for building streaks of using the same affordance.

    Example:
        >>> streak_bonus = StreakBonusConfig(
        ...     type="streak_bonus",
        ...     weight=5.0,
        ...     affordance="Bed",
        ...     min_streak=3,
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["streak_bonus"]
    weight: float = Field(gt=0.0, description="Bonus magnitude")
    affordance: str = Field(description="Target affordance to track")
    min_streak: int = Field(ge=1, description="Minimum streak length for bonus")


class DiversityBonusConfig(BaseModel):
    """Bonus for using many different affordances.

    Rewards agents for exploring diverse interactions.

    Example:
        >>> diversity_bonus = DiversityBonusConfig(
        ...     type="diversity_bonus",
        ...     weight=3.0,
        ...     min_unique_affordances=4,
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["diversity_bonus"]
    weight: float = Field(gt=0.0, description="Bonus magnitude")
    min_unique_affordances: int = Field(ge=1, description="Minimum unique affordances for bonus")


class TimeRange(BaseModel):
    """Time range for timing bonus.

    Defines a time window when using a specific affordance grants a bonus.

    Example:
        >>> nighttime_sleep = TimeRange(
        ...     start_hour=22,
        ...     end_hour=6,
        ...     affordance="Bed",
        ...     multiplier=2.0,
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    start_hour: int = Field(ge=0, le=23, description="Start hour (inclusive)")
    end_hour: int = Field(ge=0, le=23, description="End hour (inclusive)")
    affordance: str = Field(description="Affordance to reward during this window")
    multiplier: float = Field(gt=0.0, description="Bonus multiplier for this window")


class TimingBonusConfig(BaseModel):
    """Bonus for using affordance during specific time windows.

    Rewards agents for contextually appropriate timing (e.g., sleeping at night).

    Example:
        >>> timing_bonus = TimingBonusConfig(
        ...     type="timing_bonus",
        ...     weight=1.0,
        ...     time_ranges=[
        ...         TimeRange(start_hour=22, end_hour=6, affordance="Bed", multiplier=2.0),
        ...     ],
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["timing_bonus"]
    weight: float = Field(gt=0.0, description="Base bonus weight")
    time_ranges: list[TimeRange] = Field(min_length=1, description="Time windows with multipliers")


class EconomicEfficiencyConfig(BaseModel):
    """Bonus for maintaining money above threshold.

    Rewards agents for financial responsibility and resource management.

    Example:
        >>> economic_bonus = EconomicEfficiencyConfig(
        ...     type="economic_efficiency",
        ...     weight=2.0,
        ...     money_bar="money",
        ...     min_balance=0.6,
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["economic_efficiency"]
    weight: float = Field(gt=0.0, description="Bonus magnitude")
    money_bar: str = Field(description="Money bar name")
    min_balance: float = Field(ge=0.0, le=1.0, description="Minimum balance for bonus")


class BalanceBonusConfig(BaseModel):
    """Bonus for keeping multiple bars balanced (close in value).

    Rewards agents for maintaining equilibrium across multiple resources.

    Example:
        >>> balance_bonus = BalanceBonusConfig(
        ...     type="balance_bonus",
        ...     weight=5.0,
        ...     bars=["energy", "health"],
        ...     max_imbalance=0.2,
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["balance_bonus"]
    weight: float = Field(gt=0.0, description="Bonus magnitude")
    bars: list[str] = Field(min_length=2, description="Bars to balance (min 2)")
    max_imbalance: float = Field(gt=0.0, le=1.0, description="Max allowed difference between bars")


# Union type for all shaping bonuses (expand as more types are added)
ShapingBonusConfig = (
    ApproachRewardConfig
    | CompletionBonusConfig
    | EfficiencyBonusConfig
    | StateAchievementConfig
    | VFSVariableBonusConfig
    | StreakBonusConfig
    | DiversityBonusConfig
    | TimingBonusConfig
    | EconomicEfficiencyConfig
    | BalanceBonusConfig
)


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
