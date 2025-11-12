"""
Drive As Code (DAC) configuration DTOs.

This module defines Pydantic schemas for declarative reward function specifications.
DAC extracts all reward logic from Python code into composable YAML configurations.

Task 1.4: IntrinsicStrategyConfig DTO - Configuration for intrinsic curiosity strategies.
Task 1.5: Shaping Bonus DTOs - TriggerCondition, ApproachRewardConfig, CompletionBonusConfig, VFSVariableBonusConfig.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


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
