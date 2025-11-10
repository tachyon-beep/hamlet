"""Capability DTOs for affordance configuration."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class MultiTickCapability(BaseModel):
    """Multi-tick interaction semantics.

    NOTE: duration_ticks is DEPRECATED in capabilities array.
    Use root-level AffordanceConfig.duration_ticks instead (the only one used at runtime).
    This field is kept optional for backwards compatibility but should be omitted in new configs.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["multi_tick"]
    duration_ticks: int | None = Field(default=None, description="DEPRECATED: Use root-level duration_ticks instead")
    early_exit_allowed: bool = False
    resumable: bool = False

    @model_validator(mode="after")
    def validate_duration_ticks_if_present(self) -> MultiTickCapability:
        """Validate duration_ticks > 0 if provided."""
        if self.duration_ticks is not None and self.duration_ticks <= 0:
            raise ValueError("duration_ticks must be > 0 if provided")
        return self


class CooldownCapability(BaseModel):
    """Cooldown requirement after interaction completes."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["cooldown"]
    cooldown_ticks: int = Field(gt=0, description="Number of ticks before interaction can run again")
    scope: Literal["agent", "global"] = "agent"


class MeterGatedCapability(BaseModel):
    """Meter range gate for initiating an affordance."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["meter_gated"]
    meter: str
    min: float | None = None
    max: float | None = None

    @model_validator(mode="after")
    def validate_bounds(self) -> MeterGatedCapability:
        if self.min is None and self.max is None:
            raise ValueError("At least one of 'min' or 'max' must be specified for meter_gated capability")
        if self.min is not None and self.max is not None and self.min >= self.max:
            raise ValueError(f"min ({self.min}) must be < max ({self.max}) for meter_gated capability")
        return self


class SkillScalingCapability(BaseModel):
    """Scales effects based on a skill meter."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["skill_scaling"]
    skill: str
    base_multiplier: float = 1.0
    max_multiplier: float = 2.0

    @model_validator(mode="after")
    def validate_multiplier_order(self) -> SkillScalingCapability:
        if self.base_multiplier > self.max_multiplier:
            raise ValueError("base_multiplier must be <= max_multiplier")
        return self


class ProbabilisticCapability(BaseModel):
    """Probabilistic success/failure behavior."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["probabilistic"]
    success_probability: float = Field(ge=0.0, le=1.0)


class PrerequisiteCapability(BaseModel):
    """Requires completion of other affordances first."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["prerequisite"]
    required_affordances: list[str] = Field(min_length=1)

    @field_validator("required_affordances")
    @classmethod
    def validate_nonempty(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("required_affordances cannot be empty")
        return value


CapabilityConfig = (
    MultiTickCapability
    | CooldownCapability
    | MeterGatedCapability
    | SkillScalingCapability
    | ProbabilisticCapability
    | PrerequisiteCapability
)


__all__ = [
    "CapabilityConfig",
    "CooldownCapability",
    "MeterGatedCapability",
    "MultiTickCapability",
    "ProbabilisticCapability",
    "PrerequisiteCapability",
    "SkillScalingCapability",
]
