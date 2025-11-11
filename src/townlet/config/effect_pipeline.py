"""Effect pipeline DTOs for affordance configuration."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class AffordanceEffect(BaseModel):
    """Single meter effect applied during an affordance lifecycle stage."""

    model_config = ConfigDict(extra="ignore")

    meter: str = Field(min_length=1, description="Meter to modify")
    amount: float = Field(description="Delta applied to the meter (positive or negative)")


class EffectPipeline(BaseModel):
    """Multi-stage effect pipeline for advanced affordances."""

    model_config = ConfigDict(extra="ignore")

    on_start: list[AffordanceEffect] = Field(default_factory=list)
    per_tick: list[AffordanceEffect] = Field(default_factory=list)
    on_completion: list[AffordanceEffect] = Field(default_factory=list)
    on_early_exit: list[AffordanceEffect] = Field(default_factory=list)
    on_failure: list[AffordanceEffect] = Field(default_factory=list)

    def has_effects(self) -> bool:
        """Return True when any lifecycle stage is populated."""

        return any(
            (
                self.on_start,
                self.per_tick,
                self.on_completion,
                self.on_early_exit,
                self.on_failure,
            )
        )
