"""Availability and operating-mode DTOs for affordances."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class BarConstraint(BaseModel):
    """Meter-based availability constraint."""

    model_config = ConfigDict(extra="forbid")

    meter: str = Field(min_length=1)
    min: float | None = None
    max: float | None = None

    @model_validator(mode="after")
    def validate_bounds(self) -> BarConstraint:
        if self.min is None and self.max is None:
            raise ValueError("At least one of 'min' or 'max' must be specified for availability constraints")
        if self.min is not None and self.max is not None and self.min >= self.max:
            raise ValueError(f"min ({self.min}) must be < max ({self.max}) for availability constraint")
        return self


class ModeConfig(BaseModel):
    """Mode-specific affordance configuration (e.g., operating hours)."""

    model_config = ConfigDict(extra="forbid")

    hours: tuple[int, int] | None = None
    effects: dict[str, float] | None = None

    @field_validator("hours")
    @classmethod
    def validate_hours(cls, value: tuple[int, int] | None) -> tuple[int, int] | None:
        if value is None:
            return None
        start, end = value
        if not (0 <= start <= 23):
            raise ValueError(f"Start hour must be within 0-23, got {start}")
        if not (0 <= end <= 23):
            raise ValueError(f"End hour must be within 0-23, got {end}")
        return value


__all__ = ["BarConstraint", "ModeConfig"]
