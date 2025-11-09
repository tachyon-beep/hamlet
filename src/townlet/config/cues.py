"""Cues configuration DTO with no-defaults enforcement.

Defines how internal meter states map to observable cues (theory-of-mind signals).
Follows the same UNIVERSE_AS_CODE principles as other config schemas.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from townlet.config.base import format_validation_error


class CueCondition(BaseModel):
    """Single condition used to trigger a cue."""

    model_config = ConfigDict(extra="forbid")

    meter: str = Field(min_length=1, description="Meter referenced by the cue condition")
    operator: Literal["<", "<=", ">", ">=", "==", "!="]
    threshold: float


class SimpleCueConfig(BaseModel):
    """Configuration for a simple cue (single condition)."""

    model_config = ConfigDict(extra="forbid")

    cue_id: str = Field(min_length=1, description="Unique cue identifier")
    name: str = Field(min_length=1, description="Human-readable cue name")
    category: str = Field(min_length=1, description="Cue category (energy, health, etc.)")
    visibility: str = Field(min_length=1, description="Visibility scope (public, private, etc.)")
    condition: CueCondition
    description: str | None = None
    teaching_note: str | None = None
    strategic_value: str | None = None


class CompoundCueConfig(BaseModel):
    """Configuration for compound cues (multiple conditions)."""

    model_config = ConfigDict(extra="forbid")

    cue_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    category: str = Field(min_length=1)
    visibility: str = Field(min_length=1)
    logic: Literal["all_of", "any_of"]
    conditions: list[CueCondition] = Field(min_length=1)
    description: str | None = None
    teaching_note: str | None = None
    strategic_value: str | None = None

    @field_validator("conditions")
    @classmethod
    def validate_conditions_non_empty(cls, value: list[CueCondition]) -> list[CueCondition]:
        """Ensure compound cues have at least one condition."""
        if not value:
            raise ValueError("Compound cue must define at least one condition")
        return value


class VisualCueConfig(BaseModel):
    """Visual cue definition for meter range mappings."""

    model_config = ConfigDict(extra="forbid")

    range: tuple[float, float]
    label: str = Field(min_length=1)
    icon: str | None = None
    observable_effects: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_range(self) -> VisualCueConfig:
        start, end = self.range
        if not (0.0 <= start < end <= 1.0):
            raise ValueError(f"Visual cue range must lie within [0.0, 1.0] and have start < end. Got {self.range}.")
        return self


class CuesConfig(BaseModel):
    """Top-level cues configuration."""

    model_config = ConfigDict(extra="forbid")

    version: str = Field(min_length=1)
    description: str | None = None
    status: str | None = None
    simple_cues: list[SimpleCueConfig] = Field(default_factory=list)
    compound_cues: list[CompoundCueConfig] = Field(default_factory=list)
    visual_cues: dict[str, list[VisualCueConfig]] = Field(default_factory=dict)
    derived_cues: list[dict[str, Any]] | None = None
    behavioral_cues: list[dict[str, Any]] | None = None
    cue_reliability: dict[str, Any] | None = None
    training_strategy: dict[str, Any] | None = None
    teaching_value: dict[str, Any] | None = None
    game_design_insights: dict[str, Any] | None = None
    implementation_phases: dict[str, Any] | None = None
    future_extensions: dict[str, Any] | None = None
    current_status: dict[str, Any] | None = None
    next_steps: dict[str, Any] | None = None

    @property
    def total_cues(self) -> int:
        """Return total number of cues defined."""
        return len(self.simple_cues) + len(self.compound_cues)


def load_cues_config(cues_path: Path) -> CuesConfig:
    """Load and validate cues configuration from cues.yaml."""

    if not cues_path.exists():
        raise FileNotFoundError(f"Cues config not found: {cues_path}")

    with open(cues_path) as handle:
        data = yaml.safe_load(handle)

    if data is None:
        raise ValueError(f"cues.yaml is empty: {cues_path}")

    try:
        return CuesConfig(**data)
    except ValidationError as exc:
        raise ValueError(format_validation_error(exc, "cues.yaml")) from exc
