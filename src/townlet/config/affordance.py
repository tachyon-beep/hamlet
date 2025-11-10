"""Affordance configuration DTO with no-defaults enforcement.

Philosophy: All behavioral parameters must be explicitly specified.
No implicit defaults. Operator accountability.

Design: Validates affordance structure from affordances.yaml.
Cross-file validation (meter references) is still handled in the compiler (TASK-004A), but
the DTO now exposes advanced fields (capabilities, effect pipelines, availability) required
for later compiler stages.
"""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from townlet.config.affordance_masking import BarConstraint, ModeConfig
from townlet.config.base import format_validation_error, load_yaml_section
from townlet.config.capability_config import CapabilityConfig
from townlet.config.effect_pipeline import EffectPipeline
from townlet.environment.affordance_config import (
    AffordanceConfigCollection as _AffordanceConfigCollection,
)

__all__ = [
    "AffordanceConfig",
    "load_affordances_config",
    "AffordanceConfigCollection",
]


class AffordanceConfig(BaseModel):
    """Rich affordance configuration DTO used throughout the compiler pipeline."""

    model_config = ConfigDict(extra="forbid")

    # Identity / metadata
    id: str = Field(min_length=1, description="Unique affordance ID")
    name: str = Field(min_length=1, description="Display name")
    category: str | None = None
    interaction_type: str | None = None
    description: str | None = None
    teaching_note: str | None = None
    design_intent: str | None = None

    # Costs
    costs: list[dict[str, Any]] = Field(default_factory=list, description="Instant costs applied when interaction starts")
    costs_per_tick: list[dict[str, Any]] = Field(default_factory=list, description="Costs applied every tick")

    # Temporal metadata
    duration_ticks: int | None = None
    operating_hours: list[int] | None = Field(default=None, description="Operating hours [open, close] or None for always open")
    modes: dict[str, ModeConfig] = Field(default_factory=dict, description="Optional operating modes (coffee vs bar, etc.)")
    availability: list[BarConstraint] = Field(default_factory=list, description="Meter-based availability constraints")

    # Advanced behaviors
    capabilities: list[CapabilityConfig] = Field(default_factory=list)
    effect_pipeline: EffectPipeline | None = None

    # Spatial metadata
    position: list[int] | dict[str, int] | int | None = None

    @field_validator("operating_hours")
    @classmethod
    def validate_operating_hours(cls, value: list[int] | None) -> list[int] | None:
        if value is None:
            return None
        if len(value) != 2:
            raise ValueError("operating_hours must contain [open_hour, close_hour]")
        open_hour, close_hour = value
        if not (0 <= open_hour <= 23):
            raise ValueError(f"open_hour must be 0-23, got {open_hour}")
        if not (1 <= close_hour <= 28):
            raise ValueError(f"close_hour must be 1-28, got {close_hour}")
        return value

    @field_validator("position")
    @classmethod
    def validate_position_format(cls, value):
        """Ensure position can be interpreted on various substrates."""

        if value is None:
            return value
        if isinstance(value, list):
            if not value or not all(isinstance(coord, int) for coord in value):
                raise ValueError("List position must contain integer coordinates")
            if len(value) not in (2, 3):
                raise ValueError(f"List position must be 2D or 3D, got {len(value)}D")
            return value
        if isinstance(value, dict):
            if set(value.keys()) != {"q", "r"}:
                raise ValueError("Dict position must contain 'q' and 'r' keys for axial coordinates")
            if not all(isinstance(coord, int) for coord in value.values()):
                raise ValueError("Dict position values must be integers")
            return value
        if isinstance(value, int):
            if value < 0:
                raise ValueError("Integer position (graph node id) must be >= 0")
            return value
        raise ValueError(f"Invalid position format ({type(value)}). Expected list[int], dict[str, int], int, or None.")

    @model_validator(mode="after")
    def validate_no_redundant_duration_ticks(self) -> "AffordanceConfig":
        """Reject configs with duration_ticks in both root and capabilities (legacy migration artifact)."""

        if not self.capabilities:
            return self

        # Check if any multi_tick capability has duration_ticks set (not None)
        for cap in self.capabilities:
            if hasattr(cap, "type") and cap.type == "multi_tick":
                if hasattr(cap, "duration_ticks") and cap.duration_ticks is not None:
                    raise ValueError(
                        f"Affordance '{self.id}': Remove 'duration_ticks' from capabilities array. "
                        f"Only the root-level 'duration_ticks' field is used. "
                        f"(Legacy migration artifact - capabilities[].duration_ticks is ignored at runtime)"
                    )

        return self


def load_affordances_config(config_dir: Path) -> list[AffordanceConfig]:
    """Load and validate affordance configurations.

    Args:
        config_dir: Directory containing affordances.yaml

    Returns:
        List of validated AffordanceConfig objects

    Raises:
        FileNotFoundError: If affordances.yaml not found
        ValueError: If validation fails (with helpful error message)

    Example:
        >>> affordances = load_affordances_config(Path("configs/L0_0_minimal"))
        >>> print(f"Loaded {len(affordances)} affordances")
        Loaded 14 affordances
    """
    try:
        data = load_yaml_section(config_dir, "affordances.yaml", "affordances")
        return [AffordanceConfig(**affordance_data) for affordance_data in data]
    except ValidationError as e:
        raise ValueError(format_validation_error(e, "affordances.yaml")) from e


AffordanceConfigCollection = _AffordanceConfigCollection
