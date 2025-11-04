"""
Affordance configuration loader and Pydantic models.

This module provides type-safe loading of affordance definitions from YAML.
Follows the same pattern as cascade_config.py (ACTION #1).

Architecture:
- AffordanceEffect: Single meter effect (positive or negative)
- AffordanceCost: Resource cost (money, energy, etc.)
- AffordanceConfig: Complete affordance definition
- AffordanceConfigCollection: Container for all affordances
- load_affordance_config(): Type-safe YAML loader

Teaching Value:
- Data-driven affordance system enables student experimentation
- Different affordance sets teach different strategic patterns
- Config validation catches errors at load time, not runtime
"""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from townlet.environment.cascade_config import BarsConfig


class AffordanceEffect(BaseModel):
    """Single meter effect from an affordance interaction."""

    meter: str  # Meter name (e.g., "energy", "hygiene")
    amount: float  # Change amount (normalized, positive or negative)
    type: str | None = None  # "linear" for distributed effects, None for instant

    # Note: Meter name validation happens at collection level in load_affordance_config()
    # to support variable-size meter systems (TASK-001)


class AffordanceCost(BaseModel):
    """Resource cost for an affordance interaction."""

    meter: str  # Meter name (usually "money" or "energy")
    amount: float = Field(ge=0.0)  # Cost amount (must be non-negative)

    # Note: Meter name validation happens at collection level in load_affordance_config()
    # to support variable-size meter systems (TASK-001)


class AffordanceConfig(BaseModel):
    """Complete configuration for a single affordance."""

    # Identity
    id: str  # Unique identifier (e.g., "Bed", "Shower")
    name: str  # Human-readable name
    category: str  # Category (e.g., "energy_restoration", "income")

    # Interaction type
    interaction_type: Literal["instant", "multi_tick", "continuous", "dual"]

    # Multi-tick specific
    required_ticks: int | None = None  # Number of ticks to complete

    # Costs (instant or per-tick)
    costs: list[AffordanceCost] = Field(default_factory=list)
    costs_per_tick: list[AffordanceCost] = Field(default_factory=list)

    # Effects (instant or per-tick)
    effects: list[AffordanceEffect] = Field(default_factory=list)
    effects_per_tick: list[AffordanceEffect] = Field(default_factory=list)

    # Completion bonus (only for multi_tick)
    completion_bonus: list[AffordanceEffect] = Field(default_factory=list)

    # Operating hours [open_hour, close_hour]
    # Example: [8, 18] = 8am-6pm, [18, 28] = 6pm-4am (wraps midnight)
    operating_hours: list[int]

    # Optional metadata
    teaching_note: str | None = None
    design_intent: str | None = None

    @model_validator(mode="after")
    def validate_multi_tick_requirements(self) -> "AffordanceConfig":
        """Ensure multi_tick and dual affordances have required_ticks set."""
        if self.interaction_type == "multi_tick" and self.required_ticks is None:
            raise ValueError(f"Affordance '{self.id}': multi_tick type requires 'required_ticks' field")

        if self.interaction_type == "dual" and self.required_ticks is None:
            raise ValueError(f"Affordance '{self.id}': dual type requires 'required_ticks' field")

        if self.interaction_type not in ["multi_tick", "dual"] and self.required_ticks is not None:
            raise ValueError(f"Affordance '{self.id}': 'required_ticks' only valid for multi_tick or dual types")

        return self

    @model_validator(mode="after")
    def validate_operating_hours(self) -> "AffordanceConfig":
        """Ensure operating hours are valid."""
        if len(self.operating_hours) != 2:
            raise ValueError(f"Affordance '{self.id}': operating_hours must be [open, close]")

        open_hour, close_hour = self.operating_hours

        if not (0 <= open_hour < 24):
            raise ValueError(f"Affordance '{self.id}': open_hour must be 0-23, got {open_hour}")

        if not (0 < close_hour <= 28):
            raise ValueError(f"Affordance '{self.id}': close_hour must be 1-28, got {close_hour}")

        return self


class AffordanceConfigCollection(BaseModel):
    """Collection of all affordance configurations."""

    version: str
    description: str
    status: str  # e.g., "TEMPLATE", "PRODUCTION"
    affordances: list[AffordanceConfig]

    # Optional metadata
    teaching_insights: dict[str, str] | None = None
    implementation_notes: dict[str, str] | None = None

    def get_affordance(self, affordance_id: str) -> AffordanceConfig | None:
        """Look up affordance by ID."""
        for affordance in self.affordances:
            if affordance.id == affordance_id:
                return affordance
        return None

    def get_affordances_by_category(self, category: str) -> list[AffordanceConfig]:
        """Get all affordances in a category."""
        return [aff for aff in self.affordances if aff.category == category]

    def get_affordances_by_type(self, interaction_type: str) -> list[AffordanceConfig]:
        """Get all affordances of a given type."""
        return [aff for aff in self.affordances if aff.interaction_type == interaction_type]


def load_affordance_config(config_path: Path, bars_config: BarsConfig) -> AffordanceConfigCollection:
    """
    Load and validate affordance configuration from YAML.

    Args:
        config_path: Path to affordances.yaml file
        bars_config: BarsConfig to validate meter references against

    Returns:
        Validated AffordanceConfigCollection

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValidationError: If config format is invalid or meter references are invalid
        yaml.YAMLError: If YAML parsing fails
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Affordance config not found: {config_path}")

    with open(config_path) as f:
        raw_data = yaml.safe_load(f)

    # Pydantic will validate the structure
    collection = AffordanceConfigCollection(**raw_data)

    # Validate all meter references against bars_config (TASK-001 variable meter support)
    valid_meters = set(bars_config.meter_name_to_index.keys())

    for affordance in collection.affordances:
        # Check all effects
        for effect in affordance.effects + affordance.effects_per_tick + affordance.completion_bonus:
            if effect.meter not in valid_meters:
                raise ValueError(
                    f"Affordance '{affordance.id}' effect references invalid meter '{effect.meter}'. "
                    f"Valid meters: {sorted(valid_meters)}"
                )

        # Check all costs
        for cost in affordance.costs + affordance.costs_per_tick:
            if cost.meter not in valid_meters:
                raise ValueError(
                    f"Affordance '{affordance.id}' cost references invalid meter '{cost.meter}'. "
                    f"Valid meters: {sorted(valid_meters)}"
                )

    return collection


# Convenience function for common use case
def load_default_affordances() -> AffordanceConfigCollection:
    """
    Load the default affordances.yaml from the test config pack.

    Note: This requires bars.yaml to exist in the same directory for validation.
    """
    from townlet.environment.cascade_config import load_bars_config

    config_dir = Path("configs/test")
    bars_config = load_bars_config(config_dir / "bars.yaml")
    return load_affordance_config(config_dir / "affordances.yaml", bars_config)


# ============================================================================


def is_affordance_open(time_of_day: int, operating_hours: tuple[int, int]) -> bool:
    """
    Check if affordance is open at given time.

    Handles midnight wraparound (e.g., Bar: 18-4 means 6pm to 4am).

    Args:
        time_of_day: Current tick [0-23]
        operating_hours: (open_tick, close_tick)

    Returns:
        True if open, False if closed
    """
    open_tick, close_tick = operating_hours

    if open_tick < close_tick:
        # Normal hours (e.g., 8-18)
        return open_tick <= time_of_day < close_tick
    else:
        # Wraparound hours (e.g., 18-4 = 6pm to 4am)
        return time_of_day >= open_tick or time_of_day < close_tick
