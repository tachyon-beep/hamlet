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
from typing import Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, Field, model_validator


# Meter name to index mapping (same as in vectorized_env.py)
METER_NAME_TO_IDX: Dict[str, int] = {
    "energy": 0,
    "hygiene": 1,
    "satiation": 2,
    "money": 3,
    "mood": 4,
    "social": 5,
    "health": 6,
    "fitness": 7,
}


class AffordanceEffect(BaseModel):
    """Single meter effect from an affordance interaction."""

    meter: str  # Meter name (e.g., "energy", "hygiene")
    amount: float  # Change amount (normalized, positive or negative)
    type: Optional[str] = None  # "linear" for distributed effects, None for instant

    @model_validator(mode="after")
    def validate_meter_name(self) -> "AffordanceEffect":
        """Ensure meter name is valid."""
        if self.meter not in METER_NAME_TO_IDX:
            raise ValueError(
                f"Invalid meter name: {self.meter}. Valid names: {list(METER_NAME_TO_IDX.keys())}"
            )
        return self


class AffordanceCost(BaseModel):
    """Resource cost for an affordance interaction."""

    meter: str  # Meter name (usually "money" or "energy")
    amount: float = Field(ge=0.0)  # Cost amount (must be non-negative)

    @model_validator(mode="after")
    def validate_meter_name(self) -> "AffordanceCost":
        """Ensure meter name is valid."""
        if self.meter not in METER_NAME_TO_IDX:
            raise ValueError(
                f"Invalid meter name: {self.meter}. Valid names: {list(METER_NAME_TO_IDX.keys())}"
            )
        return self


class AffordanceConfig(BaseModel):
    """Complete configuration for a single affordance."""

    # Identity
    id: str  # Unique identifier (e.g., "Bed", "Shower")
    name: str  # Human-readable name
    category: str  # Category (e.g., "energy_restoration", "income")

    # Interaction type
    interaction_type: Literal["instant", "multi_tick", "continuous", "dual"]

    # Multi-tick specific
    required_ticks: Optional[int] = None  # Number of ticks to complete

    # Costs (instant or per-tick)
    costs: List[AffordanceCost] = Field(default_factory=list)
    costs_per_tick: List[AffordanceCost] = Field(default_factory=list)

    # Effects (instant or per-tick)
    effects: List[AffordanceEffect] = Field(default_factory=list)
    effects_per_tick: List[AffordanceEffect] = Field(default_factory=list)

    # Completion bonus (only for multi_tick)
    completion_bonus: List[AffordanceEffect] = Field(default_factory=list)

    # Operating hours [open_hour, close_hour]
    # Example: [8, 18] = 8am-6pm, [18, 28] = 6pm-4am (wraps midnight)
    operating_hours: List[int]

    # Optional metadata
    teaching_note: Optional[str] = None
    design_intent: Optional[str] = None

    @model_validator(mode="after")
    def validate_multi_tick_requirements(self) -> "AffordanceConfig":
        """Ensure multi_tick and dual affordances have required_ticks set."""
        if self.interaction_type == "multi_tick" and self.required_ticks is None:
            raise ValueError(
                f"Affordance '{self.id}': multi_tick type requires 'required_ticks' field"
            )

        if self.interaction_type == "dual" and self.required_ticks is None:
            raise ValueError(f"Affordance '{self.id}': dual type requires 'required_ticks' field")

        if self.interaction_type not in ["multi_tick", "dual"] and self.required_ticks is not None:
            raise ValueError(
                f"Affordance '{self.id}': 'required_ticks' only valid for multi_tick or dual types"
            )

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
    affordances: List[AffordanceConfig]

    # Optional metadata
    teaching_insights: Optional[Dict[str, str]] = None
    implementation_notes: Optional[Dict[str, str]] = None

    def get_affordance(self, affordance_id: str) -> Optional[AffordanceConfig]:
        """Look up affordance by ID."""
        for affordance in self.affordances:
            if affordance.id == affordance_id:
                return affordance
        return None

    def get_affordances_by_category(self, category: str) -> List[AffordanceConfig]:
        """Get all affordances in a category."""
        return [aff for aff in self.affordances if aff.category == category]

    def get_affordances_by_type(self, interaction_type: str) -> List[AffordanceConfig]:
        """Get all affordances of a given type."""
        return [aff for aff in self.affordances if aff.interaction_type == interaction_type]


def load_affordance_config(config_path: Path) -> AffordanceConfigCollection:
    """
    Load and validate affordance configuration from YAML.

    Args:
        config_path: Path to affordances.yaml file

    Returns:
        Validated AffordanceConfigCollection

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValidationError: If config format is invalid
        yaml.YAMLError: If YAML parsing fails
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Affordance config not found: {config_path}")

    with open(config_path, "r") as f:
        raw_data = yaml.safe_load(f)

    # Pydantic will validate the structure
    collection = AffordanceConfigCollection(**raw_data)

    return collection


# Convenience function for common use case
def load_default_affordances() -> AffordanceConfigCollection:
    """Load the default affordances.yaml from configs/ directory."""
    config_path = Path("configs/affordances.yaml")
    return load_affordance_config(config_path)


# ============================================================================
# LEGACY HARDCODED CONFIGS (Keep for backwards compatibility during transition)
# ============================================================================
# These will be removed once vectorized_env.py is migrated to use AffordanceEngine


# ============================================================================
# LEGACY HARDCODED CONFIGS (Keep for backwards compatibility during transition)
# ============================================================================
# These will be removed once vectorized_env.py is migrated to use AffordanceEngine

AFFORDANCE_CONFIGS: dict[str, dict] = {
    # === Static Affordances (24/7) ===
    "Bed": {
        "required_ticks": 5,
        "cost_per_tick": 0.01,  # $1 per tick ($5 total)
        "operating_hours": (0, 24),
        "benefits": {
            "linear": {
                "energy": +0.075,  # Per tick: (50% * 0.75) / 5
            },
            "completion": {
                "energy": +0.125,  # 50% * 0.25
                "health": +0.02,
            },
        },
    },
    "LuxuryBed": {
        "required_ticks": 5,
        "cost_per_tick": 0.022,  # $2.20 per tick ($11 total)
        "operating_hours": (0, 24),
        "benefits": {
            "linear": {
                "energy": +0.1125,  # Per tick: (75% * 0.75) / 5
            },
            "completion": {
                "energy": +0.1875,  # 75% * 0.25
                "health": +0.05,
            },
        },
    },
    "Shower": {
        "required_ticks": 3,
        "cost_per_tick": 0.01,  # $1 per tick ($3 total)
        "operating_hours": (0, 24),
        "benefits": {
            "linear": {
                "hygiene": +0.10,  # Per tick: (40% * 0.75) / 3
            },
            "completion": {
                "hygiene": +0.10,  # 40% * 0.25
            },
        },
    },
    "HomeMeal": {
        "required_ticks": 2,
        "cost_per_tick": 0.015,  # $1.50 per tick ($3 total)
        "operating_hours": (0, 24),
        "benefits": {
            "linear": {
                "satiation": +0.16875,  # Per tick: (45% * 0.75) / 2
            },
            "completion": {
                "satiation": +0.1125,  # 45% * 0.25
                "health": +0.03,
            },
        },
    },
    "Hospital": {
        "required_ticks": 3,
        "cost_per_tick": 0.05,  # $5 per tick ($15 total)
        "operating_hours": (0, 24),
        "benefits": {
            "linear": {
                "health": +0.225,  # Per tick: (60% * 0.75) / 3
            },
            "completion": {
                "health": +0.15,  # 60% * 0.25
            },
        },
    },
    "Gym": {
        "required_ticks": 4,
        "cost_per_tick": 0.02,  # $2 per tick ($8 total)
        "operating_hours": (0, 24),
        "benefits": {
            "linear": {
                "fitness": +0.1125,  # Per tick: (30% * 0.75) / 4
                "energy": -0.03,
            },
            "completion": {
                "fitness": +0.075,  # 30% * 0.25
                "mood": +0.05,
            },
        },
    },
    "FastFood": {
        "required_ticks": 1,
        "cost_per_tick": 0.10,  # $10
        "operating_hours": (0, 24),
        "benefits": {
            "linear": {
                "satiation": +0.3375,  # (45% * 0.75) / 1
                "energy": +0.1125,
            },
            "completion": {
                "satiation": +0.1125,  # 45% * 0.25
                "energy": +0.0375,
                "fitness": -0.03,
                "health": -0.02,
            },
        },
    },
    # === Business Hours Affordances (8am-6pm) ===
    "Job": {
        "required_ticks": 4,
        "cost_per_tick": 0.0,
        "operating_hours": (8, 18),
        "benefits": {
            "linear": {
                "money": +0.140625,  # Per tick: ($22.5 * 0.75) / 4
                "energy": -0.0375,
            },
            "completion": {
                "money": +0.05625,  # $22.5 * 0.25
                "social": +0.02,
                "health": -0.03,
            },
        },
    },
    "Labor": {
        "required_ticks": 4,
        "cost_per_tick": 0.0,
        "operating_hours": (8, 18),
        "benefits": {
            "linear": {
                "money": +0.1875,  # Per tick: ($30 * 0.75) / 4
                "energy": -0.05,
            },
            "completion": {
                "money": +0.075,  # $30 * 0.25
                "fitness": -0.05,
                "health": -0.05,
                "social": +0.01,
            },
        },
    },
    "Doctor": {
        "required_ticks": 2,
        "cost_per_tick": 0.04,  # $4 per tick ($8 total)
        "operating_hours": (8, 18),
        "benefits": {
            "linear": {
                "health": +0.1125,  # Per tick: (30% * 0.75) / 2
            },
            "completion": {
                "health": +0.075,  # 30% * 0.25
            },
        },
    },
    "Therapist": {
        "required_ticks": 3,
        "cost_per_tick": 0.05,  # $5 per tick ($15 total)
        "operating_hours": (8, 18),
        "benefits": {
            "linear": {
                "mood": +0.15,  # Per tick: (40% * 0.75) / 3
            },
            "completion": {
                "mood": +0.10,  # 40% * 0.25
                "social": +0.05,
            },
        },
    },
    "Recreation": {
        "required_ticks": 2,
        "cost_per_tick": 0.03,  # $3 per tick ($6 total)
        "operating_hours": (8, 22),
        "benefits": {
            "linear": {
                "mood": +0.1125,  # Per tick: (30% * 0.75) / 2
                "social": +0.075,
            },
            "completion": {
                "mood": +0.075,  # 30% * 0.25
                "social": +0.05,
            },
        },
    },
    # === Dynamic Affordances (Time-Dependent) ===
    "CoffeeShop": {
        "required_ticks": 1,
        "cost_per_tick": 0.02,  # $2
        "operating_hours": (8, 18),
        "benefits": {
            "linear": {
                "energy": +0.1125,  # (15% * 0.75) / 1
                "mood": +0.0375,
                "social": +0.045,
            },
            "completion": {
                "energy": +0.0375,  # 15% * 0.25
                "mood": +0.0125,
                "social": +0.015,
            },
        },
    },
    "Bar": {
        "required_ticks": 2,
        "cost_per_tick": 0.075,  # $7.50 per round ($15 total)
        "operating_hours": (18, 4),  # Wraps midnight
        "benefits": {
            "linear": {
                "mood": +0.075,  # Per tick: (20% * 0.75) / 2
                "social": +0.05625,
                "health": -0.01875,
            },
            "completion": {
                "mood": +0.05,  # 20% * 0.25
                "social": +0.0375,
                "health": -0.0125,
            },
        },
    },
    "Park": {
        "required_ticks": 2,
        "cost_per_tick": 0.0,
        "operating_hours": (6, 22),
        "benefits": {
            "linear": {
                "mood": +0.0975,  # Per tick: (26% * 0.75) / 2
                "social": +0.0375,
            },
            "completion": {
                "mood": +0.065,  # 26% * 0.25
                "social": +0.025,
                "fitness": +0.02,
            },
        },
    },
}


# Meter name to index mapping
METER_NAME_TO_IDX = {
    "energy": 0,
    "hygiene": 1,
    "satiation": 2,
    "money": 3,
    "mood": 4,
    "social": 5,
    "health": 6,
    "fitness": 7,
}


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
