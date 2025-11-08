"""Affordance configuration DTO with no-defaults enforcement.

Philosophy: All behavioral parameters must be explicitly specified.
No implicit defaults. Operator accountability.

Design: Validates affordance structure from affordances.yaml (BASIC version).
Advanced features (capabilities, effect pipelines, temporal mechanics) deferred to TASK-004B.
Cross-file validation (meter references) deferred to TASK-004A.
"""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from townlet.config.base import format_validation_error, load_yaml_section


class AffordanceConfig(BaseModel):
    """Affordance configuration - BASIC structural validation only.

    ALL REQUIRED FIELDS must be specified (no defaults) - operator accountability.

    This is the BASIC version for TASK-003. Advanced features deferred to TASK-004B:
    - Capabilities (visibility, requirements, unlock conditions)
    - Effect pipelines (complex multi-step effects)
    - Availability masking (temporal, spatial, state-based)
    - Multi-tick interaction mechanics

    Example:
        >>> affordance = AffordanceConfig(
        ...     id="0",
        ...     name="Bed",
        ...     category="energy_restoration",
        ...     costs=[{"meter": "money", "amount": 0.05}],
        ...     effects=[{"meter": "energy", "amount": 0.50}],
        ... )
    """

    # Allow extra fields (for temporal mechanics, operating_hours, etc.)
    model_config = ConfigDict(extra="allow")

    # Affordance identity (REQUIRED)
    id: str = Field(min_length=1, description="Unique affordance ID")
    name: str = Field(min_length=1, description="Display name")

    # Costs and effects (REQUIRED)
    costs: list[dict[str, Any]] = Field(description="Meter costs (can be empty for free affordances)")
    effects: list[dict[str, Any]] = Field(
        min_length=1,
        description="Meter effects (must have at least one effect)",
    )

    # Optional metadata
    category: str | None = None
    interaction_type: str | None = None
    description: str | None = None

    @field_validator("effects")
    @classmethod
    def validate_effects_not_empty(cls, v: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Ensure effects list has at least one element."""
        if len(v) == 0:
            raise ValueError(
                "effects cannot be empty. Every affordance must have at least one effect. "
                "Use costs=[] for free affordances, but effects must define what happens."
            )
        return v


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
