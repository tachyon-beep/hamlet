"""Cascade configuration DTO with no-defaults enforcement.

Philosophy: All behavioral parameters must be explicitly specified.
No implicit defaults. Operator accountability.

Design: Validates cascade structure from cascades.yaml (basic structural validation only).
Cross-file validation (meter references) deferred to TASK-004A.
"""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from townlet.config.base import format_validation_error, load_yaml_section
from townlet.environment.cascade_config import CascadesConfig as _CascadesConfig

__all__ = ["CascadeConfig", "load_cascades_config", "CascadesConfig"]


class CascadeConfig(BaseModel):
    """Cascade rule configuration - structural validation only.

    ALL REQUIRED FIELDS must be specified (no defaults) - operator accountability.

    Example:
        >>> cascade = CascadeConfig(
        ...     name="satiation_to_health",
        ...     description="Starvation makes you sick",
        ...     source="satiation",
        ...     target="health",
        ...     threshold=0.3,
        ...     strength=0.004,
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    # Cascade identity (REQUIRED)
    name: str = Field(min_length=1, description="Cascade rule name")
    description: str = Field(min_length=1, description="Description of cascade effect")

    # Cascade relationship (REQUIRED)
    source: str = Field(min_length=1, description="Source meter name")
    target: str = Field(min_length=1, description="Target meter name")

    # Cascade parameters (REQUIRED)
    threshold: float = Field(ge=0.0, le=1.0, description="Trigger threshold (0.0-1.0)")
    strength: float = Field(description="Effect strength (penalty per step)")

    # Optional metadata
    category: str | None = None
    source_index: int | None = None
    target_index: int | None = None
    teaching_note: str | None = None
    why_it_matters: str | None = None

    @model_validator(mode="after")
    def validate_not_self_cascade(self) -> "CascadeConfig":
        """Source and target must be different meters."""
        if self.source == self.target:
            raise ValueError(
                f"Cascade source and target cannot be the same meter. "
                f"Got: source='{self.source}', target='{self.target}'. "
                f"Self-cascades are not allowed."
            )
        return self


def load_cascades_config(config_dir: Path) -> list[CascadeConfig]:
    """Load and validate cascade configurations.

    Args:
        config_dir: Directory containing cascades.yaml

    Returns:
        List of validated CascadeConfig objects

    Raises:
        FileNotFoundError: If cascades.yaml not found
        ValueError: If validation fails (with helpful error message)

    Example:
        >>> cascades = load_cascades_config(Path("configs/L0_0_minimal"))
        >>> print(f"Loaded {len(cascades)} cascade rules")
        Loaded 10 cascade rules
    """
    try:
        data = load_yaml_section(config_dir, "cascades.yaml", "cascades")
        return [CascadeConfig(**cascade_data) for cascade_data in data]
    except ValidationError as e:
        raise ValueError(format_validation_error(e, "cascades.yaml")) from e


CascadesConfig = _CascadesConfig
