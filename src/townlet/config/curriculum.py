"""Curriculum configuration DTO with no-defaults enforcement.

Philosophy: All behavioral parameters must be explicitly specified.
No implicit defaults. Operator accountability.

Design: Validates adversarial curriculum parameters for difficulty progression.
"""

from pathlib import Path

from pydantic import BaseModel, Field, ValidationError, model_validator

from townlet.config.base import format_validation_error, load_yaml_section


class CurriculumConfig(BaseModel):
    """Curriculum (adversarial difficulty adjustment) configuration.

    ALL FIELDS REQUIRED (no defaults) - enforces operator accountability.

    Example:
        >>> config = CurriculumConfig(
        ...     max_steps_per_episode=500,
        ...     survival_advance_threshold=0.7,
        ...     survival_retreat_threshold=0.3,
        ...     entropy_gate=0.5,
        ...     min_steps_at_stage=1000,
        ... )
    """

    # Stage parameters (REQUIRED)
    max_steps_per_episode: int = Field(gt=0, description="Episode length limit (truncation)")

    # Advancement thresholds (ALL REQUIRED)
    survival_advance_threshold: float = Field(ge=0.0, le=1.0, description="Advance to next stage if survival rate >= this threshold")
    survival_retreat_threshold: float = Field(ge=0.0, le=1.0, description="Retreat to previous stage if survival rate < this threshold")
    entropy_gate: float = Field(ge=0.0, le=1.0, description="Minimum policy entropy to advance (prevents premature advancement)")
    min_steps_at_stage: int = Field(gt=0, description="Minimum training steps before stage change allowed")

    @model_validator(mode="after")
    def validate_threshold_order(self) -> "CurriculumConfig":
        """Ensure advance_threshold > retreat_threshold."""
        if self.survival_advance_threshold <= self.survival_retreat_threshold:
            raise ValueError(
                f"survival_advance_threshold ({self.survival_advance_threshold}) "
                f"must be > survival_retreat_threshold ({self.survival_retreat_threshold}). "
                f"Curriculum needs a hysteresis band between advance and retreat."
            )
        return self


def load_curriculum_config(config_dir: Path) -> CurriculumConfig:
    """Load and validate curriculum configuration.

    Args:
        config_dir: Directory containing training.yaml

    Returns:
        Validated CurriculumConfig

    Raises:
        FileNotFoundError: If training.yaml not found
        ValueError: If validation fails (with helpful error message)
    """
    try:
        data = load_yaml_section(config_dir, "training.yaml", "curriculum")
        return CurriculumConfig(**data)
    except ValidationError as e:
        raise ValueError(format_validation_error(e, "curriculum section (training.yaml)")) from e
