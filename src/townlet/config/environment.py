"""Training environment configuration DTO with no-defaults enforcement.

NOTE: Named TrainingEnvironmentConfig (not EnvironmentConfig) to avoid conflict
with existing cascade_config.EnvironmentConfig (bars + cascades mechanics).

This DTO covers training.yaml's 'environment' section which controls:
- Grid parameters (grid_size)
- Observability mode (full vs POMDP)
- Temporal mechanics (time-of-day, operating hours)
- Enabled affordances (curriculum selection)
- Action energy costs

Philosophy: All behavioral parameters must be explicitly specified.
No implicit defaults. Operator accountability.
"""

import logging
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError, model_validator

from townlet.config.base import format_validation_error, load_yaml_section

logger = logging.getLogger(__name__)


class TrainingEnvironmentConfig(BaseModel):
    """Environment configuration for training (grid, observability, affordances).

    ALL FIELDS REQUIRED (no defaults) - enforces operator accountability.
    Operator must explicitly specify all parameters that affect the environment.

    Philosophy: If it affects the universe, it's in the config. No exceptions.

    NOTE: This is for training hyperparameters (grid_size, vision_range).
    For game mechanics (bars, cascades), use cascade_config.EnvironmentConfig.

    Example:
        >>> config = TrainingEnvironmentConfig(
        ...     grid_size=8,
        ...     partial_observability=False,
        ...     vision_range=8,
        ...     enable_temporal_mechanics=False,
        ...     enabled_affordances=None,
        ...     energy_move_depletion=0.005,
        ...     energy_wait_depletion=0.001,
        ...     energy_interact_depletion=0.0,
        ... )
    """

    # Grid parameters (REQUIRED)
    grid_size: int = Field(gt=0, description="Grid dimensions (N×N square grid)")

    # Observability (REQUIRED)
    partial_observability: bool = Field(description="true = POMDP (local window), false = full grid visibility")
    vision_range: int = Field(ge=0, description="Local window radius for POMDP (e.g., 2 = 5×5 window)")

    # Temporal mechanics (REQUIRED)
    enable_temporal_mechanics: bool = Field(description="true = time-of-day, operating hours, multi-tick interactions")

    # Enabled affordances (REQUIRED - null or list)
    enabled_affordances: list[str] | None = Field(description="null = all affordances enabled, or list of affordance names for curriculum")

    # Action energy costs (ALL REQUIRED)
    energy_move_depletion: float = Field(ge=0.0, description="Energy cost per movement action (as fraction of energy meter)")
    energy_wait_depletion: float = Field(ge=0.0, description="Energy cost per WAIT action (as fraction of energy meter)")
    energy_interact_depletion: float = Field(ge=0.0, description="Energy cost per INTERACT action (as fraction of energy meter)")

    @model_validator(mode="after")
    def validate_enabled_affordances(self) -> "TrainingEnvironmentConfig":
        """Ensure enabled_affordances is null or non-empty list."""
        if self.enabled_affordances is not None:
            if len(self.enabled_affordances) == 0:
                raise ValueError(
                    "enabled_affordances cannot be an empty list. "
                    "Use null (YAML) or None (Python) to enable all affordances, "
                    "or provide a list of affordance names: ['Bed', 'Hospital']"
                )
        return self

    @model_validator(mode="after")
    def validate_pomdp_vision_range(self) -> "TrainingEnvironmentConfig":
        """Warn (not error) if POMDP vision range seems unreasonable.

        NOTE: This is a HINT, not enforcement. Operator may intentionally
        set unusual vision ranges for their experiment.

        Follows permissive semantics: allow unusual values but warn the operator.
        """
        if self.partial_observability:
            if self.vision_range > self.grid_size:
                logger.warning(
                    f"POMDP vision_range ({self.vision_range}) > grid_size ({self.grid_size}). "
                    f"Agent sees beyond grid boundaries (window larger than world). "
                    f"Typical POMDP: vision_range=2 (5×5 window) on grid_size=8. "
                    f"This may be intentional for your experiment."
                )
            elif self.vision_range == 0:
                logger.warning(
                    "POMDP vision_range=0 means agent sees only current cell (1×1 window). "
                    "This is extremely limited observability. "
                    "Typical POMDP: vision_range=2 (5×5 window). "
                    "This may be intentional for your experiment."
                )

        return self


def load_environment_config(config_dir: Path) -> TrainingEnvironmentConfig:
    """Load and validate training environment configuration.

    Args:
        config_dir: Directory containing training.yaml

    Returns:
        Validated TrainingEnvironmentConfig

    Raises:
        FileNotFoundError: If training.yaml not found
        ValueError: If validation fails (with helpful error message)

    Example:
        >>> config = load_environment_config(Path("configs/L0_0_minimal"))
        >>> print(f"Grid: {config.grid_size}×{config.grid_size}, POMDP: {config.partial_observability}")
        Grid: 3×3, POMDP: False
    """
    try:
        data = load_yaml_section(config_dir, "training.yaml", "environment")
        return TrainingEnvironmentConfig(**data)
    except ValidationError as e:
        raise ValueError(format_validation_error(e, "environment section (training.yaml)")) from e
