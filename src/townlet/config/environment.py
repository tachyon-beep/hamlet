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

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from townlet.config.base import format_validation_error, load_yaml_section

logger = logging.getLogger(__name__)


class TrainingEnvironmentConfig(BaseModel):
    """Environment configuration for training (observability, affordances, energy costs).

    ALL FIELDS REQUIRED (no defaults) - enforces operator accountability.
    Operator must explicitly specify all parameters that affect the environment.

    Philosophy: If it affects the universe, it's in the config. No exceptions.

    NOTE: This is for training hyperparameters (vision_range, energy costs).
    For spatial config (grid dimensions), use substrate.yaml.
    For game mechanics (bars, cascades), use cascade_config.EnvironmentConfig.

    Example:
        >>> config = TrainingEnvironmentConfig(
        ...     partial_observability=False,
        ...     vision_range=8,
        ...     enable_temporal_mechanics=False,
        ...     enabled_affordances=None,
        ...     energy_move_depletion=0.005,
        ...     energy_wait_depletion=0.001,
        ...     energy_interact_depletion=0.0,
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    # Observability (REQUIRED)
    partial_observability: bool = Field(description="true = POMDP (local window), false = full grid visibility")
    vision_range: int = Field(ge=0, description="Local window radius for POMDP (e.g., 2 = 5×5 window)")

    # Temporal mechanics (REQUIRED)
    enable_temporal_mechanics: bool = Field(description="true = time-of-day, operating hours, multi-tick interactions")

    # Enabled affordances (REQUIRED - null or list)
    enabled_affordances: list[str] | None = Field(description="null = all affordances enabled, or list of affordance names for curriculum")

    # Placement control (REQUIRED)
    randomize_affordances: bool = Field(description="true = shuffle affordance positions each episode, false = use config positions")

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

        NOTE: We no longer validate vision_range > grid_size here because grid
        dimensions are in substrate.yaml. The compiler handles cross-config validation.
        """
        if self.partial_observability:
            if self.vision_range == 0:
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
        >>> print(f"POMDP: {config.partial_observability}, Vision: {config.vision_range}")
        POMDP: False, Vision: 3
    """
    try:
        data = load_yaml_section(config_dir, "training.yaml", "environment")
        return TrainingEnvironmentConfig(**data)
    except ValidationError as e:
        raise ValueError(format_validation_error(e, "environment section (training.yaml)")) from e
