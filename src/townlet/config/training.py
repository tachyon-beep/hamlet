"""Training configuration DTO with no-defaults enforcement.

Philosophy: All behavioral parameters must be explicitly specified.
No implicit defaults. Operator accountability.

Design: Validates Q-learning hyperparameters, epsilon-greedy exploration,
and training infrastructure settings. Warnings guide operators without blocking.
"""

import logging
import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, model_validator

from townlet.config.base import format_validation_error, load_yaml_section

logger = logging.getLogger(__name__)


class TrainingConfig(BaseModel):
    """Training hyperparameters configuration.

    ALL FIELDS REQUIRED (no defaults) - enforces operator accountability.
    Operator must explicitly specify all parameters that affect training.

    Philosophy: If it affects the universe, it's in the config. No exceptions.

    Example:
        >>> config = TrainingConfig(
        ...     device="cuda",
        ...     max_episodes=5000,
        ...     train_frequency=4,
        ...     target_update_frequency=100,
        ...     batch_size=64,
        ...     max_grad_norm=10.0,
        ...     epsilon_start=1.0,
        ...     epsilon_decay=0.995,
        ...     epsilon_min=0.01,
        ...     sequence_length=8,
        ... )
    """

    # Compute device (REQUIRED)
    device: Literal["cpu", "cuda", "mps"] = Field(description="Compute device: 'cuda' (GPU), 'cpu' (CPU-only), 'mps' (Apple Silicon)")

    # Training duration (REQUIRED)
    max_episodes: int = Field(gt=0, description="Total episodes to train")

    # Q-learning hyperparameters (ALL REQUIRED)
    train_frequency: int = Field(gt=0, description="Train Q-network every N steps")
    target_update_frequency: int = Field(gt=0, description="Update target network every N training steps")
    batch_size: int = Field(gt=0, description="Experience replay batch size")
    max_grad_norm: float = Field(gt=0, description="Gradient clipping threshold (prevents exploding gradients)")

    # Epsilon-greedy exploration (ALL REQUIRED)
    epsilon_start: float = Field(ge=0.0, le=1.0, description="Initial exploration rate (1.0 = 100% random)")
    epsilon_decay: float = Field(gt=0.0, lt=1.0, description="Decay per episode (< 1.0)")
    epsilon_min: float = Field(ge=0.0, le=1.0, description="Minimum exploration rate (floor)")

    # Recurrent-specific (REQUIRED for all configs, used when network_type=recurrent)
    sequence_length: int = Field(gt=0, description="Length of sequences for LSTM training (recurrent networks only)")

    @model_validator(mode="after")
    def validate_epsilon_order(self) -> "TrainingConfig":
        """Ensure epsilon_start >= epsilon_min."""
        if self.epsilon_start < self.epsilon_min:
            raise ValueError(
                f"epsilon_start ({self.epsilon_start}) must be >= "
                f"epsilon_min ({self.epsilon_min}). "
                f"Exploration cannot start below the minimum threshold."
            )
        return self

    @model_validator(mode="after")
    def validate_epsilon_decay_speed(self) -> "TrainingConfig":
        """Warn (not error) if epsilon decay seems unreasonable.

        NOTE: This is a HINT, not enforcement. Operator may intentionally
        set slow/fast decay for their experiment. We validate structure, not semantics.

        Follows permissive semantics: allow unusual values but warn the operator.
        """
        # Calculate episodes to reach ε=0.1 from ε=1.0
        episodes_to_01 = math.log(0.1) / math.log(self.epsilon_decay)

        if self.epsilon_decay > 0.999:
            logger.warning(
                f"epsilon_decay={self.epsilon_decay} is very slow. "
                f"Will take {episodes_to_01:.0f} episodes to reach ε=0.1 from ε=1.0. "
                f"Typical values: 0.99 (L0 fast), 0.995 (L0.5/L1 moderate), "
                f"0.998 (L2 POMDP slow). "
                f"This may be intentional for your experiment."
            )
        elif self.epsilon_decay < 0.95:
            logger.warning(
                f"epsilon_decay={self.epsilon_decay} is very fast. "
                f"Will reach ε=0.1 in {episodes_to_01:.0f} episodes from ε=1.0. "
                f"Agent may not explore enough before exploitation. "
                f"This may be intentional for your experiment."
            )

        return self


def load_training_config(config_dir: Path) -> TrainingConfig:
    """Load and validate training configuration.

    Args:
        config_dir: Directory containing training.yaml

    Returns:
        Validated TrainingConfig

    Raises:
        FileNotFoundError: If training.yaml not found
        ValueError: If validation fails (with helpful error message)

    Example:
        >>> config = load_training_config(Path("configs/L0_0_minimal"))
        >>> print(f"Device: {config.device}, Episodes: {config.max_episodes}")
        Device: cuda, Episodes: 500
    """
    try:
        data = load_yaml_section(config_dir, "training.yaml", "training")
        return TrainingConfig(**data)
    except ValidationError as e:
        raise ValueError(format_validation_error(e, "training.yaml")) from e
