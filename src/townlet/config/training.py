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

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

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
        ...     use_double_dqn=True,
        ...     epsilon_start=1.0,
        ...     epsilon_decay=0.995,
        ...     epsilon_min=0.01,
        ...     sequence_length=8,
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    # Compute device (REQUIRED)
    device: Literal["cpu", "cuda", "mps"] = Field(description="Compute device: 'cuda' (GPU), 'cpu' (CPU-only), 'mps' (Apple Silicon)")

    # Training duration (REQUIRED)
    max_episodes: int = Field(gt=0, description="Total episodes to train")

    # Q-learning hyperparameters (ALL REQUIRED)
    train_frequency: int = Field(gt=0, description="Train Q-network every N steps")
    target_update_frequency: int = Field(gt=0, description="Update target network every N training steps")
    batch_size: int = Field(gt=0, description="Experience replay batch size")
    max_grad_norm: float = Field(gt=0, description="Gradient clipping threshold (prevents exploding gradients)")

    # Q-learning algorithm variant (REQUIRED)
    use_double_dqn: bool = Field(
        description=(
            "Use Double DQN algorithm (van Hasselt et al. 2016) instead of vanilla DQN. "
            "Double DQN reduces Q-value overestimation by using online network for action selection. "
            "True: Q_target = r + γ * Q_target(s', argmax_a Q_online(s', a)) [Double DQN] "
            "False: Q_target = r + γ * max_a Q_target(s', a) [Vanilla DQN]"
        )
    )

    # Epsilon-greedy exploration (ALL REQUIRED)
    epsilon_start: float = Field(ge=0.0, le=1.0, description="Initial exploration rate (1.0 = 100% random)")
    epsilon_decay: float = Field(gt=0.0, lt=1.0, description="Decay per episode (< 1.0)")
    epsilon_min: float = Field(ge=0.0, le=1.0, description="Minimum exploration rate (floor)")

    # Recurrent-specific (REQUIRED for all configs, used when network_type=recurrent)
    sequence_length: int = Field(gt=0, description="Length of sequences for LSTM training (recurrent networks only)")

    allow_unfeasible_universe: bool = Field(
        default=False,
        description=(
            "Set true to downgrade compiler feasibility guards (economic, sustainability) from errors to warnings. "
            "Use only when intentionally building stress-test or instructional worlds."
        ),
    )

    enabled_actions: list[str] | None = Field(
        default=None,
        description=(
            "Optional list of action names from the global vocabulary that should remain enabled for this config. "
            "Set to null to enable all actions, or [] to intentionally disable the entire action space when running diagnostics."
        ),
    )

    @field_validator("enabled_actions")
    @classmethod
    def _validate_enabled_actions(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return value
        stripped = []
        seen: set[str] = set()
        duplicates: list[str] = []
        for raw_name in value:
            name = raw_name.strip()
            if not name:
                raise ValueError("enabled_actions entries must be non-empty strings.")
            stripped.append(name)
            if name in seen:
                duplicates.append(name)
            else:
                seen.add(name)
        if duplicates:
            dup_list = ", ".join(sorted(set(duplicates)))
            raise ValueError(f"enabled_actions must not contain duplicate names: {dup_list}.")
        return stripped

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
        try:
            episodes_to_01 = math.log(0.1) / math.log(self.epsilon_decay)
        except (ZeroDivisionError, ValueError) as exc:
            logger.warning(
                "epsilon_decay=%s produced invalid episodes_to_01 calculation (%s). Skipping exact episode estimate.",
                self.epsilon_decay,
                exc,
            )
            episodes_to_01 = None

        if self.epsilon_decay > 0.999:
            if episodes_to_01 is not None:
                logger.warning(
                    f"epsilon_decay={self.epsilon_decay} is very slow. "
                    f"Will take {episodes_to_01:.0f} episodes to reach ε=0.1 from ε=1.0. "
                    f"Typical values: 0.99 (L0 fast), 0.995 (L0.5/L1 moderate), "
                    f"0.998 (L2 POMDP slow). "
                    f"This may be intentional for your experiment."
                )
            else:
                logger.warning(
                    f"epsilon_decay={self.epsilon_decay} is very slow. "
                    f"Could not estimate episodes_to_01 due to invalid epsilon_decay value. "
                    f"Typical values: 0.99 (L0 fast), 0.995 (L0.5/L1 moderate), "
                    f"0.998 (L2 POMDP slow). "
                    f"This may be intentional for your experiment."
                )
        elif self.epsilon_decay < 0.95:
            if episodes_to_01 is not None:
                logger.warning(
                    f"epsilon_decay={self.epsilon_decay} is very fast. "
                    f"Will reach ε=0.1 in {episodes_to_01:.0f} episodes from ε=1.0. "
                    f"Agent may not explore enough before exploitation. "
                    f"This may be intentional for your experiment."
                )
            else:
                logger.warning(
                    f"epsilon_decay={self.epsilon_decay} is very fast. "
                    f"Could not estimate episodes_to_01 due to invalid epsilon_decay value. "
                    f"Agent may not explore enough before exploitation. "
                    f"This may be intentional for your experiment."
                )

        return self


def load_training_config(config_dir: Path, training_config_path: Path | None = None) -> TrainingConfig:
    """Load and validate training configuration.

    Args:
        config_dir: Directory containing training.yaml (default source)
        training_config_path: Optional explicit training.yaml path overriding config_dir

    Returns:
        Validated TrainingConfig

    Raises:
        FileNotFoundError: If training.yaml not found
        ValueError: If validation fails (with helpful error message)

    Example:
        >>> config = load_training_config(Path("configs/L0_0_minimal"))
        >>> print(f"Device: {config.device}, Episodes: {config.max_episodes}")
        Device: cuda, Episodes: 500

        >>> override = Path("/tmp/custom_training.yaml")
        >>> config = load_training_config(Path("configs/L0_0_minimal"), training_config_path=override)
        >>> print(config.training.max_episodes)
        42
    """
    config_dir = Path(config_dir)
    if training_config_path is not None:
        training_config_path = Path(training_config_path)
        section_dir = training_config_path.parent
        filename = training_config_path.name
        context_label = str(training_config_path)
    else:
        section_dir = config_dir
        filename = "training.yaml"
        context_label = "training.yaml"

    try:
        data = load_yaml_section(section_dir, filename, "training")
        return TrainingConfig(**data)
    except ValidationError as e:
        raise ValueError(format_validation_error(e, context_label)) from e
