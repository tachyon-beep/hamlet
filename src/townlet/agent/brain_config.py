"""Brain configuration DTOs for declarative agent architecture.

Follows no-defaults principle: all behavioral parameters must be explicit.
Forward-compatible with future SDA (Software Defined Agent) architecture.
"""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class FeedforwardConfig(BaseModel):
    """Feedforward MLP architecture configuration.

    Example:
        >>> config = FeedforwardConfig(
        ...     hidden_layers=[256, 128],
        ...     activation="relu",
        ...     dropout=0.0,
        ...     layer_norm=True,
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    hidden_layers: list[int] = Field(min_length=1, description="Hidden layer sizes (e.g., [256, 128] for 2 hidden layers)")
    activation: Literal["relu", "gelu", "swish", "tanh", "elu"] = Field(description="Activation function")
    dropout: float = Field(ge=0.0, lt=1.0, description="Dropout probability (0.0 = no dropout)")
    layer_norm: bool = Field(description="Apply LayerNorm after each hidden layer")


class OptimizerConfig(BaseModel):
    """Optimizer configuration.

    All optimizer-specific parameters required (no defaults).

    Example:
        >>> adam = OptimizerConfig(
        ...     type="adam",
        ...     learning_rate=0.00025,
        ...     adam_beta1=0.9,
        ...     adam_beta2=0.999,
        ...     adam_eps=1e-8,
        ...     weight_decay=0.0,
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["adam", "adamw", "sgd", "rmsprop"] = Field(description="Optimizer type")
    learning_rate: float = Field(gt=0.0, description="Learning rate")

    # Adam/AdamW parameters (required for type=adam/adamw)
    adam_beta1: float | None = Field(default=None, ge=0.0, lt=1.0, description="Adam beta1 parameter (required for adam/adamw)")
    adam_beta2: float | None = Field(default=None, ge=0.0, lt=1.0, description="Adam beta2 parameter (required for adam/adamw)")
    adam_eps: float | None = Field(default=None, gt=0.0, description="Adam epsilon parameter (required for adam/adamw)")

    # SGD parameters (required for type=sgd)
    sgd_momentum: float | None = Field(default=None, ge=0.0, le=1.0, description="SGD momentum (required for sgd)")
    sgd_nesterov: bool | None = Field(default=None, description="Use Nesterov momentum (required for sgd)")

    # RMSprop parameters (required for type=rmsprop)
    rmsprop_alpha: float | None = Field(default=None, ge=0.0, lt=1.0, description="RMSprop alpha/decay (required for rmsprop)")
    rmsprop_eps: float | None = Field(default=None, gt=0.0, description="RMSprop epsilon (required for rmsprop)")

    # Common parameter
    weight_decay: float = Field(ge=0.0, description="L2 weight decay (all optimizers)")


class LossConfig(BaseModel):
    """Loss function configuration.

    Example:
        >>> mse = LossConfig(type="mse")
        >>> huber = LossConfig(type="huber", huber_delta=1.0)
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["mse", "huber", "smooth_l1"] = Field(description="Loss function type")

    huber_delta: float = Field(default=1.0, gt=0.0, description="Delta parameter for Huber loss (ignored for mse/smooth_l1)")


class ArchitectureConfig(BaseModel):
    """Neural network architecture configuration.

    Future: Will support recurrent, dueling, rainbow architectures.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["feedforward"] = Field(description="Architecture type (Phase 1: feedforward only)")

    # Architecture-specific configs (exactly one required based on type)
    feedforward: FeedforwardConfig | None = Field(default=None, description="Feedforward MLP config (required when type=feedforward)")

    @model_validator(mode="after")
    def validate_architecture_match(self) -> "ArchitectureConfig":
        """Ensure architecture config matches type."""
        if self.type == "feedforward" and self.feedforward is None:
            raise ValueError("type='feedforward' requires feedforward config")
        return self


class QLearningConfig(BaseModel):
    """Q-learning algorithm configuration."""

    model_config = ConfigDict(extra="forbid")

    gamma: float = Field(ge=0.0, le=1.0, description="Discount factor")
    target_update_frequency: int = Field(gt=0, description="Update target network every N training steps")
    use_double_dqn: bool = Field(description="Use Double DQN algorithm (van Hasselt et al. 2016)")


class BrainConfig(BaseModel):
    """Complete brain configuration.

    Top-level configuration for agent architecture, optimizer, and learning.
    All fields required (no-defaults principle).

    Example:
        >>> config = BrainConfig(
        ...     version="1.0",
        ...     description="Feedforward Q-network for L0",
        ...     architecture=ArchitectureConfig(...),
        ...     optimizer=OptimizerConfig(...),
        ...     loss=LossConfig(...),
        ...     q_learning=QLearningConfig(...),
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    version: str = Field(description="Configuration schema version (e.g., '1.0')")
    description: str = Field(description="Human-readable description of this brain configuration")
    architecture: ArchitectureConfig = Field(description="Network architecture specification")
    optimizer: OptimizerConfig = Field(description="Optimizer configuration")
    loss: LossConfig = Field(description="Loss function configuration")
    q_learning: QLearningConfig = Field(description="Q-learning algorithm parameters")


def load_brain_config(config_dir: Path) -> BrainConfig:
    """Load and validate brain configuration from brain.yaml.

    Args:
        config_dir: Directory containing brain.yaml

    Returns:
        Validated BrainConfig

    Raises:
        FileNotFoundError: If brain.yaml not found
        ValueError: If validation fails

    Example:
        >>> config = load_brain_config(Path("configs/L0_0_minimal"))
        >>> print(config.architecture.type)
        feedforward
    """
    config_path = Path(config_dir) / "brain.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"brain.yaml not found in {config_dir}. " f"Brain configuration is required for all config packs.")

    try:
        with open(config_path) as f:
            data = yaml.safe_load(f)
        return BrainConfig(**data)
    except ValidationError as e:
        # Format validation error for user-friendly output
        error_msgs = []
        for error in e.errors():
            field_path = " -> ".join(str(loc) for loc in error["loc"])
            error_msgs.append(f"  - {field_path}: {error['msg']}")

        formatted_errors = "\n".join(error_msgs)
        raise ValueError(
            f"Invalid brain.yaml in {config_dir}:\n{formatted_errors}\n\n" f"See docs/config-schemas/brain.md for valid schema."
        ) from e
