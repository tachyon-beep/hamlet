"""Brain configuration DTOs for declarative agent architecture.

Follows no-defaults principle: all behavioral parameters must be explicit.
Forward-compatible with future SDA (Software Defined Agent) architecture.
"""

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


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


class CNNEncoderConfig(BaseModel):
    """CNN encoder configuration for vision processing.

    Example:
        >>> vision = CNNEncoderConfig(
        ...     channels=[16, 32],
        ...     kernel_sizes=[3, 3],
        ...     strides=[1, 1],
        ...     padding=[1, 1],
        ...     activation="relu",
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    channels: list[int] = Field(min_length=1, description="Channel progression for CNN layers (e.g., [16, 32])")
    kernel_sizes: list[int] = Field(min_length=1, description="Kernel size for each CNN layer")
    strides: list[int] = Field(min_length=1, description="Stride for each CNN layer")
    padding: list[int] = Field(min_length=1, description="Padding for each CNN layer")
    activation: Literal["relu", "gelu", "swish"] = Field(description="Activation function for CNN")

    @field_validator("channels", "kernel_sizes", "strides")
    @classmethod
    def validate_positive_values(cls, v: list[int], info) -> list[int]:
        """Ensure channels, kernel_sizes, and strides contain only positive integers."""
        if any(x <= 0 for x in v):
            raise ValueError(f"{info.field_name} must contain only positive integers (> 0)")
        return v

    @field_validator("padding")
    @classmethod
    def validate_non_negative_padding(cls, v: list[int]) -> list[int]:
        """Ensure padding contains only non-negative integers."""
        if any(x < 0 for x in v):
            raise ValueError("padding must contain only non-negative integers (>= 0)")
        return v

    @model_validator(mode="after")
    def validate_layer_consistency(self) -> "CNNEncoderConfig":
        """Ensure all layer lists have same length."""
        lengths = {
            "channels": len(self.channels),
            "kernel_sizes": len(self.kernel_sizes),
            "strides": len(self.strides),
            "padding": len(self.padding),
        }
        unique_lengths = set(lengths.values())
        if len(unique_lengths) > 1:
            raise ValueError(f"All CNN layer lists must have same length. Got: {lengths}")
        return self


class MLPEncoderConfig(BaseModel):
    """MLP encoder configuration.

    Used for position, meter, affordance encoders, and Q-head.

    Example:
        >>> position = MLPEncoderConfig(
        ...     hidden_sizes=[32],
        ...     activation="relu",
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    hidden_sizes: list[int] = Field(min_length=1, description="Hidden layer sizes (e.g., [32] for single layer)")
    activation: Literal["relu", "gelu", "swish"] = Field(description="Activation function")

    @field_validator("hidden_sizes")
    @classmethod
    def validate_positive_hidden_sizes(cls, v: list[int]) -> list[int]:
        """Ensure hidden_sizes contains only positive integers."""
        if any(x <= 0 for x in v):
            raise ValueError("hidden_sizes must contain only positive integers (> 0)")
        return v


class LSTMConfig(BaseModel):
    """LSTM configuration for recurrent networks.

    Example:
        >>> lstm = LSTMConfig(
        ...     hidden_size=256,
        ...     num_layers=1,
        ...     dropout=0.0,
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    hidden_size: int = Field(gt=0, description="LSTM hidden state dimension")
    num_layers: int = Field(ge=1, le=4, description="Number of stacked LSTM layers (1-4)")
    dropout: float = Field(ge=0.0, lt=1.0, description="Dropout between LSTM layers (0.0 = no dropout)")


class RecurrentConfig(BaseModel):
    """Recurrent architecture configuration for POMDP.

    Architecture: CNN vision → Position MLP → Meter MLP → Affordance MLP → LSTM → Q-head

    Example:
        >>> config = RecurrentConfig(
        ...     vision_encoder=CNNEncoderConfig(...),
        ...     position_encoder=MLPEncoderConfig(...),
        ...     meter_encoder=MLPEncoderConfig(...),
        ...     affordance_encoder=MLPEncoderConfig(...),
        ...     lstm=LSTMConfig(...),
        ...     q_head=MLPEncoderConfig(...),
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    vision_encoder: CNNEncoderConfig = Field(description="CNN encoder for local vision window")
    position_encoder: MLPEncoderConfig = Field(description="MLP encoder for position (x, y, z)")
    meter_encoder: MLPEncoderConfig = Field(description="MLP encoder for meter values")
    affordance_encoder: MLPEncoderConfig = Field(description="MLP encoder for affordance types")
    lstm: LSTMConfig = Field(description="LSTM for temporal memory")
    q_head: MLPEncoderConfig = Field(description="MLP Q-value head")


class ScheduleConfig(BaseModel):
    """Learning rate schedule configuration.

    Example:
        >>> constant = ScheduleConfig(type="constant")
        >>> step = ScheduleConfig(type="step_decay", step_size=1000, gamma=0.1)
        >>> cosine = ScheduleConfig(type="cosine", t_max=5000, eta_min=0.00001)
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["constant", "step_decay", "cosine", "exponential"] = Field(description="Learning rate schedule type")

    # StepLR parameters
    step_size: int | None = Field(default=None, gt=0, description="Step size for StepLR (required for type=step_decay)")
    gamma: float | None = Field(default=None, gt=0.0, lt=1.0, description="Multiplicative factor for StepLR or ExponentialLR")

    # CosineAnnealingLR parameters
    t_max: int | None = Field(default=None, gt=0, description="Maximum number of iterations for cosine schedule")
    eta_min: float | None = Field(default=None, ge=0.0, description="Minimum learning rate for cosine schedule")

    @model_validator(mode="after")
    def validate_schedule_params(self) -> "ScheduleConfig":
        """Ensure required parameters present for each schedule type."""
        if self.type == "step_decay":
            if self.step_size is None or self.gamma is None:
                raise ValueError("type='step_decay' requires step_size and gamma")
        elif self.type == "cosine":
            if self.t_max is None or self.eta_min is None:
                raise ValueError("type='cosine' requires t_max and eta_min")
        elif self.type == "exponential":
            if self.gamma is None:
                raise ValueError("type='exponential' requires gamma")
        return self


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

    # Common parameters
    weight_decay: float = Field(ge=0.0, description="L2 weight decay (all optimizers)")

    # Learning rate schedule (TASK-005 Phase 2: REQUIRED)
    schedule: ScheduleConfig = Field(description="Learning rate schedule")

    @model_validator(mode="after")
    def validate_optimizer_params(self) -> "OptimizerConfig":
        """Ensure optimizer-specific parameters are provided for selected optimizer type."""
        if self.type in ("adam", "adamw"):
            missing = []
            if self.adam_beta1 is None:
                missing.append("adam_beta1")
            if self.adam_beta2 is None:
                missing.append("adam_beta2")
            if self.adam_eps is None:
                missing.append("adam_eps")
            if missing:
                raise ValueError(f"type='{self.type}' requires {', '.join(missing)} to be specified")

        elif self.type == "sgd":
            missing = []
            if self.sgd_momentum is None:
                missing.append("sgd_momentum")
            if self.sgd_nesterov is None:
                missing.append("sgd_nesterov")
            if missing:
                raise ValueError(f"type='sgd' requires {', '.join(missing)} to be specified")

        elif self.type == "rmsprop":
            missing = []
            if self.rmsprop_alpha is None:
                missing.append("rmsprop_alpha")
            if self.rmsprop_eps is None:
                missing.append("rmsprop_eps")
            if missing:
                raise ValueError(f"type='rmsprop' requires {', '.join(missing)} to be specified")

        return self


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

    Phase 2: Supports feedforward and recurrent (LSTM) architectures.
    Future: Will support dueling, rainbow architectures.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["feedforward", "recurrent"] = Field(description="Architecture type")

    # Architecture-specific configs (exactly one required based on type)
    feedforward: FeedforwardConfig | None = Field(default=None, description="Feedforward MLP config (required when type=feedforward)")
    recurrent: RecurrentConfig | None = Field(default=None, description="Recurrent LSTM config (required when type=recurrent)")

    @model_validator(mode="after")
    def validate_architecture_match(self) -> "ArchitectureConfig":
        """Ensure architecture config matches type."""
        if self.type == "feedforward" and self.feedforward is None:
            raise ValueError("type='feedforward' requires feedforward config")
        if self.type == "recurrent" and self.recurrent is None:
            raise ValueError("type='recurrent' requires recurrent config")
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


def compute_brain_hash(config: BrainConfig) -> str:
    """Compute SHA256 hash of brain configuration for checkpoint provenance.

    Args:
        config: Brain configuration to hash

    Returns:
        64-character hex string (SHA256 hash)

    Example:
        >>> config = BrainConfig(...)
        >>> brain_hash = compute_brain_hash(config)
        >>> len(brain_hash)
        64

    Note:
        Hash is computed from JSON-serialized config with sorted keys
        to ensure deterministic output. Similar to drive_hash for DAC configs.
    """
    # Serialize config to JSON with sorted keys for deterministic hashing
    config_dict = config.model_dump()
    config_json = json.dumps(config_dict, sort_keys=True, indent=None)

    # Compute SHA256 hash
    hash_bytes = hashlib.sha256(config_json.encode("utf-8")).digest()
    return hash_bytes.hex()
