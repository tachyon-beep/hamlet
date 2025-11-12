"""Brain configuration DTOs for declarative agent architecture.

Follows no-defaults principle: all behavioral parameters must be explicit.
Forward-compatible with future SDA (Software Defined Agent) architecture.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


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
