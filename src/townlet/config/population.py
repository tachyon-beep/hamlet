"""Population configuration DTO with no-defaults enforcement.

Philosophy: All behavioral parameters must be explicitly specified.
No implicit defaults. Operator accountability.

Design: Validates agent count, Q-learning parameters, and network architecture.
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from townlet.config.base import format_validation_error, load_yaml_section


class PopulationConfig(BaseModel):
    """Population and Q-learning configuration.

    ALL FIELDS REQUIRED (no defaults) - enforces operator accountability.

    Example:
        >>> config = PopulationConfig(
        ...     num_agents=1,
        ...     learning_rate=0.00025,
        ...     gamma=0.99,
        ...     replay_buffer_capacity=10000,
        ...     network_type="simple",
        ... )
    """

    # Agent count (REQUIRED)
    num_agents: int = Field(gt=0, description="Number of agents to train (typically 1 for single-agent RL)")

    # Q-learning parameters (ALL REQUIRED)
    learning_rate: float = Field(gt=0.0, description="Adam optimizer learning rate (Atari DQN: 0.00025)")
    gamma: float = Field(gt=0.0, le=1.0, description="Q-learning discount factor (future reward importance)")
    replay_buffer_capacity: int = Field(gt=0, description="Experience replay buffer size (number of transitions)")

    # Network architecture (REQUIRED)
    network_type: Literal["simple", "recurrent", "structured"] = Field(
        description=("'simple' (MLP for full obs), 'recurrent' (LSTM for POMDP), or 'structured' (group encoders for semantic obs)")
    )

    # Observation masking (REQUIRED)
    mask_unused_obs: bool = Field(
        description=(
            "Apply active_mask to zero out padding dimensions in observations. "
            "True: Only process curriculum-active dimensions (saves compute, improves sample efficiency). "
            "False: Process all observation dimensions (standard behavior). "
            "Currently applies to: RND networks. Future: Q-networks when network_type='structured'."
        )
    )


def load_population_config(config_dir: Path) -> PopulationConfig:
    """Load and validate population configuration.

    Args:
        config_dir: Directory containing training.yaml

    Returns:
        Validated PopulationConfig

    Raises:
        FileNotFoundError: If training.yaml not found
        ValueError: If validation fails (with helpful error message)
    """
    try:
        data = load_yaml_section(config_dir, "training.yaml", "population")
        return PopulationConfig(**data)
    except ValidationError as e:
        raise ValueError(format_validation_error(e, "population section (training.yaml)")) from e
