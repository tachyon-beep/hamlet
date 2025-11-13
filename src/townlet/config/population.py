"""Population configuration DTO with no-defaults enforcement.

Philosophy: All behavioral parameters must be explicitly specified.
No implicit defaults. Operator accountability.

Design: Validates agent count, Q-learning parameters, and network architecture.

IMPORTANT: When brain.yaml exists, gamma/learning_rate/replay_buffer_capacity
MUST NOT be specified in training.yaml - they are managed by brain.yaml.
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from townlet.config.base import format_validation_error, load_yaml_section


class PopulationConfig(BaseModel):
    """Population and Q-learning configuration.

    ALL FIELDS REQUIRED (no defaults) - enforces operator accountability.

    BREAKING CHANGE (Brain As Code): When brain.yaml exists, these fields
    are FORBIDDEN in training.yaml:population section:
    - learning_rate → Use brain.yaml:optimizer.learning_rate
    - gamma → Use brain.yaml:q_learning.gamma
    - replay_buffer_capacity → Use brain.yaml:replay.capacity

    Example:
        >>> config = PopulationConfig(
        ...     num_agents=1,
        ...     learning_rate=0.00025,
        ...     gamma=0.99,
        ...     replay_buffer_capacity=10000,
        ...     network_type="simple",
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    # Agent count (REQUIRED)
    num_agents: int = Field(gt=0, description="Number of agents to train (typically 1 for single-agent RL)")

    # Q-learning parameters (OPTIONAL when brain.yaml exists, REQUIRED otherwise)
    learning_rate: float | None = Field(
        default=None,
        description="Adam optimizer learning rate. None when managed by brain.yaml:optimizer.learning_rate",
    )
    gamma: float | None = Field(
        default=None,
        description="Q-learning discount factor. None when managed by brain.yaml:q_learning.gamma",
    )
    replay_buffer_capacity: int | None = Field(
        default=None,
        description="Experience replay buffer size. None when managed by brain.yaml:replay.capacity",
    )

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

    @model_validator(mode="before")
    @classmethod
    def reject_brain_managed_fields(cls, data: dict) -> dict:
        """Reject brain-managed fields when present.

        BREAKING CHANGE: Enforces single source of truth for Q-learning parameters.

        brain.yaml is REQUIRED for all config packs (no backwards compatibility).
        These fields MUST NOT be in training.yaml:population:
        - learning_rate → Use brain.yaml:optimizer.learning_rate
        - gamma → Use brain.yaml:q_learning.gamma
        - replay_buffer_capacity → Use brain.yaml:replay.capacity

        When brain.yaml is loaded, these fields will be None in PopulationConfig.
        This prevents silent overrides that create non-reproducible configs.
        """
        # Check if brain.yaml exists (passed via _config_dir sentinel)
        config_dir = data.get("_config_dir")
        if config_dir is not None:
            # Remove sentinel before validation
            data = {k: v for k, v in data.items() if k != "_config_dir"}

            brain_yaml_path = Path(config_dir) / "brain.yaml"
            if not brain_yaml_path.exists():
                # brain.yaml is REQUIRED - fail if missing
                raise FileNotFoundError(
                    f"brain.yaml is REQUIRED but not found at: {brain_yaml_path}\n\n"
                    f"HAMLET requires all config packs to have brain.yaml (no backwards compatibility).\n"
                    f"brain.yaml defines the neural network architecture and Q-learning hyperparameters.\n\n"
                    f"Fix: Create brain.yaml in {config_dir}/ using the template at configs/_default_brain.yaml\n"
                    f"Then calibrate the hyperparameters for your specific config pack."
                )

            # brain.yaml exists - reject brain-managed fields if present
            conflicting_fields = []
            if "learning_rate" in data and data["learning_rate"] is not None:
                conflicting_fields.append("learning_rate (use brain.yaml:optimizer.learning_rate)")
            if "gamma" in data and data["gamma"] is not None:
                conflicting_fields.append("gamma (use brain.yaml:q_learning.gamma)")
            if "replay_buffer_capacity" in data and data["replay_buffer_capacity"] is not None:
                conflicting_fields.append("replay_buffer_capacity (use brain.yaml:replay.capacity)")

            if conflicting_fields:
                raise ValueError(
                    f"training.yaml:population section contains fields managed by brain.yaml.\n"
                    f"These fields MUST be removed:\n"
                    f"  - {'\n  - '.join(conflicting_fields)}\n\n"
                    f"Reason: brain.yaml provides the single source of truth for Q-learning parameters.\n"
                    f"Having duplicate values in training.yaml creates non-reproducible configs.\n\n"
                    f"Fix: Delete these fields from training.yaml:population section."
                )

        return data


def load_population_config(config_dir: Path) -> PopulationConfig:
    """Load and validate population configuration.

    Args:
        config_dir: Directory containing training.yaml

    Returns:
        Validated PopulationConfig

    Raises:
        FileNotFoundError: If training.yaml not found
        ValueError: If validation fails (with helpful error message)
                   If brain-managed fields present when brain.yaml exists
    """
    try:
        data = load_yaml_section(config_dir, "training.yaml", "population")
        # Pass config_dir as sentinel for brain.yaml existence check
        data["_config_dir"] = str(config_dir)
        return PopulationConfig(**data)
    except ValidationError as e:
        raise ValueError(format_validation_error(e, "population section (training.yaml)")) from e
