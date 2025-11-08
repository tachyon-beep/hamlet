"""Exploration configuration DTO with no-defaults enforcement.

Philosophy: All behavioral parameters must be explicitly specified.
No implicit defaults. Operator accountability.

Design: Validates RND (Random Network Distillation) exploration parameters
and intrinsic motivation annealing settings.

NOTE: This DTO was discovered during risk assessment - present in all 12
config packs but missing from original TASK-003 plan.
"""

from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from townlet.config.base import format_validation_error, load_yaml_section


class ExplorationConfig(BaseModel):
    """Exploration (RND + adaptive intrinsic motivation) configuration.

    ALL FIELDS REQUIRED (no defaults) - enforces operator accountability.

    Example:
        >>> config = ExplorationConfig(
        ...     embed_dim=128,
        ...     initial_intrinsic_weight=1.0,
        ...     variance_threshold=100.0,
        ...     survival_window=100,
        ... )
    """

    # RND parameters (REQUIRED)
    embed_dim: int = Field(gt=0, description="Embedding dimension for RND predictor network")

    # Intrinsic reward parameters (ALL REQUIRED)
    initial_intrinsic_weight: float = Field(
        ge=0.0, description="Initial weight for intrinsic rewards (vs extrinsic). 1.0 = exploration priority"
    )
    variance_threshold: float = Field(gt=0.0, description="Survival variance threshold for annealing (higher = slower annealing)")
    survival_window: int = Field(gt=0, description="Window size for tracking survival consistency (episodes)")


def load_exploration_config(config_dir: Path) -> ExplorationConfig:
    """Load and validate exploration configuration.

    Args:
        config_dir: Directory containing training.yaml

    Returns:
        Validated ExplorationConfig

    Raises:
        FileNotFoundError: If training.yaml not found
        ValueError: If validation fails (with helpful error message)

    Example:
        >>> config = load_exploration_config(Path("configs/L0_0_minimal"))
        >>> print(f"RND embed_dim: {config.embed_dim}")
        RND embed_dim: 64
    """
    try:
        data = load_yaml_section(config_dir, "training.yaml", "exploration")
        return ExplorationConfig(**data)
    except ValidationError as e:
        raise ValueError(format_validation_error(e, "exploration section (training.yaml)")) from e
