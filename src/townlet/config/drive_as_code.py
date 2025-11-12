"""
Drive As Code (DAC) configuration DTOs.

This module defines Pydantic schemas for declarative reward function specifications.
DAC extracts all reward logic from Python code into composable YAML configurations.

Task 1.4: IntrinsicStrategyConfig DTO - Configuration for intrinsic curiosity strategies.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class IntrinsicStrategyConfig(BaseModel):
    """Configuration for intrinsic curiosity rewards.

    Supports five strategy types:
    - rnd: Random Network Distillation (novelty-seeking)
    - icm: Intrinsic Curiosity Module (prediction error)
    - count_based: State visitation counts
    - adaptive_rnd: RND with performance-based annealing
    - none: No intrinsic rewards (pure extrinsic)

    Modifiers can be applied to adjust intrinsic weight contextually
    (e.g., crisis suppression when resources are low).

    Example:
        ```yaml
        intrinsic:
          strategy: rnd
          base_weight: 0.1
          apply_modifiers: [energy_crisis]
          rnd_config:
            feature_dim: 128
            learning_rate: 0.001
        ```
    """

    model_config = ConfigDict(extra="forbid")

    # Required fields
    strategy: Literal["rnd", "icm", "count_based", "adaptive_rnd", "none"]
    base_weight: float = Field(ge=0.0, le=1.0)
    apply_modifiers: list[str] = Field(default_factory=list)

    # Optional strategy-specific configurations
    rnd_config: dict[str, Any] | None = None
    icm_config: dict[str, Any] | None = None
    count_config: dict[str, Any] | None = None
    adaptive_config: dict[str, Any] | None = None
