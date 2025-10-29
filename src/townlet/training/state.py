"""
State representations for Townlet training.

Contains DTOs for cold path (config, checkpoints, telemetry) using Pydantic
for validation, and hot path (training loop) using PyTorch tensors.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import List


class CurriculumDecision(BaseModel):
    """
    Cold path: Curriculum decision for environment configuration.

    Returned by CurriculumManager to specify environment settings.
    Validated at construction, immutable, serializable.
    """
    model_config = ConfigDict(frozen=True)

    difficulty_level: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Difficulty level from 0.0 (easiest) to 1.0 (hardest)"
    )
    active_meters: List[str] = Field(
        ...,
        min_length=1,
        max_length=6,
        description="Which meters are active (e.g., ['energy', 'hygiene'])"
    )
    depletion_multiplier: float = Field(
        ...,
        gt=0.0,
        le=10.0,
        description="Depletion rate multiplier (0.1 = 10x slower, 1.0 = normal)"
    )
    reward_mode: str = Field(
        ...,
        pattern=r'^(shaped|sparse)$',
        description="Reward mode: 'shaped' (dense) or 'sparse'"
    )
    reason: str = Field(
        ...,
        min_length=1,
        description="Human-readable explanation for this decision"
    )
