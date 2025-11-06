"""Action configuration schemas for composable action space."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ActionConfig(BaseModel):
    """Single action definition.

    Can come from:
    - Substrate (movement, INTERACT, WAIT)
    - Global custom actions (REST, MEDITATE, TELEPORT_HOME)
    - Affordances (future)

    CRITICAL: All curriculum levels share same action vocabulary.
    Disabled actions are masked out but still occupy action IDs.
    """

    id: int = Field(ge=0, description="Action ID (assigned by builder)")
    name: str = Field(min_length=1, description="Action name (UP, DOWN, REST, etc.)")
    type: Literal["movement", "interaction", "passive", "transaction"]

    # Costs: Multi-meter pattern (matches affordances.yaml effects structure)
    # NO DEFAULT: Operators must explicitly specify costs (use {} for no costs)
    costs: dict[str, float] = Field(
        description="Meter costs: {meter_name: amount}. Negative = restoration. REQUIRED: must explicitly specify (use {} for no costs).",
    )

    # Effects: Additional meter changes beyond costs
    # NO DEFAULT: Operators must explicitly specify effects (use {} for no effects)
    effects: dict[str, float] = Field(
        description="Meter effects: {meter_name: amount}. For actions with benefits. "
        "REQUIRED: must explicitly specify (use {} for no effects).",
    )

    # Movement-specific
    # NO DEFAULT: Callers must explicitly pass None for non-movement actions
    delta: list[int] | None = Field(
        description="Movement delta [dx, dy] or [dx, dy, dz] for standard movement. Pass None for non-movement.",
    )
    teleport_to: list[int] | None = Field(
        description="Teleport destination [x, y] or [x, y, z]. Overrides delta. Pass None if not teleporting.",
    )

    # Enabled/disabled state (for curriculum progression)
    # INTERNAL: Assigned by ActionSpaceBuilder based on enabled_actions list
    # Not specified in YAML configs - computed from training.yaml enabled_actions
    enabled: bool = Field(
        description="Whether this action is enabled in current config (for masking). INTERNAL: assigned by builder.",
    )

    # Metadata
    # NO DEFAULT: Callers must explicitly pass None or value
    description: str | None = Field(description="Human-readable description. Pass None if not needed.")
    icon: str | None = Field(max_length=10, description="Emoji for UI. Pass None if not needed.")
    source: Literal["substrate", "custom", "affordance"] = Field(
        description="Where this action came from. REQUIRED: must explicitly specify.",
    )
    source_affordance: str | None = Field(
        description="If source='affordance', which affordance provided it. Pass None for substrate/custom.",
    )

    @model_validator(mode="after")
    def validate_movement_actions(self) -> "ActionConfig":
        """Movement actions must have delta or teleport_to."""
        if self.type == "movement":
            if self.delta is None and self.teleport_to is None:
                raise ValueError(f"Movement action '{self.name}' must define delta or teleport_to")
        elif self.delta is not None or self.teleport_to is not None:
            raise ValueError(f"Non-movement action '{self.name}' cannot have delta or teleport_to")
        return self
