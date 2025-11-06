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
    costs: dict[str, float] = Field(
        default_factory=dict,
        description="Meter costs: {meter_name: amount}. Negative = restoration.",
    )

    # Effects: Additional meter changes beyond costs
    effects: dict[str, float] = Field(
        default_factory=dict,
        description="Meter effects: {meter_name: amount}. For actions with benefits.",
    )

    # Movement-specific
    delta: list[int] | None = Field(
        default=None,
        description="Movement delta [dx, dy] or [dx, dy, dz] for standard movement",
    )
    teleport_to: list[int] | None = Field(
        default=None,
        description="Teleport destination [x, y] or [x, y, z]. Overrides delta.",
    )

    # Enabled/disabled state (for curriculum progression)
    enabled: bool = Field(
        default=True,
        description="Whether this action is enabled in current config (for masking)",
    )

    # Metadata
    description: str | None = Field(default=None, description="Human-readable description")
    icon: str | None = Field(default=None, max_length=10, description="Emoji for UI")
    source: Literal["substrate", "custom", "affordance"] = Field(
        default="custom",
        description="Where this action came from",
    )
    source_affordance: str | None = Field(
        default=None,
        description="If source='affordance', which affordance provided it",
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
