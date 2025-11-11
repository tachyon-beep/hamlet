"""Action configuration schemas for composable action space."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, ValidationError, model_validator

# VFS Integration (Phase 1 - TASK-002C)
from townlet.vfs.schema import WriteSpec


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
    delta: list[int | float] | None = Field(
        description=(
            "Movement delta [dx, dy] or [dx, dy, dz] for standard movement. "
            "Integer for discrete grids, float for continuous substrates. Pass None for non-movement."
        ),
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

    # VFS Integration (Phase 1 - TASK-002C)
    reads: list[str] = Field(
        default_factory=list,
        description="Variables this action reads (for dependency tracking). Defaults to empty list for backward compatibility.",
    )
    writes: list[WriteSpec] = Field(
        default_factory=list,
        description="Variables this action writes (with expressions). Defaults to empty list for backward compatibility.",
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


class ActionSpaceConfig(BaseModel):
    """Collection wrapper for action definitions."""

    actions: list[ActionConfig]

    def get_action_by_name(self, name: str) -> ActionConfig:
        for action in self.actions:
            if action.name == name:
                return action
        raise KeyError(f"Action '{name}' not found")


def load_global_actions_config(global_actions_path: Path | None = None) -> ActionSpaceConfig:
    """Load configs/global_actions.yaml and return all custom actions."""

    yaml_path = Path(global_actions_path or Path("configs") / "global_actions.yaml")
    if not yaml_path.exists():
        raise FileNotFoundError(f"global_actions.yaml not found at {yaml_path}")

    try:
        with yaml_path.open() as handle:
            data = yaml.safe_load(handle) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed to parse {yaml_path}: {exc}") from exc

    custom_actions = data.get("custom_actions")
    if custom_actions is None:
        raise ValueError(f"{yaml_path} must define 'custom_actions' list")

    parsed_actions: list[ActionConfig] = []
    for next_id, raw_action in enumerate(custom_actions):
        base = {
            **raw_action,
            "id": raw_action.get("id", next_id),
            "enabled": raw_action.get("enabled", True),
            "source": raw_action.get("source", "custom"),
            "delta": raw_action.get("delta"),
            "teleport_to": raw_action.get("teleport_to"),
            "description": raw_action.get("description"),
            "icon": raw_action.get("icon"),
            "source_affordance": raw_action.get("source_affordance"),
        }
        try:
            parsed_actions.append(ActionConfig(**base))
        except ValidationError as exc:
            raise ValueError(f"Invalid action in {yaml_path}: {exc}") from exc

    return ActionSpaceConfig(actions=parsed_actions)
