"""Action space builder for composable action spaces."""

from pathlib import Path

import torch
import yaml

from townlet.environment.action_config import ActionConfig
from townlet.substrate.base import SpatialSubstrate


class ComposedActionSpace:
    """Composed action space with metadata about sources.

    CRITICAL: All curriculum levels use the SAME action space (same action_dim).
    Disabled actions are masked out but still occupy action IDs.

    Attributes:
        actions: Complete action list (substrate + custom + affordance)
        substrate_action_count: Number from substrate (6 for Grid2D, 8 for Grid3D)
        custom_action_count: Number from global_actions.yaml
        affordance_action_count: Number from affordances (future)
        enabled_action_names: Set of action names enabled in this config
    """

    def __init__(
        self,
        actions: list[ActionConfig],
        substrate_action_count: int,
        custom_action_count: int,
        affordance_action_count: int,
        enabled_action_names: set[str] | None = None,
    ):
        self.actions = actions
        self.substrate_action_count = substrate_action_count
        self.custom_action_count = custom_action_count
        self.affordance_action_count = affordance_action_count
        self.enabled_action_names = enabled_action_names

    @property
    def action_dim(self) -> int:
        """Total number of actions (including disabled ones).

        CRITICAL: This is the SAME across all curriculum levels.
        """
        return len(self.actions)

    @property
    def enabled_action_count(self) -> int:
        """Number of enabled actions in this config."""
        return sum(1 for a in self.actions if a.enabled)

    def get_action_by_id(self, action_id: int) -> ActionConfig:
        """Get action by ID."""
        return self.actions[action_id]

    def get_action_by_name(self, name: str) -> ActionConfig:
        """Get action by name."""
        for action in self.actions:
            if action.name == name:
                return action
        raise ValueError(f"Action '{name}' not found")

    def get_enabled_actions(self) -> list[ActionConfig]:
        """Get only enabled actions."""
        return [a for a in self.actions if a.enabled]

    def get_disabled_actions(self) -> list[ActionConfig]:
        """Get only disabled actions."""
        return [a for a in self.actions if not a.enabled]

    def get_substrate_actions(self) -> list[ActionConfig]:
        """Get only substrate-provided actions."""
        return [a for a in self.actions if a.source == "substrate"]

    def get_custom_actions(self) -> list[ActionConfig]:
        """Get only custom actions."""
        return [a for a in self.actions if a.source == "custom"]

    def get_base_action_mask(self, num_agents: int, device: torch.device) -> torch.Tensor:
        """Get base action mask (disabled actions masked out).

        Args:
            num_agents: Number of agents
            device: PyTorch device

        Returns:
            [num_agents, action_dim] bool tensor
            False = action disabled, True = action available
        """
        mask = torch.ones(num_agents, self.action_dim, dtype=torch.bool, device=device)

        # Mask out disabled actions
        for action_id, action in enumerate(self.actions):
            if not action.enabled:
                mask[:, action_id] = False

        return mask


class ActionSpaceBuilder:
    """Composes action space from global vocabulary.

    Action Space = Substrate Actions + Custom Actions (from global_actions.yaml)
    Enabled Actions = Subset specified in training.yaml

    CRITICAL: All curriculum levels share the SAME action vocabulary.
    This enables checkpoint transfer (same action_dim across configs).

    Examples:
        Global: 6 substrate + 4 custom = 10 total actions (all configs)
        L0: 7 enabled (UP/DOWN/LEFT/RIGHT/INTERACT/WAIT/REST)
        L1: 10 enabled (all actions available)
    """

    def __init__(
        self,
        substrate: SpatialSubstrate,
        global_actions_path: Path,
        enabled_action_names: list[str] | None = None,
    ):
        """Initialize action space builder.

        Args:
            substrate: Spatial substrate (provides substrate actions)
            global_actions_path: Path to configs/global_actions.yaml
            enabled_action_names: List of action names to enable (from training.yaml)
                                 If None, all actions are enabled.
        """
        self.substrate = substrate
        self.global_actions_path = global_actions_path
        self.enabled_action_names = set(enabled_action_names) if enabled_action_names else None

    def build(self) -> ComposedActionSpace:
        """Build complete action space from global vocabulary."""
        actions = []
        action_id = 0

        # === 1. SUBSTRATE ACTIONS (REQUIRED) ===
        substrate_actions = self.substrate.get_default_actions()
        for action in substrate_actions:
            action.id = action_id
            action.source = "substrate"
            action.enabled = self._is_enabled(action.name)
            actions.append(action)
            action_id += 1

        # === 2. CUSTOM ACTIONS (from global_actions.yaml) ===
        custom_action_count = 0
        if self.global_actions_path.exists():
            custom_actions = self._load_global_custom_actions()
            for action in custom_actions:
                action.id = action_id
                action.source = "custom"
                action.enabled = self._is_enabled(action.name)
                actions.append(action)
                action_id += 1
            custom_action_count = len(custom_actions)

        # === 3. AFFORDANCE ACTIONS (FUTURE - Deferred to TASK-003) ===
        # Will be added to global_actions.yaml when implemented

        return ComposedActionSpace(
            actions=actions,
            substrate_action_count=len(substrate_actions),
            custom_action_count=custom_action_count,
            affordance_action_count=0,  # Future
            enabled_action_names=self.enabled_action_names,
        )

    def _is_enabled(self, action_name: str) -> bool:
        """Check if action is enabled in this config."""
        if self.enabled_action_names is None:
            return True  # All actions enabled if not specified
        return action_name in self.enabled_action_names

    def _load_global_custom_actions(self) -> list[ActionConfig]:
        """Load custom actions from global_actions.yaml."""
        with open(self.global_actions_path) as f:
            data = yaml.safe_load(f)

        # Global file contains custom actions only (substrate provides its own)
        custom_action_data = data.get("custom_actions", [])

        # Parse into ActionConfig objects
        actions = []
        for action_dict in custom_action_data:
            # Add default fields if missing
            if "costs" not in action_dict:
                action_dict["costs"] = {}
            if "effects" not in action_dict:
                action_dict["effects"] = {}

            # Temporary ID (will be reassigned by build())
            action_dict["id"] = 0

            actions.append(ActionConfig(**action_dict))

        return actions
