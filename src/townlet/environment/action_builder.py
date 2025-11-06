"""Action space builder for composable action spaces."""

import torch

from townlet.environment.action_config import ActionConfig


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
