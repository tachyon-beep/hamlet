"""Aspatial substrate (no positioning - pure state machine)."""

import torch

from townlet.environment.action_config import ActionConfig
from townlet.substrate.base import SpatialSubstrate


class AspatialSubstrate(SpatialSubstrate):
    """Substrate with no spatial positioning (pure state machine).

    Key insight: The meters (bars) are the true universe. Spatial positioning
    is just an OPTIONAL overlay for navigation and affordance placement.

    An aspatial universe reveals this truth:
    - No concept of "position" or "distance"
    - All affordances are "everywhere and nowhere"
    - Agents interact directly without movement
    - Pure resource management (no navigation)

    Pedagogical value:
    - Reveals that positioning is a design choice, not fundamental
    - Simplifies universe design (no grid to configure)
    - Focuses learning on resource management, not navigation

    Use cases:
    - Abstract planning problems (no physical space)
    - Resource management games (Factorio-like)
    - State machines (FSM without spatial component)
    """

    @property
    def position_dim(self) -> int:
        """Aspatial has zero-dimensional positions (no positioning)."""
        return 0

    @property
    def position_dtype(self) -> torch.dtype:
        """Aspatial positions use torch.long (empty tensors, but typed for consistency)."""
        return torch.long

    def initialize_positions(self, num_agents: int, device: torch.device) -> torch.Tensor:
        """Return empty position tensors (agents have no position)."""
        return torch.zeros((num_agents, 0), dtype=torch.long, device=device)

    def apply_movement(
        self,
        positions: torch.Tensor,
        deltas: torch.Tensor,
    ) -> torch.Tensor:
        """No movement possible in aspatial universe (return unchanged)."""
        return positions

    def compute_distance(
        self,
        pos1: torch.Tensor,
        pos2: torch.Tensor,
    ) -> torch.Tensor:
        """Return zero distance (no spatial meaning in aspatial universe)."""
        num_agents = pos1.shape[0]
        return torch.zeros(num_agents, device=pos1.device)

    def encode_observation(
        self,
        positions: torch.Tensor,
        affordances: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Return empty observation encoding (no position to encode)."""
        num_agents = positions.shape[0]
        device = positions.device
        return torch.zeros((num_agents, 0), device=device)

    def get_observation_dim(self) -> int:
        """Aspatial has zero observation dimensions (no position encoding)."""
        return 0

    def normalize_positions(self, positions: torch.Tensor) -> torch.Tensor:
        """Normalize positions to [0, 1] range (always relative encoding).

        Args:
            positions: [num_agents, position_dim] positions

        Returns:
            [num_agents, position_dim] normalized to [0, 1]
        """
        # Aspatial has no positions, return empty tensor
        num_agents = positions.shape[0]
        device = positions.device
        return torch.zeros((num_agents, 0), dtype=torch.float32, device=device)

    def get_valid_neighbors(
        self,
        position: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Return empty list (no spatial neighbors in aspatial universe)."""
        return []

    def is_on_position(
        self,
        agent_positions: torch.Tensor,
        target_position: torch.Tensor,
    ) -> torch.Tensor:
        """Return all True (agents are 'everywhere' in aspatial universe).

        In aspatial universes, there's no concept of being "on" a position.
        All agents can interact with all affordances at any time.
        """
        num_agents = agent_positions.shape[0]
        return torch.ones(num_agents, dtype=torch.bool, device=agent_positions.device)

    def get_all_positions(self) -> list[list[int]]:
        """Return empty list (aspatial has no positions)."""
        return []

    def encode_partial_observation(
        self,
        positions: torch.Tensor,
        affordances: dict[str, torch.Tensor],
        vision_range: int,
    ) -> torch.Tensor:
        """Return empty tensor (aspatial has no position encoding).

        In aspatial universes, there's no concept of "local window" or "vision."
        All affordances are accessible without positioning.

        Returns:
            [num_agents, 0] empty tensor
        """
        num_agents = positions.shape[0]
        device = positions.device
        return torch.zeros((num_agents, 0), device=device)

    def supports_enumerable_positions(self) -> bool:
        """Aspatial substrates have no positions."""
        return False

    def get_default_actions(self) -> list[ActionConfig]:
        """Return Aspatial's 2 default actions (no movement).

        Returns:
            [INTERACT, WAIT] only (no spatial movement)
        """
        return [
            ActionConfig(
                id=0,
                name="INTERACT",
                type="interaction",
                costs={"energy": 0.003},
                description="Interact with affordance (aspatial, no position required)",
                source="substrate",
                enabled=True,
            ),
            ActionConfig(
                id=1,
                name="WAIT",
                type="passive",
                costs={"energy": 0.004},
                description="Wait (idle metabolic cost)",
                source="substrate",
                enabled=True,
            ),
        ]
