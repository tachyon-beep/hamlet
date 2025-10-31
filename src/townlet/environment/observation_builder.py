"""Observation construction for Hamlet environments.

Handles full observability, partial observability (POMDP), and temporal features.
"""

import torch
from typing import Dict


class ObservationBuilder:
    """Constructs observations for agents in vectorized Hamlet environment.

    Supports three observation modes:
    - Full observability: One-hot grid position + meters + affordance encoding
    - Partial observability (POMDP): Local 5×5 window + position + meters + affordance
    - Temporal mechanics: Adds time_of_day and interaction_progress features
    """

    def __init__(
        self,
        num_agents: int,
        grid_size: int,
        device: torch.device,
        partial_observability: bool,
        vision_range: int,
        enable_temporal_mechanics: bool,
        num_affordance_types: int,
    ):
        """Initialize observation builder.

        Args:
            num_agents: Number of parallel agents
            grid_size: Grid dimension (grid_size × grid_size)
            device: PyTorch device (cpu or cuda)
            partial_observability: If True, use local window (POMDP)
            vision_range: Radius of vision window (2 = 5×5 window)
            enable_temporal_mechanics: Add temporal features
            num_affordance_types: Number of affordance types in environment
        """
        self.num_agents = num_agents
        self.grid_size = grid_size
        self.device = device
        self.partial_observability = partial_observability
        self.vision_range = vision_range
        self.enable_temporal_mechanics = enable_temporal_mechanics
        self.num_affordance_types = num_affordance_types

    def build_observations(
        self,
        positions: torch.Tensor,
        meters: torch.Tensor,
        affordances: Dict[str, torch.Tensor],
        time_of_day: int = 0,
        interaction_progress: torch.Tensor = None,
    ) -> torch.Tensor:
        """Build observations for all agents.

        Args:
            positions: Agent positions [num_agents, 2] (x, y)
            meters: Agent meter values [num_agents, 8]
            affordances: Dict of affordance_name -> position tensor
            time_of_day: Current time (0-23) if temporal mechanics enabled
            interaction_progress: Ticks completed [num_agents] if temporal mechanics enabled

        Returns:
            observations: [num_agents, observation_dim]
        """
        if self.partial_observability:
            obs = self._build_partial_observations(positions, meters, affordances)
        else:
            obs = self._build_full_observations(positions, meters, affordances)

        # Add temporal features if enabled
        if self.enable_temporal_mechanics:
            normalized_time = torch.full(
                (self.num_agents, 1), time_of_day / 23.0, device=self.device
            )
            normalized_progress = interaction_progress.unsqueeze(1) / 10.0
            obs = torch.cat([obs, normalized_time, normalized_progress], dim=1)

        return obs

    def _build_full_observations(
        self,
        positions: torch.Tensor,
        meters: torch.Tensor,
        affordances: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Build full grid observations (Level 1).

        Args:
            positions: Agent positions [num_agents, 2]
            meters: Agent meter values [num_agents, 8]
            affordances: Dict of affordance_name -> position

        Returns:
            observations: [num_agents, grid_size² + 8 + num_affordance_types + 1]
        """
        # Grid encoding: one-hot position
        grid_encoding = torch.zeros(
            self.num_agents, self.grid_size * self.grid_size, device=self.device
        )
        flat_indices = positions[:, 1] * self.grid_size + positions[:, 0]
        grid_encoding.scatter_(1, flat_indices.unsqueeze(1), 1.0)

        # Get affordance encoding
        affordance_encoding = self._build_affordance_encoding(positions, affordances)

        # Concatenate: grid + meters + affordance
        observations = torch.cat([grid_encoding, meters, affordance_encoding], dim=1)

        return observations

    def _build_partial_observations(
        self,
        positions: torch.Tensor,
        meters: torch.Tensor,
        affordances: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Build partial observations (Level 2 POMDP).

        Agent sees only local 5×5 window centered on its position.

        Args:
            positions: Agent positions [num_agents, 2]
            meters: Agent meter values [num_agents, 8]
            affordances: Dict of affordance_name -> position

        Returns:
            observations: [num_agents, window² + 2 + 8 + num_affordance_types + 1]
        """
        window_size = 2 * self.vision_range + 1
        local_grids = []

        for agent_idx in range(self.num_agents):
            agent_pos = positions[agent_idx]
            local_grid = torch.zeros(window_size * window_size, device=self.device)

            # Extract local window centered on agent
            for dy in range(-self.vision_range, self.vision_range + 1):
                for dx in range(-self.vision_range, self.vision_range + 1):
                    world_x = agent_pos[0] + dx
                    world_y = agent_pos[1] + dy

                    # Check if position is within grid bounds
                    if 0 <= world_x < self.grid_size and 0 <= world_y < self.grid_size:
                        # Check if there's an affordance at this position
                        has_affordance = False
                        for affordance_pos in affordances.values():
                            if affordance_pos[0] == world_x and affordance_pos[1] == world_y:
                                has_affordance = True
                                break

                        # Encode in local grid (1 = affordance, 0 = empty/out-of-bounds)
                        if has_affordance:
                            local_y = dy + self.vision_range
                            local_x = dx + self.vision_range
                            local_idx = local_y * window_size + local_x
                            local_grid[local_idx] = 1.0

            local_grids.append(local_grid)

        # Stack all local grids
        local_grids_batch = torch.stack(local_grids)

        # Normalize positions to [0, 1]
        normalized_positions = positions.float() / (self.grid_size - 1)

        # Get affordance encoding
        affordance_encoding = self._build_affordance_encoding(positions, affordances)

        # Concatenate: local_grid + position + meters + affordance
        observations = torch.cat(
            [local_grids_batch, normalized_positions, meters, affordance_encoding], dim=1
        )

        return observations

    def _build_affordance_encoding(
        self,
        positions: torch.Tensor,
        affordances: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Build one-hot encoding of current affordance under each agent.

        Args:
            positions: Agent positions [num_agents, 2]
            affordances: Dict of affordance_name -> position

        Returns:
            encoding: [num_agents, num_affordance_types + 1]
                Last dimension is "none" (not on any affordance)
        """
        # Initialize with "none" (all zeros except last column)
        affordance_encoding = torch.zeros(
            self.num_agents, self.num_affordance_types + 1, device=self.device
        )
        affordance_encoding[:, -1] = 1.0  # Default to "none"

        # Check each affordance
        for affordance_idx, (affordance_name, affordance_pos) in enumerate(affordances.items()):
            distances = torch.abs(positions - affordance_pos).sum(dim=1)
            on_affordance = distances == 0
            if on_affordance.any():
                affordance_encoding[on_affordance, -1] = 0.0  # Clear "none"
                affordance_encoding[on_affordance, affordance_idx] = 1.0

        return affordance_encoding
