"""Observation construction for Hamlet environments.

Handles full observability, partial observability (POMDP), and temporal features.
"""

from __future__ import annotations

import torch


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
        affordance_names: list[str],
        substrate,  # Add substrate parameter
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
            affordance_names: Full list of affordance names (observation vocabulary)
            substrate: Spatial substrate for position operations
        """
        self.num_agents = num_agents
        self.grid_size = grid_size
        self.device = device
        self.partial_observability = partial_observability
        self.vision_range = vision_range
        self.enable_temporal_mechanics = enable_temporal_mechanics
        self.num_affordance_types = num_affordance_types
        self.affordance_names = affordance_names
        self.substrate = substrate  # Store substrate reference

    def build_observations(
        self,
        positions: torch.Tensor,
        meters: torch.Tensor,
        affordances: dict[str, torch.Tensor],
        time_of_day: int = 0,
        interaction_progress: torch.Tensor | None = None,
        lifetime_progress: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build observations for all agents.

        Args:
            positions: Agent positions [num_agents, 2] (x, y)
            meters: Agent meter values [num_agents, 8]
            affordances: Dict of affordance_name -> position tensor
            time_of_day: Current time (0-23) if temporal mechanics enabled
            interaction_progress: Ticks completed [num_agents] if temporal mechanics enabled
            lifetime_progress: Lifetime progress [num_agents] from 0.0 (birth) to 1.0 (retirement)

        Returns:
            observations: [num_agents, observation_dim]
        """
        if self.partial_observability:
            obs = self._build_partial_observations(positions, meters, affordances)
        else:
            obs = self._build_full_observations(positions, meters, affordances)

        # Always add temporal features for forward compatibility
        # Even when temporal mechanics are disabled, time cycles naturally
        # This allows networks trained at L0/L0.5 to work at L2.5 without architecture changes
        # Agent learns to use or ignore these features based on whether they're meaningful
        if interaction_progress is None:
            interaction_progress = torch.zeros(self.num_agents, device=self.device)

        if lifetime_progress is None:
            lifetime_progress = torch.zeros(self.num_agents, device=self.device)

        # Encode time_of_day as [sin, cos] for cyclical representation
        # This allows the network to understand that 23:00 and 00:00 are close
        import math

        angle = (time_of_day / 24.0) * 2 * math.pi
        time_sin = torch.full((self.num_agents, 1), math.sin(angle), device=self.device)
        time_cos = torch.full((self.num_agents, 1), math.cos(angle), device=self.device)

        normalized_progress = interaction_progress.unsqueeze(1) / 10.0
        lifetime = lifetime_progress.unsqueeze(1).clamp(0.0, 1.0)  # Clamp to [0, 1]

        obs = torch.cat([obs, time_sin, time_cos, normalized_progress, lifetime], dim=1)

        return obs

    def _build_full_observations(
        self,
        positions: torch.Tensor,
        meters: torch.Tensor,
        affordances: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Build full observations using substrate encoding.

        Args:
            positions: Agent positions [num_agents, position_dim]
            meters: Agent meter values [num_agents, 8]
            affordances: Dict of affordance_name -> position

        Returns:
            observations: [num_agents, obs_dim]
                - substrate encoding: grid positions (substrate-specific)
                - 8: Agent meter values
                - num_affordance_types + 1: Current affordance (one-hot)
        """
        # Delegate position encoding to substrate
        # Grid2D: marks agent position AND affordance positions (0=empty, 1=agent/affordance, 2=both)
        # Aspatial: returns empty tensor (no position encoding)
        grid_encoding = self.substrate.encode_observation(positions, affordances)

        # Get affordance encoding
        affordance_encoding = self._build_affordance_encoding(positions, affordances)

        # Concatenate: grid + meters + affordance
        observations = torch.cat([grid_encoding, meters, affordance_encoding], dim=1)

        return observations

    def _build_partial_observations(
        self,
        positions: torch.Tensor,
        meters: torch.Tensor,
        affordances: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Build partial observations (POMDP) using substrate encoding.

        Agent sees only local window centered on its position.

        Args:
            positions: Agent positions [num_agents, position_dim]
            meters: Agent meter values [num_agents, 8]
            affordances: Dict of affordance_name -> position

        Returns:
            observations: [num_agents, window² + position_dim + 8 + num_affordance_types + 1]
        """
        # Local window encoding from substrate
        # Grid2D: extracts (2*vision_range+1)×(2*vision_range+1) window around agent
        # Aspatial: returns empty tensor (no position encoding)
        local_grids = self.substrate.encode_partial_observation(positions, affordances, vision_range=self.vision_range)

        # Normalized positions (for recurrent network position encoder)
        # For grid substrates: normalize by grid dimensions
        # For aspatial: positions are empty, normalized_positions will be empty
        if hasattr(self.substrate, "width") and hasattr(self.substrate, "height"):
            # Grid2D substrate: normalize to [0, 1]
            normalized_positions = positions.float() / torch.tensor(
                [self.substrate.width - 1, self.substrate.height - 1],
                device=self.device,
                dtype=torch.float32,
            )
        else:
            # Aspatial substrate: no position normalization needed
            normalized_positions = positions.float()

        # Get affordance encoding
        affordance_encoding = self._build_affordance_encoding(positions, affordances)

        # Concatenate: local_grid + position + meters + affordance
        observations = torch.cat([local_grids, normalized_positions, meters, affordance_encoding], dim=1)

        return observations

    def _build_affordance_encoding(
        self,
        positions: torch.Tensor,
        affordances: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Build one-hot encoding of current affordance under each agent.

        This encodes against the FULL affordance vocabulary (from affordances.yaml),
        not just deployed affordances. This ensures observation dimensions stay
        constant across curriculum levels.

        Args:
            positions: Agent positions [num_agents, 2]
            affordances: Dict of DEPLOYED affordance_name -> position

        Returns:
            encoding: [num_agents, num_affordance_types + 1]
                Last dimension is "none" (not on any affordance)
        """
        # Initialize with "none" (all zeros except last column)
        affordance_encoding = torch.zeros(self.num_agents, self.num_affordance_types + 1, device=self.device)
        affordance_encoding[:, -1] = 1.0  # Default to "none"

        # Iterate over FULL affordance vocabulary (not just deployed)
        # This ensures consistent encoding across curriculum levels
        for affordance_idx, affordance_name in enumerate(self.affordance_names):
            # Check if this affordance is DEPLOYED (has position on grid)
            if affordance_name in affordances:
                affordance_pos = affordances[affordance_name]
                # Check which agents are on affordance (using substrate)
                on_affordance = self.substrate.is_on_position(positions, affordance_pos)
                if on_affordance.any():
                    affordance_encoding[on_affordance, -1] = 0.0  # Clear "none"
                    affordance_encoding[on_affordance, affordance_idx] = 1.0
            # If affordance NOT deployed, agent can never be "on" it, stays as "none"

        return affordance_encoding
