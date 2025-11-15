"""N-dimensional grid substrate (N≥4 dimensions)."""

import warnings
from typing import Literal

import torch

from townlet.environment.action_config import ActionConfig
from townlet.substrate.base import SpatialSubstrate


class GridNDSubstrate(SpatialSubstrate):
    """N-dimensional hypercube grid for abstract state spaces.

    GridND supports 4D to 100D discrete grid substrates. For 2D/3D grids,
    use Grid2DSubstrate or Grid3DSubstrate for better ergonomics.

    Coordinate system:
    - positions: [d0, d1, d2, ..., dN] where d0 is dimension 0, etc.
    - Origin: all zeros [0, 0, 0, ...]
    - Each dimension increases positively from 0 to (size - 1)

    Observation encoding:
    - relative: Normalized coordinates [0, 1] (N dimensions)
    - scaled: Normalized + dimension sizes (2N dimensions)
    - absolute: Raw unnormalized coordinates (N dimensions)

    Use cases:
    - High-dimensional RL research
    - Abstract state space experiments
    - Transfer learning from low-D to high-D
    """

    def __init__(
        self,
        dimension_sizes: list[int],
        boundary: Literal["clamp", "wrap", "bounce", "sticky"],
        distance_metric: Literal["manhattan", "euclidean", "chebyshev"] = "manhattan",
        observation_encoding: Literal["relative", "scaled", "absolute"] = "relative",
        topology: Literal["hypercube"] = "hypercube",  # NEW: GridND uses hypercube topology
    ):
        """Initialize N-dimensional grid substrate.

        Args:
            dimension_sizes: Size of each dimension [d0_size, d1_size, ..., dN_size]
            boundary: Boundary mode ("clamp", "wrap", "bounce", "sticky")
            distance_metric: Distance metric ("manhattan", "euclidean", "chebyshev")
            observation_encoding: Position encoding strategy ("relative", "scaled", "absolute")
            topology: Grid topology ("hypercube" for N-dimensional Cartesian grid)

        Raises:
            ValueError: If dimensions < 4 or any size <= 0

        Warnings:
            UserWarning: If dimensions >= 10 (action space size warning)
        """
        # Validate dimension count
        num_dims = len(dimension_sizes)
        if num_dims < 4:
            raise ValueError(
                f"GridND requires at least 4 dimensions, got {num_dims}. Use Grid2DSubstrate (2D) or Grid3DSubstrate (3D) instead."
            )

        if num_dims > 100:
            raise ValueError(f"GridND dimension count ({num_dims}) exceeds limit (100)")

        # Warn at N≥10 (action space grows large)
        if num_dims >= 10:
            warnings.warn(
                f"GridND with {num_dims} dimensions has {2 * num_dims + 2} actions. "
                f"Large action spaces may be challenging to train. "
                f"Verify this is intentional for your research.",
                UserWarning,
            )

        # Validate dimension sizes
        for i, size in enumerate(dimension_sizes):
            if size <= 0:
                raise ValueError(f"Dimension sizes must be positive. Dimension {i} has size {size}.")

        # Validate parameters
        if boundary not in ("clamp", "wrap", "bounce", "sticky"):
            raise ValueError(f"Unknown boundary mode: {boundary}")

        if distance_metric not in ("manhattan", "euclidean", "chebyshev"):
            raise ValueError(f"Unknown distance metric: {distance_metric}")

        if observation_encoding not in ("relative", "scaled", "absolute"):
            raise ValueError(f"Unknown observation encoding: {observation_encoding}")

        # Store configuration
        self.dimension_sizes = dimension_sizes
        self.boundary = boundary
        self.distance_metric = distance_metric
        self.observation_encoding = observation_encoding
        self.topology = topology  # NEW: Store topology

    @property
    def position_dim(self) -> int:
        """Return number of dimensions."""
        return len(self.dimension_sizes)

    @property
    def position_dtype(self) -> torch.dtype:
        """Grid positions are integers (discrete cells)."""
        return torch.long

    @property
    def action_space_size(self) -> int:
        """Return number of discrete actions: 2N + 2.

        GridND uses the standard spatial substrate action space:
        - 2N movement actions (±1 per dimension)
        - 1 INTERACT action
        - 1 WAIT action (lower energy cost than movement)

        This matches Grid2D/3D for consistency and environment compatibility.

        Returns:
            2*N + 2 where N = position_dim
        """
        return 2 * self.position_dim + 2

    # Implementation methods
    def initialize_positions(self, num_agents: int, device: torch.device) -> torch.Tensor:
        """Initialize random positions uniformly across N-dimensional grid.

        Returns:
            [num_agents, N] tensor of integer positions
        """
        return torch.stack(
            [torch.randint(0, dim_size, (num_agents,), device=device, dtype=torch.long) for dim_size in self.dimension_sizes],
            dim=1,
        )

    def apply_movement(self, positions: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
        """Apply movement deltas with boundary handling.

        Args:
            positions: [num_agents, N] current positions (long)
            deltas: [num_agents, N] movement deltas (float32)

        Returns:
            [num_agents, N] new positions after boundary handling
        """
        # Cast deltas to long for grid substrates
        new_positions = positions + deltas.long()

        # Apply boundary handling per dimension
        if self.boundary == "clamp":
            for dim_idx, dim_size in enumerate(self.dimension_sizes):
                new_positions[:, dim_idx] = torch.clamp(new_positions[:, dim_idx], 0, dim_size - 1)

        elif self.boundary == "wrap":
            for dim_idx, dim_size in enumerate(self.dimension_sizes):
                new_positions[:, dim_idx] = new_positions[:, dim_idx] % dim_size

        elif self.boundary == "bounce":
            for dim_idx, dim_size in enumerate(self.dimension_sizes):
                # Handle negative positions (reflect across 0)
                negative_mask = new_positions[:, dim_idx] < 0
                new_positions[negative_mask, dim_idx] = -new_positions[negative_mask, dim_idx]

                # Handle positions >= dim_size (reflect across upper boundary)
                exceed_mask = new_positions[:, dim_idx] >= dim_size
                new_positions[exceed_mask, dim_idx] = 2 * (dim_size - 1) - new_positions[exceed_mask, dim_idx]

                # Safety clamp (in case of large velocities)
                new_positions[:, dim_idx] = torch.clamp(new_positions[:, dim_idx], 0, dim_size - 1)

        elif self.boundary == "sticky":
            for dim_idx, dim_size in enumerate(self.dimension_sizes):
                out_of_bounds = (new_positions[:, dim_idx] < 0) | (new_positions[:, dim_idx] >= dim_size)
                new_positions[out_of_bounds, dim_idx] = positions[out_of_bounds, dim_idx]

        return new_positions

    def compute_distance(self, pos1: torch.Tensor, pos2: torch.Tensor) -> torch.Tensor:
        """Compute distance between positions using configured metric.

        Handles broadcasting: pos2 can be [N] or [batch, N].

        Args:
            pos1: [batch, N] positions
            pos2: [N] or [batch, N] positions

        Returns:
            [batch] distances
        """
        # Handle broadcasting: pos2 might be single position [N] or batch [batch, N]
        if pos2.dim() == 1:
            pos2 = pos2.unsqueeze(0)  # [N] → [1, N]

        if self.distance_metric == "manhattan":
            # L1 distance: sum of absolute differences
            return torch.abs(pos1 - pos2).sum(dim=-1)

        elif self.distance_metric == "euclidean":
            # L2 distance: sqrt(sum of squared differences)
            return torch.sqrt(((pos1 - pos2) ** 2).sum(dim=-1))

        elif self.distance_metric == "chebyshev":
            # L∞ distance: max of absolute differences
            return torch.abs(pos1 - pos2).max(dim=-1)[0]

    def get_default_actions(self) -> list[ActionConfig]:
        """Return GridND's 2N+2 default actions with default costs.

        Returns:
            [DIM0_NEG, DIM0_POS, DIM1_NEG, DIM1_POS, ..., INTERACT, WAIT]

        Example:
            4D grid: 8 movement + INTERACT + WAIT = 10 actions
            7D grid: 14 movement + INTERACT + WAIT = 16 actions
        """
        actions = []
        action_id = 0

        # Generate movement actions for each dimension
        n_dims = len(self.dimension_sizes)
        for dim_idx in range(n_dims):
            # Negative direction (DIM{N}_NEG)
            delta: list[int | float] = [0] * n_dims
            delta[dim_idx] = -1
            actions.append(
                ActionConfig(
                    id=action_id,
                    name=f"DIM{dim_idx}_NEG",
                    type="movement",
                    delta=delta,
                    teleport_to=None,
                    costs={"energy": 0.005, "hygiene": 0.003, "satiation": 0.004},
                    effects={},
                    description=f"Move -1 along dimension {dim_idx}",
                    icon=None,
                    source="substrate",
                    source_affordance=None,
                    enabled=True,
                )
            )
            action_id += 1

            # Positive direction (DIM{N}_POS)
            delta = [0] * n_dims
            delta[dim_idx] = 1
            actions.append(
                ActionConfig(
                    id=action_id,
                    name=f"DIM{dim_idx}_POS",
                    type="movement",
                    delta=delta,
                    teleport_to=None,
                    costs={"energy": 0.005, "hygiene": 0.003, "satiation": 0.004},
                    effects={},
                    description=f"Move +1 along dimension {dim_idx}",
                    icon=None,
                    source="substrate",
                    source_affordance=None,
                    enabled=True,
                )
            )
            action_id += 1

        # Core interactions
        actions.append(
            ActionConfig(
                id=action_id,
                name="INTERACT",
                type="interaction",
                delta=None,
                teleport_to=None,
                costs={"energy": 0.003},
                effects={},
                description="Interact with affordance at current position",
                icon=None,
                source="substrate",
                source_affordance=None,
                enabled=True,
            )
        )
        action_id += 1

        actions.append(
            ActionConfig(
                id=action_id,
                name="WAIT",
                type="passive",
                delta=None,
                teleport_to=None,
                costs={"energy": 0.004},
                effects={},
                description="Wait and do nothing (lower cost than movement)",
                icon=None,
                source="substrate",
                source_affordance=None,
                enabled=True,
            )
        )

        return actions

    def _encode_relative(self, positions: torch.Tensor, affordances: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode positions as normalized coordinates [0, 1] per dimension."""
        num_agents = positions.shape[0]
        device = positions.device

        normalized = torch.zeros((num_agents, len(self.dimension_sizes)), dtype=torch.float32, device=device)

        for dim_idx, dim_size in enumerate(self.dimension_sizes):
            normalized[:, dim_idx] = positions[:, dim_idx].float() / max(dim_size - 1, 1)

        return normalized

    def _encode_scaled(self, positions: torch.Tensor, affordances: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode positions as normalized coordinates + dimension sizes."""
        num_agents = positions.shape[0]
        device = positions.device

        # Get normalized positions
        relative = self._encode_relative(positions, affordances)

        # Add dimension sizes
        sizes_tensor = (
            torch.tensor(
                [float(size) for size in self.dimension_sizes],
                dtype=torch.float32,
                device=device,
            )
            .unsqueeze(0)
            .expand(num_agents, -1)
        )

        return torch.cat([relative, sizes_tensor], dim=1)

    def _encode_absolute(self, positions: torch.Tensor, affordances: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode positions as raw unnormalized coordinates."""
        return positions.float()

    def encode_observation(self, positions: torch.Tensor, affordances: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode agent positions into observation space.

        Args:
            positions: [num_agents, N] agent positions
            affordances: {name: [N]} affordance positions (currently unused)

        Returns:
            Encoded observations:
            - relative: [num_agents, N]
            - scaled: [num_agents, 2N]
            - absolute: [num_agents, N]
        """
        if self.observation_encoding == "relative":
            return self._encode_relative(positions, affordances)
        elif self.observation_encoding == "scaled":
            return self._encode_scaled(positions, affordances)
        elif self.observation_encoding == "absolute":
            return self._encode_absolute(positions, affordances)
        else:
            raise ValueError(f"Invalid observation_encoding: {self.observation_encoding}")

    def get_observation_dim(self) -> int:
        """Return dimensionality of position encoding.

        Returns:
            - relative: N (normalized coordinates)
            - scaled: 2N (normalized + sizes)
            - absolute: N (raw coordinates)
        """
        if self.observation_encoding == "relative":
            return len(self.dimension_sizes)
        elif self.observation_encoding == "scaled":
            return 2 * len(self.dimension_sizes)
        elif self.observation_encoding == "absolute":
            return len(self.dimension_sizes)
        else:
            raise ValueError(f"Invalid observation_encoding: {self.observation_encoding}")

    def normalize_positions(self, positions: torch.Tensor) -> torch.Tensor:
        """Normalize positions to [0, 1] range (always relative encoding).

        Args:
            positions: [num_agents, position_dim] positions

        Returns:
            [num_agents, position_dim] normalized to [0, 1]
        """
        return self._encode_relative(positions, {})

    def get_valid_neighbors(self, position: torch.Tensor) -> list[torch.Tensor]:
        """Get cardinal neighbors in N dimensions (2N neighbors).

        For each dimension, generates ±1 neighbors (positive and negative direction).

        Args:
            position: [N] position tensor or list

        Returns:
            List of [N] neighbor positions
            - For clamp boundary: only in-bounds neighbors
            - For other boundaries: all 2N neighbors (boundary handling in apply_movement)
        """
        if isinstance(position, torch.Tensor):
            coords = position.tolist()
        else:
            coords = list(position)

        neighbors = []

        # For each dimension, generate ±1 neighbors
        for dim_idx in range(len(self.dimension_sizes)):
            # Negative direction
            neighbor_neg = coords.copy()
            neighbor_neg[dim_idx] -= 1
            neighbors.append(neighbor_neg)

            # Positive direction
            neighbor_pos = coords.copy()
            neighbor_pos[dim_idx] += 1
            neighbors.append(neighbor_pos)

        if self.boundary == "clamp":
            # Filter out-of-bounds neighbors
            neighbors = [
                n for n in neighbors if all(0 <= n[dim_idx] < self.dimension_sizes[dim_idx] for dim_idx in range(len(self.dimension_sizes)))
            ]

        return [torch.tensor(n, dtype=torch.long) for n in neighbors]

    def is_on_position(self, agent_positions: torch.Tensor, target_position: torch.Tensor) -> torch.Tensor:
        """Check if agents are exactly on target position (exact match in N dimensions).

        Args:
            agent_positions: [num_agents, N] agent positions
            target_position: [N] target position

        Returns:
            [num_agents] boolean tensor (True if agent on target)
        """
        return (agent_positions == target_position).all(dim=-1)

    def get_all_positions(self) -> list[list[int]]:
        """Enumerate all positions in N-dimensional grid.

        Warning: Combinatorial explosion for high N or large grids!
        - 4D 3×3×3×3: 81 positions (manageable)
        - 10D 3^10: 59,049 positions (slow)
        - 10D 10^10: 10 billion positions (memory error)

        Returns:
            List of [N] positions

        Raises:
            MemoryError: If total positions > 10 million

        Warns:
            UserWarning: If total positions > 100,000
        """
        import itertools

        # Calculate total positions
        total_positions = 1
        for size in self.dimension_sizes:
            total_positions *= size

        # Error on absurd counts
        if total_positions > 10_000_000:
            raise MemoryError(
                f"get_all_positions() would generate {total_positions:,} positions. "
                f"This is too large for memory. Consider using initialize_positions() "
                f"for random sampling instead."
            )

        # Warn on large counts
        if total_positions > 100_000:
            warnings.warn(
                f"get_all_positions() generating {total_positions:,} positions. This may be slow and use significant memory.",
                UserWarning,
            )

        # Generate all combinations using Cartesian product
        ranges = [range(dim_size) for dim_size in self.dimension_sizes]
        return [list(coords) for coords in itertools.product(*ranges)]

    def get_capacity(self) -> int:
        """Calculate total positions analytically (product of dimension sizes).

        This is O(N) where N is number of dimensions, compared to O(∏ dimension_sizes)
        for get_all_positions(). For a 7D grid with [5,5,5,5,5,5,5], this is:
        - get_capacity(): 7 multiplications → instant
        - len(get_all_positions()): Generate 78,125 positions → ~50ms

        Returns:
            Total number of positions in the N-dimensional grid
        """
        capacity = 1
        for dim_size in self.dimension_sizes:
            capacity *= dim_size
        return capacity

    def supports_enumerable_positions(self) -> bool:
        return True

    def encode_partial_observation(
        self,
        positions: torch.Tensor,
        affordances: dict[str, torch.Tensor],
        vision_range: int,
    ) -> torch.Tensor:
        """Encode local window around agents for partial observability (POMDP).

        For N-dimensional grids, this would require an N-dimensional hypercube window,
        which becomes exponentially large. For example:
        - 2D with vision_range=2: 5×5 = 25 cells
        - 3D with vision_range=2: 5×5×5 = 125 cells
        - 4D with vision_range=2: 5×5×5×5 = 625 cells
        - 7D with vision_range=2: 5^7 = 78,125 cells!

        Current implementation: Not supported for N≥4. Use full observability
        with coordinate encoding instead.

        Args:
            positions: [num_agents, N] agent positions
            affordances: {name: [N]} affordance positions
            vision_range: radius of vision window

        Returns:
            Empty tensor [num_agents, 0] - partial obs not supported for ND

        Raises:
            NotImplementedError: Partial observability not supported for N≥4
        """
        raise NotImplementedError(
            f"Partial observability (POMDP) is not supported for {self.position_dim}D grids. "
            f"Local window size would be (2*{vision_range}+1)^{self.position_dim} = "
            f"{(2 * vision_range + 1) ** self.position_dim} cells, which is impractical. "
            f"Use full observability with 'relative' or 'scaled' observation_encoding instead."
        )
