"""3D cubic grid substrate with integer coordinates (x, y, z)."""

from typing import Literal

import torch

from .base import SpatialSubstrate


class Grid3DSubstrate(SpatialSubstrate):
    """3D cubic grid substrate.

    Position representation: [x, y, z] where:
    - x ∈ [0, width)
    - y ∈ [0, height)
    - z ∈ [0, depth)

    Movement actions: 6 directions (±x, ±y, ±z)

    Observation encoding: Normalized coordinates [0, 1] (not one-hot)
    - Prevents dimension explosion (3 dims instead of width*height*depth)
    - Matches Continuous substrate encoding strategy
    - Network learns spatial relationships

    Boundary modes:
    - clamp: Hard walls (position clamped to bounds)
    - wrap: Toroidal wraparound (Pac-Man in 3D)
    - bounce: Elastic reflection
    - sticky: Stay in place when hitting boundary

    Distance metrics:
    - manhattan: L1 norm, |x1-x2| + |y1-y2| + |z1-z2| (matches 6-directional movement)
    - euclidean: L2 norm, sqrt((x1-x2)² + (y1-y2)² + (z1-z2)²) (straight-line distance)
    - chebyshev: L∞ norm, max(|x1-x2|, |y1-y2|, |z1-z2|) (king's move in 3D)
    """

    position_dim = 3
    position_dtype = torch.long

    def __init__(
        self,
        width: int,
        height: int,
        depth: int,
        boundary: Literal["clamp", "wrap", "bounce", "sticky"],
        distance_metric: Literal["manhattan", "euclidean", "chebyshev"] = "manhattan",
        observation_encoding: Literal["relative", "scaled", "absolute"] = "relative",  # NEW: Phase 5C
    ):
        """Initialize 3D cubic grid.

        Args:
            width: Number of cells in X dimension
            height: Number of cells in Y dimension
            depth: Number of cells in Z dimension (floors/layers)
            boundary: Boundary mode
            distance_metric: Distance calculation method
            observation_encoding: Position encoding strategy ("relative", "scaled", "absolute")
        """
        if width <= 0 or height <= 0 or depth <= 0:
            raise ValueError(f"Grid dimensions must be positive: {width}×{height}×{depth}\nExample: width: 8, height: 8, depth: 3")
        if boundary not in ("clamp", "wrap", "bounce", "sticky"):
            raise ValueError(f"Unknown boundary mode: {boundary}")
        if distance_metric not in ("manhattan", "euclidean", "chebyshev"):
            raise ValueError(f"Unknown distance metric: {distance_metric}")

        self.width = width
        self.height = height
        self.depth = depth
        self.boundary = boundary
        self.distance_metric = distance_metric
        self.observation_encoding = observation_encoding  # NEW

    @property
    def coordinate_semantics(self) -> dict:
        """Describe what each dimension represents."""
        return {
            "X": "horizontal",  # Left/right
            "Y": "vertical",  # Up/down (screen coordinates)
            "Z": "depth",  # Floor/layer
        }

    def initialize_positions(self, num_agents: int, device: torch.device) -> torch.Tensor:
        """Randomly initialize positions in 3D grid."""
        return torch.stack(
            [
                torch.randint(0, self.width, (num_agents,), device=device),
                torch.randint(0, self.height, (num_agents,), device=device),
                torch.randint(0, self.depth, (num_agents,), device=device),
            ],
            dim=1,
        )

    def apply_movement(self, positions: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
        """Apply movement deltas with boundary handling in 3D."""
        new_positions = positions + deltas.long()

        if self.boundary == "clamp":
            new_positions[:, 0] = torch.clamp(new_positions[:, 0], 0, self.width - 1)
            new_positions[:, 1] = torch.clamp(new_positions[:, 1], 0, self.height - 1)
            new_positions[:, 2] = torch.clamp(new_positions[:, 2], 0, self.depth - 1)

        elif self.boundary == "wrap":
            new_positions[:, 0] = new_positions[:, 0] % self.width
            new_positions[:, 1] = new_positions[:, 1] % self.height
            new_positions[:, 2] = new_positions[:, 2] % self.depth

        elif self.boundary == "bounce":
            for dim, max_val in enumerate([self.width, self.height, self.depth]):
                negative_mask = new_positions[:, dim] < 0
                new_positions[negative_mask, dim] = -new_positions[negative_mask, dim]

                exceed_mask = new_positions[:, dim] >= max_val
                new_positions[exceed_mask, dim] = 2 * (max_val - 1) - new_positions[exceed_mask, dim]

                new_positions[:, dim] = torch.clamp(new_positions[:, dim], 0, max_val - 1)

        elif self.boundary == "sticky":
            for dim, max_val in enumerate([self.width, self.height, self.depth]):
                out_of_bounds = (new_positions[:, dim] < 0) | (new_positions[:, dim] >= max_val)
                new_positions[out_of_bounds, dim] = positions[out_of_bounds, dim]

        return new_positions

    def compute_distance(self, pos1: torch.Tensor, pos2: torch.Tensor) -> torch.Tensor:
        """Compute distance between positions in 3D."""
        if self.distance_metric == "manhattan":
            return torch.abs(pos1 - pos2).sum(dim=-1)
        elif self.distance_metric == "euclidean":
            return torch.sqrt(((pos1 - pos2) ** 2).sum(dim=-1))
        elif self.distance_metric == "chebyshev":
            return torch.abs(pos1 - pos2).max(dim=-1)[0]

    def is_on_position(self, positions: torch.Tensor, target_position: torch.Tensor) -> torch.Tensor:
        """Check if agents are on target position (exact match in 3D)."""
        return (positions == target_position).all(dim=-1)

    def _encode_relative(
        self,
        positions: torch.Tensor,
        affordances: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Encode positions as normalized coordinates [0, 1].

        Args:
            positions: Agent positions [num_agents, 3]
            affordances: Affordance positions (currently unused)

        Returns:
            [num_agents, 3] normalized positions
        """
        num_agents = positions.shape[0]
        device = positions.device

        normalized = torch.zeros((num_agents, 3), dtype=torch.float32, device=device)
        normalized[:, 0] = positions[:, 0].float() / max(self.width - 1, 1)
        normalized[:, 1] = positions[:, 1].float() / max(self.height - 1, 1)
        normalized[:, 2] = positions[:, 2].float() / max(self.depth - 1, 1)

        return normalized

    def _encode_scaled(
        self,
        positions: torch.Tensor,
        affordances: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Encode positions as normalized coordinates + range metadata.

        Args:
            positions: Agent positions [num_agents, 3]
            affordances: Affordance positions (currently unused)

        Returns:
            [num_agents, 6] normalized positions + range sizes
            First 3 dims: normalized [0, 1]
            Last 3 dims: (width, height, depth)
        """
        num_agents = positions.shape[0]
        device = positions.device

        # Get normalized positions
        relative = self._encode_relative(positions, affordances)

        # Add range metadata
        ranges = (
            torch.tensor(
                [float(self.width), float(self.height), float(self.depth)],
                dtype=torch.float32,
                device=device,
            )
            .unsqueeze(0)
            .expand(num_agents, -1)
        )

        return torch.cat([relative, ranges], dim=1)

    def _encode_absolute(
        self,
        positions: torch.Tensor,
        affordances: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Encode positions as raw unnormalized coordinates.

        Args:
            positions: Agent positions [num_agents, 3]
            affordances: Affordance positions (currently unused)

        Returns:
            [num_agents, 3] raw coordinates (as float)
        """
        return positions.float()

    def encode_observation(
        self,
        positions: torch.Tensor,
        affordances: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Encode agent positions and affordances into observation space.

        Args:
            positions: Agent positions [num_agents, 3]
            affordances: Dict mapping affordance names to positions [3]

        Returns:
            Encoded observations with dimensions based on encoding mode:
            - relative: [num_agents, 3]
            - scaled: [num_agents, 6]
            - absolute: [num_agents, 3]
        """
        if self.observation_encoding == "relative":
            return self._encode_relative(positions, affordances)
        elif self.observation_encoding == "scaled":
            return self._encode_scaled(positions, affordances)
        elif self.observation_encoding == "absolute":
            return self._encode_absolute(positions, affordances)
        else:
            raise ValueError(f"Invalid observation_encoding: {self.observation_encoding}. " f"Must be 'relative', 'scaled', or 'absolute'.")

    def get_observation_dim(self) -> int:
        """Return dimensionality of position encoding.

        Returns:
            - relative: 3 (normalized x, y, z)
            - scaled: 6 (normalized x, y, z, width, height, depth)
            - absolute: 3 (raw x, y, z)
        """
        if self.observation_encoding == "relative":
            return 3
        elif self.observation_encoding == "scaled":
            return 6
        elif self.observation_encoding == "absolute":
            return 3
        else:
            raise ValueError(f"Invalid observation_encoding: {self.observation_encoding}")

    def normalize_positions(self, positions: torch.Tensor) -> torch.Tensor:
        """Normalize positions to [0, 1] range (always relative encoding).

        Args:
            positions: [num_agents, 3] grid positions

        Returns:
            [num_agents, 3] normalized to [0, 1]
        """
        return self._encode_relative(positions, {})

    def get_all_positions(self) -> list[list[int]]:
        """Get all valid positions in 3D grid."""
        positions = []
        for z in range(self.depth):
            for y in range(self.height):
                for x in range(self.width):
                    positions.append([x, y, z])
        return positions

    def get_valid_neighbors(self, position: torch.Tensor) -> list[torch.Tensor]:
        """Get 6 cardinal neighbors in 3D (±x, ±y, ±z).

        Args:
            position: Position tensor [3] or list of [x, y, z]

        Returns:
            List of neighbor position tensors
        """
        if isinstance(position, torch.Tensor):
            x, y, z = position.tolist()
        else:
            x, y, z = position

        neighbors = [
            [x, y - 1, z],  # Negative Y
            [x, y + 1, z],  # Positive Y
            [x - 1, y, z],  # Negative X
            [x + 1, y, z],  # Positive X
            [x, y, z - 1],  # Negative Z
            [x, y, z + 1],  # Positive Z
        ]

        if self.boundary == "clamp":
            neighbors = [n for n in neighbors if 0 <= n[0] < self.width and 0 <= n[1] < self.height and 0 <= n[2] < self.depth]

        return [torch.tensor(n, dtype=torch.long) for n in neighbors]

    def encode_partial_observation(
        self,
        positions: torch.Tensor,
        affordances: dict[str, torch.Tensor],
        vision_range: int,
    ) -> torch.Tensor:
        """Encode local 3D window around each agent (POMDP).

        Extracts a local (2*vision_range+1)³ cube centered on each agent's
        position. Affordances within the window are marked.

        Args:
            positions: [num_agents, 3] agent positions
            affordances: {name: [3]} affordance positions
            vision_range: radius of vision (e.g., 2 for 5×5×5 window)

        Returns:
            [num_agents, window_size³] local grid encoding
            where window_size = 2*vision_range + 1

        Note: Handles boundary cases - if agent near edge, out-of-bounds
        cells are marked as empty.
        """
        num_agents = positions.shape[0]
        device = positions.device
        window_size = 2 * vision_range + 1

        # Initialize local grids for all agents
        local_grids = torch.zeros(
            (num_agents, window_size, window_size, window_size),
            device=device,
            dtype=torch.float32,
        )

        # For each agent, extract local window
        for agent_idx in range(num_agents):
            agent_x, agent_y, agent_z = positions[agent_idx]

            # Mark affordances in local window
            for affordance_pos in affordances.values():
                aff_x = affordance_pos[0].item()
                aff_y = affordance_pos[1].item()
                aff_z = affordance_pos[2].item()

                # Compute relative position in local window
                rel_x = aff_x - agent_x + vision_range
                rel_y = aff_y - agent_y + vision_range
                rel_z = aff_z - agent_z + vision_range

                # Check if affordance is within vision window
                if 0 <= rel_x < window_size and 0 <= rel_y < window_size and 0 <= rel_z < window_size:
                    local_grids[agent_idx, rel_z, rel_y, rel_x] = 1.0

        # Flatten local grids: [num_agents, W, W, W] → [num_agents, W³]
        return local_grids.reshape(num_agents, -1)
