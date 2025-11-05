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
    ):
        """Initialize 3D cubic grid.

        Args:
            width: Number of cells in X dimension
            height: Number of cells in Y dimension
            depth: Number of cells in Z dimension (floors/layers)
            boundary: Boundary mode
            distance_metric: Distance calculation method
        """
        if width <= 0 or height <= 0 or depth <= 0:
            raise ValueError(f"Grid dimensions must be positive: {width}×{height}×{depth}\n" f"Example: width: 8, height: 8, depth: 3")
        if boundary not in ("clamp", "wrap", "bounce", "sticky"):
            raise ValueError(f"Unknown boundary mode: {boundary}")
        if distance_metric not in ("manhattan", "euclidean", "chebyshev"):
            raise ValueError(f"Unknown distance metric: {distance_metric}")

        self.width = width
        self.height = height
        self.depth = depth
        self.boundary = boundary
        self.distance_metric = distance_metric

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

    def encode_observation(self, positions: torch.Tensor, affordances: dict[str, torch.Tensor]) -> torch.Tensor:
        """Normalize 3D positions to [0, 1] for each dimension.

        Args:
            positions: Agent positions [num_agents, 3]
            affordances: Dict of affordance positions (not used for Grid3D encoding)

        Returns [num_agents, 3] tensor (constant size regardless of grid dimensions).

        This avoids dimension explosion from one-hot encoding:
        - One-hot: 8×8×3 = 192 dims
        - Normalized: 3 dims

        Network must learn spatial relationships (no explicit topology),
        but representation is more flexible and scales to large grids.
        """
        num_agents = positions.shape[0]
        normalized = torch.zeros((num_agents, 3), dtype=torch.float32, device=positions.device)

        normalized[:, 0] = positions[:, 0].float() / max(self.width - 1, 1)
        normalized[:, 1] = positions[:, 1].float() / max(self.height - 1, 1)
        normalized[:, 2] = positions[:, 2].float() / max(self.depth - 1, 1)

        return normalized

    def get_observation_dim(self) -> int:
        """Grid3D observation is normalized coordinates (constant 3 dims).

        Unlike Grid2D which uses one-hot encoding (width × height dims),
        Grid3D uses normalized coordinates to avoid dimension explosion.
        """
        return 3

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
