"""2D square grid substrate (replicates current HAMLET behavior)."""

from typing import Literal

import torch

from townlet.substrate.base import SpatialSubstrate


class Grid2DSubstrate(SpatialSubstrate):
    """2D square grid with configurable boundaries and distance metrics.

    This replicates the current hardcoded behavior of HAMLET (vectorized_env.py).

    Coordinate system:
    - positions: [x, y] where x is column, y is row
    - Origin: top-left corner is [0, 0]
    - x increases rightward, y increases downward

    Supported boundaries:
    - clamp: Hard walls (clamp to grid edge, current behavior)
    - wrap: Toroidal wraparound (Pac-Man style)
    - bounce: Elastic reflection (agent reflects back from boundary)
    - sticky: Sticky walls (agent stays in place when hitting boundary)

    Supported distance metrics:
    - manhattan: |x1-x2| + |y1-y2| (default, current behavior)
    - euclidean: sqrt((x1-x2)² + (y1-y2)²)
    - chebyshev: max(|x1-x2|, |y1-y2|)
    """

    def __init__(
        self,
        width: int,
        height: int,
        boundary: Literal["clamp", "wrap", "bounce", "sticky"],
        distance_metric: Literal["manhattan", "euclidean", "chebyshev"],
    ):
        """Initialize 2D grid substrate.

        Args:
            width: Grid width (number of columns)
            height: Grid height (number of rows)
            boundary: Boundary mode ("clamp", "wrap", "bounce", "sticky")
            distance_metric: Distance metric ("manhattan", "euclidean", "chebyshev")
        """
        if width <= 0 or height <= 0:
            raise ValueError(f"Grid dimensions must be positive: width={width}, height={height}")

        if boundary not in ("clamp", "wrap", "bounce", "sticky"):
            raise ValueError(f"Unknown boundary mode: {boundary}")

        if distance_metric not in ("manhattan", "euclidean", "chebyshev"):
            raise ValueError(f"Unknown distance metric: {distance_metric}")

        self.width = width
        self.height = height
        self.boundary = boundary
        self.distance_metric = distance_metric

    @property
    def position_dim(self) -> int:
        """2D grid has 2-dimensional positions."""
        return 2

    @property
    def position_dtype(self) -> torch.dtype:
        """Grid positions are integers (discrete cells)."""
        return torch.long

    def initialize_positions(self, num_agents: int, device: torch.device) -> torch.Tensor:
        """Initialize random positions uniformly across the grid."""
        return torch.stack(
            [
                torch.randint(0, self.width, (num_agents,), device=device),
                torch.randint(0, self.height, (num_agents,), device=device),
            ],
            dim=1,
        )

    def apply_movement(
        self,
        positions: torch.Tensor,
        deltas: torch.Tensor,
    ) -> torch.Tensor:
        """Apply movement deltas with boundary handling."""
        new_positions = positions + deltas

        if self.boundary == "clamp":
            # Hard walls: clamp to valid range
            new_positions[:, 0] = torch.clamp(new_positions[:, 0], 0, self.width - 1)
            new_positions[:, 1] = torch.clamp(new_positions[:, 1], 0, self.height - 1)

        elif self.boundary == "wrap":
            # Toroidal wraparound
            new_positions[:, 0] = new_positions[:, 0] % self.width
            new_positions[:, 1] = new_positions[:, 1] % self.height

        elif self.boundary == "bounce":
            # Elastic reflection: agent reflects back from boundaries
            # Negative positions: reflect to positive (mirror across 0)
            # Positions >= max: reflect back from upper boundary

            # Handle x-axis bouncing
            x_neg = new_positions[:, 0] < 0
            x_over = new_positions[:, 0] >= self.width
            new_positions[x_neg, 0] = torch.abs(new_positions[x_neg, 0])
            new_positions[x_over, 0] = 2 * (self.width - 1) - new_positions[x_over, 0]

            # Handle y-axis bouncing
            y_neg = new_positions[:, 1] < 0
            y_over = new_positions[:, 1] >= self.height
            new_positions[y_neg, 1] = torch.abs(new_positions[y_neg, 1])
            new_positions[y_over, 1] = 2 * (self.height - 1) - new_positions[y_over, 1]

        elif self.boundary == "sticky":
            # Sticky walls: if out of bounds, stay in place
            out_of_bounds_x = (new_positions[:, 0] < 0) | (new_positions[:, 0] >= self.width)
            out_of_bounds_y = (new_positions[:, 1] < 0) | (new_positions[:, 1] >= self.height)

            new_positions[out_of_bounds_x, 0] = positions[out_of_bounds_x, 0]
            new_positions[out_of_bounds_y, 1] = positions[out_of_bounds_y, 1]

        return new_positions

    def compute_distance(
        self,
        pos1: torch.Tensor,
        pos2: torch.Tensor,
    ) -> torch.Tensor:
        """Compute distance between positions using configured metric."""
        # Handle broadcasting: pos2 might be single position [2] or batch [N, 2]
        if pos2.dim() == 1:
            pos2 = pos2.unsqueeze(0)  # [2] → [1, 2]

        if self.distance_metric == "manhattan":
            # L1 distance: |x1-x2| + |y1-y2|
            return torch.abs(pos1 - pos2).sum(dim=-1)

        elif self.distance_metric == "euclidean":
            # L2 distance: sqrt((x1-x2)² + (y1-y2)²)
            return torch.sqrt(((pos1 - pos2) ** 2).sum(dim=-1))

        elif self.distance_metric == "chebyshev":
            # L∞ distance: max(|x1-x2|, |y1-y2|)
            return torch.abs(pos1 - pos2).max(dim=-1)[0]

    def encode_observation(
        self,
        positions: torch.Tensor,
        affordances: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Encode positions as one-hot grid cells.

        Creates a grid_size × grid_size one-hot encoding where:
        - Affordances are marked with 1.0
        - Agent position adds 1.0 (so agent on affordance = 2.0)
        """
        num_agents = positions.shape[0]
        device = positions.device

        # Initialize grid encoding [num_agents, width * height]
        grid_encoding = torch.zeros(num_agents, self.width * self.height, device=device)

        # Mark affordance positions
        for affordance_pos in affordances.values():
            affordance_flat_idx = affordance_pos[1] * self.width + affordance_pos[0]
            grid_encoding[:, affordance_flat_idx] = 1.0

        # Mark agent positions (add 1.0, so overlaps become 2.0)
        flat_indices = positions[:, 1] * self.width + positions[:, 0]
        ones = torch.ones(num_agents, 1, device=device)
        grid_encoding.scatter_add_(1, flat_indices.unsqueeze(1), ones)

        return grid_encoding

    def get_observation_dim(self) -> int:
        """Grid observation is width × height (flattened)."""
        return self.width * self.height

    def get_valid_neighbors(
        self,
        position: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Get valid 4-connected neighbors (UP, DOWN, LEFT, RIGHT).

        For clamp boundary: only returns in-bounds neighbors
        For wrap/bounce: returns all 4 neighbors (wrapping/bouncing handled in apply_movement)
        """
        x, y = position[0].item(), position[1].item()

        neighbors = [
            torch.tensor([x, y - 1], dtype=torch.long, device=position.device),  # UP
            torch.tensor([x, y + 1], dtype=torch.long, device=position.device),  # DOWN
            torch.tensor([x - 1, y], dtype=torch.long, device=position.device),  # LEFT
            torch.tensor([x + 1, y], dtype=torch.long, device=position.device),  # RIGHT
        ]

        if self.boundary == "clamp":
            # Filter out-of-bounds neighbors
            neighbors = [n for n in neighbors if 0 <= n[0] < self.width and 0 <= n[1] < self.height]

        return neighbors

    def is_on_position(
        self,
        agent_positions: torch.Tensor,
        target_position: torch.Tensor,
    ) -> torch.Tensor:
        """Check if agents are exactly on target position (exact match)."""
        # For discrete grids, agents must be on exact cell
        distances = self.compute_distance(agent_positions, target_position)
        return distances == 0

    def get_all_positions(self) -> list[list[int]]:
        """Return all grid positions."""
        return [[x, y] for x in range(self.width) for y in range(self.height)]

    def encode_partial_observation(
        self,
        positions: torch.Tensor,
        affordances: dict[str, torch.Tensor],
        vision_range: int,
    ) -> torch.Tensor:
        """Encode local window around each agent (POMDP).

        Extracts a local (2*vision_range+1)×(2*vision_range+1) window centered
        on each agent's position. Affordances within the window are marked.

        Args:
            positions: [num_agents, 2] agent positions
            affordances: {name: [2]} affordance positions
            vision_range: radius of vision (e.g., 2 for 5×5 window)

        Returns:
            [num_agents, window_size²] local grid encoding
            where window_size = 2*vision_range + 1

        Note: Handles boundary cases - if agent near edge, out-of-bounds
        cells are marked as empty.
        """
        num_agents = positions.shape[0]
        device = positions.device
        window_size = 2 * vision_range + 1

        # Initialize local grids for all agents
        local_grids = torch.zeros(
            (num_agents, window_size, window_size),
            device=device,
            dtype=torch.float32,
        )

        # For each agent, extract local window
        for agent_idx in range(num_agents):
            agent_x, agent_y = positions[agent_idx]

            # Mark affordances in local window
            for affordance_pos in affordances.values():
                aff_x, aff_y = affordance_pos[0].item(), affordance_pos[1].item()

                # Compute relative position in local window
                rel_x = aff_x - agent_x + vision_range
                rel_y = aff_y - agent_y + vision_range

                # Check if affordance is within vision window
                if 0 <= rel_x < window_size and 0 <= rel_y < window_size:
                    local_grids[agent_idx, rel_y, rel_x] = 1.0

        # Flatten local grids: [num_agents, window_size, window_size] → [num_agents, window_size²]
        return local_grids.reshape(num_agents, -1)
