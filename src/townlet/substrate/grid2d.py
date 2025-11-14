"""2D square grid substrate (replicates current HAMLET behavior)."""

from typing import Literal, cast

import torch

from townlet.environment.action_config import ActionConfig
from townlet.environment.affordance_layout import iter_affordance_positions
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
        observation_encoding: Literal["relative", "scaled", "absolute"] = "relative",
        topology: Literal["square"] = "square",  # NEW: Grid2D is always square topology
    ):
        """Initialize 2D grid substrate.

        Args:
            width: Grid width (number of columns)
            height: Grid height (number of rows)
            boundary: Boundary mode ("clamp", "wrap", "bounce", "sticky")
            distance_metric: Distance metric ("manhattan", "euclidean", "chebyshev")
            observation_encoding: Position encoding strategy ("relative", "scaled", "absolute")
            topology: Grid topology ("square" for 2D Cartesian grid)
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
        self.observation_encoding = observation_encoding
        self.topology = topology  # NEW: Store topology

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
        # Cast deltas to long for grid substrates (deltas come in as float32)
        new_positions = positions + deltas.long()

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

    def get_default_actions(self) -> list[ActionConfig]:
        """Return Grid2D's 6 default actions with default costs.

        Returns:
            [UP, DOWN, LEFT, RIGHT, INTERACT, WAIT] with standard 2D costs.
        """
        return [
            ActionConfig(
                id=0,  # Temporary, reassigned by builder
                name="UP",
                type="movement",
                delta=[0, -1],
                teleport_to=None,
                costs={"energy": 0.005},  # JANK-02: Only energy cost (works with all meter configs)
                effects={},
                description="Move one cell upward (north)",
                icon=None,
                source="substrate",
                source_affordance=None,
                enabled=True,
            ),
            ActionConfig(
                id=1,
                name="DOWN",
                type="movement",
                delta=[0, 1],
                teleport_to=None,
                costs={"energy": 0.005},  # JANK-02: Only energy cost (works with all meter configs)
                effects={},
                description="Move one cell downward (south)",
                icon=None,
                source="substrate",
                source_affordance=None,
                enabled=True,
            ),
            ActionConfig(
                id=2,
                name="LEFT",
                type="movement",
                delta=[-1, 0],
                teleport_to=None,
                costs={"energy": 0.005},  # JANK-02: Only energy cost (works with all meter configs)
                effects={},
                description="Move one cell left (west)",
                icon=None,
                source="substrate",
                source_affordance=None,
                enabled=True,
            ),
            ActionConfig(
                id=3,
                name="RIGHT",
                type="movement",
                delta=[1, 0],
                teleport_to=None,
                costs={"energy": 0.005},  # JANK-02: Only energy cost (works with all meter configs)
                effects={},
                description="Move one cell right (east)",
                icon=None,
                source="substrate",
                source_affordance=None,
                enabled=True,
            ),
            ActionConfig(
                id=4,
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
            ),
            ActionConfig(
                id=5,
                name="WAIT",
                type="passive",
                delta=None,
                teleport_to=None,
                costs={"energy": 0.004},
                effects={},
                description="Wait in place (idle metabolic cost)",
                icon=None,
                source="substrate",
                source_affordance=None,
                enabled=True,
            ),
        ]

    def _encode_relative(
        self,
        positions: torch.Tensor,
        affordances: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Encode positions as normalized coordinates [0, 1].

        Args:
            positions: Agent positions [num_agents, 2]
            affordances: Affordance positions (currently unused)

        Returns:
            [num_agents, 2] normalized positions
        """
        num_agents = positions.shape[0]
        device = positions.device

        normalized = torch.zeros((num_agents, 2), dtype=torch.float32, device=device)
        normalized[:, 0] = positions[:, 0].float() / max(self.width - 1, 1)
        normalized[:, 1] = positions[:, 1].float() / max(self.height - 1, 1)

        return normalized

    def _encode_scaled(
        self,
        positions: torch.Tensor,
        affordances: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Encode positions as normalized coordinates + range metadata.

        Args:
            positions: Agent positions [num_agents, 2]
            affordances: Affordance positions (currently unused)

        Returns:
            [num_agents, 4] normalized positions + range sizes
            First 2 dims: normalized [0, 1]
            Last 2 dims: (width, height)
        """
        num_agents = positions.shape[0]
        device = positions.device

        # Get normalized positions
        relative = self._encode_relative(positions, affordances)

        # Add range metadata
        ranges = (
            torch.tensor(
                [float(self.width), float(self.height)],
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
            positions: Agent positions [num_agents, 2]
            affordances: Affordance positions (currently unused)

        Returns:
            [num_agents, 2] raw coordinates (as float)
        """
        return positions.float()

    def _encode_full_grid(
        self,
        positions: torch.Tensor,
        affordances: dict[str, torch.Tensor] | object,
    ) -> torch.Tensor:
        """Global occupancy grid encoding.

        Builds a flattened ``width × height`` grid for each agent where:
            - 0.0 → empty cell
            - 1.0 → affordance present
            - 2.0 → agent present (or agent + affordance overlap)
        """

        num_agents = positions.shape[0]
        device = positions.device
        grid_area = self.width * self.height

        affordance_grid = torch.zeros(grid_area, dtype=torch.float32, device=device)

        for affordance_pos in iter_affordance_positions(affordances):
            if affordance_pos.numel() < 2:
                continue

            pos = torch.as_tensor(affordance_pos, device=device, dtype=torch.long)
            x = int(pos[0].item())
            y = int(pos[1].item())

            if 0 <= x < self.width and 0 <= y < self.height:
                idx = y * self.width + x
                affordance_grid[idx] = 1.0

        # Broadcast grid to all agents and clone so agent annotations do not alias
        global_grid = affordance_grid.unsqueeze(0).expand(num_agents, -1).clone()

        if num_agents == 0:
            return global_grid

        agent_x = positions[:, 0].long()
        agent_y = positions[:, 1].long()

        in_bounds = (agent_x >= 0) & (agent_x < self.width) & (agent_y >= 0) & (agent_y < self.height)

        if not torch.all(in_bounds):
            invalid = torch.stack([agent_x[~in_bounds], agent_y[~in_bounds]], dim=1).tolist()
            raise ValueError(f"Agent positions out of bounds for Grid2D: {invalid}")

        agent_indices = agent_y * self.width + agent_x
        batch_indices = torch.arange(num_agents, device=device)

        current_values = global_grid[batch_indices, agent_indices]
        global_grid[batch_indices, agent_indices] = torch.clamp(current_values + 1.0, max=2.0)

        return global_grid

    def _encode_position_features(
        self,
        positions: torch.Tensor,
        affordances: dict[str, torch.Tensor] | object,
    ) -> torch.Tensor:
        """Encode agent-centric features based on configured mode."""
        # The affordances parameter accepts object for flexibility with affordance_layout helpers,
        # but the encoding methods expect dict. This is safe because encode_observation validates the type.
        affordances_dict = cast(dict[str, torch.Tensor], affordances)

        if self.observation_encoding == "relative":
            return self._encode_relative(positions, affordances_dict)
        if self.observation_encoding == "scaled":
            return self._encode_scaled(positions, affordances_dict)
        if self.observation_encoding == "absolute":
            return self._encode_absolute(positions, affordances_dict)

        raise ValueError(f"Invalid observation_encoding: {self.observation_encoding}. Must be 'relative', 'scaled', or 'absolute'.")

    def encode_observation(
        self,
        positions: torch.Tensor,
        affordances: dict[str, torch.Tensor] | object,
    ) -> torch.Tensor:
        """Encode agent positions and affordances into observation space."""

        global_grid = self._encode_full_grid(positions, affordances)
        position_features = self._encode_position_features(positions, affordances)

        if position_features.numel() == 0:
            return global_grid

        return torch.cat([global_grid, position_features], dim=1)

    def get_observation_dim(self) -> int:
        """Return dimensionality of position encoding.

        Returns:
            - relative: 2 (normalized x, y)
            - scaled: 4 (normalized x, y, width, height)
            - absolute: 2 (raw x, y)
        """
        grid_dim = self.width * self.height

        if self.observation_encoding == "relative":
            return grid_dim + 2
        if self.observation_encoding == "scaled":
            return grid_dim + 4
        if self.observation_encoding == "absolute":
            return grid_dim + 2

        raise ValueError(f"Invalid observation_encoding: {self.observation_encoding}")

    def normalize_positions(self, positions: torch.Tensor) -> torch.Tensor:
        """Normalize positions to [0, 1] range (always relative encoding).

        Returns normalized coordinates regardless of observation_encoding mode.
        Used by POMDP for position context in recurrent networks.

        Args:
            positions: [num_agents, 2] grid positions

        Returns:
            [num_agents, 2] normalized to [0, 1]
        """
        return self._encode_relative(positions, {})

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

    def get_capacity(self) -> int:
        """Calculate total positions analytically (width × height)."""
        return self.width * self.height

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
            for affordance_pos in iter_affordance_positions(affordances):
                if affordance_pos.numel() < 2:
                    continue

                aff_tensor = torch.as_tensor(affordance_pos, device=device, dtype=torch.long)
                aff_x, aff_y = int(aff_tensor[0].item()), int(aff_tensor[1].item())

                # Compute relative position in local window
                rel_x = aff_x - agent_x + vision_range
                rel_y = aff_y - agent_y + vision_range

                # Check if affordance is within vision window
                if 0 <= rel_x < window_size and 0 <= rel_y < window_size:
                    local_grids[agent_idx, rel_y, rel_x] = 1.0

        # Flatten local grids: [num_agents, window_size, window_size] → [num_agents, window_size²]
        return local_grids.reshape(num_agents, -1)

    def supports_enumerable_positions(self) -> bool:
        """Grid substrates have finite enumerable positions."""
        return True
