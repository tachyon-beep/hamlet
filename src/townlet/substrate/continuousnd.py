"""N-dimensional continuous substrate (N≥4 dimensions)."""

import warnings
from typing import Literal

import torch

from townlet.substrate.base import SpatialSubstrate


class ContinuousNDSubstrate(SpatialSubstrate):
    """N-dimensional continuous space for abstract state spaces.

    ContinuousND supports 4D to 100D continuous substrates with float coordinates.
    For 1D/2D/3D continuous spaces, use Continuous1D/2D/3DSubstrate for better ergonomics.

    Coordinate system:
    - positions: [d0, d1, d2, ..., dN] where d0 is dimension 0, etc.
    - Each dimension has configurable bounds: (min, max)
    - Coordinates are continuous floats, not discrete integers

    Movement:
    - Discrete actions move agent by fixed movement_delta
    - MOVE_D0_NEGATIVE = delta = (-movement_delta, 0, 0, ...)
    - MOVE_D0_POSITIVE = delta = (+movement_delta, 0, 0, ...)

    Interaction:
    - Agent must be within interaction_radius of affordance
    - Uses distance metric (euclidean, manhattan, chebyshev)
    - Proximity-based, not exact position match

    Observation encoding:
    - relative: Normalized coordinates [0, 1] per dimension (N dimensions)
    - scaled: Normalized + range sizes (2N dimensions)
    - absolute: Raw unnormalized coordinates (N dimensions)

    Use cases:
    - High-dimensional continuous control
    - Abstract state space experiments
    - Robotics simulation in high dimensions
    """

    def __init__(
        self,
        bounds: list[tuple[float, float]],
        boundary: Literal["clamp", "wrap", "bounce", "sticky"],
        movement_delta: float,
        interaction_radius: float,
        distance_metric: Literal["euclidean", "manhattan", "chebyshev"] = "euclidean",
        observation_encoding: Literal["relative", "scaled", "absolute"] = "relative",
    ):
        """Initialize N-dimensional continuous substrate.

        Args:
            bounds: List of (min, max) tuples for each dimension
            boundary: Boundary mode ("clamp", "wrap", "bounce", "sticky")
            movement_delta: Distance discrete actions move agent
            interaction_radius: Distance threshold for affordance interaction
            distance_metric: Distance metric ("euclidean", "manhattan", "chebyshev")
            observation_encoding: Position encoding strategy ("relative", "scaled", "absolute")

        Raises:
            ValueError: If dimensions < 4 or bounds invalid

        Warnings:
            UserWarning: If dimensions >= 10 (action space size warning)
        """
        # Validate dimension count
        num_dims = len(bounds)
        if num_dims < 4:
            raise ValueError(
                f"ContinuousND requires at least 4 dimensions, got {num_dims}. "
                f"Use Continuous1DSubstrate (1D), Continuous2DSubstrate (2D), or "
                f"Continuous3DSubstrate (3D) instead."
            )

        if num_dims > 100:
            raise ValueError(f"ContinuousND dimension count ({num_dims}) exceeds limit (100)")

        # Warn at N≥10 (action space grows large)
        if num_dims >= 10:
            warnings.warn(
                f"ContinuousND with {num_dims} dimensions has {2*num_dims+2} actions. "
                f"Large action spaces may be challenging to train. "
                f"Verify this is intentional for your research.",
                UserWarning,
            )

        # Validate bounds
        for i, (min_val, max_val) in enumerate(bounds):
            if min_val >= max_val:
                raise ValueError(f"Bound {i} invalid: min ({min_val}) must be < max ({max_val})")

            # Check space is large enough for interaction
            range_size = max_val - min_val
            if range_size < interaction_radius:
                raise ValueError(
                    f"Dimension {i} range ({range_size}) < interaction_radius ({interaction_radius}). "
                    f"Space too small for affordance interaction."
                )

        # Validate parameters
        if boundary not in ("clamp", "wrap", "bounce", "sticky"):
            raise ValueError(f"Unknown boundary mode: {boundary}")

        if distance_metric not in ("euclidean", "manhattan", "chebyshev"):
            raise ValueError(f"Unknown distance metric: {distance_metric}")

        if observation_encoding not in ("relative", "scaled", "absolute"):
            raise ValueError(f"Unknown observation encoding: {observation_encoding}")

        if movement_delta <= 0:
            raise ValueError(f"movement_delta must be positive, got {movement_delta}")

        if interaction_radius <= 0:
            raise ValueError(f"interaction_radius must be positive, got {interaction_radius}")

        # Warn if interaction_radius < movement_delta
        if interaction_radius < movement_delta:
            warnings.warn(
                f"interaction_radius ({interaction_radius}) < movement_delta ({movement_delta}). "
                f"Agent may step over affordances without interaction. "
                f"This may be intentional for challenge, but verify configuration.",
                UserWarning,
            )

        # Store configuration
        self.bounds = bounds
        self.boundary = boundary
        self.movement_delta = movement_delta
        self.interaction_radius = interaction_radius
        self.distance_metric = distance_metric
        self.observation_encoding = observation_encoding

    @property
    def position_dim(self) -> int:
        """Return number of dimensions."""
        return len(self.bounds)

    @property
    def position_dtype(self) -> torch.dtype:
        """Continuous positions are float32."""
        return torch.float32

    @property
    def action_space_size(self) -> int:
        """Return number of discrete actions: 2N + 2.

        ContinuousND uses the standard spatial substrate action space:
        - 2N movement actions (±movement_delta per dimension)
        - 1 INTERACT action
        - 1 WAIT action (lower energy cost than movement)

        This matches Continuous1D/2D/3D for consistency and environment compatibility.

        Returns:
            2*N + 2 where N = position_dim
        """
        return 2 * self.position_dim + 2

    # Implementation methods
    def initialize_positions(self, num_agents: int, device: torch.device) -> torch.Tensor:
        """Initialize random positions uniformly in continuous bounds.

        Returns:
            [num_agents, N] tensor of float positions
        """
        positions = []
        for min_val, max_val in self.bounds:
            dim_positions = torch.rand(num_agents, device=device, dtype=torch.float32) * (max_val - min_val) + min_val
            positions.append(dim_positions)

        return torch.stack(positions, dim=1)

    def apply_movement(self, positions: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
        """Apply movement deltas with boundary handling.

        Args:
            positions: [num_agents, N] current positions (float32)
            deltas: [num_agents, N] movement deltas (float32)

        Returns:
            [num_agents, N] new positions after boundary handling
        """
        # Scale deltas by movement_delta
        scaled_deltas = deltas.float() * self.movement_delta
        new_positions = positions + scaled_deltas

        # Apply boundary handling per dimension
        for dim_idx, (min_val, max_val) in enumerate(self.bounds):
            if self.boundary == "clamp":
                new_positions[:, dim_idx] = torch.clamp(new_positions[:, dim_idx], min_val, max_val)

            elif self.boundary == "wrap":
                # Toroidal wraparound
                range_size = max_val - min_val
                # Shift to [0, range_size), wrap, shift back
                new_positions[:, dim_idx] = ((new_positions[:, dim_idx] - min_val) % range_size) + min_val

            elif self.boundary == "bounce":
                # Elastic reflection
                range_size = max_val - min_val

                # Normalize to [0, range_size)
                normalized = new_positions[:, dim_idx] - min_val

                # Reflect about boundaries (multiple bounces)
                # Fold into [0, 2*range_size)
                normalized = normalized % (2 * range_size)

                # If in second half, reflect back
                exceed_half = normalized >= range_size
                normalized[exceed_half] = 2 * range_size - normalized[exceed_half]

                # Denormalize back
                new_positions[:, dim_idx] = normalized + min_val

            elif self.boundary == "sticky":
                # Stay in place if out of bounds
                out_of_bounds = (new_positions[:, dim_idx] < min_val) | (new_positions[:, dim_idx] > max_val)
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

        if self.distance_metric == "euclidean":
            # L2 distance: sqrt(sum of squared differences)
            return torch.sqrt(((pos1 - pos2) ** 2).sum(dim=-1))

        elif self.distance_metric == "manhattan":
            # L1 distance: sum of absolute differences
            return torch.abs(pos1 - pos2).sum(dim=-1)

        elif self.distance_metric == "chebyshev":
            # L∞ distance: max of absolute differences
            return torch.abs(pos1 - pos2).max(dim=-1)[0]

    def _encode_relative(self, positions: torch.Tensor, affordances: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode positions as normalized coordinates [0, 1] per dimension."""
        num_agents = positions.shape[0]
        device = positions.device

        normalized = torch.zeros((num_agents, len(self.bounds)), dtype=torch.float32, device=device)

        for dim_idx, (min_val, max_val) in enumerate(self.bounds):
            range_size = max_val - min_val
            normalized[:, dim_idx] = (positions[:, dim_idx] - min_val) / range_size

        return normalized

    def _encode_scaled(self, positions: torch.Tensor, affordances: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode positions as normalized coordinates + range metadata."""
        num_agents = positions.shape[0]
        device = positions.device

        # Get normalized positions
        relative = self._encode_relative(positions, affordances)

        # Add range metadata
        ranges = []
        for min_val, max_val in self.bounds:
            ranges.append(max_val - min_val)

        ranges_tensor = torch.tensor(ranges, dtype=torch.float32, device=device).unsqueeze(0).expand(num_agents, -1)

        return torch.cat([relative, ranges_tensor], dim=1)

    def _encode_absolute(self, positions: torch.Tensor, affordances: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode positions as raw unnormalized coordinates."""
        return positions

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
            - scaled: 2N (normalized + range sizes)
            - absolute: N (raw coordinates)
        """
        if self.observation_encoding == "relative":
            return len(self.bounds)
        elif self.observation_encoding == "scaled":
            return 2 * len(self.bounds)
        elif self.observation_encoding == "absolute":
            return len(self.bounds)
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
        """Raise error - continuous space has no discrete neighbors.

        Args:
            position: Position tensor (not used)

        Raises:
            NotImplementedError: Continuous substrates don't have discrete neighbors
        """
        raise NotImplementedError(
            "ContinuousND has continuous positions. "
            "No discrete neighbors exist. "
            "Use compute_distance() and interaction_radius for proximity detection."
        )

    def is_on_position(self, agent_positions: torch.Tensor, target_position: torch.Tensor) -> torch.Tensor:
        """Check if agents are within interaction radius of target (proximity-based).

        Args:
            agent_positions: [num_agents, N] agent positions
            target_position: [N] target position

        Returns:
            [num_agents] boolean tensor (True if agent within interaction_radius)
        """
        distance = self.compute_distance(agent_positions, target_position)
        return distance <= self.interaction_radius

    def get_all_positions(self) -> list[list[float]]:
        """Raise error - continuous space has infinite positions."""
        raise NotImplementedError(
            "ContinuousND has infinite positions (continuous space). "
            "Use random sampling for affordance placement instead. "
            "See vectorized_env.py randomize_affordance_positions()."
        )

    def supports_enumerable_positions(self) -> bool:
        """Continuous substrates have infinite positions."""
        return False

    def encode_partial_observation(
        self,
        positions: torch.Tensor,
        affordances: dict[str, torch.Tensor],
        vision_range: int,
    ) -> torch.Tensor:
        """Encode local window around agents for partial observability (POMDP).

        For N-dimensional continuous spaces, local windows are impractical:
        - No discrete grid to window into
        - Infinite positions in any local region
        - Exponential growth of window volume with dimensions

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
            f"Partial observability (POMDP) is not supported for {self.position_dim}D continuous spaces. "
            f"Continuous spaces have infinite positions in any local window. "
            f"Use full observability with 'relative' or 'scaled' observation_encoding instead."
        )
