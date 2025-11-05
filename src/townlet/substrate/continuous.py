"""Continuous space substrates with float-based positioning."""

from typing import Literal

import torch

from .base import SpatialSubstrate


class ContinuousSubstrate(SpatialSubstrate):
    """Base class for continuous space substrates.

    Position representation: float coordinates in bounded space
    - 1D: [x] where x ∈ [min_x, max_x]
    - 2D: [x, y] where x ∈ [min_x, max_x], y ∈ [min_y, max_y]
    - 3D: [x, y, z] where x ∈ [min_x, max_x], y ∈ [min_y, max_y], z ∈ [min_z, max_z]

    Movement: Discrete actions move agent by fixed `movement_delta`
    - MOVE_X_NEGATIVE = delta = (-movement_delta, 0, 0)
    - MOVE_X_POSITIVE = delta = (+movement_delta, 0, 0)
    - etc.

    Interaction: Agent must be within `interaction_radius` of affordance
    - Uses distance metric (euclidean or manhattan)
    - Proximity-based, not exact position match

    Observation encoding: Normalized coordinates [0, 1] per dimension
    - Same as Grid3D (consistent representation)
    - Constant size regardless of bounds
    """

    position_dtype = torch.float32

    def __init__(
        self,
        dimensions: int,
        bounds: list[tuple[float, float]],
        boundary: Literal["clamp", "wrap", "bounce", "sticky"],
        movement_delta: float,
        interaction_radius: float,
        distance_metric: Literal["euclidean", "manhattan"] = "euclidean",
    ):
        """Initialize continuous substrate.

        Args:
            dimensions: Number of dimensions (1, 2, or 3)
            bounds: List of (min, max) tuples for each dimension
            boundary: Boundary handling mode
            movement_delta: Distance discrete actions move agent
            interaction_radius: Distance threshold for affordance interaction
            distance_metric: Distance calculation method
        """
        if dimensions not in (1, 2, 3):
            raise ValueError(f"Continuous substrates support 1-3 dimensions, got {dimensions}")

        if len(bounds) != dimensions:
            raise ValueError(
                f"Number of bounds ({len(bounds)}) must match dimensions ({dimensions}). Example for 2D: bounds=[(0.0, 10.0), (0.0, 10.0)]"
            )

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

        if boundary not in ("clamp", "wrap", "bounce", "sticky"):
            raise ValueError(f"Unknown boundary mode: {boundary}")

        if distance_metric not in ("euclidean", "manhattan"):
            raise ValueError(f"Unknown distance metric: {distance_metric}")

        if movement_delta <= 0:
            raise ValueError(f"movement_delta must be positive, got {movement_delta}")

        if interaction_radius <= 0:
            raise ValueError(f"interaction_radius must be positive, got {interaction_radius}")

        # Warn if interaction_radius < movement_delta
        if interaction_radius < movement_delta:
            import warnings

            warnings.warn(
                f"interaction_radius ({interaction_radius}) < movement_delta ({movement_delta}). "
                f"Agent may step over affordances without interaction. "
                f"This may be intentional for challenge, but verify configuration.",
                UserWarning,
            )

        self.dimensions = dimensions
        self.bounds = bounds
        self.boundary = boundary
        self.movement_delta = movement_delta
        self.interaction_radius = interaction_radius
        self.distance_metric = distance_metric

    @property
    def position_dim(self) -> int:
        """Number of dimensions."""
        return self.dimensions

    @property
    def coordinate_semantics(self) -> dict:
        """Describe what each dimension represents."""
        names = {1: {"X": "position"}, 2: {"X": "horizontal", "Y": "vertical"}, 3: {"X": "horizontal", "Y": "vertical", "Z": "depth"}}
        return names.get(self.dimensions, {})

    def initialize_positions(self, num_agents: int, device: torch.device) -> torch.Tensor:
        """Randomly initialize positions in continuous space."""
        positions = []
        for min_val, max_val in self.bounds:
            dim_positions = torch.rand(num_agents, device=device) * (max_val - min_val) + min_val
            positions.append(dim_positions)

        return torch.stack(positions, dim=1)

    def apply_movement(self, positions: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
        """Apply continuous movement with boundary handling."""
        # Scale deltas by movement_delta
        scaled_deltas = deltas.float() * self.movement_delta
        new_positions = positions + scaled_deltas

        for dim in range(self.dimensions):
            min_val, max_val = self.bounds[dim]

            if self.boundary == "clamp":
                new_positions[:, dim] = torch.clamp(new_positions[:, dim], min_val, max_val)

            elif self.boundary == "wrap":
                # Toroidal wraparound
                range_size = max_val - min_val
                # Shift to [0, range_size), wrap, shift back
                new_positions[:, dim] = ((new_positions[:, dim] - min_val) % range_size) + min_val

            elif self.boundary == "bounce":
                # Elastic reflection
                range_size = max_val - min_val

                # Normalize to [0, range_size)
                normalized = new_positions[:, dim] - min_val

                # Reflect about boundaries (multiple bounces)
                # Fold into [0, 2*range_size)
                normalized = normalized % (2 * range_size)

                # If in second half, reflect back
                exceed_half = normalized >= range_size
                normalized[exceed_half] = 2 * range_size - normalized[exceed_half]

                # Denormalize back
                new_positions[:, dim] = normalized + min_val

            elif self.boundary == "sticky":
                # Stay in place if out of bounds
                out_of_bounds = (new_positions[:, dim] < min_val) | (new_positions[:, dim] > max_val)
                new_positions[out_of_bounds, dim] = positions[out_of_bounds, dim]

        return new_positions

    def compute_distance(self, pos1: torch.Tensor, pos2: torch.Tensor) -> torch.Tensor:
        """Compute distance between positions in continuous space."""
        if self.distance_metric == "euclidean":
            return torch.sqrt(((pos1 - pos2) ** 2).sum(dim=-1))
        elif self.distance_metric == "manhattan":
            return torch.abs(pos1 - pos2).sum(dim=-1)

    def is_on_position(self, positions: torch.Tensor, target_position: torch.Tensor) -> torch.Tensor:
        """Check if agents are within interaction radius of target.

        For continuous space, this is proximity-based (not exact match).
        """
        distance = self.compute_distance(positions, target_position)
        return distance <= self.interaction_radius

    def encode_observation(self, positions: torch.Tensor, affordances: dict[str, torch.Tensor]) -> torch.Tensor:
        """Normalize positions to [0, 1] for each dimension.

        Args:
            positions: Agent positions [num_agents, dimensions]
            affordances: Dict of affordance positions (not used for Continuous encoding)

        Returns [num_agents, dimensions] tensor.
        Matches Grid3D encoding strategy (constant size).
        """
        num_agents = positions.shape[0]
        normalized = torch.zeros((num_agents, self.dimensions), dtype=torch.float32, device=positions.device)

        for dim in range(self.dimensions):
            min_val, max_val = self.bounds[dim]
            range_size = max_val - min_val
            normalized[:, dim] = (positions[:, dim] - min_val) / range_size

        return normalized

    def get_observation_dim(self) -> int:
        """Return observation dimensionality (number of dimensions)."""
        return self.dimensions

    def encode_partial_observation(self, positions: torch.Tensor, affordances: dict[str, torch.Tensor], vision_range: int) -> torch.Tensor:
        """Encode partial observation for POMDP support.

        For continuous space, returns normalized positions (no local window concept).
        POMDP support for continuous substrates is limited.
        """
        # For continuous space, no discrete grid to window into
        # Just return normalized positions (same as full observability)
        return self.encode_observation(positions, affordances)

    def get_all_positions(self) -> list[list[float]]:
        """Raise error - continuous space has infinite positions."""
        raise NotImplementedError(
            f"{self.__class__.__name__} has infinite positions (continuous space). "
            f"Use random sampling for affordance placement instead. "
            f"See vectorized_env.py randomize_affordance_positions()."
        )

    def get_valid_neighbors(self, position: torch.Tensor) -> list[torch.Tensor]:
        """Raise error - continuous space has no discrete neighbors.

        Args:
            position: Position tensor (not used)

        Raises:
            NotImplementedError: Continuous substrates don't have discrete neighbors
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} has continuous positions. "
            f"No discrete neighbors exist. "
            f"Use compute_distance() and interaction_radius for proximity detection."
        )

    def supports_enumerable_positions(self) -> bool:
        """Continuous substrates have infinite positions."""
        return False


class Continuous1DSubstrate(ContinuousSubstrate):
    """1D continuous line."""

    def __init__(
        self,
        min_x: float,
        max_x: float,
        boundary: Literal["clamp", "wrap", "bounce", "sticky"],
        movement_delta: float,
        interaction_radius: float,
        distance_metric: Literal["euclidean", "manhattan"] = "euclidean",
    ):
        super().__init__(
            dimensions=1,
            bounds=[(min_x, max_x)],
            boundary=boundary,
            movement_delta=movement_delta,
            interaction_radius=interaction_radius,
            distance_metric=distance_metric,
        )
        self.min_x = min_x
        self.max_x = max_x


class Continuous2DSubstrate(ContinuousSubstrate):
    """2D continuous plane."""

    def __init__(
        self,
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
        boundary: Literal["clamp", "wrap", "bounce", "sticky"],
        movement_delta: float,
        interaction_radius: float,
        distance_metric: Literal["euclidean", "manhattan"] = "euclidean",
    ):
        super().__init__(
            dimensions=2,
            bounds=[(min_x, max_x), (min_y, max_y)],
            boundary=boundary,
            movement_delta=movement_delta,
            interaction_radius=interaction_radius,
            distance_metric=distance_metric,
        )
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y


class Continuous3DSubstrate(ContinuousSubstrate):
    """3D continuous space."""

    def __init__(
        self,
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
        min_z: float,
        max_z: float,
        boundary: Literal["clamp", "wrap", "bounce", "sticky"],
        movement_delta: float,
        interaction_radius: float,
        distance_metric: Literal["euclidean", "manhattan"] = "euclidean",
    ):
        super().__init__(
            dimensions=3,
            bounds=[(min_x, max_x), (min_y, max_y), (min_z, max_z)],
            boundary=boundary,
            movement_delta=movement_delta,
            interaction_radius=interaction_radius,
            distance_metric=distance_metric,
        )
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        self.min_z = min_z
        self.max_z = max_z
