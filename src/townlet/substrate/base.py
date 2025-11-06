"""Abstract base class for spatial substrates."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from townlet.environment.action_config import ActionConfig


class SpatialSubstrate(ABC):
    """Abstract interface for spatial substrates.

    A spatial substrate defines:
    - How positions are represented (dimensionality, dtype)
    - How positions are initialized (random, fixed, etc.)
    - How movement is applied (deltas, boundaries)
    - How distance is computed (Manhattan, Euclidean, graph distance)
    - How positions are encoded in observations

    Key insight: The substrate is OPTIONAL. Aspatial universes (pure state
    machines without positioning) are valid and reveal that meters (bars)
    are the true universe.

    Design Principles:
    - Conceptual Agnosticism: Don't assume 2D, Euclidean, or grid-based
    - Permissive Semantics: Allow 3D, hexagonal, continuous, graph, aspatial
    - Structural Enforcement: Validate tensor shapes, boundary behaviors
    """

    @property
    @abstractmethod
    def position_dim(self) -> int:
        """Dimensionality of position vectors.

        Returns:
            0 for aspatial (no positioning)
            2 for 2D grids
            3 for 3D grids
            N for N-dimensional continuous spaces
        """
        pass

    @property
    @abstractmethod
    def position_dtype(self) -> torch.dtype:
        """Data type of position tensors.

        Returns:
            torch.long for discrete grids (integer coordinates)
            torch.float32 for continuous spaces (float coordinates)

        This enables substrates to mix int and float positioning without dtype errors.

        Example:
            Grid2D: torch.long (positions are integers)
            Continuous2D: torch.float32 (positions are floats)
        """
        pass

    @property
    def action_space_size(self) -> int:
        """Return number of discrete actions supported by this substrate.

        Action spaces are determined by substrate dimensionality:
        - Spatial substrates: 2 * position_dim + 2 (±movement per dimension + INTERACT + WAIT)
        - Aspatial substrates: 2 (INTERACT + WAIT)

        Examples:
            Grid2D (position_dim=2): 6 actions (UP/DOWN/LEFT/RIGHT/INTERACT/WAIT)
            Grid3D (position_dim=3): 8 actions (±X/±Y/±Z/INTERACT/WAIT)
            Continuous1D (position_dim=1): 4 actions (±X/INTERACT/WAIT)
            Continuous2D (position_dim=2): 6 actions (±X/±Y/INTERACT/WAIT)
            Continuous3D (position_dim=3): 8 actions (±X/±Y/±Z/INTERACT/WAIT)
            Aspatial (position_dim=0): 2 actions (INTERACT/WAIT)

        This enables dynamic action space sizing for N-dimensional substrates.
        VectorizedHamletEnv queries this property instead of hardcoding action counts.

        Returns:
            Integer count of discrete actions
        """
        if self.position_dim == 0:
            # Aspatial: INTERACT + WAIT actions (no movement)
            return 2
        # Spatial: 2N + 2 (±movement per dimension + INTERACT + WAIT)
        return 2 * self.position_dim + 2

    @abstractmethod
    def get_default_actions(self) -> list["ActionConfig"]:
        """Return substrate's default action space with default costs.

        **CANONICAL ORDERING CONTRACT:**
        All substrates MUST emit actions in this order:
        1. Movement actions (substrate-specific, in any order)
        2. INTERACT (second-to-last position)
        3. WAIT (last position)

        This ordering enables downstream systems (ActionSpaceBuilder, environment)
        to consistently identify meta-actions by position:
        - actions[-2] is always INTERACT
        - actions[-1] is always WAIT
        - actions[:-2] are always movement actions (if any)

        Special case: Aspatial substrates have NO movement actions, only [INTERACT, WAIT].

        Returns:
            List of ActionConfig instances with substrate-provided actions.
            IDs are temporary (will be reassigned by ActionSpaceBuilder).

        Examples:
            Grid2D: [UP, DOWN, LEFT, RIGHT, INTERACT, WAIT] (6 actions)
            Grid3D: [UP, DOWN, LEFT, RIGHT, UP_Z, DOWN_Z, INTERACT, WAIT] (8 actions)
            GridND(7D): [DIM0_NEG, DIM0_POS, ..., DIM6_POS, INTERACT, WAIT] (16 actions)
            Aspatial: [INTERACT, WAIT] (2 actions, no movement)
        """
        pass

    @abstractmethod
    def initialize_positions(self, num_agents: int, device: torch.device) -> torch.Tensor:
        """Initialize random positions for agents.

        Args:
            num_agents: Number of agents to initialize
            device: PyTorch device (cuda/cpu)

        Returns:
            Tensor of shape [num_agents, position_dim]
            For aspatial substrates: [num_agents, 0]
        """
        pass

    @abstractmethod
    def apply_movement(
        self,
        positions: torch.Tensor,
        deltas: torch.Tensor,
    ) -> torch.Tensor:
        """Apply movement deltas to positions, respecting boundaries.

        Args:
            positions: [num_agents, position_dim] current positions
            deltas: [num_agents, position_dim] movement deltas

        Returns:
            [num_agents, position_dim] new positions after movement

        Boundary handling (clamp, wrap, bounce) is substrate-specific.
        """
        pass

    @abstractmethod
    def compute_distance(
        self,
        pos1: torch.Tensor,
        pos2: torch.Tensor,
    ) -> torch.Tensor:
        """Compute distance between positions.

        Args:
            pos1: [num_agents, position_dim] or [position_dim]
            pos2: [num_agents, position_dim] or [position_dim]

        Returns:
            [num_agents] tensor of distances

        Distance metric is substrate-specific:
        - Grid: Manhattan, Euclidean, or Chebyshev
        - Graph: Shortest path distance
        - Aspatial: Zero (no meaningful distance)
        """
        pass

    @abstractmethod
    def encode_observation(
        self,
        positions: torch.Tensor,
        affordances: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Encode positions and affordances into observation space.

        Args:
            positions: [num_agents, position_dim] agent positions
            affordances: {name: [position_dim]} affordance positions

        Returns:
            [num_agents, observation_dim] grid + position features

        observation_dim is substrate-specific (grid + position):
        - Grid2D (8×8, relative): 66 (64 grid cells + 2 normalized position)
        - Grid3D (8×8×3, relative): 195 (192 grid cells + 3 normalized position)
        - Aspatial: 0 (no position encoding)
        """
        pass

    @abstractmethod
    def get_observation_dim(self) -> int:
        """Return the dimensionality of grid + position encoding in observations.

        Returns:
            Number of features in observation (grid + position):
            - Grid2D (8×8, relative): 66 (64 grid + 2 position)
            - Grid3D (8×8×3, relative): 195 (192 grid + 3 position)
            - Aspatial: 0
        """
        pass

    @abstractmethod
    def normalize_positions(self, positions: torch.Tensor) -> torch.Tensor:
        """Normalize positions to [0, 1] range (always relative encoding).

        This method ALWAYS returns relative encoding (normalized to [0,1]),
        regardless of the substrate's observation_encoding mode.

        Used by POMDP for position context in recurrent networks, which
        requires normalized positions regardless of how full observations
        are encoded.

        Args:
            positions: [num_agents, position_dim] agent positions

        Returns:
            [num_agents, position_dim] normalized to [0, 1]
            For aspatial substrates: [num_agents, 0] (empty)

        Example:
            Grid2D (8×8): position [3, 4] → [3/7, 4/7] ≈ [0.43, 0.57]
            Continuous2D: position [5.5, 3.2] on [0,10] → [0.55, 0.32]
            Aspatial: any position → [] (empty)
        """
        pass

    @abstractmethod
    def get_valid_neighbors(
        self,
        position: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Get valid neighbor positions for action validation.

        Args:
            position: [position_dim] single position

        Returns:
            List of [position_dim] neighbor positions

        Used for action masking (boundary checks).
        """
        pass

    @abstractmethod
    def is_on_position(
        self,
        agent_positions: torch.Tensor,
        target_position: torch.Tensor,
    ) -> torch.Tensor:
        """Check if agents are on the target position (for interactions).

        Args:
            agent_positions: [num_agents, position_dim]
            target_position: [position_dim]

        Returns:
            [num_agents] bool tensor (True if on target)

        For discrete grids: exact match
        For continuous spaces: proximity threshold
        For aspatial: always True (no positioning concept)
        """
        pass

    @abstractmethod
    def get_all_positions(self) -> list[list[int]] | list[list[float]]:
        """Return all valid positions in the substrate.

        Returns:
            List of positions, where each position is [x, y, ...] (position_dim elements).
            For aspatial substrates, returns empty list.
            For discrete grids (3×3), returns [[0,0], [0,1], [0,2], [1,0], ...] (9 positions) as ints.
            For continuous spaces, raises NotImplementedError (infinite positions).

        Used for affordance randomization to ensure valid placement.
        """
        pass

    @abstractmethod
    def encode_partial_observation(
        self,
        positions: torch.Tensor,
        affordances: dict[str, torch.Tensor],
        vision_range: int,
    ) -> torch.Tensor:
        """Encode local window around agents for partial observability (POMDP).

        Args:
            positions: [num_agents, position_dim] agent positions
            affordances: {name: [position_dim]} affordance positions
            vision_range: radius of vision window (e.g., 2 for 5×5 window)

        Returns:
            [num_agents, window_size] local grid encoding

            window_size depends on substrate:
            - Grid2D: (2*vision_range + 1)²  (e.g., 5×5 = 25)
            - Aspatial: 0 (no position encoding)

        Used for:
        - Level 2 POMDP observations (5×5 local window)
        - Partial observability training

        Example:
            Grid2D with vision_range=2:
            - Agent at (4, 4) sees cells (2,2) to (6,6)
            - Encodes 5×5 = 25 cells relative to agent
        """
        pass
