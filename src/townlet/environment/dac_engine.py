"""DAC Engine - Runtime reward computation from declarative specs.

The DACEngine compiles declarative DAC specs into optimized GPU-native
computation graphs for reward calculation.

Design:
- All operations vectorized across agents (batch dimension)
- Modifier evaluation uses torch.where for range lookups
- VFS integration via runtime registry with reader="engine"
- Intrinsic weight modulation for crisis suppression

Formula:
    total_reward = extrinsic + (intrinsic * effective_intrinsic_weight) + shaping

Where:
    effective_intrinsic_weight = base_weight * modifier1 * modifier2 * ...
"""

from collections.abc import Callable

import torch

from townlet.config.drive_as_code import DriveAsCodeConfig
from townlet.vfs.registry import VariableRegistry


class DACEngine:
    """Drive As Code reward computation engine.

    Compiles declarative DAC specs into optimized GPU-native computation graphs.

    Example:
        >>> engine = DACEngine(
        ...     dac_config=dac_config,
        ...     vfs_registry=vfs_registry,
        ...     device=torch.device("cpu"),
        ...     num_agents=4,
        ... )
        >>> total_rewards, intrinsic_weights, components = engine.calculate_rewards(
        ...     step_counts=step_counts,
        ...     dones=dones,
        ...     meters=meters,
        ...     intrinsic_raw=intrinsic_raw,
        ... )
    """

    def __init__(
        self,
        dac_config: DriveAsCodeConfig,
        vfs_registry: VariableRegistry,
        device: torch.device,
        num_agents: int,
    ):
        """Initialize DAC engine.

        Args:
            dac_config: DAC configuration
            vfs_registry: VFS runtime registry for variable access
            device: PyTorch device (cpu or cuda)
            num_agents: Number of agents in population
        """
        self.dac_config = dac_config
        self.vfs_registry = vfs_registry
        self.device = device
        self.num_agents = num_agents
        self.vfs_reader = "engine"  # DAC reads as engine, not agent

        # Compile modifiers into lookup tables
        self.modifiers = self._compile_modifiers()

        # Compile extrinsic strategy
        self.extrinsic_fn = self._compile_extrinsic()

        # Compile shaping bonuses
        self.shaping_fns = self._compile_shaping()

        # Logging
        self.log_components = dac_config.composition.log_components
        self.log_modifiers = dac_config.composition.log_modifiers

    def _compile_modifiers(self) -> dict[str, Callable]:
        """Compile modifiers into efficient lookup functions.

        Returns:
            Dictionary of modifier functions
        """
        # TODO: Implement in next task
        return {}

    def _compile_extrinsic(self) -> Callable:
        """Compile extrinsic strategy into computation function.

        Returns:
            Function that computes extrinsic rewards
        """

        # TODO: Implement in next sub-phase
        def placeholder(meters: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
            return torch.zeros(self.num_agents, device=self.device)

        return placeholder

    def _compile_shaping(self) -> list[Callable]:
        """Compile shaping bonuses into computation functions.

        Returns:
            List of shaping bonus functions
        """
        # TODO: Implement in later sub-phase
        return []

    def calculate_rewards(
        self,
        step_counts: torch.Tensor,
        dones: torch.Tensor,
        meters: torch.Tensor,
        intrinsic_raw: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Calculate total rewards with DAC.

        Args:
            step_counts: [num_agents] current step count
            dones: [num_agents] agent death flags
            meters: [num_agents, meter_count] normalized meter values
            intrinsic_raw: [num_agents] raw intrinsic curiosity values
            **kwargs: Additional context (positions, affordance states, etc.)

        Returns:
            total_rewards: [num_agents] final rewards
            intrinsic_weights: [num_agents] effective intrinsic weights
            components: dict of reward components
        """
        # TODO: Implement full calculation in next tasks
        # For now, return zeros
        total_rewards = torch.zeros(self.num_agents, device=self.device)
        intrinsic_weights = torch.ones(self.num_agents, device=self.device)
        components = {
            "extrinsic": torch.zeros(self.num_agents, device=self.device),
            "intrinsic": torch.zeros(self.num_agents, device=self.device),
            "shaping": torch.zeros(self.num_agents, device=self.device),
        }

        return total_rewards, intrinsic_weights, components
