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

from townlet.config.drive_as_code import DriveAsCodeConfig, ModifierConfig
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

        Uses torch.where for GPU-optimized range evaluation.
        Modifiers return multipliers that can be chained together.

        Returns:
            Dictionary mapping modifier name to evaluation function
        """
        compiled = {}

        for mod_name, mod_config in self.dac_config.modifiers.items():
            # Closure captures mod_config
            def create_modifier_fn(config: ModifierConfig) -> Callable:
                # Pre-compute range boundaries and multipliers as tensors
                ranges = sorted(config.ranges, key=lambda r: r.min)

                def evaluate_modifier(meters: torch.Tensor) -> torch.Tensor:
                    """Evaluate modifier for all agents.

                    Args:
                        meters: [num_agents, meter_count] normalized meter values

                    Returns:
                        [num_agents] multipliers for this modifier
                    """
                    # Get source value
                    if config.bar:
                        # Bar source: Index into meters tensor
                        bar_idx = self._get_bar_index(config.bar)
                        source_value = meters[:, bar_idx]  # [num_agents]
                    elif config.variable:
                        # VFS variable source: Read from registry
                        source_value = self.vfs_registry.get(config.variable, reader=self.vfs_reader)  # [num_agents]
                    else:
                        raise ValueError(f"Modifier has no source: {mod_name}")

                    # Evaluate ranges using torch.where for GPU efficiency
                    # Start with last range as default (fallback)
                    multiplier = torch.full_like(source_value, ranges[-1].multiplier, dtype=torch.float32)

                    # Work backwards through ranges using nested torch.where
                    for r in reversed(ranges[:-1]):
                        condition = (source_value >= r.min) & (source_value < r.max)
                        multiplier = torch.where(condition, r.multiplier, multiplier)

                    return multiplier

                return evaluate_modifier

            compiled[mod_name] = create_modifier_fn(mod_config)

        return compiled

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

    def _get_bar_index(self, bar_id: str) -> int:
        """Get meter index for a bar ID.

        Args:
            bar_id: Bar identifier (e.g., "energy", "health")

        Returns:
            Index into meters tensor

        Raises:
            ValueError: If bar_id not found
        """
        # NOTE: This assumes bars are in the order defined in bars.yaml
        # The compiler should have validated that bar_id exists
        bar_ids = self.dac_config.extrinsic.bars

        # For now, use a simple lookup (will be optimized in later task)
        # In production, we'd get this from universe metadata
        try:
            return bar_ids.index(bar_id)
        except ValueError:
            raise ValueError(f"Bar '{bar_id}' not found in extrinsic bars list")

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
