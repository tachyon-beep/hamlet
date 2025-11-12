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
            Function that computes extrinsic rewards [num_agents]
        """
        strategy = self.dac_config.extrinsic

        if strategy.type == "multiplicative":
            # reward = base * bar1 * bar2 * ...
            base = strategy.base if strategy.base is not None else 1.0
            bar_ids = strategy.bars

            def compute_multiplicative(meters: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
                """Multiplicative: reward = base * product(bars)"""
                # Start with base
                reward = torch.full((self.num_agents,), base, device=self.device, dtype=torch.float32)

                # Multiply by each bar
                for bar_id in bar_ids:
                    bar_idx = self._get_bar_index(bar_id)
                    reward = reward * meters[:, bar_idx]

                # Dead agents get 0.0
                reward = torch.where(dones, torch.zeros_like(reward), reward)

                return reward

            return compute_multiplicative

        elif strategy.type == "constant_base_with_shaped_bonus":
            # reward = base + sum(bar_bonuses) + sum(variable_bonuses)
            base_reward = strategy.base_reward if strategy.base_reward is not None else 1.0
            bar_bonuses = strategy.bar_bonuses if strategy.bar_bonuses is not None else []
            variable_bonuses = strategy.variable_bonuses if strategy.variable_bonuses is not None else []

            def compute_constant_base(meters: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
                """Constant base + shaped bonuses"""
                # Start with base reward
                reward = torch.full((self.num_agents,), base_reward, device=self.device, dtype=torch.float32)

                # Add bar bonuses: bonus = scale * (bar_value - center)
                for bonus_config in bar_bonuses:
                    bar_idx = self._get_bar_index(bonus_config.bar)
                    bar_value = meters[:, bar_idx]
                    bonus = bonus_config.scale * (bar_value - bonus_config.center)
                    reward = reward + bonus

                # Add variable bonuses (from VFS): bonus = weight * var_value
                for bonus_config in variable_bonuses:
                    var_value = self.vfs_registry.get(bonus_config.variable, reader=self.vfs_reader)
                    bonus = bonus_config.weight * var_value
                    reward = reward + bonus

                # Dead agents get 0.0
                reward = torch.where(dones, torch.zeros_like(reward), reward)

                return reward

            return compute_constant_base

        elif strategy.type == "additive_unweighted":
            # reward = base + sum(bars)
            base = strategy.base if strategy.base is not None else 0.0
            bar_ids = strategy.bars

            def compute_additive(meters: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
                """Additive unweighted: reward = base + sum(bars)"""
                reward = torch.full((self.num_agents,), base, device=self.device, dtype=torch.float32)

                # Sum all bars
                for bar_id in bar_ids:
                    bar_idx = self._get_bar_index(bar_id)
                    reward = reward + meters[:, bar_idx]

                # Dead agents get 0.0
                reward = torch.where(dones, torch.zeros_like(reward), reward)

                return reward

            return compute_additive

        elif strategy.type == "weighted_sum":
            # reward = sum(weight_i * bar_i) using bar_bonuses for weights
            # (center is ignored, scale is the weight)
            base = strategy.base if strategy.base is not None else 0.0
            bar_bonuses = strategy.bar_bonuses if strategy.bar_bonuses is not None else []

            def compute_weighted_sum(meters: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
                """Weighted sum: reward = sum(weight_i * bar_i)"""
                reward = torch.full((self.num_agents,), base, device=self.device, dtype=torch.float32)

                # Add weighted bars (using scale as weight, ignoring center)
                for bonus_config in bar_bonuses:
                    bar_idx = self._get_bar_index(bonus_config.bar)
                    bar_value = meters[:, bar_idx]
                    weighted_value = bonus_config.scale * bar_value
                    reward = reward + weighted_value

                # Dead agents get 0.0
                reward = torch.where(dones, torch.zeros_like(reward), reward)

                return reward

            return compute_weighted_sum

        elif strategy.type == "polynomial":
            # reward = sum(weight_i * bar_i^exponent_i)
            # Using bar_bonuses: scale=weight, center=exponent
            base = strategy.base if strategy.base is not None else 0.0
            bar_bonuses = strategy.bar_bonuses if strategy.bar_bonuses is not None else []

            def compute_polynomial(meters: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
                """Polynomial: reward = sum(weight_i * bar_i^exponent_i)"""
                reward = torch.full((self.num_agents,), base, device=self.device, dtype=torch.float32)

                # Add polynomial terms (scale=weight, center=exponent)
                for bonus_config in bar_bonuses:
                    bar_idx = self._get_bar_index(bonus_config.bar)
                    bar_value = meters[:, bar_idx]
                    exponent = bonus_config.center
                    weight = bonus_config.scale
                    term = weight * torch.pow(bar_value, exponent)
                    reward = reward + term

                # Dead agents get 0.0
                reward = torch.where(dones, torch.zeros_like(reward), reward)

                return reward

            return compute_polynomial

        elif strategy.type == "threshold_based":
            # reward = base + sum(bonuses if bar >= threshold)
            base = strategy.base if strategy.base is not None else 0.0
            bar_bonuses = strategy.bar_bonuses if strategy.bar_bonuses is not None else []

            def compute_threshold(meters: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
                """Threshold-based: bonus when bar crosses threshold"""
                reward = torch.full((self.num_agents,), base, device=self.device, dtype=torch.float32)

                # Add threshold bonuses (center=threshold, scale=bonus)
                for bonus_config in bar_bonuses:
                    bar_idx = self._get_bar_index(bonus_config.bar)
                    bar_value = meters[:, bar_idx]
                    threshold = bonus_config.center
                    bonus = bonus_config.scale

                    # Apply bonus where bar >= threshold
                    above_threshold = bar_value >= threshold
                    reward = reward + torch.where(above_threshold, bonus, 0.0)

                # Dead agents get 0.0
                reward = torch.where(dones, torch.zeros_like(reward), reward)

                return reward

            return compute_threshold

        elif strategy.type == "aggregation":
            # reward = base + min(bars)  (simplified - always uses min)
            # Full implementation would have operation field (min/max/mean/product)
            base = strategy.base if strategy.base is not None else 0.0
            bar_ids = strategy.bars

            def compute_aggregation(meters: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
                """Aggregation: reward = base + min(bars)"""
                reward = torch.full((self.num_agents,), base, device=self.device, dtype=torch.float32)

                if bar_ids:
                    # Stack bars for aggregation: [num_agents, num_bars]
                    bar_values = torch.stack([meters[:, self._get_bar_index(bar_id)] for bar_id in bar_ids], dim=1)

                    # Apply min aggregation (could be extended to max/mean/product)
                    aggregated = torch.min(bar_values, dim=1).values
                    reward = reward + aggregated

                # Dead agents get 0.0
                reward = torch.where(dones, torch.zeros_like(reward), reward)

                return reward

            return compute_aggregation

        elif strategy.type == "vfs_variable":
            # reward = sum(weight_i * vfs[variable_i])
            base = strategy.base if strategy.base is not None else 0.0
            variable_bonuses = strategy.variable_bonuses if strategy.variable_bonuses is not None else []

            def compute_vfs_variable(meters: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
                """VFS variable: reward from VFS-computed values (escape hatch)"""
                reward = torch.full((self.num_agents,), base, device=self.device, dtype=torch.float32)

                # Add weighted VFS variables
                for bonus_config in variable_bonuses:
                    var_value = self.vfs_registry.get(bonus_config.variable, reader=self.vfs_reader)
                    weighted_value = bonus_config.weight * var_value
                    reward = reward + weighted_value

                # Dead agents get 0.0
                reward = torch.where(dones, torch.zeros_like(reward), reward)

                return reward

            return compute_vfs_variable

        elif strategy.type == "hybrid":
            # Simplified hybrid: base + weighted bars with optional centering
            # Full implementation would compose multiple sub-strategies
            base = strategy.base if strategy.base is not None else 0.0
            bar_bonuses = strategy.bar_bonuses if strategy.bar_bonuses is not None else []

            def compute_hybrid(meters: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
                """Hybrid: combine multiple approaches (simplified version)"""
                reward = torch.full((self.num_agents,), base, device=self.device, dtype=torch.float32)

                # For each bar bonus, apply weighted calculation
                # If center is 0.0, treat as linear weight
                # If center is non-zero, treat as shaped bonus (value - center)
                for bonus_config in bar_bonuses:
                    bar_idx = self._get_bar_index(bonus_config.bar)
                    bar_value = meters[:, bar_idx]

                    if abs(bonus_config.center) < 1e-6:
                        # Linear weight (center ≈ 0)
                        term = bonus_config.scale * bar_value
                    else:
                        # Shaped bonus (value - center)
                        term = bonus_config.scale * (bar_value - bonus_config.center)

                    reward = reward + term

                # Dead agents get 0.0
                reward = torch.where(dones, torch.zeros_like(reward), reward)

                return reward

            return compute_hybrid

        # Fallback for unimplemented strategies
        def placeholder(meters: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
            return torch.zeros(self.num_agents, device=self.device)

        return placeholder

    def _compile_shaping(self) -> list[Callable]:
        """Compile shaping bonuses into computation functions.

        Returns:
            List of shaping bonus functions
        """
        shaping_fns = []

        for bonus_config in self.dac_config.shaping:
            if bonus_config.type == "approach_reward":
                # Use closure factory to capture config correctly
                def create_approach_reward_fn(config):
                    weight = config.weight
                    target_affordance = config.target_affordance
                    max_distance = config.max_distance

                    def compute_approach_reward(**kwargs) -> torch.Tensor:
                        """Compute approach reward bonus for all agents."""
                        # Extract kwargs
                        agent_positions = kwargs.get("agent_positions")
                        affordance_positions = kwargs.get("affordance_positions", {})

                        # Initialize bonus to zeros
                        bonus = torch.zeros(self.num_agents, device=self.device)

                        # Check if target affordance exists
                        if target_affordance not in affordance_positions:
                            return bonus

                        # Get target position
                        target_pos = affordance_positions[target_affordance]

                        # Calculate distances using Euclidean norm
                        distances = torch.norm(agent_positions - target_pos, dim=1)

                        # Compute bonus: weight * (1.0 - distance / max_distance), clamped to [0, weight]
                        bonus = weight * (1.0 - distances / max_distance)
                        bonus = torch.clamp(bonus, min=0.0, max=weight)

                        return bonus

                    return compute_approach_reward

                shaping_fns.append(create_approach_reward_fn(bonus_config))

            elif bonus_config.type == "completion_bonus":
                # Use closure factory to capture config correctly
                def create_completion_bonus_fn(config):
                    weight = config.weight
                    target_affordance = config.affordance

                    def compute_completion_bonus(**kwargs) -> torch.Tensor:
                        """Compute completion bonus for all agents."""
                        # Extract kwargs
                        last_action_affordance = kwargs.get("last_action_affordance")

                        # Initialize bonus to zeros
                        bonus = torch.zeros(self.num_agents, device=self.device)

                        # Null check for missing kwarg
                        if last_action_affordance is None:
                            return bonus

                        # Vectorize the comparison using list comprehension + tensor creation
                        # Note: Can't fully vectorize string comparison, but keep it minimal
                        matches = torch.tensor(
                            [1.0 if aff == target_affordance else 0.0 for aff in last_action_affordance],
                            device=self.device,
                        )
                        bonus = weight * matches

                        return bonus

                    return compute_completion_bonus

                shaping_fns.append(create_completion_bonus_fn(bonus_config))

            elif bonus_config.type == "efficiency_bonus":
                # Use closure factory to capture config correctly
                def create_efficiency_bonus_fn(config):
                    weight = config.weight
                    bar_id = config.bar
                    threshold = config.threshold

                    def compute_efficiency_bonus(**kwargs) -> torch.Tensor:
                        """Compute efficiency bonus for all agents."""
                        # Extract kwargs
                        meters = kwargs.get("meters")

                        # Null check for missing kwarg
                        if meters is None:
                            return torch.zeros(self.num_agents, device=self.device)

                        # Get bar index
                        # NOTE: This uses the flawed _get_bar_index() from Phase 3B
                        # It will be fixed in Phase 3D when bar_index_map is added
                        bar_idx = self._get_bar_index(bar_id)

                        # Get bar values
                        bar_values = meters[:, bar_idx]

                        # Bonus if bar >= threshold
                        above_threshold = bar_values >= threshold
                        bonus = torch.where(above_threshold, weight, 0.0)

                        return bonus

                    return compute_efficiency_bonus

                shaping_fns.append(create_efficiency_bonus_fn(bonus_config))

            elif bonus_config.type == "state_achievement":
                # Use closure factory to capture config correctly
                def create_state_achievement_fn(config):
                    weight = config.weight
                    conditions = config.conditions

                    def compute_state_achievement(**kwargs) -> torch.Tensor:
                        """Compute state achievement bonus for all agents."""
                        # Extract kwargs
                        meters = kwargs.get("meters")

                        # Null check for missing kwarg
                        if meters is None:
                            return torch.zeros(self.num_agents, device=self.device)

                        # Start with all agents meeting all conditions
                        all_conditions_met = torch.ones(self.num_agents, device=self.device, dtype=torch.bool)

                        # Check each condition
                        for condition in conditions:
                            # Get bar index
                            # NOTE: This uses the flawed _get_bar_index() from Phase 3B
                            # It will be fixed in Phase 3D when bar_index_map is added
                            bar_idx = self._get_bar_index(condition.bar)

                            # Get bar values
                            bar_values = meters[:, bar_idx]

                            # Check if condition met
                            condition_met = bar_values >= condition.min_value

                            # Use logical AND to accumulate conditions
                            all_conditions_met = all_conditions_met & condition_met

                        # Convert boolean mask to bonus
                        bonus = torch.where(all_conditions_met, weight, 0.0)

                        return bonus

                    return compute_state_achievement

                shaping_fns.append(create_state_achievement_fn(bonus_config))

        return shaping_fns

    def _get_bar_index(self, bar_id: str) -> int:
        """Get meter index for a bar ID.

        TODO(CRITICAL - Phase 3D): This implementation has a critical flaw identified
        in code review. It returns index within extrinsic.bars list, NOT index into
        the full meters tensor from universe metadata. This will cause silent data
        corruption when extrinsic config doesn't reference all bars or references
        them in different order than bars.yaml.

        Fix required: Add bar_index_map: dict[str, int] to DACEngine constructor,
        populated from universe metadata in Phase 3D integration.

        Args:
            bar_id: Bar identifier (e.g., "energy", "health")

        Returns:
            Index into meters tensor (CURRENTLY WRONG - returns config index)

        Raises:
            ValueError: If bar_id not found
        """
        # NOTE: This assumes bars are in the order defined in bars.yaml
        # The compiler should have validated that bar_id exists

        # Build bar list from config (either bars field or bar_bonuses)
        bar_ids = self.dac_config.extrinsic.bars
        if not bar_ids and self.dac_config.extrinsic.bar_bonuses:
            # For strategies using bar_bonuses, extract bar IDs
            bar_ids = [bonus.bar for bonus in self.dac_config.extrinsic.bar_bonuses]

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
