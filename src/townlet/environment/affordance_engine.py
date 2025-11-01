"""
AffordanceEngine: Config-driven affordance interaction system.

This module processes affordance interactions using YAML configuration instead
of hardcoded logic. Follows the same pattern as CascadeEngine (ACTION #1).

Architecture:
- Load affordance configs at initialization
- Pre-build lookup maps and tensors for GPU performance
- Apply instant interactions (single-step effects)
- Apply multi-tick interactions (progressive effects)
- Handle operating hours and affordability checks

Teaching Value:
- Students can modify affordances by editing YAML, not Python
- Different affordance sets create different strategic environments
- Demonstrates data-driven game design

Status: Ready for integration with vectorized_env.py
"""

from pathlib import Path

import torch

from townlet.environment.affordance_config import (
    METER_NAME_TO_IDX,
    AffordanceConfigCollection,
    load_affordance_config,
)


class AffordanceEngine:
    """
    Config-driven affordance interaction processor.

    Handles instant and multi-tick affordance interactions based on YAML configuration.
    All operations are vectorized for GPU performance.
    """

    def __init__(
        self,
        affordance_config: AffordanceConfigCollection,
        num_agents: int,
        device: torch.device,
    ):
        """
        Initialize AffordanceEngine.

        Args:
            affordance_config: Loaded affordance configuration
            num_agents: Number of agents in parallel
            device: torch.device for GPU/CPU
        """
        self.num_agents = num_agents
        self.device = device
        self.affordances = affordance_config.affordances

        # Build lookup maps
        self._build_lookup_maps()

        # Pre-compute tensors for common operations (future optimization)
        # For now, we compute on-the-fly for clarity

    def _build_lookup_maps(self) -> None:
        """
        Build efficient lookup maps for affordances.

        The affordance order is determined BY THE CONFIG FILE, not hardcoded.
        This makes the config the single source of truth.
        """
        # Map affordance name to index (order from config file)
        # This is now dynamically built from the config - no hardcoding!
        self.affordance_name_to_idx = {aff.name: idx for idx, aff in enumerate(self.affordances)}

        # Map affordance ID to config object
        self.affordance_map_by_id = {aff.id: aff for aff in self.affordances}

        # Map affordance NAME to config object (for apply_instant_interaction calls)
        self.affordance_map = {aff.name: aff for aff in self.affordances}

    def get_affordance(self, affordance_id: str):
        """Get affordance config by ID."""
        return self.affordance_map_by_id.get(affordance_id)

    def is_affordance_open(self, affordance_name: str, time_of_day: int) -> bool:
        """
        Check if affordance is open at given time.

        Args:
            affordance_name: Name of affordance (e.g., "Job", "Bar")
            time_of_day: Current hour [0-23]

        Returns:
            True if open, False if closed
        """
        affordance = self.affordance_map.get(affordance_name)
        if affordance is None:
            return False

        open_hour, close_hour = affordance.operating_hours

        # Handle midnight wraparound (e.g., Bar: [18, 28] = 6pm-4am)
        if close_hour > 24:
            close_hour_adjusted = close_hour % 24
            # Open if: time >= open_hour OR time < close_hour_adjusted
            return time_of_day >= open_hour or time_of_day < close_hour_adjusted
        else:
            # Normal hours: time >= open_hour AND time < close_hour
            return open_hour <= time_of_day < close_hour

    def apply_instant_interaction(
        self,
        meters: torch.Tensor,
        affordance_name: str,
        agent_mask: torch.Tensor,
        check_affordability: bool = False,
    ) -> torch.Tensor:
        """
        Apply instant affordance interaction.

        Args:
            meters: [num_agents, 8] current meter values
            affordance_name: Name of affordance (e.g., "Shower")
            agent_mask: [num_agents] bool mask of agents to apply to
            check_affordability: If True, check if agents can afford costs

        Returns:
            updated_meters: [num_agents, 8] after effects applied
        """
        affordance = self.affordance_map.get(affordance_name)
        if affordance is None:
            return meters

        if affordance.interaction_type not in ["instant", "dual"]:
            raise ValueError(
                f"Affordance '{affordance_name}' is {affordance.interaction_type}, "
                f"not instant or dual. Use apply_multi_tick_interaction instead."
            )

        # Clone meters to avoid modifying input
        updated_meters = meters.clone()

        # Check affordability if requested
        if check_affordability and len(affordance.costs) > 0:
            can_afford = self._check_affordability(meters, affordance.costs)
            agent_mask = agent_mask & can_afford

        # Apply costs
        for cost in affordance.costs:
            meter_idx = METER_NAME_TO_IDX[cost.meter]
            updated_meters[agent_mask, meter_idx] -= cost.amount

        # Apply effects
        for effect in affordance.effects:
            meter_idx = METER_NAME_TO_IDX[effect.meter]
            updated_meters[agent_mask, meter_idx] += effect.amount

        # Clamp meters to [0, 1]
        updated_meters = torch.clamp(updated_meters, 0.0, 1.0)

        return updated_meters

    def apply_multi_tick_interaction(
        self,
        meters: torch.Tensor,
        affordance_name: str,
        current_tick: int,
        agent_mask: torch.Tensor,
        check_affordability: bool = False,
    ) -> torch.Tensor:
        """
        Apply multi-tick affordance interaction for a single tick.

        Args:
            meters: [num_agents, 8] current meter values
            affordance_name: Name of affordance (e.g., "Bed", "Job")
            current_tick: Current tick number [0, required_ticks-1]
            agent_mask: [num_agents] bool mask of agents to apply to
            check_affordability: If True, check if agents can afford costs

        Returns:
            updated_meters: [num_agents, 8] after per-tick effects applied
        """
        affordance = self.affordance_map.get(affordance_name)
        if affordance is None:
            return meters

        if affordance.interaction_type not in ["multi_tick", "dual"]:
            raise ValueError(
                f"Affordance '{affordance_name}' is {affordance.interaction_type}, "
                f"not multi_tick or dual. Use apply_instant_interaction instead."
            )

        # Clone meters
        updated_meters = meters.clone()

        # Check affordability if requested
        if check_affordability and len(affordance.costs_per_tick) > 0:
            can_afford = self._check_affordability(meters, affordance.costs_per_tick)
            agent_mask = agent_mask & can_afford

        # Apply per-tick costs
        for cost in affordance.costs_per_tick:
            meter_idx = METER_NAME_TO_IDX[cost.meter]
            updated_meters[agent_mask, meter_idx] -= cost.amount

        # Apply per-tick effects
        for effect in affordance.effects_per_tick:
            meter_idx = METER_NAME_TO_IDX[effect.meter]
            updated_meters[agent_mask, meter_idx] += effect.amount

        # Check if this is the final tick - if so, apply completion bonus
        is_final_tick = current_tick == (affordance.required_ticks - 1)
        if is_final_tick and len(affordance.completion_bonus) > 0:
            for effect in affordance.completion_bonus:
                meter_idx = METER_NAME_TO_IDX[effect.meter]
                updated_meters[agent_mask, meter_idx] += effect.amount

        # Clamp meters to [0, 1]
        updated_meters = torch.clamp(updated_meters, 0.0, 1.0)

        return updated_meters

    def _check_affordability(self, meters: torch.Tensor, costs: list) -> torch.Tensor:
        """
        Check if agents can afford the costs.

        Args:
            meters: [batch_size, 8] current meter values
            costs: List of AffordanceCost objects

        Returns:
            can_afford: [batch_size] bool tensor
        """
        batch_size = meters.shape[0]
        can_afford = torch.ones(batch_size, dtype=torch.bool, device=self.device)

        for cost in costs:
            meter_idx = METER_NAME_TO_IDX[cost.meter]
            can_afford = can_afford & (meters[:, meter_idx] >= cost.amount)

        return can_afford

    def get_action_masks(
        self,
        meters: torch.Tensor,
        time_of_day: int,
        check_affordability: bool = True,
        check_hours: bool = True,
    ) -> torch.Tensor:
        """
        Get action masks for all agents considering affordability and operating hours.

        Args:
            meters: [batch_size, 8] current meter values
            time_of_day: Current hour [0-23]
            check_affordability: If True, mask unaffordable actions
            check_hours: If True, mask closed affordances

        Returns:
            action_masks: [batch_size, num_actions] bool tensor
                         Actions include: 4 movement + 15 affordances = 19 total
        """
        batch_size = meters.shape[0]
        num_movement_actions = 4  # UP, DOWN, LEFT, RIGHT
        num_affordances = 15
        num_actions = num_movement_actions + num_affordances

        # Start with all actions available
        action_masks = torch.ones((batch_size, num_actions), dtype=torch.bool, device=self.device)

        # Movement actions always available
        # (boundary checks happen separately in environment)

        # Check each affordance
        for affordance_name, affordance_idx in self.affordance_name_to_idx.items():
            affordance = self.affordance_map.get(affordance_name)
            if affordance is None:
                continue

            action_idx = num_movement_actions + affordance_idx

            # Check operating hours
            if check_hours:
                is_open = self.is_affordance_open(affordance_name, time_of_day)
                if not is_open:
                    action_masks[:, action_idx] = False
                    continue

            # Check affordability
            if check_affordability:
                # Check instant costs
                if len(affordance.costs) > 0:
                    can_afford = self._check_affordability(meters, affordance.costs)
                    action_masks[:, action_idx] = action_masks[:, action_idx] & can_afford

                # Check per-tick costs (for multi-tick affordances)
                elif len(affordance.costs_per_tick) > 0:
                    can_afford = self._check_affordability(meters, affordance.costs_per_tick)
                    action_masks[:, action_idx] = action_masks[:, action_idx] & can_afford

        return action_masks

    def get_affordance_action_map(self) -> dict[str, int]:
        """
        Get the mapping of affordance names to action indices.

        The environment should use this to build its action space,
        ensuring it's always in sync with the config file.

        Returns:
            Dict mapping affordance name to action index
            Example: {"Bed": 0, "Shower": 1, ...}
        """
        return self.affordance_name_to_idx.copy()

    def get_num_affordances(self) -> int:
        """Get the number of affordances defined in config."""
        return len(self.affordances)

    def get_affordance_cost(self, affordance_name: str, cost_mode: str = "instant") -> float:
        """
        Get the monetary cost for an affordance interaction.

        Args:
            affordance_name: Name of affordance
            cost_mode: "instant" or "per_tick"

        Returns:
            Normalized cost [0, 1] where 1.0 = $100
        """
        affordance = self.affordance_map.get(affordance_name)
        if affordance is None:
            return 0.0

        # Get costs list based on mode
        costs = affordance.costs if cost_mode == "instant" else affordance.costs_per_tick

        # Find money cost (most affordances only have money cost)
        for cost in costs:
            if cost.meter == "money":
                return cost.amount

        return 0.0

    def get_required_ticks(self, affordance_name: str) -> int:
        """
        Get the required number of ticks for a multi-tick affordance.

        Args:
            affordance_name: Name of affordance

        Returns:
            Number of required ticks (1 for instant affordances)
        """
        affordance = self.affordance_map.get(affordance_name)
        if affordance is None or affordance.required_ticks is None:
            return 1
        return affordance.required_ticks

    def apply_interaction(
        self,
        meters: torch.Tensor,
        affordance_name: str,
        agent_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply affordance effects to agent meters.

        This method applies the effects and costs defined in the config
        for a given affordance to the specified agents.

        Args:
            meters: [num_agents, 8] meter values
            affordance_name: Name of the affordance being interacted with
            agent_mask: [num_agents] bool mask indicating which agents interact

        Returns:
            Updated meters tensor [num_agents, 8]

        Raises:
            ValueError: If affordance_name is not recognized
        """
        # Validate affordance exists
        if affordance_name not in self.affordance_name_to_idx:
            raise ValueError(f"Unknown affordance: {affordance_name}")

        # Get affordance config
        affordance = self.affordances[self.affordance_name_to_idx[affordance_name]]

        # Clone meters to avoid in-place modification
        result_meters = meters.clone()

        # Apply effects (all affordances in corrected config are instant)
        for effect in affordance.effects:
            meter_idx = METER_NAME_TO_IDX[effect.meter]
            result_meters[agent_mask, meter_idx] = torch.clamp(
                result_meters[agent_mask, meter_idx] + effect.amount,
                0.0,
                1.0,
            )

        # Apply costs
        for cost in affordance.costs:
            meter_idx = METER_NAME_TO_IDX[cost.meter]
            result_meters[agent_mask, meter_idx] -= cost.amount

        return result_meters


def create_affordance_engine(
    config_path: Path | None = None,
    num_agents: int = 1,
    device: torch.device = torch.device("cpu"),
) -> AffordanceEngine:
    """
    Convenience function to create AffordanceEngine from config file.

    Args:
        config_path: Path to affordances.yaml (default: configs/affordances.yaml)
        num_agents: Number of agents
        device: torch device

    Returns:
        Initialized AffordanceEngine
    """
    if config_path is None:
        config_path = Path("configs/affordances.yaml")

    affordance_config = load_affordance_config(config_path)
    return AffordanceEngine(affordance_config, num_agents, device)
