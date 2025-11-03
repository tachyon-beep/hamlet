"""
Vectorized Hamlet environment for GPU-native training.

Batches multiple independent Hamlet environments into a single vectorized
environment with tensor operations [num_agents, ...].
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from townlet.environment.affordance_config import load_affordance_config
from townlet.environment.affordance_engine import AffordanceEngine
from townlet.environment.meter_dynamics import MeterDynamics
from townlet.environment.observation_builder import ObservationBuilder
from townlet.environment.reward_strategy import RewardStrategy

if TYPE_CHECKING:
    from townlet.population.runtime_registry import AgentRuntimeRegistry


class VectorizedHamletEnv:
    """
    GPU-native vectorized Hamlet environment.

    Batches multiple independent environments for parallel execution.
    All state is stored as PyTorch tensors on specified device.
    """

    def __init__(
        self,
        num_agents: int,
        grid_size: int = 8,
        device: torch.device = torch.device("cpu"),
        partial_observability: bool = False,
        vision_range: int = 2,
        enable_temporal_mechanics: bool = False,
        enabled_affordances: list[str] | None = None,
        move_energy_cost: float = 0.005,
        wait_energy_cost: float = 0.001,
        interact_energy_cost: float = 0.0,
        agent_lifespan: int = 1000,
        config_pack_path: Path | None = None,
    ):
        """
        Initialize vectorized environment.

        Args:
            num_agents: Number of parallel agents
            grid_size: Grid dimension (grid_size × grid_size)
            device: PyTorch device (cpu or cuda)
            partial_observability: If True, agent sees only local window (POMDP)
            vision_range: Radius of vision window (2 = 5×5 window)
            enable_temporal_mechanics: Enable time-based mechanics and multi-tick interactions
            enabled_affordances: List of affordance names to enable (None = all affordances)
            move_energy_cost: Energy cost per movement action (default 0.005 = 0.5%)
            wait_energy_cost: Energy cost per WAIT action (default 0.001 = 0.1%)
            interact_energy_cost: Energy cost per INTERACT action (default 0.0 = free)
            agent_lifespan: Maximum lifetime in steps (default 1000) - provides retirement incentive
        """
        project_root = Path(__file__).parent.parent.parent.parent
        default_pack = project_root / "configs" / "test"

        self.config_pack_path = Path(config_pack_path) if config_pack_path else default_pack
        if not self.config_pack_path.exists():
            raise FileNotFoundError(f"Config pack directory not found: {self.config_pack_path}")

        self.num_agents = num_agents
        self.grid_size = grid_size
        self.device = device
        self.partial_observability = partial_observability
        self.vision_range = vision_range
        self.enable_temporal_mechanics = enable_temporal_mechanics
        self.agent_lifespan = agent_lifespan

        # Configurable energy costs
        self.move_energy_cost = move_energy_cost
        self.wait_energy_cost = wait_energy_cost
        self.interact_energy_cost = interact_energy_cost

        if self.wait_energy_cost >= self.move_energy_cost:
            raise ValueError("wait_energy_cost must be less than move_energy_cost to preserve WAIT as a low-cost recovery action")

        # Define which affordances exist (positions assigned by randomize_affordance_positions())
        all_affordance_names = [
            # Basic survival (tiered)
            "Bed",  # Energy restoration tier 1
            "LuxuryBed",  # Energy restoration tier 2
            "Shower",
            "HomeMeal",
            "FastFood",
            # Income sources
            "Job",  # Office work
            "Labor",  # Physical labor
            # Fitness/Social builders (secondary meters)
            "Gym",
            "Bar",
            "Park",
            # Mood restoration (primary meter - tier 1 & 2)
            "Recreation",
            "Therapist",
            # Health restoration (primary meter - tier 1 & 2)
            "Doctor",
            "Hospital",
        ]

        # Filter affordances if enabled_affordances is specified
        if enabled_affordances is not None:
            affordance_names_to_use = [name for name in all_affordance_names if name in enabled_affordances]
        else:
            affordance_names_to_use = all_affordance_names

        # Initialize affordances dict with placeholder positions (randomized at episode start)
        # Positions will be shuffled by randomize_affordance_positions() before first use
        self.affordances = {
            name: torch.tensor([0, 0], device=device, dtype=torch.long)
            for name in affordance_names_to_use
        }

        # Create ordered list of affordance names for consistent encoding
        self.affordance_names = list(self.affordances.keys())
        self.num_affordance_types = len(self.affordance_names)

        # Observation dimensions depend on observability mode
        if partial_observability:
            # Level 2 POMDP: local window + position + meters + current affordance type
            window_size = 2 * vision_range + 1  # 5×5 for vision_range=2
            # Grid + position + 8 meters + affordance type one-hot (N+1 for "none")
            self.observation_dim = window_size * window_size + 2 + 8 + (self.num_affordance_types + 1)
        else:
            # Level 1: full grid one-hot + meters + current affordance type
            # Grid one-hot + 8 meters + affordance type (N+1 for "none")
            self.observation_dim = grid_size * grid_size + 8 + (self.num_affordance_types + 1)

        # Always add temporal features for forward compatibility (4 features)
        # time_sin, time_cos, interaction_progress, lifetime_progress
        self.observation_dim += 4

        # Initialize observation builder
        self.observation_builder = ObservationBuilder(
            num_agents=num_agents,
            grid_size=grid_size,
            device=device,
            partial_observability=partial_observability,
            vision_range=vision_range,
            enable_temporal_mechanics=enable_temporal_mechanics,
            num_affordance_types=self.num_affordance_types,
        )

        # Initialize reward strategy (P2.1: per-agent baseline support)
        self.reward_strategy = RewardStrategy(device=device, num_agents=num_agents)
        self.runtime_registry: AgentRuntimeRegistry | None = None  # Injected by population/inference controllers
        self._cached_baseline_tensor = torch.full((num_agents,), 100.0, dtype=torch.float32, device=device)

        # Initialize meter dynamics
        self.meter_dynamics = MeterDynamics(
            num_agents=num_agents,
            device=device,
            cascade_config_dir=self.config_pack_path,
        )

        # Initialize affordance engine
        # Path from src/townlet/environment/ → project root
        config_path = self.config_pack_path / "affordances.yaml"
        affordance_config = load_affordance_config(config_path)
        self.affordance_engine = AffordanceEngine(affordance_config, num_agents, device)

        self.action_dim = 6  # UP, DOWN, LEFT, RIGHT, INTERACT, WAIT

        # State tensors (initialized in reset)
        self.positions = torch.zeros((self.num_agents, 2), dtype=torch.long, device=self.device)
        self.meters = torch.zeros((self.num_agents, 8), dtype=torch.float32, device=self.device)
        self.dones = torch.zeros(self.num_agents, dtype=torch.bool, device=self.device)
        self.step_counts = torch.zeros(self.num_agents, dtype=torch.long, device=self.device)

        # Temporal mechanics state
        self.interaction_progress = torch.zeros(self.num_agents, dtype=torch.long, device=self.device)
        self.last_interaction_affordance: list[str | None] = [None] * self.num_agents
        self.last_interaction_position = torch.zeros((self.num_agents, 2), dtype=torch.long, device=self.device)
        self.time_of_day = 0

        if not self.enable_temporal_mechanics:
            # When temporal mechanics are disabled, interaction progress is unused but kept for typing consistency.
            self.interaction_progress.zero_()

        # Randomize affordance positions on initialization (will be re-randomized each episode)
        self.randomize_affordance_positions()

    def attach_runtime_registry(self, registry: "AgentRuntimeRegistry") -> None:
        """Attach runtime registry for telemetry-aware reward baselines."""
        if registry.get_baseline_tensor().shape != (self.num_agents,):
            raise ValueError(f"Registry baseline shape {registry.get_baseline_tensor().shape} does not match num_agents={self.num_agents}")

        self.runtime_registry = registry

        # Initialise registry with current cached baseline if empty (fresh attach).
        baseline_tensor = registry.get_baseline_tensor()
        if torch.allclose(baseline_tensor, torch.zeros_like(baseline_tensor)):
            registry.set_baselines(self._cached_baseline_tensor.clone())

    def reset(self) -> torch.Tensor:
        """
        Reset all environments.

        Returns:
            observations: [num_agents, observation_dim]
        """
        # Random starting positions
        self.positions = torch.randint(0, self.grid_size, (self.num_agents, 2), device=self.device)

        # Initial meter values (normalized to [0, 1])
        # [energy, hygiene, satiation, money, mood, social, health, fitness]
        # NOTE: money=1.0 corresponds to $100 in range [0, 100] (no debt allowed)
        # All meters start at 100% (healthy agent) except money starts at $50
        self.meters = torch.ones((self.num_agents, 8), device=self.device)
        self.meters[:, 3] = 0.5  # Money starts at $50 (0.5 normalized)

        self.dones = torch.zeros(self.num_agents, dtype=torch.bool, device=self.device)
        self.step_counts = torch.zeros(self.num_agents, dtype=torch.long, device=self.device)

        # Reset temporal mechanics state
        if self.enable_temporal_mechanics:
            self.time_of_day = 0
            self.interaction_progress.fill_(0)
            self.last_interaction_affordance = [None] * self.num_agents
            self.last_interaction_position.fill_(0)

        return self._get_observations()

    def _get_observations(self) -> torch.Tensor:
        """
        Construct observation vector.

        Returns:
            observations: [num_agents, observation_dim]
        """
        # Calculate lifetime progress: 0.0 at birth, 1.0 at retirement
        # This allows agent to learn temporal planning based on remaining lifespan
        lifetime_progress = (self.step_counts.float() / self.agent_lifespan).clamp(0.0, 1.0)

        # Delegate to observation builder
        return self.observation_builder.build_observations(
            positions=self.positions,
            meters=self.meters,
            affordances=self.affordances,
            time_of_day=self.time_of_day if self.enable_temporal_mechanics else 0,
            interaction_progress=self.interaction_progress if self.enable_temporal_mechanics else None,
            lifetime_progress=lifetime_progress,
        )

    def get_action_masks(self) -> torch.Tensor:
        """
        Get action masks for all agents (invalid actions = False).

        Action masking prevents agents from selecting movements that would
        take them off the grid. This saves exploration budget and speeds learning.

        Returns:
            action_masks: [num_agents, 6] bool tensor
                True = valid action, False = invalid
                Actions: [UP, DOWN, LEFT, RIGHT, INTERACT, WAIT]
        """
        action_masks = torch.ones(self.num_agents, 6, dtype=torch.bool, device=self.device)

        # Check boundary constraints
        # positions[:, 0] = x (column), positions[:, 1] = y (row)
        at_top = self.positions[:, 1] == 0  # y == 0
        at_bottom = self.positions[:, 1] == self.grid_size - 1  # y == max
        at_left = self.positions[:, 0] == 0  # x == 0
        at_right = self.positions[:, 0] == self.grid_size - 1  # x == max

        # Mask invalid movements
        action_masks[at_top, 0] = False  # Can't go UP at top edge
        action_masks[at_bottom, 1] = False  # Can't go DOWN at bottom edge
        action_masks[at_left, 2] = False  # Can't go LEFT at left edge
        action_masks[at_right, 3] = False  # Can't go RIGHT at right edge

        # Mask INTERACT (action 4) - only valid when on an open affordance
        # P1.4: Removed affordability check - agents can attempt INTERACT even when broke
        # Affordability is checked inside interaction handlers; failing to afford just
        # wastes a turn (passive decay) and teaches economic planning
        on_valid_affordance = torch.zeros(self.num_agents, dtype=torch.bool, device=self.device)

        # Check each affordance using AffordanceEngine
        for affordance_name, affordance_pos in self.affordances.items():
            distances = torch.abs(self.positions - affordance_pos).sum(dim=1)
            on_this_affordance = distances == 0

            # Check operating hours using AffordanceEngine
            if self.enable_temporal_mechanics:
                if not self.affordance_engine.is_affordance_open(affordance_name, self.time_of_day):
                    # Affordance is closed, skip
                    continue

            # Valid if on affordance AND is open (affordability checked in handler)
            on_valid_affordance |= on_this_affordance

        action_masks[:, 4] = on_valid_affordance

        # P3.1: Mask all actions for dead agents (health <= 0 OR energy <= 0)
        # This must be LAST to override all other masking
        dead_agents = (self.meters[:, 6] <= 0.0) | (self.meters[:, 0] <= 0.0)  # health OR energy
        action_masks[dead_agents] = False

        return action_masks

    def step(
        self,
        actions: torch.Tensor,  # [num_agents]
        depletion_multiplier: float = 1.0,  # Curriculum difficulty
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Execute one step for all agents.

        Args:
            actions: [num_agents] tensor of actions (0-4)
            depletion_multiplier: Curriculum difficulty multiplier (0.2 = 20% difficulty)

        Returns:
            observations: [num_agents, observation_dim]
            rewards: [num_agents]
            dones: [num_agents] bool
            info: dict with metadata
        """
        # 1. Execute actions and track successful interactions
        successful_interactions = self._execute_actions(actions)

        # 2. Deplete meters (base passive decay with curriculum difficulty)
        self.meters = self.meter_dynamics.deplete_meters(self.meters, depletion_multiplier)

        # 3. Cascading effects (coupled differential equations!)
        self.meters = self.meter_dynamics.apply_secondary_to_primary_effects(self.meters)
        self.meters = self.meter_dynamics.apply_tertiary_to_secondary_effects(self.meters)
        self.meters = self.meter_dynamics.apply_tertiary_to_primary_effects(self.meters)

        # 4. Check terminal conditions
        self.dones = self.meter_dynamics.check_terminal_conditions(self.meters, self.dones)

        # 5. Increment step counts (before retirement check)
        self.step_counts += 1

        # 5.5. Check for retirement (reached maximum lifespan)
        # Agents that reach their lifespan retire with a bonus reward
        retired = self.step_counts >= self.agent_lifespan

        # 6. Calculate rewards (steps lived - baseline)
        rewards = self._calculate_shaped_rewards()
        rewards = torch.where(retired, rewards + 1.0, rewards)  # +1 retirement bonus
        self.dones = torch.logical_or(self.dones, retired)

        # 6. Increment time of day (always cycles, but only affects mechanics if enabled)
        self.time_of_day = (self.time_of_day + 1) % 24

        observations = self._get_observations()

        info = {
            "step_counts": self.step_counts.clone(),
            "positions": self.positions.clone(),
            "successful_interactions": successful_interactions,  # {agent_idx: affordance_name}
        }

        return observations, rewards, self.dones, info

    def _execute_actions(self, actions: torch.Tensor) -> dict:
        """
        Execute movement, interaction, and wait actions.

        Args:
            actions: [num_agents] tensor
                0=UP, 1=DOWN, 2=LEFT, 3=RIGHT, 4=INTERACT, 5=WAIT

        Returns:
            Dictionary mapping agent indices to affordance names for successful interactions
        """
        # Store old positions for temporal mechanics progress tracking
        old_positions = self.positions.clone() if self.enable_temporal_mechanics else None

        # Movement deltas (x, y) coordinates
        # x = horizontal (column), y = vertical (row)
        deltas = torch.tensor(
            [
                [0, -1],  # UP - decreases y, x unchanged
                [0, 1],  # DOWN - increases y, x unchanged
                [-1, 0],  # LEFT - decreases x, y unchanged
                [1, 0],  # RIGHT - increases x, y unchanged
                [0, 0],  # INTERACT (no movement)
                [0, 0],  # WAIT (no movement)
            ],
            device=self.device,
        )

        # Apply movement
        movement_deltas = deltas[actions]  # [num_agents, 2]
        new_positions = self.positions + movement_deltas

        # Clamp to grid boundaries
        new_positions = torch.clamp(new_positions, 0, self.grid_size - 1)

        self.positions = new_positions

        # Reset progress for agents that moved away (temporal mechanics)
        if self.enable_temporal_mechanics and old_positions is not None:
            for agent_idx in range(self.num_agents):
                if not torch.equal(old_positions[agent_idx], self.positions[agent_idx]):
                    self.interaction_progress[agent_idx] = 0
                    self.last_interaction_affordance[agent_idx] = None

        # Apply action costs (configurable)
        # Movement (UP, DOWN, LEFT, RIGHT)
        movement_mask = actions < 4
        if movement_mask.any():
            movement_costs = torch.tensor(
                [
                    self.move_energy_cost,  # energy (configurable, default 0.5%)
                    0.003,  # hygiene: -0.3%
                    0.004,  # satiation: -0.4%
                    0.0,  # money: no cost
                    0.0,  # mood: no cost
                    0.0,  # social: no cost
                    0.0,  # health: no cost
                    0.0,  # fitness: no cost
                ],
                device=self.device,
            )
            self.meters[movement_mask] -= movement_costs.unsqueeze(0)
            self.meters = torch.clamp(self.meters, 0.0, 1.0)

        # WAIT action (action 5) - lighter energy cost
        wait_mask = actions == 5
        if wait_mask.any():
            wait_costs = torch.tensor(
                [
                    self.wait_energy_cost,  # energy (configurable, default 0.1%)
                    0.0,  # hygiene: no cost
                    0.0,  # satiation: no cost
                    0.0,  # money: no cost
                    0.0,  # mood: no cost
                    0.0,  # social: no cost
                    0.0,  # health: no cost
                    0.0,  # fitness: no cost
                ],
                device=self.device,
            )
            self.meters[wait_mask] -= wait_costs.unsqueeze(0)
            self.meters = torch.clamp(self.meters, 0.0, 1.0)

        # Handle INTERACT actions
        successful_interactions = {}
        interact_mask = actions == 4
        if interact_mask.any():
            successful_interactions = self._handle_interactions(interact_mask)

        return successful_interactions

    def _handle_interactions(self, interact_mask: torch.Tensor) -> dict:
        """
        Handle INTERACT actions with multi-tick accumulation.

        Args:
            interact_mask: [num_agents] bool mask

        Returns:
            Dictionary mapping agent indices to affordance names
        """
        if not self.enable_temporal_mechanics:
            # Instant interactions (Level 1-2)
            return self._handle_interactions_legacy(interact_mask)

        # Multi-tick interaction logic using AffordanceEngine
        successful_interactions = {}

        for affordance_name, affordance_pos in self.affordances.items():
            # Distance to affordance
            distances = torch.abs(self.positions - affordance_pos).sum(dim=1)
            at_affordance = (distances == 0) & interact_mask

            if not at_affordance.any():
                continue

            # Check affordability using AffordanceEngine
            cost_per_tick = self.affordance_engine.get_affordance_cost(affordance_name, cost_mode="per_tick")
            can_afford = self.meters[:, 3] >= cost_per_tick
            at_affordance = at_affordance & can_afford

            if not at_affordance.any():
                continue

            # Get required ticks from AffordanceEngine
            required_ticks = self.affordance_engine.get_required_ticks(affordance_name)

            # Track successful interactions
            agent_indices = torch.where(at_affordance)[0]

            for agent_idx in agent_indices:
                agent_idx_int = agent_idx.item()
                current_pos = self.positions[agent_idx]

                # Check if continuing same affordance at same position
                if self.last_interaction_affordance[agent_idx_int] == affordance_name and torch.equal(
                    current_pos, self.last_interaction_position[agent_idx_int]
                ):
                    # Continue progress
                    self.interaction_progress[agent_idx] += 1
                else:
                    # New affordance - reset progress
                    self.interaction_progress[agent_idx] = 1
                    self.last_interaction_affordance[agent_idx_int] = affordance_name
                    self.last_interaction_position[agent_idx_int] = current_pos.clone()

                ticks_done = int(self.interaction_progress[agent_idx].item())

                # Create single-agent mask for this agent
                single_agent_mask = torch.zeros(self.num_agents, dtype=torch.bool, device=self.device)
                single_agent_mask[agent_idx] = True

                # Apply multi-tick interaction using AffordanceEngine
                # This applies per-tick effects and costs
                self.meters = self.affordance_engine.apply_multi_tick_interaction(
                    meters=self.meters,
                    affordance_name=affordance_name,
                    current_tick=ticks_done - 1,  # 0-indexed
                    agent_mask=single_agent_mask,
                    check_affordability=False,  # Already checked above
                )

                # Reset progress if completed
                if ticks_done == required_ticks:
                    self.interaction_progress[agent_idx] = 0
                    self.last_interaction_affordance[agent_idx_int] = None

                successful_interactions[agent_idx_int] = affordance_name

        return successful_interactions

    def _handle_interactions_legacy(self, interact_mask: torch.Tensor) -> dict:
        """
        Handle INTERACT action at affordances (instant mode).

        Uses AffordanceEngine for all logic - no hardcoded costs!

        Args:
            interact_mask: [num_agents] bool mask

        Returns:
            Dictionary mapping agent indices to affordance names for successful interactions
        """
        # Track successful interactions for this step
        successful_interactions = {}  # {agent_idx: affordance_name}

        # Check each affordance
        for affordance_name, affordance_pos in self.affordances.items():
            # Distance to affordance
            distances = torch.abs(self.positions - affordance_pos).sum(dim=1)
            at_affordance = (distances == 0) & interact_mask

            if not at_affordance.any():
                continue

            # Check affordability using AffordanceEngine
            cost_normalized = self.affordance_engine.get_affordance_cost(affordance_name, cost_mode="instant")
            if cost_normalized > 0:
                can_afford = self.meters[:, 3] >= cost_normalized
                at_affordance = at_affordance & can_afford

                if not at_affordance.any():
                    # No one at this affordance can afford it, skip
                    continue

            # Track successful interactions
            agent_indices = torch.where(at_affordance)[0]
            for agent_idx in agent_indices:
                successful_interactions[agent_idx.item()] = affordance_name

            # Apply affordance effects using AffordanceEngine
            self.meters = self.affordance_engine.apply_interaction(
                meters=self.meters,
                affordance_name=affordance_name,
                agent_mask=at_affordance,
            )

        return successful_interactions

    def _get_current_baseline_tensor(self) -> torch.Tensor:
        """Return current baseline tensor from registry or cached fallback."""
        if self.runtime_registry is not None:
            return self.runtime_registry.get_baseline_tensor()
        return self._cached_baseline_tensor

    def _calculate_shaped_rewards(self) -> torch.Tensor:
        """
        Calculate interoception-aware rewards.

        Delegates to RewardStrategy for calculation.

        Returns:
            rewards: [num_agents]
        """
        baseline_tensor = self._get_current_baseline_tensor()
        return self.reward_strategy.calculate_rewards(
            step_counts=self.step_counts,
            dones=self.dones,
            baseline_steps=baseline_tensor,
            meters=self.meters,  # Pass meters for interoception-aware rewards
        )

    def calculate_baseline_survival(self, depletion_multiplier: float = 1.0) -> float:
        """
        Calculate baseline survival steps (R) for random-walking agent.

        This is the expected survival time if agent does nothing but move randomly
        until death (no affordance interactions).

        Args:
            depletion_multiplier: Curriculum difficulty multiplier

        Returns:
            baseline_steps: Expected survival time in steps
        """
        # Energy is the most restrictive death condition
        # Base depletion comes from the active config pack (bars.yaml)
        cascade_engine = self.meter_dynamics.cascade_engine
        energy_base_depletion = cascade_engine.get_base_depletion("energy")

        # Longest survival assumes the agent repeatedly takes the cheapest non-affordance action.
        min_action_cost = min(self.move_energy_cost, self.wait_energy_cost, self.interact_energy_cost)

        total_energy_depletion_per_step = (energy_base_depletion * depletion_multiplier) + min_action_cost

        # Starting energy: 1.0
        # Steps until death: 1.0 / depletion_per_step
        baseline_steps = 1.0 / total_energy_depletion_per_step

        return baseline_steps

    def update_baseline_for_curriculum(self, depletion_multipliers: torch.Tensor | float):
        """
        Update reward baseline when curriculum stage changes.

        P2.1: Now supports per-agent baselines for multi-agent curriculum.

        Args:
            depletion_multipliers: Curriculum difficulty multiplier(s)
                - torch.Tensor[num_agents]: Per-agent multipliers
                - float: Shared multiplier (broadcasts to all agents)
        """
        if isinstance(depletion_multipliers, torch.Tensor):
            multipliers = depletion_multipliers.to(self.device, dtype=torch.float32)
            baselines = torch.stack(
                [
                    torch.tensor(
                        self.calculate_baseline_survival(multiplier.item()),
                        dtype=torch.float32,
                        device=self.device,
                    )
                    for multiplier in multipliers
                ]
            )
        else:
            baseline_value = self.calculate_baseline_survival(float(depletion_multipliers))
            baselines = torch.full(
                (self.num_agents,),
                float(baseline_value),
                dtype=torch.float32,
                device=self.device,
            )

        self._cached_baseline_tensor = baselines
        if self.runtime_registry is not None:
            self.runtime_registry.set_baselines(baselines.clone())

        return baselines

    def get_affordance_positions(self) -> dict:
        """Get current affordance positions (P1.1 checkpointing).

        Returns:
            Dictionary with 'positions' and 'ordering' keys:
            - 'positions': Dict mapping affordance names to [x, y] positions
            - 'ordering': List of affordance names in consistent order
        """
        positions = {}
        for name, pos_tensor in self.affordances.items():
            # Convert tensor position to list (for JSON serialization)
            pos = pos_tensor.cpu().tolist()
            positions[name] = [int(pos[0]), int(pos[1])]

        # Include affordance ordering for consistent observation encoding
        return {
            "positions": positions,
            "ordering": self.affordance_names,
        }

    def set_affordance_positions(self, checkpoint_data: dict) -> None:
        """Set affordance positions from checkpoint (P1.1 checkpointing).

        Args:
            checkpoint_data: Dictionary with 'positions' and optionally 'ordering':
                - If 'ordering' provided, rebuild affordances dict in that order
                - Otherwise, use current affordance_names (backwards compatible)
        """
        # Handle backwards compatibility: checkpoint might be old format (just positions dict)
        if "positions" in checkpoint_data:
            positions = checkpoint_data["positions"]
            ordering = checkpoint_data.get("ordering", self.affordance_names)
        else:
            # Old format: checkpoint_data is the positions dict directly
            positions = checkpoint_data
            ordering = self.affordance_names

        # Restore ordering first (critical for consistent observation encoding)
        self.affordance_names = ordering
        self.num_affordance_types = len(self.affordance_names)

        # Rebuild affordances dict in correct order
        for name, pos in positions.items():
            if name in self.affordances:
                self.affordances[name] = torch.tensor(pos, device=self.device, dtype=torch.long)

    def randomize_affordance_positions(self):
        """Randomize affordance positions for generalization testing.

        Ensures no two affordances occupy the same position.
        """
        import random

        # Validate that grid has enough cells for all affordances (need +1 for agent)
        num_affordances = len(self.affordances)
        total_cells = self.grid_size * self.grid_size
        if num_affordances >= total_cells:
            raise ValueError(
                f"Grid has {total_cells} cells but {num_affordances} affordances + 1 agent need space. "
                f"Reduce affordances or increase grid_size to at least {int((num_affordances + 1) ** 0.5) + 1}."
            )

        # Generate list of all grid positions
        all_positions = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size)]

        # Shuffle and assign to affordances
        random.shuffle(all_positions)

        # Assign new positions to affordances
        for i, affordance_name in enumerate(self.affordances.keys()):
            new_pos = all_positions[i]
            self.affordances[affordance_name] = torch.tensor(new_pos, dtype=torch.long, device=self.device)
