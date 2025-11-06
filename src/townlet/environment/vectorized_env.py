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
        grid_size: int,
        partial_observability: bool,
        vision_range: int,
        enable_temporal_mechanics: bool,
        move_energy_cost: float,
        wait_energy_cost: float,
        interact_energy_cost: float,
        agent_lifespan: int,
        device: torch.device = torch.device("cpu"),
        enabled_affordances: list[str] | None = None,
        config_pack_path: Path | None = None,
    ):
        """
        Initialize vectorized environment.

        Args:
            num_agents: Number of parallel agents
            grid_size: Grid dimension (grid_size × grid_size)
            device: PyTorch device (default: cpu). Infrastructure default - PDR-002 exemption.
            partial_observability: If True, agent sees only local window (POMDP)
            vision_range: Radius of vision window (2 = 5×5 window)
            enable_temporal_mechanics: Enable time-based mechanics and multi-tick interactions
            enabled_affordances: List of affordance names to enable (None = all affordances). Semantic default.
            move_energy_cost: Energy cost per movement action
            wait_energy_cost: Energy cost per WAIT action
            interact_energy_cost: Energy cost per INTERACT action
            agent_lifespan: Maximum lifetime in steps (provides retirement incentive)
            config_pack_path: Path to config pack (default: configs/test). Infrastructure fallback - PDR-002 exemption.

        Note (PDR-002 Compliance):
            - device and config_pack_path have infrastructure defaults (exempted from no-defaults principle)
            - enabled_affordances=None is a semantic default (None means "all affordances enabled")
            - All other parameters are UAC behavioral parameters and MUST be explicitly provided
        """
        project_root = Path(__file__).parent.parent.parent.parent
        default_pack = project_root / "configs" / "test"

        self.config_pack_path = Path(config_pack_path) if config_pack_path else default_pack
        if not self.config_pack_path.exists():
            raise FileNotFoundError(f"Config pack directory not found: {self.config_pack_path}")

        # BREAKING CHANGE: substrate.yaml is now REQUIRED
        substrate_config_path = self.config_pack_path / "substrate.yaml"
        if not substrate_config_path.exists():
            raise FileNotFoundError(
                f"substrate.yaml is required but not found in {self.config_pack_path}.\n\n"
                f"All config packs must define their spatial substrate.\n\n"
                f"Quick fix:\n"
                f"  1. Copy template: cp configs/templates/substrate.yaml {self.config_pack_path}/\n"
                f"  2. Edit substrate.yaml to match your grid_size from training.yaml\n"
                f"  3. See CLAUDE.md 'Configuration System' for details\n\n"
                f"This is a breaking change from TASK-002A. Previous configs without\n"
                f"substrate.yaml will no longer work. See CHANGELOG.md for migration guide."
            )

        from townlet.substrate.config import load_substrate_config
        from townlet.substrate.factory import SubstrateFactory

        substrate_config = load_substrate_config(substrate_config_path)
        self.substrate = SubstrateFactory.build(substrate_config, device=device)

        # Load action labels (optional - defaults to "gaming" preset if not specified)
        from townlet.environment.action_labels import get_labels

        action_labels_config_path = self.config_pack_path / "action_labels.yaml"
        if action_labels_config_path.exists():
            # Load custom action labels from config
            import yaml

            with open(action_labels_config_path) as f:
                action_labels_data = yaml.safe_load(f)

            from townlet.substrate.config import ActionLabelConfig

            label_config = ActionLabelConfig(**action_labels_data)

            # Get labels for substrate dimensionality
            if label_config.preset:
                self.action_labels = get_labels(preset=label_config.preset, substrate_position_dim=self.substrate.position_dim)
            else:
                self.action_labels = get_labels(custom_labels=label_config.custom, substrate_position_dim=self.substrate.position_dim)
        else:
            # Default to gaming preset if no action_labels.yaml
            self.action_labels = get_labels(preset="gaming", substrate_position_dim=self.substrate.position_dim)

        # Update grid_size from substrate (for backward compatibility with other code)
        # For aspatial substrates, keep parameter value; for grid substrates, use substrate
        self.grid_size = grid_size  # Default to parameter (for aspatial or backward compat)
        if hasattr(self.substrate, "width") and hasattr(self.substrate, "height"):
            if self.substrate.width != self.substrate.height:
                raise ValueError(f"Non-square grids not yet supported: {self.substrate.width}×{self.substrate.height}")
            self.grid_size = self.substrate.width  # Override with substrate for grid

        self.num_agents = num_agents
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

        # Load bars configuration to get meter_count for observation dimensions
        # This must happen before observation dimension calculation
        from townlet.environment.cascade_config import load_bars_config

        bars_config_path = self.config_pack_path / "bars.yaml"
        bars_config = load_bars_config(bars_config_path)
        self.bars_config = bars_config  # Store for use in affordance validation (TASK-001)
        self.meter_count = bars_config.meter_count
        meter_count = self.meter_count  # Keep local variable for backward compatibility in this method

        # Load affordance configuration to get FULL universe vocabulary
        # This must also happen before observation dimension calculation
        # Pass bars_config for meter reference validation (TASK-001)
        config_path = self.config_pack_path / "affordances.yaml"
        affordance_config = load_affordance_config(config_path, bars_config)

        # Extract ALL affordance names from YAML (defines observation vocabulary)
        # This is the FULL universe - what the agent can observe and reason about
        all_affordance_names = [aff.name for aff in affordance_config.affordances]

        # Filter affordances for DEPLOYMENT (which ones actually exist on the grid)
        # enabled_affordances from training.yaml controls what the agent can interact with
        if enabled_affordances is not None:
            affordance_names_to_deploy = [name for name in all_affordance_names if name in enabled_affordances]
        else:
            affordance_names_to_deploy = all_affordance_names

        # DEPLOYED affordances: have positions on grid, can be interacted with
        # Positions will be randomized by randomize_affordance_positions() before first use
        default_position = torch.zeros(self.substrate.position_dim, dtype=self.substrate.position_dtype, device=device)
        self.affordances = {name: default_position.clone() for name in affordance_names_to_deploy}

        # OBSERVATION VOCABULARY: Full list from YAML, used for fixed observation encoding
        # This stays constant across all curriculum levels for transfer learning
        self.affordance_names = all_affordance_names
        self.num_affordance_types = len(all_affordance_names)

        # Validate partial observability support
        if partial_observability and self.substrate.position_dim >= 4:
            raise ValueError(
                f"Partial observability (POMDP) is not supported for {self.substrate.position_dim}D substrates. "
                f"Local window size would be (2*{vision_range}+1)^{self.substrate.position_dim} = "
                f"{(2*vision_range+1)**self.substrate.position_dim} cells, which is impractical. "
                f"Use full observability (partial_observability=False) with 'relative' or 'scaled' "
                f"observation_encoding instead. See substrate configuration docs for details."
            )

        # Validate Grid3D POMDP vision range (prevent memory explosion)
        if partial_observability and self.substrate.position_dim == 3:
            window_volume = (2 * vision_range + 1) ** 3
            if window_volume > 125:  # 5×5×5 = 125 is the threshold
                raise ValueError(
                    f"Grid3D POMDP with vision_range={vision_range} requires {window_volume} cells "
                    f"(window size {2*vision_range+1}×{2*vision_range+1}×{2*vision_range+1}), which is excessive. "
                    f"Use vision_range ≤ 2 (5×5×5 = 125 cells) for Grid3D partial observability, "
                    f"or disable partial_observability."
                )

        # Validate observation_encoding compatibility with POMDP
        if partial_observability and hasattr(self.substrate, "observation_encoding"):
            if self.substrate.observation_encoding != "relative":
                raise ValueError(
                    f"Partial observability (POMDP) requires observation_encoding='relative', "
                    f"but substrate is configured with observation_encoding='{self.substrate.observation_encoding}'. "
                    f"POMDP uses normalized positions for recurrent network position encoder. "
                    f"Set observation_encoding='relative' in substrate.yaml or disable partial_observability."
                )

        # Observation dimensions depend on observability mode
        if partial_observability:
            # Level 2 POMDP: local window + position + meters + current affordance type
            window_size = 2 * vision_range + 1  # 5×5 for vision_range=2
            # Local window + normalized position + meters + affordance type one-hot (N+1 for "none")
            # Position dimension is substrate-specific (2 for Grid2D, 0 for Aspatial)
            self.observation_dim = (
                window_size * window_size
                + self.substrate.position_dim  # 2 for Grid2D, 0 for Aspatial
                + meter_count
                + (self.num_affordance_types + 1)
            )
        else:
            # Level 1: full grid encoding + meters + current affordance type
            # Grid encoding dimension is substrate-specific (width*height for Grid2D, 0 for Aspatial)
            self.observation_dim = (
                self.substrate.get_observation_dim() + meter_count + (self.num_affordance_types + 1)  # Substrate-specific grid encoding
            )

        # Always add temporal features for forward compatibility (4 features)
        # time_sin, time_cos, interaction_progress, lifetime_progress
        self.observation_dim += 4

        # Initialize observation builder
        self.observation_builder = ObservationBuilder(
            num_agents=num_agents,
            grid_size=self.grid_size,
            device=device,
            partial_observability=partial_observability,
            vision_range=vision_range,
            enable_temporal_mechanics=enable_temporal_mechanics,
            num_affordance_types=self.num_affordance_types,
            affordance_names=self.affordance_names,
            substrate=self.substrate,  # Pass substrate to observation builder
        )

        # Initialize reward strategy (TASK-001: variable meters)
        # Get meter indices from bars_config for dynamic action costs and death detection
        meter_name_to_index = bars_config.meter_name_to_index
        self.energy_idx = meter_name_to_index.get("energy", 0)  # Default to 0 if not found
        self.health_idx = meter_name_to_index.get("health", min(6, meter_count - 1))  # Default to 6 or last meter
        self.hygiene_idx = meter_name_to_index.get("hygiene", None)  # Optional meter
        self.satiation_idx = meter_name_to_index.get("satiation", None)  # Optional meter
        self.money_idx = meter_name_to_index.get("money", None)  # Optional meter

        self.reward_strategy = RewardStrategy(
            device=device, num_agents=num_agents, meter_count=meter_count, energy_idx=self.energy_idx, health_idx=self.health_idx
        )
        self.runtime_registry: AgentRuntimeRegistry | None = None  # Injected by population/inference controllers

        # Initialize meter dynamics
        self.meter_dynamics = MeterDynamics(
            num_agents=num_agents,
            device=device,
            cascade_config_dir=self.config_pack_path,
        )

        # Initialize affordance engine (reuse affordance_config loaded above)
        # Pass meter_name_to_index for dynamic meter lookups (TASK-001)
        self.affordance_engine = AffordanceEngine(
            affordance_config,
            num_agents,
            device,
            bars_config.meter_name_to_index,
        )

        # Action space size is determined by substrate
        # Substrate reports number of discrete actions via action_space_size property
        # This enables dynamic action spaces for N-dimensional substrates
        self.action_dim = self.substrate.action_space_size

        # State tensors (initialized in reset)
        self.positions = torch.zeros(
            (self.num_agents, self.substrate.position_dim),
            dtype=self.substrate.position_dtype,
            device=self.device,
        )
        self.meters = torch.zeros((self.num_agents, meter_count), dtype=torch.float32, device=self.device)
        self.dones = torch.zeros(self.num_agents, dtype=torch.bool, device=self.device)
        self.step_counts = torch.zeros(self.num_agents, dtype=torch.long, device=self.device)

        # Temporal mechanics state
        self.interaction_progress = torch.zeros(self.num_agents, dtype=torch.long, device=self.device)
        self.last_interaction_affordance: list[str | None] = [None] * self.num_agents
        self.last_interaction_position = torch.zeros(
            (self.num_agents, self.substrate.position_dim),
            dtype=self.substrate.position_dtype,
            device=self.device,
        )
        self.time_of_day = 0

        if not self.enable_temporal_mechanics:
            # When temporal mechanics are disabled, interaction progress is unused but kept for typing consistency.
            self.interaction_progress.zero_()

        # Randomize affordance positions on initialization (will be re-randomized each episode)
        self.randomize_affordance_positions()

    def attach_runtime_registry(self, registry: AgentRuntimeRegistry) -> None:
        """Attach runtime registry for telemetry tracking."""
        self.runtime_registry = registry

    def get_action_label_names(self) -> dict[int, str]:
        """Get action label names for current substrate.

        Returns:
            Dictionary mapping action indices to user-facing labels.

        Example:
            >>> env = VectorizedHamletEnv(...)
            >>> labels = env.get_action_label_names()
            >>> print(labels)
            {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT', 4: 'INTERACT', 5: 'WAIT'}
        """
        return self.action_labels.get_all_labels()

    def reset(self) -> torch.Tensor:
        """
        Reset all environments.

        Returns:
            observations: [num_agents, observation_dim]
        """
        # Use substrate for position initialization (supports grid and aspatial)
        self.positions = self.substrate.initialize_positions(self.num_agents, self.device)

        # Initial meter values (normalized to [0, 1])
        # [energy, hygiene, satiation, money, mood, social, health, fitness]
        # Read initial values from bars.yaml config
        initial_values = self.meter_dynamics.cascade_engine.get_initial_meter_values()
        self.meters = initial_values.unsqueeze(0).expand(self.num_agents, -1).clone()

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
            action_masks: [num_agents, action_dim] bool tensor
                True = valid action, False = invalid
                Grid2D (6 actions): [UP, DOWN, LEFT, RIGHT, INTERACT, WAIT]
                Grid3D (8 actions): [UP, DOWN, LEFT, RIGHT, INTERACT, WAIT, UP_Z, DOWN_Z]
        """
        action_masks = torch.ones(self.num_agents, self.action_dim, dtype=torch.bool, device=self.device)

        # Check boundary constraints (only for spatial substrates)
        if self.substrate.position_dim >= 2:
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

        # 3D-specific: mask Z-axis movements at floor/ceiling
        if self.substrate.position_dim == 3:
            at_floor = self.positions[:, 2] == 0  # z == 0
            # Assume depth from substrate
            if hasattr(self.substrate, "depth"):
                at_ceiling = self.positions[:, 2] == self.substrate.depth - 1
            else:
                at_ceiling = torch.zeros(self.num_agents, dtype=torch.bool, device=self.device)

            action_masks[at_ceiling, 6] = False  # Can't go UP_Z at ceiling
            action_masks[at_floor, 7] = False  # Can't go DOWN_Z at floor

        # Mask INTERACT - only valid when on an open affordance
        # P1.4: Removed affordability check - agents can attempt INTERACT even when broke
        # Affordability is checked inside interaction handlers; failing to afford just
        # wastes a turn (passive decay) and teaches economic planning

        # Determine INTERACT action index based on substrate dimensionality
        # 1D: INTERACT = 2, 2D: INTERACT = 4, 3D: INTERACT = 4, Aspatial: INTERACT = 0
        if self.substrate.position_dim == 1:
            interact_action_idx = 2
        elif self.substrate.position_dim == 0:
            interact_action_idx = 0
        else:  # 2D or 3D
            interact_action_idx = 4

        on_valid_affordance = torch.zeros(self.num_agents, dtype=torch.bool, device=self.device)

        # Check each affordance using AffordanceEngine
        for affordance_name, affordance_pos in self.affordances.items():
            # Check if on affordance for interaction (using substrate)
            on_this_affordance = self.substrate.is_on_position(self.positions, affordance_pos)

            # Check operating hours using AffordanceEngine
            if self.enable_temporal_mechanics:
                if not self.affordance_engine.is_affordance_open(affordance_name, self.time_of_day):
                    # Affordance is closed, skip
                    continue

            # Valid if on affordance AND is open (affordability checked in handler)
            on_valid_affordance |= on_this_affordance

        action_masks[:, interact_action_idx] = on_valid_affordance

        # P3.1: Mask all actions for dead agents (health <= 0 OR energy <= 0)
        # This must be LAST to override all other masking
        # TASK-001: Use dynamic meter indices instead of hardcoded 0 and 6
        dead_agents = (self.meters[:, self.health_idx] <= 0.0) | (self.meters[:, self.energy_idx] <= 0.0)  # health OR energy
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

        # 6. Calculate rewards (interoception-aware)
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

        # Movement deltas - dynamically sized based on substrate dimensionality
        # Always float32 - substrates cast to their dtype as needed
        # For 1D (Continuous1D): 2 directions (X-/X+) + interact/wait
        # For 2D (Grid2D, Continuous2D): 4 directions + interact/wait
        # For 3D (Grid3D, Continuous3D): 6 directions + interact/wait
        if self.substrate.position_dim == 1:
            deltas = torch.tensor(
                [
                    [-1],  # MOVE_X_NEGATIVE (action 0)
                    [1],  # MOVE_X_POSITIVE (action 1)
                    [0],  # INTERACT (no movement, action 2)
                    [0],  # WAIT (no movement, action 3)
                ],
                device=self.device,
                dtype=torch.float32,  # Always float, substrates cast if needed
            )
        elif self.substrate.position_dim == 2:
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
                dtype=torch.float32,  # Always float, substrates cast if needed
            )
        elif self.substrate.position_dim == 3:
            deltas = torch.tensor(
                [
                    [0, -1, 0],  # UP - decreases y (Grid2D compat: action 0)
                    [0, 1, 0],  # DOWN - increases y (Grid2D compat: action 1)
                    [-1, 0, 0],  # LEFT - decreases x (Grid2D compat: action 2)
                    [1, 0, 0],  # RIGHT - increases x (Grid2D compat: action 3)
                    [0, 0, 0],  # INTERACT (no movement) (Grid2D compat: action 4)
                    [0, 0, 0],  # WAIT (no movement) (Grid2D compat: action 5)
                    [0, 0, 1],  # UP_Z - increases z (new for 3D: action 6)
                    [0, 0, -1],  # DOWN_Z - decreases z (new for 3D: action 7)
                ],
                device=self.device,
                dtype=torch.float32,  # Always float, substrates cast if needed
            )
        elif self.substrate.position_dim == 0:
            # Aspatial: no movement deltas needed, but provide dummy array
            deltas = torch.zeros((6, 0), device=self.device, dtype=torch.float32)
        else:
            raise ValueError(
                f"Unsupported substrate position_dim: {self.substrate.position_dim}. "
                f"Expected 0 (aspatial), 1 (1D continuous), 2 (Grid2D/Continuous2D), or 3 (Grid3D/Continuous3D)"
            )

        # Apply movement with substrate-specific boundary handling
        movement_deltas = deltas[actions]  # [num_agents, position_dim]
        self.positions = self.substrate.apply_movement(self.positions, movement_deltas)

        # Reset progress for agents that moved away (temporal mechanics)
        if self.enable_temporal_mechanics and old_positions is not None:
            for agent_idx in range(self.num_agents):
                if not torch.equal(old_positions[agent_idx], self.positions[agent_idx]):
                    self.interaction_progress[agent_idx] = 0
                    self.last_interaction_affordance[agent_idx] = None

        # Apply action costs (configurable)
        # Determine movement actions directly from the movement deltas to support
        # substrates where non-movement actions appear before all movement actions
        # (e.g., 3D where INTERACT/WAIT sit at indices < last movement deltas).
        movement_actions = deltas.abs().sum(dim=1) > 0
        movement_mask = movement_actions[actions]
        if movement_mask.any():
            # TASK-001: Create dynamic cost tensor based on meter_count
            movement_costs = torch.zeros(self.meter_count, device=self.device)
            movement_costs[self.energy_idx] = self.move_energy_cost  # Energy (configurable, default 0.5%)
            if self.hygiene_idx is not None:
                movement_costs[self.hygiene_idx] = 0.003  # Hygiene: -0.3%
            if self.satiation_idx is not None:
                movement_costs[self.satiation_idx] = 0.004  # Satiation: -0.4%

            self.meters[movement_mask] -= movement_costs.unsqueeze(0)
            self.meters = torch.clamp(self.meters, 0.0, 1.0)

        # WAIT action - lighter energy cost
        # Index derived from substrate dimensionality:
        # 0D (aspatial): WAIT = 1
        # 1D: WAIT = 3
        # 2D: WAIT = 5
        # 3D: WAIT = 5
        if self.substrate.position_dim == 0:
            wait_action_idx = 1
        elif self.substrate.position_dim == 1:
            wait_action_idx = 3
        elif self.substrate.position_dim in (2, 3):
            wait_action_idx = 5
        else:
            wait_action_idx = 2 * self.substrate.position_dim + 1

        wait_mask = actions == wait_action_idx
        if wait_mask.any():
            # TASK-001: Create dynamic cost tensor based on meter_count
            wait_costs = torch.zeros(self.meter_count, device=self.device)
            wait_costs[self.energy_idx] = self.wait_energy_cost  # Energy (configurable, default 0.1%)

            self.meters[wait_mask] -= wait_costs.unsqueeze(0)
            self.meters = torch.clamp(self.meters, 0.0, 1.0)

        # Handle INTERACT actions
        # Index derived from substrate dimensionality:
        # 0D (aspatial): INTERACT = 0
        # 1D: INTERACT = 2
        # 2D: INTERACT = 4
        # 3D: INTERACT = 4
        if self.substrate.position_dim == 0:
            interact_action_idx = 0
        elif self.substrate.position_dim == 1:
            interact_action_idx = 2
        elif self.substrate.position_dim in (2, 3):
            interact_action_idx = 4
        else:
            interact_action_idx = 2 * self.substrate.position_dim

        successful_interactions = {}
        interact_mask = actions == interact_action_idx
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
            # Check if still on same affordance (using substrate)
            at_affordance = self.substrate.is_on_position(self.positions, affordance_pos) & interact_mask

            if not at_affordance.any():
                continue

            # Check affordability using AffordanceEngine
            cost_per_tick = self.affordance_engine.get_affordance_cost(affordance_name, cost_mode="per_tick")
            # TASK-001: Use dynamic money index (if money meter exists)
            if self.money_idx is not None:
                can_afford = self.meters[:, self.money_idx] >= cost_per_tick
                at_affordance = at_affordance & can_afford
            # else: no money meter, affordability always passes

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
            # Check which agents are on this affordance (using substrate)
            at_affordance = self.substrate.is_on_position(self.positions, affordance_pos) & interact_mask

            if not at_affordance.any():
                continue

            # Check affordability using AffordanceEngine
            cost_normalized = self.affordance_engine.get_affordance_cost(affordance_name, cost_mode="instant")
            if cost_normalized > 0:
                # TASK-001: Use dynamic money index (if money meter exists)
                if self.money_idx is not None:
                    can_afford = self.meters[:, self.money_idx] >= cost_normalized
                    at_affordance = at_affordance & can_afford
                # else: no money meter, affordability always passes

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

    def _calculate_shaped_rewards(self) -> torch.Tensor:
        """
        Calculate interoception-aware rewards.

        Delegates to RewardStrategy for calculation.

        Returns:
            rewards: [num_agents]
        """
        return self.reward_strategy.calculate_rewards(
            step_counts=self.step_counts,
            dones=self.dones,
            meters=self.meters,  # Pass meters for interoception-aware rewards
        )

    def get_affordance_positions(self) -> dict:
        """Get current affordance positions (substrate-agnostic checkpointing).

        Returns:
            Dictionary with 'positions', 'ordering', and 'position_dim' keys:
            - 'positions': Dict mapping affordance names to position lists
            - 'ordering': List of affordance names in consistent order
            - 'position_dim': Dimensionality for validation (0=aspatial, 2=2D, 3=3D)
        """
        positions = {}
        for name, pos_tensor in self.affordances.items():
            # Convert tensor to list (handles any dimensionality)
            pos = pos_tensor.cpu().tolist()

            # Ensure pos is a list (even for 0-dimensional positions)
            if isinstance(pos, int | float):
                pos = [pos]
            elif self.substrate.position_dim == 0:
                pos = []

            positions[name] = [int(x) for x in pos] if pos else []

        return {
            "positions": positions,
            "ordering": self.affordance_names,
            "position_dim": self.substrate.position_dim,  # For validation
        }

    def set_affordance_positions(self, checkpoint_data: dict) -> None:
        """Set affordance positions from checkpoint (Phase 4+ only).

        BREAKING CHANGE: Only loads Phase 4+ checkpoints with position_dim field.
        Legacy checkpoints will not load.

        Args:
            checkpoint_data: Dictionary with 'positions', 'ordering', and 'position_dim'

        Raises:
            ValueError: If checkpoint missing position_dim or incompatible with substrate
        """
        # Validate position_dim exists (no default fallback)
        if "position_dim" not in checkpoint_data:
            raise ValueError(
                "Checkpoint missing 'position_dim' field.\n"
                "This is a legacy checkpoint (pre-Phase 4).\n"
                "\n"
                "BREAKING CHANGE: Phase 4 changed checkpoint format.\n"
                "Legacy checkpoints (Version 2) are no longer compatible.\n"
                "\n"
                "Action required:\n"
                "  1. Delete old checkpoint directories: checkpoints_level*/\n"
                "  2. Retrain models from scratch with Phase 4+ code\n"
                "\n"
                "If you need to preserve old models, checkout pre-Phase 4 git commit."
            )

        # Validate compatibility (no backward compatibility)
        checkpoint_position_dim = checkpoint_data["position_dim"]
        if checkpoint_position_dim != self.substrate.position_dim:
            raise ValueError(
                f"Checkpoint position_dim mismatch: checkpoint has {checkpoint_position_dim}D, "
                f"but current substrate requires {self.substrate.position_dim}D."
            )

        # Simple loading (no backward compat branches)
        positions = checkpoint_data["positions"]
        ordering = checkpoint_data["ordering"]

        self.affordance_names = ordering
        self.num_affordance_types = len(self.affordance_names)

        for name, pos in positions.items():
            if name in self.affordances:
                self.affordances[name] = torch.tensor(pos, device=self.device, dtype=self.substrate.position_dtype)

    def randomize_affordance_positions(self):
        """Randomize affordance positions for generalization testing.

        Grid substrates: Shuffle all positions
        Continuous substrates: Random sampling
        Aspatial: No positions

        Ensures no two affordances occupy the same position (for grid substrates).
        """
        import random

        # Skip if no affordances to randomize
        num_affordances = len(self.affordances)
        if num_affordances == 0:
            return  # Nothing to randomize

        # Aspatial substrates don't have positions
        if self.substrate.position_dim == 0:
            # Aspatial: no positions, affordances don't need placement
            return

        # Check if substrate supports enumerable positions
        if hasattr(self.substrate, "supports_enumerable_positions") and self.substrate.supports_enumerable_positions():
            # Grid substrates: shuffle all positions
            all_positions = self.substrate.get_all_positions()

            # Validate that grid has enough cells for all affordances (need +1 for agent)
            total_cells = len(all_positions)
            if num_affordances >= total_cells:
                raise ValueError(
                    f"Grid has {total_cells} cells but {num_affordances} affordances + 1 agent need space. "
                    f"Reduce affordances or increase grid_size to at least {int((num_affordances + 1) ** 0.5) + 1}."
                )

            # Shuffle and assign to affordances
            random.shuffle(all_positions)

            # Assign new positions to affordances
            for i, affordance_name in enumerate(self.affordances.keys()):
                new_pos = all_positions[i]
                self.affordances[affordance_name] = torch.tensor(new_pos, dtype=self.substrate.position_dtype, device=self.device)
        else:
            # Continuous/other: random sampling
            # Use substrate's initialize_positions for random placement
            affordance_positions_tensor = self.substrate.initialize_positions(num_agents=num_affordances, device=self.device)

            # Assign to affordances
            for i, affordance_name in enumerate(self.affordances.keys()):
                self.affordances[affordance_name] = affordance_positions_tensor[i]
