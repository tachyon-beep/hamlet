"""
Vectorized Hamlet environment for GPU-native training.

Batches multiple independent Hamlet environments into a single vectorized
environment with tensor operations [num_agents, ...].
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
import yaml

from townlet.environment.action_builder import ActionSpaceBuilder
from townlet.environment.affordance_config import load_affordance_config
from townlet.environment.affordance_engine import AffordanceEngine
from townlet.environment.meter_dynamics import MeterDynamics
from townlet.environment.reward_strategy import RewardStrategy
from townlet.substrate.continuous import ContinuousSubstrate
from townlet.vfs import VariableRegistry, VFSObservationSpecBuilder
from townlet.vfs.schema import VariableDef

if TYPE_CHECKING:
    from townlet.environment.action_config import ActionConfig
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

        # VFS INTEGRATION: Load variables from config pack
        # BREAKING CHANGE: variables_reference.yaml is now REQUIRED
        variables_path = self.config_pack_path / "variables_reference.yaml"
        if not variables_path.exists():
            raise FileNotFoundError(
                f"variables_reference.yaml is required but not found in {self.config_pack_path}.\n\n"
                f"All config packs must define their VFS variables.\n\n"
                f"Quick fix:\n"
                f"  1. Copy reference: cp configs/L1_full_observability/variables_reference.yaml {self.config_pack_path}/\n"
                f"  2. Edit to match your configuration\n"
                f"  3. See docs/config-schemas/variables.md for schema details\n\n"
                f"This is a breaking change from VFS Phase 1 integration. Previous configs without\n"
                f"variables_reference.yaml will no longer work. See docs/vfs-integration-guide.md."
            )

        with open(variables_path) as f:
            variables_data = yaml.safe_load(f)

        self.vfs_variables = [VariableDef(**var_data) for var_data in variables_data["variables"]]

        # Build exposure configuration for observation spec
        self.vfs_exposures = {}
        if "exposed_observations" in variables_data:
            for obs in variables_data["exposed_observations"]:
                var_id = obs["source_variable"]
                self.vfs_exposures[var_id] = {
                    "normalization": obs.get("normalization"),
                }
        else:
            # Fallback: expose all agent-readable variables
            for var in self.vfs_variables:
                if "agent" in var.readable_by:
                    self.vfs_exposures[var.id] = {"normalization": None}

        # Filter exposures based on observability mode
        # Full obs uses grid_encoding, POMDP uses local_window
        if partial_observability:
            # POMDP: Remove grid_encoding if present (use local_window instead)
            self.vfs_exposures.pop("grid_encoding", None)
        else:
            # Full obs: Remove local_window if present (use grid_encoding instead)
            self.vfs_exposures.pop("local_window", None)

        # Load action labels (optional - defaults to "gaming" preset if not specified)
        from townlet.environment.action_labels import get_labels

        action_labels_config_path = self.config_pack_path / "action_labels.yaml"
        if action_labels_config_path.exists():
            # Load custom action labels from config
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
        if partial_observability and self.substrate.position_dim == 0:
            raise ValueError(
                "Partial observability (POMDP) is not supported for aspatial substrates. "
                "A local vision window requires at least 1 spatial dimension. "
                "Set partial_observability=False when using an aspatial substrate."
            )
        if partial_observability and isinstance(self.substrate, ContinuousSubstrate):
            raise ValueError(
                "Partial observability (POMDP) is not supported for continuous substrates. "
                "Continuous spaces have infinite positions within any local window, making discrete vision grids undefined. "
                "Use partial_observability=False with 'relative' or 'scaled' observation_encoding instead."
            )
        if partial_observability and self.substrate.position_dim >= 4:
            window_size = 2 * vision_range + 1
            cell_count = window_size**self.substrate.position_dim
            raise ValueError(
                f"Partial observability (POMDP) is not supported for {self.substrate.position_dim}D substrates. "
                f"\n\nProblem: Local window size grows EXPONENTIALLY with dimensionality:"
                f"\n  - 2D: {window_size}×{window_size} = {window_size**2} cells (practical)"
                f"\n  - 3D: {window_size}×{window_size}×{window_size} = {window_size**3} cells (supported up to vision_range=2)"
                f"\n  - {self.substrate.position_dim}D: {window_size}^{self.substrate.position_dim} = {cell_count:,} cells (IMPRACTICAL)"
                f"\n\nThis creates:"
                f"\n  - Network input explosion ({cell_count:,} vision features + position + meters)"
                f"\n  - Memory explosion (each agent's observation is massive)"
                f"\n  - Training slowdown (gradient computation over huge inputs)"
                f"\n\nSolution: Use full observability (partial_observability=False) with normalized position encoding:"
                f"\n  - observation_encoding='relative': Just {self.substrate.position_dim} dims (normalized coordinates)"
                f"\n  - observation_encoding='scaled': {self.substrate.position_dim*2} dims (coordinates + grid sizes)"
                f"\n  - Enables dimension-independent learning WITHOUT exponential curse"
                f"\n\nSee docs/manual/pomdp_compatibility_matrix.md for details."
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

        # VFS INTEGRATION: Initialize variable registry
        # Registry holds runtime state for all VFS variables
        self.vfs_registry = VariableRegistry(variables=self.vfs_variables, num_agents=num_agents, device=device)

        # VFS INTEGRATION: Build observation spec from variables
        # This replaces hardcoded observation dimension calculation
        obs_builder = VFSObservationSpecBuilder()
        self.vfs_observation_spec = obs_builder.build_observation_spec(self.vfs_variables, self.vfs_exposures)

        # Calculate observation_dim from VFS spec
        self.observation_dim = sum(field.shape[0] if field.shape else 1 for field in self.vfs_observation_spec)

        # Store partial observability settings for observation construction
        self.partial_observability = partial_observability
        self.vision_range = vision_range

        # Initialize reward strategy (TASK-001: variable meters)
        # Get meter indices from bars_config for dynamic action costs and death detection
        meter_name_to_index = bars_config.meter_name_to_index
        # Store full mapping for dynamic meter lookups (custom actions, telemetry, etc.)
        self.meter_name_to_index: dict[str, int] = dict(meter_name_to_index)
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

        # Build composed action space from substrate + global custom actions
        # TODO(Phase 4.2): Load enabled_actions from training.yaml when TrainingConfig DTO exists
        # For now, all actions enabled by default (None = all enabled)
        enabled_actions = None  # Will be loaded from training.yaml in Task 4.2

        global_actions_path = Path("configs/global_actions.yaml")
        builder = ActionSpaceBuilder(
            substrate=self.substrate,
            global_actions_path=global_actions_path,
            enabled_action_names=enabled_actions,
        )
        self.action_space = builder.build()
        self.action_dim = self.action_space.action_dim

        # Cache action indices for fast lookup (replaces hardcoded formulas from Task 1.6)
        self.interact_action_idx = self.action_space.get_action_by_name("INTERACT").id
        self.wait_action_idx = self.action_space.get_action_by_name("WAIT").id
        self.up_z_action_idx = self._get_optional_action_idx("UP_Z")
        self.down_z_action_idx = self._get_optional_action_idx("DOWN_Z")

        # Build movement deltas from ActionConfig (dynamic, not hardcoded)
        self._movement_deltas = self._build_movement_deltas()

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

    def _get_optional_action_idx(self, action_name: str) -> int | None:
        """Return action index if available in composed action space."""
        try:
            return self.action_space.get_action_by_name(action_name).id
        except ValueError:
            return None

    def _build_movement_deltas(self) -> torch.Tensor:
        """Build movement delta tensor from substrate default actions.

        Returns:
            [action_space_size, position_dim] tensor of movement deltas
        """
        substrate_actions = self.substrate.get_default_actions()
        action_space_size = self.substrate.action_space_size
        position_dim = self.substrate.position_dim

        # Initialize zero deltas for all actions
        deltas = torch.zeros(
            (action_space_size, position_dim),
            device=self.device,
            dtype=self.substrate.position_dtype,
        )

        # Fill in deltas from ActionConfig
        for action in substrate_actions:
            if action.delta is not None:
                deltas[action.id] = torch.tensor(
                    action.delta,
                    device=self.device,
                    dtype=self.substrate.position_dtype,
                )

        return deltas

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
        Construct observation vector using VFS registry.

        Returns:
            observations: [num_agents, observation_dim]
        """
        import math

        # Calculate lifetime progress: 0.0 at birth, 1.0 at retirement
        # This allows agent to learn temporal planning based on remaining lifespan
        lifetime_progress = (self.step_counts.float() / self.agent_lifespan).clamp(0.0, 1.0)

        # Update VFS registry with current state
        # Grid encoding (full or partial depending on POMDP setting)
        if self.partial_observability:
            # POMDP: Local window encoding (returns only the local window, no position)
            local_window = self.substrate.encode_partial_observation(self.positions, self.affordances, vision_range=self.vision_range)
            # POMDP uses "local_window" variable name
            self.vfs_registry.set("local_window", local_window, writer="engine")
        else:
            # Full observability: Complete grid encoding (grid only, position handled separately in VFS)
            # Use internal _encode_full_grid to get ONLY the grid, not grid+position
            if hasattr(self.substrate, "_encode_full_grid"):
                grid_encoding = self.substrate._encode_full_grid(self.positions, self.affordances)
            else:
                # Fallback for substrates without _encode_full_grid (e.g., aspatial)
                grid_encoding = self.substrate.encode_observation(self.positions, self.affordances)
            # Full obs uses "grid_encoding" variable name
            self.vfs_registry.set("grid_encoding", grid_encoding, writer="engine")

        # Position (normalized for POMDP, substrate-specific for full obs)
        if self.partial_observability:
            normalized_positions = self.substrate.normalize_positions(self.positions)
            self.vfs_registry.set("position", normalized_positions, writer="engine")
        else:
            self.vfs_registry.set("position", self.positions.float(), writer="engine")

        # Meters (write each meter individually)
        for meter_idx, meter_name in enumerate(self.meter_name_to_index.keys()):
            self.vfs_registry.set(meter_name, self.meters[:, meter_idx], writer="engine")

        # Affordance encoding (one-hot of current affordance)
        affordance_encoding = self._build_affordance_encoding()
        self.vfs_registry.set("affordance_at_position", affordance_encoding, writer="engine")

        # Temporal features
        time_of_day = self.time_of_day if self.enable_temporal_mechanics else 0
        time_angle = (time_of_day / 24.0) * 2 * math.pi
        time_sin = torch.full((self.num_agents,), math.sin(time_angle), device=self.device)
        time_cos = torch.full((self.num_agents,), math.cos(time_angle), device=self.device)

        self.vfs_registry.set("time_sin", time_sin, writer="engine")
        self.vfs_registry.set("time_cos", time_cos, writer="engine")

        if self.enable_temporal_mechanics and self.interaction_progress is not None:
            normalized_progress = self.interaction_progress.float() / 10.0
        else:
            normalized_progress = torch.zeros(self.num_agents, device=self.device)

        self.vfs_registry.set("interaction_progress", normalized_progress, writer="engine")
        self.vfs_registry.set("lifetime_progress", lifetime_progress, writer="engine")

        # Build observations from VFS registry according to observation spec
        observations = []
        for field in self.vfs_observation_spec:
            value = self.vfs_registry.get(field.source_variable, reader="agent")

            # Apply normalization if specified
            if field.normalization:
                if field.normalization.kind == "minmax":
                    min_val = field.normalization.min
                    max_val = field.normalization.max
                    # Convert to tensors if not already
                    if not isinstance(min_val, torch.Tensor):
                        min_val = torch.tensor(min_val, device=self.device, dtype=value.dtype)
                    if not isinstance(max_val, torch.Tensor):
                        max_val = torch.tensor(max_val, device=self.device, dtype=value.dtype)
                    value = (value - min_val) / (max_val - min_val + 1e-8)  # Add epsilon to avoid division by zero
                elif field.normalization.kind == "zscore":
                    mean = field.normalization.mean
                    std = field.normalization.std
                    # Convert to tensors if not already
                    if not isinstance(mean, torch.Tensor):
                        mean = torch.tensor(mean, device=self.device, dtype=value.dtype)
                    if not isinstance(std, torch.Tensor):
                        std = torch.tensor(std, device=self.device, dtype=value.dtype)
                    value = (value - mean) / (std + 1e-8)  # Add epsilon to avoid division by zero

            # Ensure 2D shape [num_agents, *]
            if value.ndim == 1:
                value = value.unsqueeze(1)

            observations.append(value)

        return torch.cat(observations, dim=1)

    def _build_affordance_encoding(self) -> torch.Tensor:
        """Build one-hot encoding of current affordance under each agent.

        This encodes against the FULL affordance vocabulary (from affordances.yaml),
        not just deployed affordances. This ensures observation dimensions stay
        constant across curriculum levels.

        Returns:
            encoding: [num_agents, num_affordance_types + 1]
                Last dimension is "none" (not on any affordance)
        """
        # Initialize with "none" (all zeros except last column)
        affordance_encoding = torch.zeros(self.num_agents, self.num_affordance_types + 1, device=self.device)
        affordance_encoding[:, -1] = 1.0  # Default to "none"

        # Iterate over FULL affordance vocabulary (not just deployed)
        # This ensures consistent encoding across curriculum levels
        for affordance_idx, affordance_name in enumerate(self.affordance_names):
            # Check if this affordance is DEPLOYED (has position on grid)
            if affordance_name in self.affordances:
                affordance_pos = self.affordances[affordance_name]
                # Check which agents are on affordance (using substrate)
                on_affordance = self.substrate.is_on_position(self.positions, affordance_pos)
                if on_affordance.any():
                    affordance_encoding[on_affordance, -1] = 0.0  # Clear "none"
                    affordance_encoding[on_affordance, affordance_idx] = 1.0
            # If affordance NOT deployed, agent can never be "on" it, stays as "none"

        return affordance_encoding

    def get_action_masks(self) -> torch.Tensor:
        """
        Get action masks for all agents (invalid actions = False).

        Action masking prevents agents from selecting movements that would
        take them off the grid. This saves exploration budget and speeds learning.

        Returns:
            action_masks: [num_agents, action_dim] bool tensor
                True = valid action, False = invalid
                Grid2D (6 actions): [UP, DOWN, LEFT, RIGHT, INTERACT, WAIT]
                Grid3D (8 actions): [UP, DOWN, LEFT, RIGHT, UP_Z, DOWN_Z, INTERACT, WAIT]
        """
        # Start with base mask (disabled actions = False)
        action_masks = self.action_space.get_base_action_mask(
            num_agents=self.num_agents,
            device=self.device,
        )

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

            if self.up_z_action_idx is not None:
                action_masks[at_ceiling, self.up_z_action_idx] = False  # Can't go UP_Z at ceiling
            if self.down_z_action_idx is not None:
                action_masks[at_floor, self.down_z_action_idx] = False  # Can't go DOWN_Z at floor

        # Mask INTERACT - only valid when on an open affordance
        # P1.4: Removed affordability check - agents can attempt INTERACT even when broke
        # Affordability is checked inside interaction handlers; failing to afford just
        # wastes a turn (passive decay) and teaches economic planning

        # Use cached INTERACT index (from ActionSpaceBuilder)
        interact_action_idx = self.interact_action_idx

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
        # === CUSTOM ACTION DISPATCH (early) ===
        # Custom actions start after substrate actions
        custom_action_start_id = self.action_space.substrate_action_count
        custom_mask = actions >= custom_action_start_id

        if custom_mask.any():
            custom_agent_indices = torch.where(custom_mask)[0]
            for agent_idx in custom_agent_indices:
                action_id = int(actions[agent_idx].item())
                action = self.action_space.get_action_by_id(action_id)

                # Apply custom action costs/effects/teleportation
                self._apply_custom_action(agent_idx, action)

        # Store old positions for temporal mechanics progress tracking
        old_positions = self.positions.clone() if self.enable_temporal_mechanics else None

        # Apply movement using pre-built delta tensor from ActionConfig
        # Only for substrate actions (custom actions already handled above)
        substrate_mask = actions < custom_action_start_id
        if substrate_mask.any():
            movement_deltas = self._movement_deltas[actions[substrate_mask]]  # [num_substrate_agents, position_dim]
            self.positions[substrate_mask] = self.substrate.apply_movement(self.positions[substrate_mask], movement_deltas)

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
        # Only apply to substrate actions (not custom actions)
        movement_actions = self._movement_deltas.ne(0).any(dim=1)
        # Create a full mask (initialize to False for all agents)
        movement_mask = torch.zeros(self.num_agents, dtype=torch.bool, device=self.device)
        # Only check movement for substrate actions
        if substrate_mask.any():
            movement_mask[substrate_mask] = movement_actions[actions[substrate_mask]]
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
        # Use cached WAIT index (from ActionSpaceBuilder)
        wait_action_idx = self.wait_action_idx

        wait_mask = (actions == wait_action_idx) & substrate_mask
        if wait_mask.any():
            # TASK-001: Create dynamic cost tensor based on meter_count
            wait_costs = torch.zeros(self.meter_count, device=self.device)
            wait_costs[self.energy_idx] = self.wait_energy_cost  # Energy (configurable, default 0.1%)

            self.meters[wait_mask] -= wait_costs.unsqueeze(0)
            self.meters = torch.clamp(self.meters, 0.0, 1.0)

        # Handle INTERACT actions
        # Use cached INTERACT index (from ActionSpaceBuilder)
        interact_action_idx = self.interact_action_idx

        successful_interactions = {}
        interact_mask = (actions == interact_action_idx) & substrate_mask
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

    def _get_meter_index(self, meter_name: str) -> int | None:
        """Get meter index by name.

        Args:
            meter_name: Meter name (e.g., "energy", "mood")

        Returns:
            Meter index, or None if meter doesn't exist
        """
        mapping: dict[str, int] | None = getattr(self, "meter_name_to_index", None)
        if mapping is None:
            return None
        result: int | None = mapping.get(meter_name)
        return result

    def _apply_custom_action(self, agent_idx: int, action: ActionConfig):
        """Apply custom action effects, movement delta, and teleportation.

        Args:
            agent_idx: Agent index
            action: Custom action config
        """
        # Apply costs (negative costs = restoration)
        for meter_name, cost in action.costs.items():
            meter_idx = self._get_meter_index(meter_name)
            if meter_idx is not None:
                self.meters[agent_idx, meter_idx] -= cost  # Subtract cost (negative = add)

        # Apply effects
        for meter_name, effect in action.effects.items():
            meter_idx = self._get_meter_index(meter_name)
            if meter_idx is not None:
                self.meters[agent_idx, meter_idx] += effect  # Add effect

        # Apply movement delta (for movement-type custom actions like SPRINT)
        if action.delta is not None:
            delta_tensor = torch.tensor(action.delta, device=self.device, dtype=self.substrate.position_dtype)
            new_position = self.substrate.apply_movement(self.positions[agent_idx : agent_idx + 1], delta_tensor.unsqueeze(0))
            self.positions[agent_idx] = new_position.squeeze(0)

        # Handle teleportation (overrides movement delta if both present)
        if action.teleport_to is not None:
            target_pos = torch.tensor(
                action.teleport_to,
                device=self.device,
                dtype=self.substrate.position_dtype,
            )
            self.positions[agent_idx] = target_pos

        # Clamp meters to [0, 1]
        self.meters = torch.clamp(self.meters, 0.0, 1.0)

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
