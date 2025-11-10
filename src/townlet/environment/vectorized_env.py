"""
Vectorized Hamlet environment for GPU-native training.

Batches multiple independent Hamlet environments into a single vectorized
environment with tensor operations [num_agents, ...].
"""

from __future__ import annotations

import random
from collections.abc import Callable
from copy import deepcopy
from numbers import Number
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch

from townlet.environment.action_builder import ComposedActionSpace
from townlet.environment.affordance_config import AffordanceConfig, AffordanceConfigCollection
from townlet.environment.affordance_engine import AffordanceEngine
from townlet.environment.meter_dynamics import MeterDynamics
from townlet.environment.reward_strategy import RewardStrategy
from townlet.substrate.continuous import ContinuousSubstrate
from townlet.vfs.registry import VariableRegistry

if TYPE_CHECKING:
    from townlet.environment.action_config import ActionConfig, ActionSpaceConfig
    from townlet.population.runtime_registry import AgentRuntimeRegistry
    from townlet.universe.compiled import CompiledUniverse


def _build_affordance_collection(raw_affordances: tuple[Any, ...]) -> AffordanceConfigCollection:
    """Convert compiler affordance DTOs into runtime collection without touching disk."""

    affordance_entries: list[AffordanceConfig] = []
    for raw in raw_affordances:
        data = raw.model_dump()
        interaction_type = data.get("interaction_type")
        if not interaction_type:
            raise ValueError(f"Affordance '{raw.id}' missing interaction_type; required for runtime execution")
        operating_hours = data.get("operating_hours")
        if operating_hours is None:
            raise ValueError(f"Affordance '{raw.id}' missing operating_hours; runtime requires explicit hours")

        payload = {
            "id": raw.id,
            "name": raw.name,
            "category": data.get("category") or "unspecified",
            "interaction_type": interaction_type,
            "required_ticks": data.get("required_ticks"),
            "costs": data.get("costs") or [],
            "costs_per_tick": data.get("costs_per_tick") or [],
            "effects": data.get("effects") or [],
            "effects_per_tick": data.get("effects_per_tick") or [],
            "completion_bonus": data.get("completion_bonus") or [],
            "operating_hours": operating_hours,
            "teaching_note": data.get("teaching_note"),
            "design_intent": data.get("design_intent"),
            "position": data.get("position"),
        }
        affordance_entries.append(AffordanceConfig(**payload))

    return AffordanceConfigCollection(
        version="runtime",
        description="Compiled affordance set",
        status="COMPILED",
        affordances=affordance_entries,
    )


def _resolve_deployable_affordances(
    all_affordance_names: list[str],
    enabled_affordances: list[str] | None,
    name_to_id: dict[str, str],
) -> list[str]:
    """Return affordances that should be deployed, respecting IDs and names in config."""

    if enabled_affordances is None:
        return all_affordance_names

    enabled_lookup = {str(entry) for entry in enabled_affordances}
    deployable: list[str] = []
    for name in all_affordance_names:
        if name in enabled_lookup:
            deployable.append(name)
            continue
        aff_id = name_to_id.get(name)
        if aff_id is not None and aff_id in enabled_lookup:
            deployable.append(name)
    return deployable


class VectorizedHamletEnv:
    """
    GPU-native vectorized Hamlet environment.

    Batches multiple independent environments for parallel execution.
    All state is stored as PyTorch tensors on specified device.
    """

    def __init__(
        self,
        *,
        universe: CompiledUniverse,
        num_agents: int,
        device: torch.device | str = torch.device("cpu"),
    ):
        """
        Initialize vectorized environment.

        Args:
            universe: CompiledUniverse artifact produced by UniverseCompiler
            num_agents: Number of parallel agents to simulate
            device: PyTorch device or device string (defaults to CPU). Infrastructure default - PDR-002 exemption.

        Note (PDR-002 Compliance):
            - device retains an infrastructure default (exempted from no-defaults principle)
            - Behavioral parameters (grid size, observability, energy costs, affordance selection)
              now flow exclusively from the compiled universe
        """
        torch_device = torch.device(device) if isinstance(device, str) else device

        runtime = universe.to_runtime()
        self.runtime = runtime
        self.universe = universe
        self.config_pack_path = Path(universe.config_dir)
        self.num_agents = num_agents
        self.device = torch_device
        self.optimization_data = universe.optimization_data

        env_cfg = runtime.clone_environment_config()
        curriculum = runtime.clone_curriculum_config()
        enabled_affordances = env_cfg.enabled_affordances
        cascade_env_config = runtime.clone_environment_cascade_config()
        self.bars_config = cascade_env_config.bars
        randomize_setting = getattr(env_cfg, "randomize_affordances", None)
        if randomize_setting is None:
            raise ValueError(
                "training.environment.randomize_affordances must be explicitly specified (true/false). "
                "Implicit defaults are disallowed by PDR-002."
            )
        self.randomize_affordances = bool(randomize_setting)

        self.partial_observability = env_cfg.partial_observability
        self.vision_range = env_cfg.vision_range
        self.enable_temporal_mechanics = env_cfg.enable_temporal_mechanics
        self.move_energy_cost = env_cfg.energy_move_depletion
        self.wait_energy_cost = env_cfg.energy_wait_depletion
        self.interact_energy_cost = env_cfg.energy_interact_depletion
        self.agent_lifespan = curriculum.max_steps_per_episode
        partial_observability = self.partial_observability
        vision_range = self.vision_range

        from townlet.substrate.factory import SubstrateFactory

        self.substrate = SubstrateFactory.build(runtime.clone_substrate_config(), device=torch_device)

        from townlet.environment.action_labels import get_labels

        action_labels_config = runtime.clone_action_labels_config()
        if action_labels_config and action_labels_config.custom:
            self.action_labels = get_labels(
                custom_labels=action_labels_config.custom,
                substrate_position_dim=self.substrate.position_dim,
            )
        elif action_labels_config and action_labels_config.preset:
            self.action_labels = get_labels(preset=action_labels_config.preset, substrate_position_dim=self.substrate.position_dim)
        else:
            self.action_labels = get_labels(preset="gaming", substrate_position_dim=self.substrate.position_dim)

        self.metadata = runtime.metadata

        # Update grid_size from compiler metadata / substrate (handles aspatial vs grid)
        self.grid_size = self.metadata.grid_size or env_cfg.grid_size
        if hasattr(self.substrate, "width") and hasattr(self.substrate, "height"):
            if self.substrate.width != self.substrate.height:
                raise ValueError(f"Non-square grids not yet supported: {self.substrate.width}×{self.substrate.height}")
            self.grid_size = self.substrate.width  # Override with substrate for grid

        self.vfs_variables = [var.model_copy(deep=True) for var in runtime.variables_reference]
        self.vfs_observation_spec = [deepcopy(field) for field in runtime.vfs_observation_fields]

        computed_observation_dim = sum(field.shape[0] if field.shape else 1 for field in self.vfs_observation_spec)
        if computed_observation_dim != self.metadata.observation_dim:
            raise ValueError(
                f"Observation dimension mismatch between compiled metadata ({self.metadata.observation_dim}) "
                f"and VFS exposures ({computed_observation_dim})."
            )

        self.meter_count = self.metadata.meter_count
        meter_count = self.meter_count
        self.base_depletions = self.optimization_data.base_depletions.to(self.device)

        affordance_config = _build_affordance_collection(runtime.clone_affordance_configs())
        metadata_affordance_lookup = dict(self.metadata.affordance_id_to_index)
        self.affordance_name_to_id = {aff.name: aff.id for aff in affordance_config.affordances}
        self.affordance_name_to_mask_idx = {
            name: metadata_affordance_lookup.get(aff_id)
            for name, aff_id in self.affordance_name_to_id.items()
            if metadata_affordance_lookup.get(aff_id) is not None
        }
        self.affordance_positions_from_config = {aff.name: aff.position for aff in affordance_config.affordances}
        optimization_position_map = getattr(self.optimization_data, "affordance_position_map", {})
        self.affordance_positions_from_optimization = {
            name: optimization_position_map.get(aff_id) for name, aff_id in self.affordance_name_to_id.items()
        }

        # Extract ALL affordance names from YAML (defines observation vocabulary)
        # This is the FULL universe - what the agent can observe and reason about
        all_affordance_names = [aff.name for aff in affordance_config.affordances]

        # Filter affordances for DEPLOYMENT (which ones actually exist on the grid)
        # enabled_affordances from training.yaml controls what the agent can interact with
        affordance_names_to_deploy = _resolve_deployable_affordances(
            all_affordance_names,
            enabled_affordances,
            self.affordance_name_to_id,
        )

        # DEPLOYED affordances: have positions on grid, can be interacted with
        # Positions will be randomized by randomize_affordance_positions() before first use
        default_position = torch.zeros(self.substrate.position_dim, dtype=self.substrate.position_dtype, device=self.device)
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
                f"\n  - observation_encoding='scaled': {self.substrate.position_dim * 2} dims (coordinates + grid sizes)"
                f"\n  - Enables dimension-independent learning WITHOUT exponential curse"
                f"\n\nSee docs/manual/pomdp_compatibility_matrix.md for details."
            )

        # Validate Grid3D POMDP vision range (prevent memory explosion)
        if partial_observability and self.substrate.position_dim == 3:
            window_volume = (2 * vision_range + 1) ** 3
            if window_volume > 125:  # 5×5×5 = 125 is the threshold
                raise ValueError(
                    f"Grid3D POMDP with vision_range={vision_range} requires {window_volume} cells "
                    f"(window size {2 * vision_range + 1}×{2 * vision_range + 1}×{2 * vision_range + 1}), which is excessive. "
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

        self.observation_dim = self.metadata.observation_dim

        # VFS INTEGRATION: Initialize variable registry
        self.vfs_registry = VariableRegistry(variables=self.vfs_variables, num_agents=num_agents, device=self.device)

        # Initialize reward strategy (TASK-001: variable meters)
        meter_name_to_index = dict(self.metadata.meter_name_to_index)
        self.meter_name_to_index = meter_name_to_index
        self.energy_idx = meter_name_to_index.get("energy", 0)  # Default to 0 if not found
        self.health_idx = meter_name_to_index.get("health", min(6, meter_count - 1))  # Default to 6 or last meter
        self.hygiene_idx = meter_name_to_index.get("hygiene", None)  # Optional meter
        self.satiation_idx = meter_name_to_index.get("satiation", None)  # Optional meter
        self.money_idx = meter_name_to_index.get("money", None)  # Optional meter

        self.reward_strategy = RewardStrategy(
            device=self.device, num_agents=num_agents, meter_count=meter_count, energy_idx=self.energy_idx, health_idx=self.health_idx
        )
        self.runtime_registry: AgentRuntimeRegistry | None = None  # Injected by population/inference controllers

        # Precompute meter initialization tensor from bars config
        self.initial_meter_values = torch.zeros(meter_count, dtype=torch.float32, device=self.device)
        for bar in self.bars_config.bars:
            self.initial_meter_values[bar.index] = bar.initial

        # Build terminal conditions lookup once (compiler already validated names)
        terminal_specs: list[dict[str, Any]] = []
        for condition in self.bars_config.terminal_conditions:
            meter_idx = meter_name_to_index.get(condition.meter)
            if meter_idx is None:
                continue
            terminal_specs.append(
                {
                    "meter_idx": meter_idx,
                    "operator": condition.operator,
                    "value": condition.value,
                }
            )

        # Initialize meter dynamics directly from optimization tensors
        self.meter_dynamics = MeterDynamics(
            base_depletions=self.optimization_data.base_depletions,
            cascade_data=self.optimization_data.cascade_data,
            modulation_data=self.optimization_data.modulation_data,
            terminal_conditions=terminal_specs,
            meter_name_to_index=meter_name_to_index,
            device=self.device,
        )

        # Cache action mask table (24 × affordance_count) for temporal mechanics
        self.action_mask_table = self.optimization_data.action_mask_table.to(self.device).clone()
        self.hours_per_day = self.action_mask_table.shape[0] if self.action_mask_table.ndim > 0 else 24

        # Initialize affordance engine (reuse affordance_config loaded above)
        # Pass meter_name_to_index for dynamic meter lookups (TASK-001)
        self.affordance_engine = AffordanceEngine(
            affordance_config,
            num_agents,
            self.device,
            self.meter_name_to_index,
        )

        # Build composed action space from substrate + global custom actions
        # Compiler metadata already encodes enabled flags from training.enabled_actions
        action_metadata = universe.action_space_metadata
        enabled_names = [action.name for action in action_metadata.actions if action.enabled]
        enabled_param = None if len(enabled_names) == action_metadata.total_actions else enabled_names

        global_actions = runtime.clone_global_actions()
        self.action_space = self._compose_action_space(global_actions, enabled_param)
        self.action_dim = self.metadata.action_count
        if self.action_space.action_dim != self.action_dim:
            raise ValueError(
                f"Action dimension mismatch between compiled metadata ({self.action_dim}) "
                f"and composed action space ({self.action_space.action_dim})."
            )

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

        # TASK-004B Phase B: Cooldown and prerequisite tracking
        # cooldown_state[affordance_name] = tensor of tick when each agent can use this affordance again
        # Value of 0 means ready now, >0 means must wait until global_tick >= cooldown_state
        self.cooldown_state: dict[str, torch.Tensor] = {}

        # completed_affordances[affordance_name] = boolean tensor of which agents have completed this affordance
        # Used for prerequisite capability validation
        self.completed_affordances: dict[str, torch.Tensor] = {}

        # TASK-004B Phase C: Resumable multi-tick progress tracking
        # saved_progress[affordance_name] = tensor of saved progress (ticks completed) for each agent
        # Used when affordance has resumable=true capability to persist progress when interrupted
        self.saved_progress: dict[str, torch.Tensor] = {}

        # Initialize cooldown, completion, and saved progress tracking for all affordances
        for affordance_name in all_affordance_names:
            self.cooldown_state[affordance_name] = torch.zeros(self.num_agents, dtype=torch.long, device=self.device)
            self.completed_affordances[affordance_name] = torch.zeros(self.num_agents, dtype=torch.bool, device=self.device)
            self.saved_progress[affordance_name] = torch.zeros(self.num_agents, dtype=torch.long, device=self.device)

        # Global tick counter for cooldown management (persists across episodes)
        self.global_tick = 0

        # Initialize affordance positions per configuration
        if self.randomize_affordances:
            self.randomize_affordance_positions()
        else:
            self._apply_configured_affordance_positions()

    def attach_runtime_registry(self, registry: AgentRuntimeRegistry) -> None:
        """Attach runtime registry for telemetry tracking."""
        self.runtime_registry = registry

    def _get_optional_action_idx(self, action_name: str) -> int | None:
        """Return action index if available in composed action space."""
        try:
            return self.action_space.get_action_by_name(action_name).id
        except ValueError:
            return None

    def _apply_configured_affordance_positions(self) -> None:
        """Load static affordance positions from config/optimization data."""

        if self.substrate.position_dim == 0:
            empty = torch.zeros(0, dtype=self.substrate.position_dtype, device=self.device)
            for name in self.affordances.keys():
                self.affordances[name] = empty.clone()
            return

        for name in self.affordances.keys():
            source = self.affordance_positions_from_config.get(name)
            if source is None:
                optimization_tensor = self.affordance_positions_from_optimization.get(name)
                if isinstance(optimization_tensor, torch.Tensor):
                    source = optimization_tensor.tolist()
                else:
                    source = optimization_tensor

            tensor = self._position_to_tensor(source, name)
            self.affordances[name] = tensor

    def _position_to_tensor(self, raw_position: Any, affordance_name: str) -> torch.Tensor:
        """Convert raw config/optimization positions into substrate tensors."""

        if self.substrate.position_dim == 0:
            return torch.zeros(0, dtype=self.substrate.position_dtype, device=self.device)

        if raw_position is None:
            raise ValueError(f"Affordance '{affordance_name}' requires explicit position when randomize_affordances is disabled.")

        if isinstance(raw_position, torch.Tensor):
            tensor = raw_position.to(device=self.device, dtype=self.substrate.position_dtype)
        elif isinstance(raw_position, dict):
            if set(raw_position.keys()) == {"q", "r"}:
                coords = [raw_position["q"], raw_position["r"]]
            else:
                raise ValueError(
                    f"Affordance '{affordance_name}' provided unsupported position mapping keys: {sorted(raw_position.keys())}."
                )
            tensor = torch.tensor(coords, dtype=self.substrate.position_dtype, device=self.device)
        elif isinstance(raw_position, list | tuple):
            tensor = torch.tensor(list(raw_position), dtype=self.substrate.position_dtype, device=self.device)
        elif isinstance(raw_position, Number):
            tensor = torch.tensor([raw_position], dtype=self.substrate.position_dtype, device=self.device)
        else:
            raise ValueError(f"Affordance '{affordance_name}' provided unsupported position type: {type(raw_position)!r}.")

        if tensor.numel() != self.substrate.position_dim:
            raise ValueError(
                f"Affordance '{affordance_name}' position has {tensor.numel()} dims but substrate requires {self.substrate.position_dim}."
            )

        return tensor

    def _is_affordance_open(self, affordance_name: str, hour: int | None = None) -> bool:
        """
        Return True if an affordance is open for the specified (or current) hour.

        TASK-004B Phase E: Supports mode-based hours if defined.
        """

        if not self.enable_temporal_mechanics:
            return True

        # TASK-004B Phase E: Check modes if defined
        affordance_config = self.affordance_engine.affordance_map.get(affordance_name)
        if affordance_config is not None and hasattr(affordance_config, "modes") and affordance_config.modes:
            # Modes defined - check if any mode is active for current hour
            active_hour = self.time_of_day if hour is None else hour

            for mode_name, mode_config in affordance_config.modes.items():
                if hasattr(mode_config, "hours") and mode_config.hours is not None:
                    start_hour, end_hour = mode_config.hours

                    # Handle midnight wrap (e.g., 18-2 means 6pm to 2am)
                    if start_hour <= end_hour:
                        # Normal range (e.g., 9-17 means 9am to 5pm)
                        if start_hour <= active_hour <= end_hour:
                            return True
                    else:
                        # Wraps midnight (e.g., 18-2 means 6pm to 2am next day)
                        if active_hour >= start_hour or active_hour <= end_hour:
                            return True

            # No mode is active for this hour
            return False

        # Fallback to action_mask_table (precomputed operating_hours)
        if self.action_mask_table.shape[1] == 0:
            return False

        idx = self.affordance_name_to_mask_idx.get(affordance_name)
        if idx is None or idx >= self.action_mask_table.shape[1]:
            # Missing metadata should not block interactions
            return True

        active_hour = self.time_of_day if hour is None else hour
        hour_idx = active_hour % self.hours_per_day
        return bool(self.action_mask_table[hour_idx, idx].item())

    def _compose_action_space(
        self,
        global_actions: ActionSpaceConfig,
        enabled_action_names: list[str] | None,
    ) -> ComposedActionSpace:
        enabled_set = None if enabled_action_names is None else set(enabled_action_names)
        cloned_actions = [action.model_copy(deep=True) for action in global_actions.actions]
        for action in cloned_actions:
            action.enabled = True if enabled_set is None else action.name in enabled_set

        substrate_count = sum(1 for action in cloned_actions if action.source == "substrate")
        custom_count = sum(1 for action in cloned_actions if action.source == "custom")
        affordance_count = sum(1 for action in cloned_actions if action.source == "affordance")

        return ComposedActionSpace(
            actions=cloned_actions,
            substrate_action_count=substrate_count,
            custom_action_count=custom_count,
            affordance_action_count=affordance_count,
            enabled_action_names=enabled_set,
        )

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
        # Refresh affordance layout each episode so randomization/configured layouts stay in sync
        if self.randomize_affordances:
            self.randomize_affordance_positions()
        else:
            self._apply_configured_affordance_positions()

        # Use substrate for position initialization (supports grid and aspatial)
        self.positions = self.substrate.initialize_positions(self.num_agents, self.device)

        # Initial meter values (normalized to [0, 1]) from compiled bars config
        self.meters = self.initial_meter_values.unsqueeze(0).expand(self.num_agents, -1).clone()

        self.dones = torch.zeros(self.num_agents, dtype=torch.bool, device=self.device)
        self.step_counts = torch.zeros(self.num_agents, dtype=torch.long, device=self.device)

        # Reset temporal mechanics state
        if self.enable_temporal_mechanics:
            self.time_of_day = 0
            self.interaction_progress.fill_(0)
            self.last_interaction_affordance = [None] * self.num_agents
            self.last_interaction_position.fill_(0)

        # TASK-004B Phase B: Reset cooldown and prerequisite tracking
        # Reset global tick counter at episode start
        self.global_tick = 0

        # Reset all cooldown timers
        for affordance_name in self.cooldown_state:
            self.cooldown_state[affordance_name].fill_(0)

        # Reset all prerequisite completion tracking
        for affordance_name in self.completed_affordances:
            self.completed_affordances[affordance_name].fill_(False)

        # TASK-004B Phase C: Reset saved progress for resumable interactions
        for affordance_name in self.saved_progress:
            self.saved_progress[affordance_name].fill_(0)

        return self._get_observations()

    @classmethod
    def from_universe(
        cls,
        universe: CompiledUniverse,
        *,
        num_agents: int,
        device: torch.device | str = "cpu",
    ) -> VectorizedHamletEnv:
        """Instantiate environment using metadata from a compiled universe."""

        torch_device = torch.device(device) if isinstance(device, str) else device

        return cls(
            universe=universe,
            num_agents=num_agents,
            device=torch_device,
        )

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
            # Full obs uses "grid_encoding" variable name (if present in VFS config)
            if "grid_encoding" in self.vfs_registry._definitions:
                self.vfs_registry.set("grid_encoding", grid_encoding, writer="engine")

        # Position features (respect substrate observation_encoding/payload)
        position_features = self._encode_position_observation()
        if position_features is not None:
            self.vfs_registry.set("position", position_features, writer="engine")

        # Meters (write each meter individually)
        for meter_name, meter_idx in self.meter_name_to_index.items():
            self.vfs_registry.set(meter_name, self.meters[:, meter_idx], writer="engine")

        # Affordance encoding (one-hot of current affordance)
        affordance_encoding = self._build_affordance_encoding()
        self.vfs_registry.set("affordance_at_position", affordance_encoding, writer="engine")

        # Temporal features
        time_of_day = self.time_of_day if self.enable_temporal_mechanics else 0
        time_angle = (time_of_day / 24.0) * 2 * math.pi
        time_sin = torch.tensor(math.sin(time_angle), device=self.device)
        time_cos = torch.tensor(math.cos(time_angle), device=self.device)

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
        var_defs = self.vfs_registry.variables
        for field in self.vfs_observation_spec:
            var_def = var_defs[field.source_variable]

            if "agent" not in var_def.readable_by:
                raise PermissionError(
                    f"Variable '{field.source_variable}' is exposed to agents but not readable_by them. "
                    "Update variables_reference.yaml to include 'agent' in readable_by."
                )

            reader_role = "engine" if var_def.scope == "agent_private" else "agent"
            value = self.vfs_registry.get(field.source_variable, reader=reader_role)

            if var_def.scope == "agent_private" and value.shape[0] != self.num_agents:
                raise ValueError(
                    f"agent_private variable '{field.source_variable}' must have leading dimension num_agents="
                    f"{self.num_agents}, got shape {tuple(value.shape)}"
                )

            # Broadcast global scalars to all agents
            if value.ndim == 0:
                value = value.expand(self.num_agents).clone()

            # Apply normalization if specified
            if field.normalization:
                if field.normalization.kind == "minmax":
                    min_val = field.normalization.min
                    max_val = field.normalization.max
                    # Convert to tensors if not already
                    if not isinstance(min_val, torch.Tensor):
                        min_val = torch.tensor(min_val, device=self.device, dtype=value.dtype)  # type: ignore[assignment]
                    if not isinstance(max_val, torch.Tensor):
                        max_val = torch.tensor(max_val, device=self.device, dtype=value.dtype)  # type: ignore[assignment]
                    value = (value - min_val) / (max_val - min_val + 1e-8)  # type: ignore[operator] # Add epsilon to avoid division by zero
                elif field.normalization.kind == "zscore":
                    mean = field.normalization.mean
                    std = field.normalization.std
                    # Convert to tensors if not already
                    if not isinstance(mean, torch.Tensor):
                        mean = torch.tensor(mean, device=self.device, dtype=value.dtype)  # type: ignore[assignment]
                    if not isinstance(std, torch.Tensor):
                        std = torch.tensor(std, device=self.device, dtype=value.dtype)  # type: ignore[assignment]
                    value = (value - mean) / (std + 1e-8)  # type: ignore[operator] # Add epsilon to avoid division by zero

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

    def _encode_position_observation(self) -> torch.Tensor | None:
        """Encode position variable using substrate-native encoding metadata."""
        if "position" not in self.vfs_registry._definitions:
            return None

        # Aspatial substrates have no positional encoding
        if getattr(self.substrate, "position_dim", 0) == 0:
            return None

        encode_fn = Callable[[torch.Tensor, dict[str, torch.Tensor]], torch.Tensor]

        encoder = getattr(self.substrate, "_encode_position_features", None)
        if callable(encoder):
            typed_encoder = cast(encode_fn, encoder)
            return typed_encoder(self.positions, self.affordances)

        public_encoder = getattr(self.substrate, "encode_position_features", None)
        if callable(public_encoder):
            typed_public = cast(encode_fn, public_encoder)
            return typed_public(self.positions, self.affordances)

        encode_observation = getattr(self.substrate, "encode_observation", None)
        if callable(encode_observation):
            typed_encode_obs = cast(encode_fn, encode_observation)
            return typed_encode_obs(self.positions, self.affordances)

        normalizer = getattr(self.substrate, "normalize_positions", None)
        if callable(normalizer):
            typed_normalizer = cast(Callable[[torch.Tensor], torch.Tensor], normalizer)
            return typed_normalizer(self.positions)

        return None

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
            if self.enable_temporal_mechanics and not self._is_affordance_open(affordance_name):
                continue

            on_this_affordance = self.substrate.is_on_position(self.positions, affordance_pos)
            on_valid_affordance |= on_this_affordance

        base_interact_mask = action_masks[:, interact_action_idx].clone()
        # Respect config-disabled INTERACT entries by preserving the base mask.
        action_masks[:, interact_action_idx] = base_interact_mask & on_valid_affordance

        # TASK-004B Phase B: Mask INTERACT for cooldown and prerequisite constraints
        # For each affordance, check capabilities and mask accordingly
        for affordance_name, affordance_pos in self.affordances.items():
            # Get affordance config from engine
            affordance_config = self.affordance_engine.affordance_map.get(affordance_name)
            if affordance_config is None:
                continue

            # Identify agents on this affordance
            agents_on_affordance = self.substrate.is_on_position(self.positions, affordance_pos)

            # Check cooldown capability
            cooldown_cap = self.affordance_engine._get_capability(affordance_config, "cooldown")
            if cooldown_cap is not None:
                # Mask INTERACT for agents whose cooldown hasn't expired
                cooldown_tensor = self.cooldown_state.get(affordance_name)
                if cooldown_tensor is not None:
                    on_cooldown = self.global_tick < cooldown_tensor
                    # Mask agents that are on this affordance AND on cooldown
                    agents_to_mask = agents_on_affordance & on_cooldown
                    action_masks[agents_to_mask, interact_action_idx] = False

            # Check prerequisite capability
            prereq_cap = self.affordance_engine._get_capability(affordance_config, "prerequisite")
            if prereq_cap is not None and hasattr(prereq_cap, "required_affordances"):
                # Check if all required affordances have been completed by each agent
                has_all_prereqs = torch.ones(self.num_agents, dtype=torch.bool, device=self.device)

                for required_aff_id in prereq_cap.required_affordances:
                    # Find the affordance name from ID (prerequisite uses IDs, not names)
                    required_aff_name = None
                    for aff_name, aff_id in self.affordance_name_to_id.items():
                        if aff_id == required_aff_id:
                            required_aff_name = aff_name
                            break

                    if required_aff_name is not None:
                        completed_tensor = self.completed_affordances.get(required_aff_name)
                        if completed_tensor is not None:
                            # Agent must have completed this prerequisite
                            has_all_prereqs &= completed_tensor

                # Mask agents that are on this affordance BUT lack prerequisites
                agents_to_mask = agents_on_affordance & ~has_all_prereqs
                action_masks[agents_to_mask, interact_action_idx] = False

            # TASK-004B Phase D: Check meter_gated capability
            meter_gated_cap = self.affordance_engine._get_capability(affordance_config, "meter_gated")
            if meter_gated_cap is not None and hasattr(meter_gated_cap, "meter"):
                # Get meter index
                meter_name = meter_gated_cap.meter
                meter_idx = self.meter_name_to_index.get(meter_name)

                if meter_idx is not None:
                    # Get meter values for all agents
                    meter_values = self.meters[:, meter_idx]

                    # Check bounds (at least one must be specified per DTO validation)
                    within_bounds = torch.ones(self.num_agents, dtype=torch.bool, device=self.device)

                    if hasattr(meter_gated_cap, "min") and meter_gated_cap.min is not None:
                        # Meter must be >= min
                        within_bounds &= (meter_values >= meter_gated_cap.min)

                    if hasattr(meter_gated_cap, "max") and meter_gated_cap.max is not None:
                        # Meter must be <= max
                        within_bounds &= (meter_values <= meter_gated_cap.max)

                    # Mask agents that are on this affordance BUT outside meter bounds
                    agents_to_mask = agents_on_affordance & ~within_bounds
                    action_masks[agents_to_mask, interact_action_idx] = False

            # TASK-004B Phase E: Check availability constraints (BarConstraint list)
            if hasattr(affordance_config, "availability") and affordance_config.availability:
                # All availability constraints must be satisfied
                satisfies_availability = torch.ones(self.num_agents, dtype=torch.bool, device=self.device)

                for bar_constraint in affordance_config.availability:
                    meter_name = bar_constraint.meter
                    meter_idx = self.meter_name_to_index.get(meter_name)

                    if meter_idx is not None:
                        meter_values = self.meters[:, meter_idx]

                        # Check min bound
                        if hasattr(bar_constraint, "min") and bar_constraint.min is not None:
                            satisfies_availability &= (meter_values >= bar_constraint.min)

                        # Check max bound
                        if hasattr(bar_constraint, "max") and bar_constraint.max is not None:
                            satisfies_availability &= (meter_values <= bar_constraint.max)

                # Mask agents that don't satisfy availability constraints
                agents_to_mask = agents_on_affordance & ~satisfies_availability
                action_masks[agents_to_mask, interact_action_idx] = False

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
        # TASK-004B Phase B: Increment global tick counter for cooldown management
        self.global_tick += 1

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

        # TASK-004B Phase C: Handle early exit and resumable for agents that moved away
        if self.enable_temporal_mechanics and old_positions is not None:
            for agent_idx in range(self.num_agents):
                if not torch.equal(old_positions[agent_idx], self.positions[agent_idx]):
                    # Agent moved - check if they were in a multi-tick interaction
                    affordance_name = self.last_interaction_affordance[agent_idx]
                    ticks_completed = int(self.interaction_progress[agent_idx].item())

                    if affordance_name is not None and ticks_completed > 0:
                        # Handle early exit (apply effects, save progress if resumable)
                        self._handle_early_exit(agent_idx, affordance_name, ticks_completed)

                    # Reset current interaction state (saved progress preserved in _handle_early_exit if resumable)
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

    def _handle_early_exit(self, agent_idx: int, affordance_name: str, ticks_completed: int) -> None:
        """
        Handle early exit from multi-tick interaction (TASK-004B Phase C).

        When agent leaves before completing a multi-tick interaction:
        1. Check if affordance has early_exit_allowed capability
        2. If yes, apply on_early_exit effects from effect pipeline
        3. Check if affordance has resumable capability
        4. If resumable, save progress for later resumption
        5. If not resumable, progress is lost (handled by caller)

        Args:
            agent_idx: Agent leaving the interaction
            affordance_name: Name of the affordance being exited
            ticks_completed: Number of ticks completed before exit
        """
        # Get affordance config
        affordance_config = self.affordance_engine.affordance_map.get(affordance_name)
        if affordance_config is None:
            return

        # Check if this is a multi-tick interaction
        multi_tick_cap = self.affordance_engine._get_capability(affordance_config, "multi_tick")
        if multi_tick_cap is None:
            return

        # Only process early exit if ticks were completed but not finished
        required_ticks = getattr(multi_tick_cap, "duration_ticks", 0)
        if ticks_completed == 0 or ticks_completed >= required_ticks:
            return  # Not an early exit scenario

        # Apply early_exit effects if allowed
        if getattr(multi_tick_cap, "early_exit_allowed", False):
            effect_pipeline = getattr(affordance_config, "effect_pipeline", None)
            if effect_pipeline is not None and hasattr(effect_pipeline, "on_early_exit"):
                # Create single-agent mask
                agent_mask = torch.zeros(self.num_agents, dtype=torch.bool, device=self.device)
                agent_mask[agent_idx] = True

                # Apply early_exit effects
                for effect in effect_pipeline.on_early_exit:
                    meter_idx = self.affordance_engine.meter_name_to_idx.get(effect.meter)
                    if meter_idx is not None:
                        self.meters[agent_mask, meter_idx] += effect.amount

                self.meters = torch.clamp(self.meters, 0.0, 1.0)

        # Save progress if resumable
        if getattr(multi_tick_cap, "resumable", False):
            if affordance_name in self.saved_progress:
                self.saved_progress[affordance_name][agent_idx] = ticks_completed

    def _track_affordance_completion(self, affordance_name: str, agent_mask: torch.Tensor) -> None:
        """
        Track affordance completion for prerequisite and cooldown capabilities.

        TASK-004B Phase B: After successful affordance completion:
        1. Mark affordance as completed for prerequisite tracking
        2. Set cooldown timer if affordance has cooldown capability

        Args:
            affordance_name: Name of completed affordance
            agent_mask: [num_agents] bool tensor indicating which agents completed it
        """
        # Mark completion for prerequisite tracking
        if affordance_name in self.completed_affordances:
            self.completed_affordances[affordance_name] |= agent_mask

        # Set cooldown timer if affordance has cooldown capability
        affordance_config = self.affordance_engine.affordance_map.get(affordance_name)
        if affordance_config is not None:
            cooldown_cap = self.affordance_engine._get_capability(affordance_config, "cooldown")
            if cooldown_cap is not None and hasattr(cooldown_cap, "cooldown_ticks"):
                cooldown_duration = cooldown_cap.cooldown_ticks
                # Set cooldown expiration time for agents that completed this affordance
                if affordance_name in self.cooldown_state:
                    # Agents on cooldown can't use this until global_tick >= cooldown_expiration
                    cooldown_expiration = self.global_tick + cooldown_duration
                    # Update only for agents in the mask
                    self.cooldown_state[affordance_name] = torch.where(
                        agent_mask,
                        torch.tensor(cooldown_expiration, dtype=torch.long, device=self.device),
                        self.cooldown_state[affordance_name],
                    )

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
            if not self._is_affordance_open(affordance_name):
                continue

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
                    # TASK-004B Phase C: Check for saved progress (resumable interactions)
                    saved = int(self.saved_progress.get(affordance_name, torch.zeros(self.num_agents, device=self.device))[agent_idx].item())

                    if saved > 0:
                        # Resume from saved progress
                        self.interaction_progress[agent_idx] = saved + 1  # Continue from next tick
                        # Clear saved progress (now active again)
                        if affordance_name in self.saved_progress:
                            self.saved_progress[affordance_name][agent_idx] = 0
                    else:
                        # New affordance - start from tick 1
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

                    # TASK-004B Phase C: Clear saved progress on normal completion
                    if affordance_name in self.saved_progress:
                        self.saved_progress[affordance_name][agent_idx] = 0

                    # TASK-004B Phase B: Track completion for cooldown and prerequisite
                    self._track_affordance_completion(affordance_name, single_agent_mask)

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
            if self.enable_temporal_mechanics and not self._is_affordance_open(affordance_name):
                continue

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

            # TASK-004B Phase B: Track completion for cooldown and prerequisite
            self._track_affordance_completion(affordance_name, at_affordance)

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

    def randomize_affordance_positions(self) -> None:
        """Randomize affordance positions using substrate-provided layouts."""

        if not self.randomize_affordances:
            return

        if not self.affordances:
            return

        # Aspatial universes have no coordinates; store empty tensors for consistency
        if self.substrate.position_dim == 0:
            empty = torch.zeros(0, dtype=self.substrate.position_dtype, device=self.device)
            for name in self.affordances.keys():
                self.affordances[name] = empty.clone()
            return

        try:
            all_positions = self.substrate.get_all_positions()
        except NotImplementedError:
            all_positions = None

        if all_positions:
            total_positions = len(all_positions)
            required_slots = len(self.affordances) + self.num_agents
            if required_slots > total_positions:
                raise ValueError(
                    f"Substrate exposes {total_positions} positions but {len(self.affordances)} affordances + "
                    f"{self.num_agents} agents require more space."
                )

            random.shuffle(all_positions)
            for idx, name in enumerate(self.affordances.keys()):
                tensor_pos = torch.tensor(
                    all_positions[idx],
                    dtype=self.substrate.position_dtype,
                    device=self.device,
                )
                self.affordances[name] = tensor_pos
            return

        # Continuous substrates expose infinite positions; sample using initializer
        sampled = self.substrate.initialize_positions(len(self.affordances), self.device)
        for idx, name in enumerate(self.affordances.keys()):
            self.affordances[name] = sampled[idx].clone()
