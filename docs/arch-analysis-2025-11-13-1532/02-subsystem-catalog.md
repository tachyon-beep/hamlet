# Subsystem Catalog - HAMLET Architecture

**Analysis Date**: November 13, 2025
**Scope**: Production subsystems in `src/townlet/`
**Purpose**: Detailed component-level architecture documentation

---

## Universe Compiler (UAC)

**Location:** `src/townlet/universe/`

**Responsibility:** Transforms YAML configuration packs into immutable, validated CompiledUniverse artifacts with GPU-optimized computation graphs and comprehensive provenance tracking.

**Key Components:**
- `compiler.py` (2542 lines) - Main 7-stage pipeline orchestrator with security validations
- `compiled.py` - CompiledUniverse frozen dataclass (immutable output artifact with drive_hash provenance)
- `symbol_table.py` - UniverseSymbolTable for cross-file entity registration and reference resolution
- `cues_compiler.py` - CuesCompiler sub-module for UI metadata validation (Stage 4 helper)
- `errors.py` - CompilationError/CompilationErrorCollector for batch diagnostic reporting
- `optimization.py` - OptimizationData container for pre-computed GPU tensors (depletions, cascades, action masks)
- `dto/*.py` - Metadata DTOs (UniverseMetadata, ObservationSpec, ActionMetadata, MeterMetadata, AffordanceMetadata)
- `adapters/vfs_adapter.py` - VFS↔Compiler integration layer (observation spec translation)
- `compiler_inputs.py` - RawConfigs aggregator for YAML→DTO parsing
- `compiler/__main__.py` - CLI entry point providing compile/inspect/validate commands

**Pipeline Architecture (7 Stages + Phase 0):**
- **Phase 0**: YAML syntax validation (grammar check before semantic analysis, prevents cryptic errors)
- **Stage 1**: Parse individual YAML files into Pydantic DTOs via RawConfigs loader
- **Stage 2**: Build symbol tables (register all meters, affordances, cascades, variables, actions for cross-validation)
- **Stage 3**: Resolve references (validate affordances→meters, DAC→bars/variables/affordances, enabled_affordances existence)
- **Stage 4**: Cross-validate constraints (meter sustainability, affordance capabilities, economic balance, position bounds, substrate-action compatibility)
- **Stage 5**: Compute metadata (observation specs via VFS integration, action/meter/affordance metadata, provenance IDs with git SHA/config hash)
- **Stage 6**: Optimize (pre-compute GPU tensors: base_depletions, cascade_data, modulation_data, action_mask_table[24×N], affordance_position_map)
- **Stage 7**: Emit CompiledUniverse (frozen dataclass with all metadata + drive_hash for checkpoint validation)

**Dependencies:**
- **Inbound** (who depends on UAC):
  - `demo/runner.py` (DemoRunner.compile_universe) - compiles config before training starts
  - `demo/live_inference.py` (LiveInference) - compiles config for inference server initialization
  - `compiler/__main__.py` - CLI tool for manual compilation/inspection/validation workflows
  - `environment/vectorized_env.py` - consumes CompiledUniverse artifact to initialize environment
  - `training/checkpoint_utils.py` - validates CompiledUniverse compatibility for checkpoint resume
- **Outbound** (what UAC depends on):
  - Config DTO layer: `config/{bar,cascade,affordance,drive_as_code,hamlet,exploration,curriculum,training}.py` (Pydantic schemas)
  - VFS subsystem: `vfs/{schema,registry,observation_builder}.py` for observation spec generation and variable definitions
  - DAC subsystem: `config/drive_as_code.py` for reward function configuration and drive_hash computation
  - Substrate layer: `substrate/config.py`, `environment/substrate_action_validator.py` for spatial topology validation
  - Environment layer: `environment/{cascade_config,action_config}.py` for meter relationships and action space construction

**Patterns Observed:**
- **7-Stage Pipeline**: Separates parsing → symbol resolution → cross-validation → metadata generation → optimization → emission (clean separation of concerns)
- **Error Collection Pattern**: CompilationErrorCollector batches all diagnostics before raising (shows complete error report instead of fail-fast)
- **Immutable Output Artifact**: CompiledUniverse is frozen dataclass with `__post_init__` validation (no post-compilation mutation possible)
- **Dual Cache Invalidation**: Fast mtime check first, then SHA256 config_hash + provenance_id for accuracy (avoids expensive hash on every compile)
- **Provenance Tracking**: Embeds config_hash, drive_hash, compiler_version, git_sha, python_version, torch_version for full reproducibility
- **Symbol Table Registry**: Stage 2 builds central entity registry used by Stages 3-7 for reference resolution (hub pattern)
- **Auto-Generation of Standard Variables**: Compiler generates spatial (position, velocity, grid_encoding, local_window), meter, affordance, and temporal variables from substrate/bars/affordances configs
- **GPU Tensor Pre-Computation**: Stage 6 optimizes runtime by moving constant computations to compile-time (depletions, cascades, action masks on GPU)
- **Security Validations**: Path traversal protection, cache bomb protection (10MB limit), DoS protection (max 10K grid cells, max 300 actions, max 100 meters)
- **CLI Tool Integration**: Provides `python -m townlet.compiler {compile,inspect,validate}` for config pack development workflow
- **No-Defaults Principle Enforcement**: All behavioral parameters must be explicit in YAML; compiler raises errors on missing required fields

**Concerns:**
- **Large Monolithic File**: compiler.py at 2542 lines handles all 7 stages in one module (could benefit from stage extraction into `stages/*.py` modules)
- **No Incremental Compilation**: Cache invalidation is all-or-nothing; any file change triggers full recompile (no dependency graph tracking)
- **Stage 6 Optimization Scope**: Currently pre-computes basic tensors; could optimize more complex DAC/cascade computation graphs
- **Error Code Documentation**: Error codes like "UAC-VAL-001" scattered throughout code with no central registry or docs/error-codes.md reference
- **Stage 4 Completeness**: Some validations marked as "subset of spec for TASK-004A" suggesting incomplete implementation of cross-validation rules
- **VFS Adapter Coupling**: VFS integration uses adapter pattern but still tightly coupled to compiler internals (vfs_adapter.py accesses many internal structures)

**Confidence:** High - Comprehensive analysis of 2542-line compiler with all 7 stages examined, dependencies mapped via grep/code inspection, CLI/cache/testing infrastructure understood, security validations identified.

---

## Variable & Feature System (VFS)

**Location:** `src/townlet/vfs/`

**Responsibility:** Declarative state space configuration system providing compile-time observation spec generation and runtime GPU tensor storage with access control enforcement.

**Key Components:**
- `schema.py` (307 lines) - Pydantic models: VariableDef (variable declarations), ObservationField (observation specs), NormalizationSpec (minmax/zscore normalization), WriteSpec (action write effects, Phase 2 placeholder)
- `registry.py` (280 lines) - VariableRegistry class for runtime GPU tensor storage with scope-aware shapes and access control enforcement (readable_by/writable_by validation)
- `observation_builder.py` (202 lines) - VFSObservationSpecBuilder class for compile-time spec generation (shapes inference, normalization validation, BAC integration)
- `__init__.py` (27 lines) - Public API exports

**Dependencies:**
- **Inbound** (who depends on VFS):
  - `universe/compiler.py` - Auto-generates standard variables (spatial, meters, affordances, temporal) from substrate/bars/affordances configs in Stage 2; validates variable references in Stage 3
  - `universe/adapters/vfs_adapter.py` - Converts VFS ObservationField specs to compiler ObservationSpec DTOs via vfs_to_observation_spec function
  - `universe/compiled.py` - Embeds list[VariableDef] in CompiledUniverse artifact for runtime initialization
  - `universe/symbol_table.py` - Registers variables in UniverseSymbolTable for cross-validation
  - `environment/vectorized_env.py` - Instantiates VariableRegistry with num_agents and device for runtime storage
  - `environment/dac_engine.py` - Reads VFS variables via vfs_registry.get() for reward computation (vfs_variable extrinsic strategy)
  - `environment/action_config.py` - Uses WriteSpec for action effect definitions (Phase 2)
- **Outbound** (what VFS depends on):
  - Pydantic - Schema validation (BaseModel, Field, model_validator, ValidationError)
  - PyTorch - GPU tensor storage (torch.Tensor, torch.device, torch.float32, torch.long, torch.bool)
  - PyYAML - Config file parsing (load_variables_reference_config function uses yaml.safe_load)

**Patterns Observed:**
- **Three-Scope Pattern**: Variables have scope semantics: `global` (single shared value), `agent` (per-agent public values), `agent_private` (per-agent hidden state)
- **Scope-Aware Shape Inference**: Registry automatically determines tensor shapes: global scalar → [], agent scalar → [num_agents], global vector → [dims], agent vector → [num_agents, dims]
- **Access Control Enforcement**: Runtime registry.get()/set() validates reader/writer against readable_by/writable_by lists; raises PermissionError on violations
- **Six-Type System**: scalar (float), bool, vec2i (2D int), vec3i (3D int), vecNi (N-D int with dims field), vecNf (N-D float with dims field)
- **Compile-Time vs Runtime Split**: VFSObservationSpecBuilder generates schemas (specs) at compile-time for BAC; VariableRegistry manages GPU tensors at runtime for environment
- **Auto-Generation Pattern**: Compiler generates standard variables from substrate/bars/affordances configs; users can override via variables_reference.yaml
- **Normalization Flexibility**: Supports two modes: minmax (linear scaling to [min, max]) and zscore (standardization via mean/std); can be scalar or vector
- **Defensive Storage**: Registry clones tensors on get/set to prevent aliasing bugs (no shared references)
- **Curriculum Masking**: ObservationField.curriculum_active (bool) enables structured masking for inactive affordances/meters; used by RND and structured encoders
- **Adapter Pattern**: VFSAdapter translates VFS schemas to compiler DTOs; vfs_to_observation_spec flattens fields and computes start/end indices
- **Semantic Type Inference**: vfs_adapter._semantic_from_name uses heuristic string matching to infer semantic_type ("energy" in name → "meter", "position" in name → "position")
- **Dimension Validation**: VFSObservationSpecBuilder._validate_normalization_shape ensures normalization params (min/max/mean/std) match observation shape

**Concerns:**
- **Phase 2 Not Implemented**: WriteSpec.expression stored as string (no AST parsing or execution); action effects cannot reference variable expressions yet
- **Limited Expressiveness**: Phase 1 only supports static variables; no derived features (e.g., energy_deficit = 1.0 - energy) or environmental phenomena (e.g., raining, rush_hour)
- **variables_reference.yaml Currently Empty**: All production config packs use auto-generated variables only; no custom variables in curriculum levels yet (design validated but unused in practice)
- **Dual ObservationField Types**: VFS has VFSObservationField (vfs/schema.py) and compiler has CompilerObservationField (universe/dto/observation_spec.py); adapter translates between them (duplication risk)
- **Semantic Type Heuristic**: _semantic_from_name uses fragile string matching ("energy" in lowercased name → "meter") instead of explicit metadata; could misclassify custom variables
- **No Lifetime Enforcement**: VariableDef.lifetime ("tick" vs "episode") is metadata-only; runtime doesn't enforce tick-scoped recomputation (engine must manually invalidate tick-scoped variables)
- **Shape Validation Asymmetry**: NormalizationSpec validates minmax/zscore params match shape, but no validation that variable defaults match declared dims/type (e.g., default=[0,0,0] for dims=2)

**Confidence:** High - Comprehensive analysis of all 4 VFS files (817 total lines), dependencies mapped via grep across 11 files, integration patterns understood through adapter/compiler/environment code inspection, test suite examined (5 test files including test_registry.py, test_schema.py, test_observation_builder.py), documentation reviewed (docs/config-schemas/variables.md), production config packs inspected (variables_reference.yaml currently template-only). VFS is a well-documented, production-ready subsystem (TASK-002C complete) with clear separation between compile-time spec generation and runtime storage, though Phase 2 features (expression evaluation, derived variables) remain unimplemented.

---
## Vectorized Environment

**Location:** `src/townlet/environment/`

**Responsibility:** Implements GPU-native vectorized HAMLET environment providing the core RL loop (reset/step/observe) with multi-substrate support, configuration-driven behavior, and integrated reward/meter/affordance/cascade computation engines.

**Key Components:**
- `vectorized_env.py` (1531 lines) - VectorizedHamletEnv class: Main RL environment with Gymnasium-style interface, manages agent state (positions, meters, dones), orchestrates step execution, builds observations via VFS, integrates all sub-engines
- `dac_engine.py` (916 lines) - DACEngine: Runtime reward computation from declarative DAC specs (extrinsic strategies, intrinsic modulation, shaping bonuses, modifier evaluation)
- `affordance_engine.py` (506 lines) - AffordanceEngine: Config-driven affordance interaction processor (instant/multi-tick/dual interactions, operating hours, affordability checks, effect application)
- `meter_dynamics.py` (187 lines) - MeterDynamics: Tensor-driven meter depletion/cascade/modulation/terminal condition checking (wraps optimization_data tensors from compiler)
- `cascade_engine.py` (331 lines) - CascadeEngine: Legacy config-driven cascade processor (being replaced by MeterDynamics, still used in some paths)
- `action_builder.py` (229 lines) - ComposedActionSpace & ActionSpaceBuilder: Composes substrate + custom actions into global vocabulary with enable/disable flags
- `action_labels.py` (359 lines) - Action label presets (gaming, 6dof, cardinal, math) and custom label resolution
- `action_config.py` (140 lines) - ActionConfig/ActionSpaceConfig Pydantic models for action metadata
- `affordance_config.py` (254 lines) - AffordanceConfig/AffordanceConfigCollection models + YAML loader
- `affordance_layout.py` (73 lines) - Affordance positioning utilities (randomization, static layouts)
- `cascade_config.py` (318 lines) - EnvironmentConfig aggregator for bars/cascades/terminal conditions
- `substrate_action_validator.py` (95 lines) - Validates action space compatibility with substrate types (POMDP checks, dimension validation)

**Dependencies:**
- **Inbound** (who depends on this):
  - `population/vectorized.py` (VectorizedPopulation) - Creates env, calls reset/step in training loop, injects exploration module
  - `demo/runner.py` (DemoRunner) - Instantiates env from compiled universe for training orchestration
  - `demo/live_inference.py` (LiveInference) - Creates env for inference server, broadcasts observations/rewards via WebSocket
  - `tests/test_townlet/integration/` - Integration tests for full training pipeline
- **Outbound** (what this depends on):
  - `universe/compiled.py` (CompiledUniverse) - Source artifact for all configuration data (observation specs, action metadata, optimization tensors)
  - `substrate/{base,grid2d,grid3d,gridnd,continuous,continuousnd,aspatial}.py` - Spatial topology implementations (position init, movement, distance, observation encoding)
  - `vfs/registry.py` (VariableRegistry) - Runtime variable storage with access control, observation building
  - `config/drive_as_code.py` (DriveAsCodeConfig) - DAC specification DTOs for reward computation
  - `config/{bar,cascade,affordance,hamlet,training}.py` - Configuration DTOs consumed by env initialization
  - `exploration/base.py` (ExplorationStrategy) - Optional intrinsic reward computation (RND, ICM, epsilon-greedy)
  - `population/runtime_registry.py` (AgentRuntimeRegistry) - Optional telemetry tracking

**Patterns Observed:**
- **GPU-Native Vectorization**: All state tensors [num_agents, ...] on device, batch operations throughout (positions, meters, dones, observations)
- **Modular Engine Architecture**: Separate engines for DAC (reward), AffordanceEngine (interactions), MeterDynamics (depletion/cascades), each GPU-accelerated and config-driven
- **Configuration-Driven Behavior**: Zero hardcoded logic - all parameters from CompiledUniverse (grid size, affordance effects, meter dynamics, action space, observation specs)
- **Fixed Observation Vocabulary**: All curriculum levels observe same affordance count (15) even if not deployed, enabling transfer learning across levels
- **VFS Integration**: Observation building via VariableRegistry with access control (engine writes, agent reads), respects variable scopes (global/agent/agent_private)
- **Composable Action Space**: Substrate actions + custom actions + affordance actions (future), global vocabulary with enable/disable flags preserves action_dim across configs
- **POMDP Support with Validation**: Partial observability via local vision windows (Grid2D/Grid3D only), extensive validation prevents unsupported configs (Continuous, GridND N≥4, wrong encoding modes)
- **Temporal Mechanics Integration**: Operating hours (action_mask_table[24×affordance_count]), time_of_day cycling, interaction progress tracking
- **Affordance Tracking for DAC**: Tracks last_affordances, affordance_streaks, unique_affordances_count per agent for DAC shaping bonuses (diversity, streak, completion)
- **Gymnasium Interface Compliance**: reset() → observations, step(actions) → (observations, rewards, dones, info)
- **Substrate Abstraction**: Position encoding delegates to substrate (relative/scaled/absolute modes), boundary handling varies by substrate
- **Action Masking**: Invalid movements masked at boundaries (Grid2D/3D), INTERACT masked when not on open affordance, all actions masked for dead agents
- **Cached Action Indices**: Pre-computes action indices (INTERACT, WAIT, UP_Z, DOWN_Z) at init for fast lookup during step execution
- **Dynamic Meter Indexing**: Uses meter_name_to_index from metadata instead of hardcoded indices (TASK-001 compliance), handles optional meters (money, hygiene, satiation)
- **Retirement Bonus**: Agents reaching max_steps_per_episode get +1.0 reward before episode termination (pedagogical: surviving = success)

**Concerns:**
- **Large Monolithic File**: vectorized_env.py at 1531 lines handles initialization (200+ lines), observation building (150+ lines), action execution (300+ lines), reward computation (100+ lines), affordance tracking (150+ lines) - could benefit from extraction into modules (e.g., `env_observation_builder.py`, `env_action_executor.py`, `env_state_manager.py`)
- **Complex Initialization**: __init__ spans 200+ lines with multiple subsystem integrations (substrate, VFS, DAC, affordance, meter dynamics, action space, POMDP validation) - difficult to unit test in isolation
- **Tight Coupling with Many Subsystems**: Imports from 8+ different subsystems (universe, substrate, vfs, config, exploration, population, environment sub-engines) - changes in any propagate to env
- **POMDP Validation Scattered**: POMDP checks in __init__ (lines 252-303), get_action_masks() (no current check), substrate validator (separate file) - validation logic fragmented
- **Legacy Fallback Patterns**: hasattr() checks for substrate methods (_encode_full_grid, encode_position_features, normalize_positions) suggest substrate interface not fully stabilized
- **Dual Cascade Systems**: Both CascadeEngine (legacy) and MeterDynamics (modern) exist - MeterDynamics preferred but CascadeEngine not fully removed (technical debt)
- **Affordance Position Dual Sources**: Loads positions from both affordance_positions_from_config and affordance_positions_from_optimization with fallback logic - single source of truth unclear
- **Action Execution Complexity**: _execute_actions() handles 4 action types (substrate movement, custom actions, INTERACT, WAIT) with different code paths - could benefit from action handler registry pattern
- **Observation Building Indirection**: _get_observations() has 2 modes (POMDP vs full obs) with different variable names (local_window vs grid_encoding), conditional VFS updates, fallback chains for substrate encoding methods - hard to trace observation structure
- **Missing Docstrings**: Several internal methods (_update_affordance_tracking, _get_last_action_affordances, _get_affordance_streaks) lack docstrings explaining parameters/returns
- **Hard-to-Test State Management**: Agent state split across multiple tensors (positions, meters, dones, step_counts, intrinsic_weights, interaction_progress, _last_affordances, _affordance_streaks, _unique_affordances_count, _affordances_seen) - difficult to verify state consistency in tests
- **Temporal Mechanics Always Active**: time_of_day always cycles even when enable_temporal_mechanics=False (line 967), but mechanics only enforced conditionally - potential confusion
- **Meter Index Fallbacks**: energy_idx defaults to 0, health_idx defaults to "min(6, meter_count-1)" if meters not found (lines 313-314) - silent fallbacks could hide config errors
- **Affordability Check Location**: Affordability for INTERACT checked in _execute_actions() (lines 1202-1214) after action masking already excluded some agents - could optimize by moving to get_action_masks()
- **Aspatial Special Cases**: Multiple conditionals checking "self.substrate.position_dim == 0" scattered throughout - aspatial substrate might benefit from dedicated AspatialEnv subclass
- **No Batch Size Validation**: Accepts any num_agents without upper bound validation (contrast with compiler's 10K cell limit for grid size) - large batches could OOM GPU

**Confidence:** High - Comprehensive analysis of 1531-line core environment plus 12 supporting modules (4940 total lines), examined initialization flow, reset/step execution, observation building, action handling, engine integration, dependencies mapped via grep/code inspection, patterns/anti-patterns identified from code structure and comments, POMDP validation logic traced across multiple locations, substrate abstraction understood, VFS/DAC/affordance/meter integration points verified.

---

## Vectorized Training Loop (Population)

**Location:** `src/townlet/population/`

**Responsibility:** Orchestrates batched training for multiple agents with GPU-native vectorized operations, managing Q-networks, replay buffers, curriculum progression, and exploration strategies in a unified training loop.

**Key Components:**
- `vectorized.py` (1155 lines) - VectorizedPopulation class (main training orchestrator coordinating network training, replay sampling, curriculum decisions, exploration, and episode lifecycle)
- `runtime_registry.py` (135 lines) - AgentRuntimeRegistry class (GPU tensor storage for per-agent telemetry: survival_time, curriculum_stage, epsilon, intrinsic_weight with JSON-safe snapshot API for WebSocket broadcast)
- `base.py` (70 lines) - PopulationManager ABC (abstract interface defining step_population() and get_checkpoint() contracts for population management implementations)

**Dependencies:**
- **Inbound** (who depends on this):
  - `demo/runner.py` (DemoRunner) - instantiates VectorizedPopulation for multi-day training orchestration
  - `demo/live_inference.py` (LiveInference) - uses VectorizedPopulation for inference server with trained checkpoints
  - Tests: `tests/test_townlet/{integration,unit}/population/*.py` - extensive coverage of training loop, action selection, recurrent networks, Double DQN
- **Outbound** (what this depends on):
  - Environment: `environment/vectorized_env.py` (VectorizedHamletEnv) - for env.step(), env.reset(), get_action_masks(), observation generation
  - Agent networks: `agent/networks.py` (SimpleQNetwork, RecurrentSpatialQNetwork, StructuredQNetwork) via `agent/network_factory.py`
  - Replay buffers: `training/{replay_buffer,sequential_replay_buffer,prioritized_replay_buffer}.py` (strategy pattern for experience replay)
  - Exploration: `exploration/{base,rnd,adaptive_intrinsic,action_selection}.py` (for action selection, intrinsic reward computation, epsilon decay)
  - Curriculum: `curriculum/{base,adversarial,static}.py` (for difficulty adjustment, depletion_multiplier decisions)
  - Training state: `training/state.py` (BatchedAgentState, PopulationCheckpoint, CurriculumDecision DTOs)
  - Config: `agent/brain_config.py` (TASK-005 Brain As Code), `config/training.py` (legacy parameters)
  - Factories: `agent/{network_factory,optimizer_factory,loss_factory}.py` (for network/optimizer/loss instantiation from brain.yaml)
  - Checkpoint utils: `training/checkpoint_utils.py` (for safe_torch_load, verify_checkpoint_digest, universe metadata validation)

**Patterns Observed:**
- **Dual Network Architecture**: Online Q-network for training, target network for stability (synced every target_update_frequency steps, default 100)
- **Replay Buffer Strategy Pattern**: Supports 3 buffer types (ReplayBuffer for feedforward, SequentialReplayBuffer for LSTM with temporal dependencies, PrioritizedReplayBuffer for importance sampling)
- **Episode Accumulation for Recurrent**: Accumulates full episodes in memory (current_episodes list) before flushing to SequentialReplayBuffer for LSTM training with hidden state continuity
- **Immediate Storage for Feedforward**: Stores transitions immediately after env.step() in ReplayBuffer or PrioritizedReplayBuffer (no episode accumulation overhead)
- **Training Loop Decoupling**: train_frequency parameter (default 4) controls Q-network updates frequency (not every step) for sample efficiency
- **Double DQN Support**: Configurable algorithm variant via brain_config.q_learning.use_double_dqn (vanilla DQN: target network for both selection+evaluation; Double DQN: online network for selection, target network for evaluation to reduce overestimation bias)
- **Loss Function Abstraction**: Loss type configurable via brain_config.loss (MSE, Huber, Smooth L1) with functional API for per-sample losses in PER path
- **Curriculum Integration**: Calls curriculum.get_batch_decisions() every step, passes depletion_multiplier to env.step() for difficulty adjustment
- **Telemetry Registry Pattern**: AgentRuntimeRegistry provides GPU-native tensor storage with to_dict() snapshots for JSON-safe WebSocket broadcast to frontend
- **Brain As Code Integration**: TASK-005 migrates hardcoded hyperparameters to brain.yaml (network architecture, optimizer with scheduler, loss function, replay config including PER alpha/beta)
- **Checkpoint Provenance**: Stores universe_metadata (meter_count, meter_names, obs_dim, action_dim, version) for compatibility validation on checkpoint load
- **Hidden State Management**: LSTM hidden state persists during episode rollout, resets at episode boundaries (done signal), resets during batch training (separate hidden state for training batch vs rollout batch)
- **Action Masking**: All action selection paths (select_greedy_actions, select_epsilon_greedy_actions, exploration.select_actions) use environment action masks to prevent boundary violations and invalid actions
- **Post-Terminal Masking**: For recurrent networks, applies batch["mask"] to loss computation to prevent gradients from post-terminal garbage (P2.2 pattern for LSTM training on variable-length episodes)
- **Gradient Clipping**: max_grad_norm=10.0 prevents exploding gradients via torch.nn.utils.clip_grad_norm_ (especially important for LSTM backpropagation through time)
- **DAC Reward Composition Integration**: Environment returns fully composed rewards (extrinsic + intrinsic + shaping), population stores zeros for intrinsic_rewards in replay buffer to avoid double-counting, comments emphasize this throughout
- **Scheduler Support**: Supports learning rate schedulers via brain_config.optimizer.scheduler (steps after each optimizer.step() call)
- **PER Beta Annealing**: Prioritized replay buffer anneals importance sampling weight (beta) from initial value toward 1.0 over training (requires max_episodes and max_steps_per_episode for annealing schedule)
- **Training Metrics Tracking**: Stores last_td_error, last_loss, last_q_values_mean, last_rnd_loss for TensorBoard logging and monitoring
- **Selective TensorBoard Logging**: Logs network weights/gradients histograms every 100 training steps (not every step) to reduce I/O overhead

**Concerns:**
- **Large Monolithic Class**: VectorizedPopulation at 1155 lines handles network instantiation, training loop, episode management, checkpointing, telemetry sync (could benefit from extraction into PopulationTrainer, EpisodeManager, CheckpointManager classes)
- **TODO Comments**: Multiple "TODO(BRAIN_AS_CODE): Should come from config" comments (lines 181, 188, 189, 192, 248, 255, 256, 261) for hardcoded hidden dimensions in legacy code paths - partially resolved in TASK-005 but legacy paths still exist when brain_config=None
- **Dual Initialization Paths**: Network instantiation has two code paths (brain_config vs legacy parameters) creating maintenance burden and testing complexity - legacy path should be deprecated post-launch per pre-release policy
- **PER Beta Annealing Coupling**: PER beta annealing requires max_episodes and max_steps_per_episode from constructor (tight coupling to training duration, fragile if training loop changes)
- **Recurrent Training Overhead**: LSTM training with Double DQN requires 3 forward passes (online prediction for gradients, online selection for action choice, target evaluation for Q-targets) vs 2 for vanilla DQN - 50% more compute
- **Episode Container Memory**: For recurrent networks, accumulates full episodes on CPU (current_episodes list) before flushing to GPU buffer - potential memory pressure for long episodes (e.g., survival time > 1000 steps)
- **No Explicit Flush on Max Steps**: Episode flush relies on done signal from environment; if max_steps_per_episode reached without done, episode stays in current_episodes (potential memory leak, though environment should emit done)
- **Global Epsilon Assumption**: _get_current_epsilon_value() returns single float, but current_epsilons is a tensor [num_agents] - suggests per-agent epsilon exploration not fully supported (all agents share same epsilon)
- **Intrinsic Reward Double-Counting Prevention**: Complex logic to avoid double-counting intrinsic rewards (DAC already includes them in env.step() rewards) - stores zeros for intrinsic_rewards in replay buffer, extensive comments warn about this throughout (lines 657-673, 706-707) - fragile pattern requiring careful coordination with DAC engine
- **Checkpoint Version Migration**: No automatic migration from old checkpoint formats - fails loudly with ValueError instead (intentional per pre-release zero-backwards-compatibility policy, but means all old checkpoints must be retrained)
- **Recurrent Network Type Checking**: Extensive use of cast(RecurrentSpatialQNetwork, self.q_network) and is_recurrent flag checks throughout code - suggests network polymorphism could be improved with better interface design
- **Training Metrics Staleness**: last_td_error, last_loss, etc. are only updated when training occurs (every train_frequency steps), not every step - telemetry consumers must check last_training_step to detect staleness
- **RND Buffer Management**: RND predictor training accumulates observations in rnd.obs_buffer (CPU) across all agents, trains when buffer full - potential memory pressure if predictor training is slower than observation generation
- **Curriculum Decision Caching**: current_curriculum_decisions stored as instance variable, only refreshed in step_population() - stale if curriculum state queried outside training loop

**Confidence:** High - Comprehensive analysis of 1155-line orchestrator with all training paths examined (feedforward/recurrent, standard/PER replay, vanilla/Double DQN), dependencies mapped via code inspection and grep, integration points with environment/curriculum/exploration understood, telemetry registry pattern analyzed, Brain As Code migration status documented (TASK-005), checkpoint provenance validation confirmed, episode lifecycle for LSTM training traced through code.

---

## Agent Networks & Q-Learning

**Location:** `src/townlet/agent/`

**Responsibility:** Provides neural network architectures and Q-learning algorithm implementations for training action-value functions, with declarative configuration via Brain As Code (brain.yaml) enabling configurable network types, optimizer strategies, loss functions, and DQN algorithm variants.

**Key Components:**
- `networks.py` (511 lines) - Four neural architectures: SimpleQNetwork (feedforward MLP for full observability), RecurrentSpatialQNetwork (CNN+LSTM for POMDP with vision window/position/meter/affordance encoders), DuelingQNetwork (value/advantage stream decomposition), StructuredQNetwork (semantic group encoders leveraging observation structure)
- `network_factory.py` (189 lines) - NetworkFactory class with factory methods (build_feedforward, build_recurrent, build_dueling) for instantiating networks from brain.yaml BrainConfig specifications
- `brain_config.py` (489 lines) - Pydantic DTOs for declarative architecture specs: BrainConfig (top-level), ArchitectureConfig (network type + architecture), OptimizerConfig (optimizer + scheduler), LossConfig (loss function + params), QLearningConfig (gamma + target_update_frequency + use_double_dqn), ReplayConfig (capacity + PER params), FeedforwardConfig, RecurrentConfig (CNN/MLP/LSTM sub-configs), DuelingConfig (shared/value/advantage streams), plus compute_brain_hash for checkpoint provenance
- `optimizer_factory.py` (163 lines) - OptimizerFactory class building PyTorch optimizers (Adam/AdamW/SGD/RMSprop) and LR schedulers (constant/step_decay/cosine/exponential) from brain.yaml with parameter validation ensuring all optimizer-specific params explicit
- `loss_factory.py` (41 lines) - LossFactory class building PyTorch loss modules (MSELoss/HuberLoss/SmoothL1Loss) from brain.yaml LossConfig with configurable Huber delta
- `__init__.py` (27 lines) - Public API exports

**Dependencies:**
- **Inbound** (who depends on this):
  - `population/vectorized.py` (VectorizedPopulation) - Instantiates q_network and target_network via NetworkFactory.build_* methods, trains online network with optimizer from OptimizerFactory, computes loss with LossFactory, implements vanilla DQN vs Double DQN Q-target computation, syncs target network periodically, checkpoints network state_dicts
  - `exploration/rnd.py` (RND) - Uses SimpleQNetwork-like predictor/target networks for novelty detection (Random Network Distillation)
  - `demo/runner.py` (DemoRunner) - Loads BrainConfig via load_brain_config, passes to VectorizedPopulation constructor
  - Tests: `tests/test_townlet/unit/agent/test_networks.py`, `tests/test_townlet/integration/test_double_dqn_training.py`, `tests/test_townlet/unit/population/test_double_dqn_algorithm.py` - Extensive coverage of architectures and algorithm variants
- **Outbound** (what this depends on):
  - PyTorch: `torch.nn.Module`, `torch.nn.{Linear,LayerNorm,ReLU,Conv2d,LSTM,Sequential,ModuleDict}`, `torch.optim.{Adam,AdamW,SGD,RMSprop}`, `torch.optim.lr_scheduler.{StepLR,CosineAnnealingLR,ExponentialLR}`
  - Pydantic: `BaseModel`, `Field`, `ValidationError`, `field_validator`, `model_validator` for schema validation enforcing no-defaults principle
  - `universe/dto.py` (ObservationActivity) - Used by StructuredQNetwork for semantic group slicing (spatial/bars/affordances/temporal/custom groups)
  - PyYAML: `yaml.safe_load` in load_brain_config for parsing brain.yaml files
  - Standard library: `hashlib` (SHA256 brain_hash for checkpoint provenance), `json` (deterministic config serialization for hashing), `pathlib` (config file paths)

**Patterns Observed:**
- **Dual Network Architecture**: Online Q-network for action selection and training, target network for stable Q-target computation (synced every target_update_frequency steps, default 100) to reduce moving target problem in TD learning
- **Factory Pattern**: NetworkFactory/OptimizerFactory/LossFactory encapsulate complex instantiation logic with validation, enabling declarative brain.yaml configs to drive network construction without code changes
- **Brain As Code (TASK-005)**: Declarative configuration via brain.yaml with Pydantic validation enforces all behavioral parameters explicit (learning_rate, layer sizes, activation functions, optimizer betas/eps, scheduler params, loss delta, gamma, target_update_frequency, use_double_dqn) following no-defaults principle
- **Algorithm Variants via Configuration**: Vanilla DQN (Q-target = r + γ × max_a Q_target(s', a)) vs Double DQN (Q-target = r + γ × Q_target(s', argmax_a Q_online(s', a))) controlled by brain.yaml:q_learning.use_double_dqn boolean; Double DQN reduces Q-value overestimation bias from max operator (van Hasselt et al. 2016)
- **Architecture Selection Based on Observability**: SimpleQNetwork (MLP, ~26K params) for full observability (L0/L0.5/L1), RecurrentSpatialQNetwork (CNN+LSTM, ~650K params) for partial observability with POMDP (L2/L3), DuelingQNetwork (value/advantage decomposition) for improved learning efficiency, StructuredQNetwork (group encoders) for leveraging observation semantics
- **Recurrent Hidden State Management**: LSTM hidden state (h, c) resets at episode start, persists during episode rollout for online action selection, resets per transition during batch training to maintain temporal dependencies in sequential replay buffer (important for LSTM backpropagation through time)
- **Gradient Clipping**: max_grad_norm=10.0 in VectorizedPopulation prevents exploding gradients during backpropagation (especially critical for LSTM with long sequences)
- **Fixed Observation Vocabulary**: All Grid2D configs use 29→8 architecture (29-dim observations including 15 affordance types even if not deployed, 8 actions) enabling checkpoint transfer across curriculum levels without retraining observation encoder
- **Checkpoint Provenance**: brain_hash (SHA256 of brain.yaml config) embedded in checkpoints for reproducibility verification, preventing accidental loading of checkpoints with mismatched network architectures or hyperparameters
- **Learning Rate Scheduling**: Supports constant, step_decay (StepLR), cosine (CosineAnnealingLR), exponential (ExponentialLR) schedules with required params validation; scheduler.step() called after each optimizer.step() in training loop
- **Positional Encoding for Recurrent Networks**: RecurrentSpatialQNetwork handles variable position_dim (2 for Grid2D, 3 for Grid3D, 0 for Aspatial) with conditional position_encoder instantiation, enabling multi-substrate support
- **Multi-Stream Architecture**: DuelingQNetwork separates shared feature extraction → value stream V(s) [scalar] + advantage stream A(s,a) [action_dim vector] → aggregation Q(s,a) = V(s) + (A(s,a) - mean(A(s,:))) ensuring identifiability (Wang et al. 2016)
- **Semantic Group Processing**: StructuredQNetwork processes observation groups (spatial/bars/affordances/temporal/custom) with separate MLPs before concatenation, leveraging observation structure for better inductive bias vs uniform SimpleQNetwork
- **Modular Loss Functions**: Loss type (mse/huber/smooth_l1) configurable in brain.yaml with functional API (F.mse_loss/huber_loss with reduction='none') for per-sample losses in PER importance sampling weighting path

**Concerns:**
- **Dual Initialization Paths**: Network instantiation has two code paths (brain_config vs legacy parameters via network_type argument) creating maintenance burden and testing complexity (lines 143-192 in vectorized.py); legacy path should be deprecated post-launch per pre-release zero-backwards-compatibility policy
- **TODO Comments for Legacy Hardcoded Dimensions**: Multiple "TODO(BRAIN_AS_CODE): Should come from config" comments in vectorized.py lines 181, 188-189, 192 for hardcoded hidden_dim=256/128, group_embed_dim=32, q_head_hidden_dim=128 in legacy paths; partially resolved in TASK-005 but legacy paths still exist when brain_config=None
- **RecurrentSpatialQNetwork Fixed Internal Architecture**: Phase 2 only makes LSTM hidden_size configurable via RecurrentConfig.lstm.hidden_size; vision_encoder (CNN channels/kernels/strides), position_encoder/meter_encoder/affordance_encoder (MLP layer sizes), q_head dimensions remain hardcoded (lines 114-164 in networks.py); future phases should make full architecture configurable from RecurrentConfig sub-configs (vision_encoder: CNNEncoderConfig, position_encoder: MLPEncoderConfig, etc.)
- **Recurrent Network Type Checking**: Extensive use of cast(RecurrentSpatialQNetwork, self.q_network) and is_recurrent flag checks throughout population/vectorized.py suggests network polymorphism could be improved with better interface design (common forward() signature for feedforward vs recurrent, or abstract base class with reset_hidden_state() method)
- **Double DQN Recurrent Training Overhead**: LSTM with Double DQN requires 3 forward passes per training step (online network prediction for gradients, online network selection for action choice, target network evaluation for Q-targets) vs 2 for vanilla DQN (online prediction, target evaluation) → 50% more compute overhead for recurrent+Double DQN combination
- **LSTM Hidden State Lifecycle Complexity**: Hidden state management spans multiple methods (reset_hidden_state in environment reset, set_hidden_state/get_hidden_state in action selection, separate hidden states for rollout vs training batch) creating potential for state inconsistencies if not carefully synchronized
- **Checkpoint Version Migration**: No automatic migration from old checkpoint formats with different network architectures - fails loudly with ValueError on network.load_state_dict() mismatch (intentional per pre-release zero-backwards-compatibility policy, but means all old checkpoints must be retrained after architecture changes)
- **StructuredQNetwork Limited Adoption**: Defined in networks.py but only used in legacy network_type='structured' path (line 183-190 in vectorized.py); no brain.yaml support yet in ArchitectureConfig (only feedforward/recurrent/dueling types) suggesting incomplete integration in TASK-005
- **Positional Encoding Conditional Logic**: RecurrentSpatialQNetwork has conditional position_encoder instantiation (lines 126-136) and conditional position_features concatenation (lines 223-236) with if position_dim > 0 checks scattered throughout forward() - Aspatial substrate special case could benefit from dedicated network variant or cleaner abstraction
- **Activation Function String Mapping**: NetworkFactory._get_activation and DuelingQNetwork._get_activation maintain duplicate activation string→module dicts (relu/gelu/swish/tanh/elu) suggesting need for shared activation registry or utility module
- **No Gradient Norm Logging**: VectorizedPopulation applies gradient clipping (max_norm=10.0) but doesn't log pre-clipped gradient norms to TensorBoard, making it hard to diagnose if clipping threshold is appropriate or if network has vanishing/exploding gradient issues
- **Target Network Sync Validation**: target_network.load_state_dict(q_network.state_dict()) in VectorizedPopulation lines 264, 843, 928 has no verification that architectures match (e.g., if brain_config changed between checkpoint creation and loading); could benefit from explicit architecture hash check before sync
- **PER Beta Annealing Coupling**: PER beta annealing in ReplayConfig requires max_episodes and max_steps_per_episode from VectorizedPopulation constructor (tight coupling to training duration), but BrainConfig has no place to specify these - PER beta schedule should be in brain.yaml or accept training duration separately
- **Loss Delta Hard Default**: LossConfig.huber_delta defaults to 1.0 (line 327 in brain_config.py) violating no-defaults principle slightly; should be required field for consistency, though this is minor exemption for common Huber loss default

**Confidence:** High - Comprehensive analysis of all 5 agent files (1389 total lines), network architectures examined (SimpleQNetwork MLP structure, RecurrentSpatialQNetwork CNN+LSTM+encoder design with 192-dim LSTM input from 128 vision + 32 position + 32 meters + 32 affordance features, DuelingQNetwork value/advantage streams, StructuredQNetwork group processing), factory patterns understood, Brain As Code integration analyzed (BrainConfig schema with 7 top-level fields, ArchitectureConfig supporting 3 types, OptimizerConfig for 4 optimizer types with scheduler support), Q-learning algorithm variants traced (vanilla DQN vs Double DQN implementation in vectorized.py lines 866-876), dependencies mapped via grep/code inspection across population/exploration/demo/tests, checkpoint provenance via brain_hash confirmed, test coverage verified (3 test files specifically for agent networks and Double DQN).

---

## Substrate Implementations

**Location:** `src/townlet/substrate/`

**Responsibility:** Provides pluggable spatial topology abstractions for HAMLET environments, defining how agents navigate space (discrete grids, continuous spaces, or no space), with GPU-native vectorized operations and configurable observation encoding modes.

**Key Components:**
- `base.py` (316 lines) - SpatialSubstrate abstract interface defining 14 methods (position initialization, movement, distance computation, observation encoding, POMDP local windows, action space size)
- `config.py` (392 lines) - Pydantic DTOs (SubstrateConfig, GridConfig, GridNDConfig, ContinuousConfig, AspatialSubstrateConfig, ActionLabelConfig) with No-Defaults Principle validation
- `factory.py` (152 lines) - SubstrateFactory.build() instantiates concrete substrate from config (device-agnostic construction, device specified per-operation)
- `grid2d.py` (542 lines) - Grid2DSubstrate for 2D square grids (Manhattan/Euclidean/Chebyshev distance metrics, 4 boundary modes, 3 observation encoding modes, POMDP local window support)
- `grid3d.py` (554 lines) - Grid3DSubstrate for 3D cubic grids (extends Grid2D patterns with Z-axis, 8 actions including UP_Z/DOWN_Z)
- `gridnd.py` (521 lines) - GridNDSubstrate for 4D-100D hypercube grids (generic N-dimensional movement, 2N+2 action space, position_dim validation)
- `continuous.py` (916 lines) - Continuous1D/2D/3DSubstrate for float-based positioning with bounded space, proximity-based interaction, configurable action discretization (8-32 directions × 3-7 magnitudes)
- `continuousnd.py` (500 lines) - ContinuousNDSubstrate for 4D-100D continuous spaces (generic bounds validation, interaction radius enforcement)
- `aspatial.py` (172 lines) - AspatialSubstrate for position-free universes (position_dim=0, reveals meters as true universe state, returns empty position tensors, all agents "everywhere")

**Dependencies:**
- **Inbound** (who depends on this):
  - `environment/vectorized_env.py` (VectorizedHamletEnv) - calls substrate.initialize_positions(), apply_movement(), compute_distance(), encode_observation(), normalize_positions(), get_valid_neighbors(), is_on_position(), encode_partial_observation() for spatial operations
  - `universe/compiler.py` (UniverseCompiler) - validates substrate compatibility with POMDP configs (Stage 4 cross-validation), generates observation specs (Stage 5 metadata), pre-computes action masks for temporal mechanics (Stage 6 optimization)
  - `environment/action_builder.py` (ActionSpaceBuilder) - queries substrate.get_default_actions() for substrate-provided action vocabulary, uses action_space_size property for validation
  - `environment/substrate_action_validator.py` - validates substrate type supports requested features (POMDP, observation encoding modes, action space size limits)
  - `universe/compiler_inputs.py` (RawConfigs) - loads substrate.yaml via SubstrateConfig.load_substrate_config()
  - Tests: `tests/test_townlet/{unit,integration}/substrate/` - 27 test files with comprehensive coverage (boundary cases, scaling, encoding modes, POMDP validation)
- **Outbound** (what this depends on):
  - PyTorch - GPU tensor operations (torch.Tensor, torch.device, torch.randint, torch.clamp, torch.stack, torch.cat for vectorized operations)
  - Pydantic - Configuration validation (BaseModel, Field, model_validator for SubstrateConfig/GridConfig/ContinuousConfig DTOs)
  - PyYAML - Config file parsing (yaml.safe_load in load_substrate_config)
  - `environment/action_config.py` (ActionConfig) - action metadata DTOs returned by get_default_actions()
  - `environment/affordance_layout.py` (iter_affordance_positions) - utility for iterating affordance positions in observation encoding

**Patterns Observed:**
- **Abstract Factory Pattern**: SubstrateFactory.build() instantiates concrete substrate from SubstrateConfig, enabling runtime substrate selection without environment code changes
- **Strategy Pattern**: Distance metrics (manhattan/euclidean/chebyshev), boundary modes (clamp/wrap/bounce/sticky), observation encodings (relative/scaled/absolute) configurable per substrate instance
- **Device-Agnostic Construction**: Substrates instantiated without device parameter; methods accept device argument for tensor creation (enables multi-GPU flexibility)
- **Canonical Action Ordering Contract**: All substrates emit actions as [movement_actions..., INTERACT, WAIT] enabling predictable action indexing (actions[-2]=INTERACT, actions[-1]=WAIT)
- **GPU-Native Vectorization**: All operations batch-agnostic with [num_agents, ...] tensor shapes, no CPU↔GPU transfers in hot paths
- **Three Observation Encoding Modes**: relative (normalized [0,1], default for transfer learning), scaled (normalized + range metadata for network to learn grid size matters), absolute (raw coordinates for physical simulation)
- **POMDP Local Window Extraction**: encode_partial_observation() provides (2×vision_range+1)² local grids for partial observability (Grid2D/3D only, validated in compiler Stage 4)
- **Aspatial Special Case**: position_dim=0 reveals philosophical insight that spatial positioning is optional overlay on meter-based universe (pedagogically valuable for teaching RL fundamentals)
- **No-Defaults Principle Enforcement**: All SubstrateConfig fields required (width, height, boundary, distance_metric, observation_encoding), Pydantic validators prevent missing parameters
- **Boundary Mode Implementations**: clamp (torch.clamp to valid range), wrap (modulo for toroidal topology), bounce (reflection with absolute value and mirroring), sticky (revert to original position if out of bounds)
- **Distance Metric Abstractions**: manhattan (L1 norm, sum of absolute deltas), euclidean (L2 norm, sqrt of squared deltas), chebyshev (L∞ norm, max absolute delta)
- **Topology Metadata Consistency**: All substrates store topology field (square/cubic/hypercube) for compiler metadata generation and documentation
- **Action Space Size Formula**: action_space_size = 2×position_dim + 2 for spatial (±movement per dimension + INTERACT + WAIT), = 2 for aspatial (INTERACT + WAIT only)
- **Continuous Action Discretization**: Optional action_discretization config (num_directions: 8-32, num_magnitudes: 3-7) expands action space from legacy 8-way to 195 actions (32×7 for maximum freedom)
- **Defensive Validation**: GridND/ContinuousND validate dimension counts (4-100 range), grid sizes (prevent DoS with 10K cell limit in compiler), bounds (min < max, space large enough for interaction_radius)

**Concerns:**
- **Grid Encoding Memory Scaling**: Grid2D._encode_full_grid() creates [num_agents, width×height] tensors; 8×8 grid = 64 cells manageable, but 100×100 = 10K cells per agent (compiler enforces 10K cell limit, but still memory-intensive for large batches)
- **Continuous POMDP Not Supported**: encode_partial_observation() unimplemented for continuous substrates (local window concept unclear for float positions, would require spatial indexing structure like KD-tree)
- **GridND POMDP Memory Explosion**: 4D grid with vision_range=2 → 5⁴=625 cells per agent, 7D → 5⁷=78,125 cells (compiler validation prevents this, but limitation documented in tests/unit/environment/test_pomdp_validation.py)
- **Action Discretization Not Validated**: ContinuousConfig.action_discretization accepts arbitrary num_directions/num_magnitudes, no validation that product < max_actions limit (300), could OOM action space without compiler check
- **Bounce Boundary Limitations**: Grid2D/3D bounce mode uses simple reflection (2×max - position), but doesn't handle multi-step bounces correctly if delta > grid size (agent could reflect multiple times)
- **No Substrate Versioning**: SubstrateConfig has version field but no version-specific validation logic; old configs might break silently if substrate behavior changes
- **Observation Encoding Inconsistency**: Grid2D._encode_full_grid() always used (global occupancy grid), but _encode_position_features() varies by observation_encoding mode; observation dim varies (relative=2, scaled=4, absolute=2) complicating observation spec generation
- **Sticky Boundary Asymmetry**: Sticky mode checks out-of-bounds per dimension independently (agent can move in X but not Y), different semantics from bounce/clamp/wrap (all-or-nothing movement)
- **Distance Computation Broadcasting**: compute_distance() handles pos2 as single position [position_dim] or batch [num_agents, position_dim] via unsqueeze(0), but no validation that batch dimensions match pos1
- **get_all_positions() Not Scalable**: Grid2D returns list of [x,y] for all cells (8×8=64 positions manageable, 100×100=10K positions in Python list inefficient); continuous substrates raise NotImplementedError (infinite positions)
- **INTERACT Cost Hardcoded**: get_default_actions() returns ActionConfig with hardcoded costs (energy: 0.003), but affordances.yaml overrides costs per affordance; potential confusion about cost source of truth
- **No Hexagonal/Graph Topologies**: Only Cartesian grids (square/cubic/hypercube) and continuous spaces supported; hexagonal grids, graph substrates, irregular topologies require new substrate implementations (Phase 9/10 tasks in backlog)
- **Aspatial Always Returns True for is_on_position()**: Pedagogically correct (agents "everywhere"), but means affordance interaction logic must handle aspatial case specially in environment (no proximity checks)

**Confidence:** High - Comprehensive analysis of all 9 substrate files (4096 total lines), abstract interface and all concrete implementations examined, dependencies mapped via grep across 80 files, patterns identified from code structure and comments, test suite scanned (27 test files, no skipped tests or TODOs), POMDP validation logic traced, observation encoding modes compared, boundary mode implementations verified, action space size formula validated, continuous action discretization understood, concerns identified from code inspection and test coverage gaps.

---
## Exploration Strategies

**Location:** `src/townlet/exploration/`

**Responsibility:** Provides intrinsic motivation and action selection strategies to drive exploration behavior through novelty-seeking rewards and epsilon-greedy policies.

**Key Components:**
- `base.py` (119 lines) - ExplorationStrategy ABC defining interface: select_actions(), compute_intrinsic_rewards(), update(), checkpoint_state(), load_state()
- `rnd.py` (344 lines) - RNDExploration class implementing Random Network Distillation with dual networks (fixed random target, trained predictor), prediction error as novelty signal, RunningMeanStd for reward normalization, obs_buffer for mini-batch training
- `adaptive_intrinsic.py` (229 lines) - AdaptiveIntrinsicExploration class wrapping RNDExploration with performance-based annealing (survival variance tracking, dual-gate annealing: low variance + high survival required, exponential decay of intrinsic weight)
- `epsilon_greedy.py` (119 lines) - EpsilonGreedyExploration class providing simple baseline (exponential epsilon decay, no intrinsic rewards, zero network overhead)
- `action_selection.py` (79 lines) - Shared epsilon_greedy_action_selection() utility with vectorized action masking (multinomial sampling from valid actions, 10-100× faster than Python loops)
- `__init__.py` (17 lines) - Public API exports

**Dependencies:**
- **Inbound** (who depends on this):
  - `population/vectorized.py` (VectorizedPopulation) - Instantiates exploration strategy at init, calls select_actions() every step during rollouts, calls compute_intrinsic_rewards() for DAC integration (though DAC now handles intrinsic internally), calls update() after replay buffer sampling, manages epsilon decay, stores/loads checkpoint state
  - `demo/runner.py` (DemoRunner) - Passes exploration config to population, coordinates checkpoint save/load
  - `demo/live_inference.py` (LiveInference) - Instantiates exploration for inference server (epsilon=0 for greedy actions)
  - `environment/vectorized_env.py` (VectorizedHamletEnv) - Receives exploration module for optional intrinsic reward computation (legacy path, DAC now preferred)
- **Outbound** (what this depends on):
  - `training/state.py` (BatchedAgentState) - Reads per-agent epsilons for action selection
  - PyTorch - GPU tensor operations (forward passes, MSE loss, optimizer steps, action sampling)
  - NumPy - Running statistics for RND reward normalization (Welford's algorithm via RunningMeanStd)

**Patterns Observed:**
- **Strategy Pattern**: ExplorationStrategy ABC enables swapping between RND/AdaptiveIntrinsic/EpsilonGreedy without population code changes
- **Composition over Inheritance**: AdaptiveIntrinsicExploration contains RNDExploration instance, delegates novelty computation (avoids diamond inheritance)
- **Dual Network Architecture (RND)**: Fixed random network (frozen) + predictor network (trained) - prediction error = novelty
- **GPU-Native Vectorization**: All operations on batched tensors [batch_size, ...] on device (no CPU transfers in hot paths)
- **Running Statistics Normalization**: RND uses Welford's online algorithm to normalize intrinsic rewards to comparable scale with extrinsic rewards
- **Mini-Batch Training Accumulation**: RND accumulates observations in obs_buffer (CPU), trains predictor when buffer reaches training_batch_size (decouples rollout from training frequency)
- **Active Mask Pattern**: RNDNetwork accepts active_mask buffer for observation padding dimensions (suggests observation space padding workarounds for structured encoders)
- **Defensive Annealing Gate**: Adaptive intrinsic requires BOTH low variance AND high survival (prevents premature annealing during "stable failure" phase)
- **Checkpointing Support**: All strategies implement checkpoint_state()/load_state() for resumable training (network weights, optimizer state, epsilon, normalization stats, survival history)
- **Hot Path Optimization**: select_actions() marked as hot path (runs every step for all agents) - minimal overhead, no validation, no CPU transfers
- **Shared Utility Extraction**: epsilon_greedy_action_selection() factored out to avoid duplication across 3 strategies (RND/AdaptiveIntrinsic/EpsilonGreedy)

**Concerns:**
- **RND Buffer Management on CPU**: obs_buffer accumulates observations on CPU (lines rnd.py:164, 236-237) before GPU training batch - potential memory pressure for large batch_size × obs_dim configurations, CPU↔GPU transfers not eliminated
- **Global Epsilon Assumption**: action_selection.py operates on per-agent epsilons tensor [batch_size] but epsilon decay methods return single float (rnd.py:302-304, epsilon_greedy.py:93-94) - suggests per-agent epsilon exploration not fully supported, all agents likely share epsilon
- **Active Mask Workaround**: RNDNetwork accepts active_mask to zero out padding dimensions (rnd.py:75-91) - suggests observation space has variable-length fields padded to fixed dim, fragile coupling to observation structure
- **Intrinsic Reward Double-Counting Risk**: compute_intrinsic_rewards() docstring warns "Weight is applied in replay buffer sampling, NOT here" (adaptive_intrinsic.py:112-113) to prevent double-weighting - fragile coordination with DAC engine, requires careful documentation
- **Survival History Unbounded Growth**: AdaptiveIntrinsicExploration.survival_history list keeps only recent window via slicing (adaptive_intrinsic.py:145-146) but checkpoint state includes full history (line 210) - could grow unbounded if training paused/resumed frequently
- **Incremental Window Heuristic**: Adaptive annealing uses incremental window checking (min 10 episodes, lines 163-169) instead of waiting for full survival_window - undocumented behavioral change when window size reached, could cause confusion
- **Hardcoded Novelty Map Grid Size**: get_novelty_map() assumes 8×8 grid (rnd.py:271-300), hardcodes meter dimensions (lines 294), creates placeholder meters (line 294) - visualization utility but fragile assumptions about environment structure
- **No Convergence Guarantees**: RND predictor trained to minimize MSE but no early stopping or convergence criteria - predictor could overfit on limited experience, reducing novelty signal quality
- **Epsilon Decay Not Per-Agent**: decay_epsilon() modifies single global epsilon value (rnd.py:303, epsilon_greedy.py:93) - per-agent exploration rates not supported despite BatchedAgentState.epsilons being tensor
- **Adaptive Threshold Tuning**: Adaptive intrinsic uses hardcoded variance_threshold=100.0, min_survival_fraction=0.4 (adaptive_intrinsic.py:29-30) with comments "Increased from 10.0 to prevent premature annealing" - suggests manual tuning required per environment, not auto-configured

**Confidence:** High - Comprehensive analysis of 5 core files (888 total lines), dependencies mapped via grep across 7 integration points, integration with population/environment/DAC understood through code inspection, RND algorithm (dual networks, prediction error, normalization) traced through implementation, adaptive annealing logic (dual gate: variance + survival) analyzed with survival tracking code, action selection vectorization patterns verified, checkpoint save/load examined for all strategies, concerns identified from comments and implementation patterns (CPU buffer management, epsilon decay scope, double-counting risks).

---

## Curriculum Strategies

**Location:** `src/townlet/curriculum/`

**Responsibility:** Adapts environment difficulty progression based on agent performance metrics (survival rate, learning progress, policy entropy) through configurable stage-based or static difficulty policies.

**Key Components:**
- `base.py` (74 lines) - CurriculumManager ABC defining interface: get_batch_decisions() returns list[CurriculumDecision] (difficulty_level, active_meters, depletion_multiplier, reward_mode), checkpoint_state(), load_state()
- `adversarial.py` (485 lines) - AdversarialCurriculum class implementing 5-stage progression (basic needs → all meters shaped → sparse rewards), PerformanceTracker class for GPU-native per-agent metrics (episode_rewards, episode_steps, prev_avg_reward, last_survival_rate, agent_stages, steps_at_stage, episodes_at_stage), multi-signal decision logic (survival rate > 70% + positive learning progress + policy entropy < 0.5 for advance; survival < 30% OR negative learning for retreat)
- `static.py` (96 lines) - StaticCurriculum class providing trivial baseline (returns same CurriculumDecision for all agents, no adaptation, used for interface validation and ablation studies)
- `__init__.py` (24 lines) - Public API exports

**Dependencies:**
- **Inbound** (who depends on this):
  - `population/vectorized.py` (VectorizedPopulation) - Calls get_batch_decisions_with_qvalues() every step to get depletion_multiplier/active_meters/reward_mode for environment configuration, updates PerformanceTracker.update_step() with survival times after env.step(), manages checkpoint state
  - `demo/runner.py` (DemoRunner) - Instantiates curriculum from config, coordinates checkpoint save/load
  - `demo/live_inference.py` (LiveInference) - Instantiates curriculum for inference server
  - `training/state.py` - Imports CurriculumDecision dataclass (lines 85-92) for structured decision communication
- **Outbound** (what this depends on):
  - `training/state.py` (BatchedAgentState, CurriculumDecision) - Reads agent state for decision logic, returns structured decisions
  - PyTorch - GPU tensor operations for performance tracking (rewards, steps, stages, survival rates all on device)
  - PyYAML - Config file loading via from_yaml() classmethod (adversarial.py:158-184)
  - Pydantic - StageConfig model for stage definitions (adversarial.py:17-24)

**Patterns Observed:**
- **Strategy Pattern**: CurriculumManager ABC enables swapping between Adversarial/Static without population code changes
- **5-Stage Pedagogical Progression**: Stage 1 (energy+hygiene, 20% depletion) → Stage 2 (+satiation, 50%) → Stage 3 (+money, 80%) → Stage 4 (all meters, 100%, shaped) → Stage 5 (sparse rewards graduation)
- **Multi-Signal Decision Logic**: Advances require ALL of: high survival (>70%), positive learning progress (reward improvement), low entropy (<0.5 policy convergence) - prevents premature advancement (adversarial.py:224-229)
- **Dual-Direction Adaptation**: Agents can both advance (when succeeding) and retreat (when struggling) - bidirectional curriculum unlike monotonic progressions
- **GPU-Native Performance Tracking**: PerformanceTracker stores all metrics as GPU tensors [num_agents] for vectorized operations (adversarial.py:70-83)
- **Per-Agent Stage Independence**: Each agent progresses through stages independently based on individual performance (agent_stages tensor [num_agents], adversarial.py:81)
- **Entropy as Convergence Signal**: Uses softmax(Q-values) to compute action distribution entropy, normalized by log(num_actions) to [0,1] range (adversarial.py:381-404) - low entropy indicates policy convergence
- **Transition Event Logging**: Records structured telemetry for all stage transitions (agent_id, from_stage, to_stage, reason, metrics) in transition_events list (adversarial.py:339-363) for analysis
- **Minimum Steps Gate**: Requires min_steps_at_stage (default 1000) before allowing transitions to prevent thrashing (adversarial.py:216-218, 244-246)
- **Welford-Style Statistics**: PerformanceTracker.update_step() updates running averages for baseline comparison (adversarial.py:85-108)
- **Baseline Capture on Transition**: Updates prev_avg_reward only for transitioning agents (adversarial.py:289-301) to avoid polluting learning progress signal
- **YAML Config Loading**: AdversarialCurriculum.from_yaml() provides declarative configuration (adversarial.py:158-184)
- **PyTorch-Style Aliases**: Provides state_dict()/load_state_dict() aliases for checkpoint_state()/load_state() API consistency with PyTorch conventions (adversarial.py:478-484)
- **Testing-Friendly Fallback**: get_batch_decisions() (without Q-values) creates dummy peaked Q-values for testing without full population context (adversarial.py:365-379)

**Concerns:**
- **Hardcoded Stage Configs**: STAGE_CONFIGS list defined in adversarial.py (lines 28-64) with hardcoded meter names ("energy", "hygiene", "satiation", "money", "mood", "social") and multipliers - should be loaded from YAML per No-Defaults Principle, prevents customization without code changes
- **Transition Events Unbounded Growth**: transition_events list (adversarial.py:156, 352-363) accumulates indefinitely during training - no size limit or rotation policy, could cause memory pressure in long-running training (e.g., 1M episodes × 64 agents × 5 stages = 320M events)
- **Meter Name Hardcoding**: Stage configs reference specific meter names ("energy", "hygiene") without validation that these meters exist in universe - could fail silently if custom config uses different meter names
- **Global Entropy Calculation**: _calculate_action_entropy() computes entropy for all agents then indexes per-agent (adversarial.py:381-404) - could optimize with per-agent computation to avoid full batch softmax
- **Stage Advancement Coupling**: _should_advance() requires ALL three conditions (survival + learning + entropy) with no weight tuning - rigid decision rule may not suit all environments, no per-stage threshold configuration
- **Learning Progress Fragility**: Learning progress = current_avg - prev_avg_reward (adversarial.py:115-118) is sensitive to reward scale and noise - single bad episode can trigger retreat even if policy improving overall
- **No Stage Skip Logic**: Agents must progress through all 5 stages sequentially - cannot skip stages even if demonstrating mastery (e.g., agent that immediately achieves >90% survival at Stage 1 still must progress 1→2→3→4→5)
- **Survival Rate Comment Mismatch**: update_step() comment says "rewards here are actually survival_steps" (adversarial.py:88-89) but implementation divides by 100.0 (line 101) - suggests hardcoded max_steps assumption, fragile if max_steps_per_episode changes
- **Dual Decision Methods**: get_batch_decisions() and get_batch_decisions_with_qvalues() both exist (adversarial.py:258-263, 365-379) - former creates dummy Q-values for testing, latter is production path from population - API confusion, callers must know which to use
- **Retreat Priority Asymmetry**: Retreat check happens before advance check (adversarial.py:286-293) giving retreat priority - agent oscillating at threshold could retreat more often than advance (hysteresis not implemented)
- **Initialization Coupling**: initialize_population() must be called before any get_batch_decisions() or tracker methods (adversarial.py:186-189, 206-207) - fragile initialization order, raises RuntimeError if violated, could use lazy initialization
- **Device Parameter Ignored**: device parameter passed to __init__ (adversarial.py:153) but not used to initialize PerformanceTracker until initialize_population() called - inconsistent device handling between constructor and initialization
- **get_stage_info() Single Agent Query**: get_stage_info(agent_idx=0) returns dict for single agent (adversarial.py:434-476) - no batch API to query all agents simultaneously, forces loop for population-wide analysis
- **Normalized Difficulty Mapping**: Stage (1-5) mapped to difficulty_level (0.0-1.0) via (stage-1)/4.0 (adversarial.py:322) - linear mapping assumes uniform difficulty spacing across stages, actual difficulty may be non-linear (sparse rewards in Stage 5 much harder than Stage 4)

**Confidence:** High - Comprehensive analysis of 3 core files (655 total lines), dependencies mapped via grep across 6 integration points, 5-stage progression logic traced through STAGE_CONFIGS and decision methods, multi-signal advancement criteria (survival + learning + entropy) analyzed with PerformanceTracker implementation, GPU tensor tracking patterns verified, transition event logging examined, integration with population training loop understood through get_batch_decisions_with_qvalues() calls, static curriculum analyzed as trivial baseline, concerns identified from hardcoded configs and comments about survival_steps division.

---

## Recording & Replay System

**Location:** `src/townlet/recording/`

**Responsibility:** Capture, compress, store, and replay training episode trajectories with selective recording criteria for documentation, debugging, and video export.

**Key Components:**
- `recorder.py` (308 lines) - EpisodeRecorder (non-blocking async queue producer) + RecordingWriter (background thread consumer for msgpack serialization and LZ4 compression)
- `criteria.py` (225 lines) - RecordingCriteria evaluator with 4 criterion types: periodic (every N episodes), stage_transitions (before/after curriculum changes), performance (top/bottom percentile rewards), stage_boundaries (first/last N episodes per stage)
- `replay.py` (223 lines) - ReplayManager for loading/decompressing msgpack.lz4 files with step-by-step playback control (seek, next, reset, unload)
- `data_structures.py` (109 lines) - Frozen dataclasses (RecordedStep with __slots__ for ~100-150 bytes/step, EpisodeMetadata for criteria evaluation, EpisodeEndMarker sentinel)
- `video_export.py` (253 lines) - MP4 export pipeline via FFmpeg (H.264, yuv420p, CRF 18 for YouTube) with single-episode and batch export modes
- `video_renderer.py` (343 lines) - Matplotlib-based frame rendering with 16:9 YouTube-optimized layout (grid view, meter bars, Q-values chart, episode info panel)
- `__main__.py` (148 lines) - CLI entry point providing `python -m townlet.recording {export,batch}` commands

**Dependencies:**
- **Inbound** (who uses this):
  - `demo/runner.py` (DemoRunner) - Instantiates EpisodeRecorder when recording.enabled=true in config, calls record_step() during env.step() for selected agents, finish_episode() on termination, shutdown() during cleanup
  - CLI users - `python -m townlet.recording export <episode_id>` for single video, `batch --stage 3 --min-reward 100` for filtered batch exports
  - Tests - `tests/test_townlet/{unit,integration}/recording/*.py` (9 test files covering recorder, replay, criteria, video export)
- **Outbound** (what this depends on):
  - `demo/database.py` (DemoDatabase) - Indexes recordings in SQLite (episode_id, file_path, metadata, reason, file_size, compressed_size for queryable catalog)
  - `curriculum/{base,adversarial,static}.py` - Queries curriculum.get_stage_info() for stage_transitions criterion lookahead (predict imminent transitions)
  - msgpack + lz4.frame - Serialization/compression stack achieving ~10-30× size reduction (compression_level=0 for speed)
  - FFmpeg - External video encoder (subprocess calls to ffmpeg with glob pattern input for frame sequences)
  - Matplotlib - Frame rendering (matplotlib.use("Agg") non-interactive backend, Figure/Canvas for numpy array generation)
  - PyTorch - Tensor cloning (.tolist() conversion) to prevent GPU blocking during record_step() hot path

**Patterns Observed:**
- **Producer-Consumer Pattern**: EpisodeRecorder.queue (bounded Queue maxsize=1000) enables non-blocking record_step() from training loop (producer); RecordingWriter.writer_loop() runs in daemon thread handling expensive I/O (consumer)
- **Graceful Degradation**: Queue full → log warning + drop frame (recorder.py:141-143) instead of blocking training loop (prioritizes training throughput over recording completeness)
- **OR Logic Criteria**: RecordingCriteria.should_record() returns True if ANY enabled criterion matches (periodic OR stage_transitions OR performance OR stage_boundaries)
- **Frozen Dataclasses**: RecordedStep(frozen=True, slots=True) for minimal memory footprint (~100-150 bytes/step) and immutability guarantees
- **Two-Phase Compression**: msgpack.packb() serialization → lz4.frame.compress() (~10-30× reduction for typical episodes with repetitive state data)
- **Database Indexing**: SQLite metadata enables queryable recordings (ReplayManager.list_recordings() filters by stage/reason/reward thresholds with LIMIT clause)
- **Stateful Criteria Evaluation**: RecordingCriteria maintains deque history (episode_history for performance percentile), set tracking (transition_episodes for transition windows), stage counters (stage_episode_counts for boundaries)
- **Lookahead via Curriculum Integration**: stage_transitions criterion calls curriculum.get_stage_info(agent_idx=0) to predict imminent transitions (likely_transition_soon flag) enabling pre-recording before stage change
- **CLI Subcommand Pattern**: export (single episode) vs batch (query + filter + export loop) commands with shared parameters (fps, speed, dpi, style)
- **Matplotlib Headless Rendering**: matplotlib.use("Agg") backend → Figure/FigureCanvasAgg → canvas.buffer_rgba() → numpy array → PIL.Image.save() pipeline
- **16:9 Grid Layout**: GridSpec(2, 3, width_ratios=[2,1,1], height_ratios=[3,1]) with grid view (left, spans 2 rows), meters/info (top-right), Q-values (bottom-right spanning 2 columns)
- **FFmpeg Subprocess Integration**: Glob pattern input (-pattern_type glob -i "frame_*.png"), H.264 codec (libx264, yuv420p pixel format, CRF 18 high quality, preset slow)
- **Tensor Cloning Hot Path**: record_step() clones GPU tensors to Python tuples (positions.tolist(), meters.tolist()) before queueing to avoid blocking training loop on I/O

**Concerns:**
- **Criteria Evaluator Not Integrated**: RecordingCriteria fully implemented in criteria.py (4 criterion types with OR logic) but RecordingWriter._should_record_episode() uses simple inline periodic check (recorder.py:254-262) instead of calling criteria evaluator - suggests partial implementation (TASK-005A Phase 2 incomplete)
- **Hardcoded 8 Meter Assumption**: video_renderer.py._render_meters() assumes 8 standard meters with hardcoded names ["energy", "hygiene", "satiation", "money", "health", "fitness", "mood", "social"] (line 216) - fragile if custom configs use different meter counts/names, no dynamic meter name resolution from metadata
- **Grid Size Auto-Detection Fragility**: video_export.py infers grid size from max(affordance_positions) (lines 69-76) - could fail if affordances don't span full grid (e.g., 8×8 grid with affordances only in 3×3 corner), no fallback to substrate.yaml
- **No Aspatial Rendering Support**: video_renderer.py._render_grid() assumes 2D positions (agent_x, agent_y from position[0], position[1], lines 186-187) - would crash on aspatial substrates (position_dim=0, empty position tuple), no conditional logic for substrate type
- **Queue Size Not Configurable**: max_queue_size defaults to 1000 (recorder.py line 51) with no config override - could cause frame drops in high-frequency recording (e.g., recording every step instead of every episode) or large batch sizes
- **Transition Episodes Unbounded Growth**: RecordingCriteria.transition_episodes set accumulates episode IDs indefinitely (criteria.py line 224) - memory leak for long-running training (1M episodes = 1M integers in set), no rotation or cleanup policy
- **Performance Window Fixed Size**: deque(maxlen=window_size) with window_size default 100 (criteria.py line 50) - recent 100 episodes may not be statistically significant for percentile calculation in early training or small batch sizes
- **Compression Level Hardcoded**: lz4.frame.compress(compression_level=0) (recorder.py line 285) - fast but lower compression ratio (~10-15× instead of 20-30×), no config tuning for storage vs speed tradeoff
- **FFmpeg Hard Dependency**: video_export.py subprocess calls require FFmpeg installed (lines 136-145) with no graceful fallback or pure-Python alternative (e.g., opencv-python), dependency not enforced in pyproject.toml
- **Database Coupling**: ReplayManager requires DemoDatabase instance (replay.py line 26) instead of generic metadata store interface - tightly coupled to demo subsystem instead of standalone recording library
- **No Video Progress Reporting**: export_episode_video() renders all frames before encoding (lines 87-108) with no progress callback - user sees no feedback during long episode exports (e.g., 1000-step episode = 1000 frames × rendering time)
- **Matplotlib Font Warnings**: ACTION_NAMES uses simple characters to "avoid font glyph warnings with DejaVu Sans" (video_renderer.py line 46) suggesting arrow unicode (↑↓←→) may not render on all systems - platform-dependent font issues
- **Episode Buffer Accumulation**: RecordingWriter.episode_buffer list accumulates all steps in memory (recorder.py line 193) before serialization - potential memory pressure for very long episodes (e.g., 10K-step episodes × 100 bytes/step = 1MB per episode × queue backlog)
- **Stage Info Single Agent Query**: stage_transitions criterion calls get_stage_info(agent_idx=0) (criteria.py line 146) - always queries first agent, may not represent population-wide transition behavior in multi-agent curriculum
- **No Frame Rate Validation**: export_episode_video() accepts arbitrary fps parameter (default 30) with no validation that speed × fps produces reasonable effective framerate (e.g., speed=10.0 × fps=30 = 300 fps may exceed display capabilities)

**Confidence:** Medium-High - Comprehensive analysis of all 7 recording files (1609 total lines), dependencies mapped via grep across 21 files, integration with DemoRunner confirmed via code inspection (lines 477-489 in runner.py), compression/serialization stack understood (msgpack→lz4 pipeline), video export pipeline traced (Matplotlib→PNG→FFmpeg), criteria evaluation logic analyzed (4 types with OR logic + stateful tracking), replay manager playback control examined (seek/next/reset API), concerns identified from implementation gaps (criteria.py not called by recorder.py) and hardcoded assumptions (8 meters, grid size inference, aspatial unsupported). Medium rating due to uncertainty about RecordingCriteria integration status (fully implemented but only simple periodic check used in production path, suggesting TASK-005A Phase 2 incomplete).

---

## Training Infrastructure

**Location:** `src/townlet/training/`

**Responsibility:** Provides stateful training management through experience replay buffers, checkpoint persistence with provenance verification, training state DTOs, and TensorBoard metrics logging for reproducible deep RL experiments.

**Key Components:**
- `state.py` (159 lines) - Training state DTOs: BatchedAgentState (hot path GPU tensors with __slots__ for performance), CurriculumDecision (cold path Pydantic immutable config), PopulationCheckpoint (serializable population state with curriculum/exploration/Pareto frontier metadata), ExplorationConfig (epsilon-greedy/RND/adaptive intrinsic parameters)
- `checkpoint_utils.py` (165 lines) - Checkpoint security and provenance: attach_universe_metadata (embeds config_hash/drive_hash/observation_field_uuids/dimensions), persist_checkpoint_digest/verify_checkpoint_digest (SHA256 verification with 1MiB chunked hashing), safe_torch_load (weights_only=True with numpy type allowlisting for PyTorch 2.6+), assert_checkpoint_dimensions (validates obs_dim/action_dim/field UUIDs match universe)
- `replay_buffer.py` (216 lines) - ReplayBuffer class: FIFO circular buffer with dual rewards (extrinsic + intrinsic stored separately, combined at sample time with configurable intrinsic_weight), GPU-native tensor storage [capacity, obs_dim], lazy initialization on first push, serialize/load_from_serialized for checkpointing
- `sequential_replay_buffer.py` (249 lines) - SequentialReplayBuffer class: Episode-based replay for LSTM training, stores complete episodes with temporal structure, samples consecutive sequences of length seq_len, post-terminal masking (mask[terminal_idx+1:] = False) to prevent gradients from garbage observations, FIFO eviction by num_transitions capacity
- `prioritized_replay_buffer.py` (206 lines) - PrioritizedReplayBuffer class (Schaul et al. 2016): TD-error-based sampling (priority = |TD_error| + 1e-6), alpha exponent (0=uniform, 1=full prioritization), beta importance sampling weights (anneals 0.4→1.0 over training), new transitions assigned max_priority, update_priorities after each training batch, CPU storage with Python lists (not GPU-native like other buffers)
- `tensorboard_logger.py` (340 lines) - TensorBoardLogger class: Structured metrics logging (log_episode for survival/rewards/curriculum/epsilon, log_training_step for TD_error/Q_values/loss, log_network_stats for weights/gradients/learning_rate, log_curriculum_transitions for stage advancement rationale), auto-flush every N episodes, context manager support, multi-agent logging with agent_id prefixes, histogram/gradient logging toggles
- `__init__.py` (2 lines) - Package docstring only (no exports)

**Dependencies:**
- **Inbound** (who depends on this):
  - `population/vectorized.py` (VectorizedPopulation) - Uses all 3 replay buffer types (ReplayBuffer for feedforward, SequentialReplayBuffer for LSTM, PrioritizedReplayBuffer for importance sampling), creates BatchedAgentState for training loop hot path, reads CurriculumDecision from curriculum manager, serializes PopulationCheckpoint for save_checkpoint()
  - `demo/runner.py` (DemoRunner) - Uses checkpoint_utils (attach_universe_metadata, persist_checkpoint_digest, verify_checkpoint_digest, safe_torch_load, assert_checkpoint_dimensions, config_hash_warning) for checkpoint save/load/validation, creates TensorBoardLogger for training metrics, passes BatchedAgentState between population/environment
  - `demo/live_inference.py` (LiveInference) - Uses checkpoint_utils for loading trained checkpoints, creates BatchedAgentState for inference loop
  - `curriculum/{adversarial,static}.py` - Returns CurriculumDecision from get_batch_decisions() methods
  - `exploration/{rnd,adaptive_intrinsic,epsilon_greedy}.py` - Uses ExplorationConfig (though currently defined in state.py, may migrate to exploration/config.py)
  - Tests - `tests/test_townlet/{unit,integration}/training/*.py`, `tests/test_townlet/integration/test_tensorboard_logger.py` (comprehensive coverage of all buffers, checkpoint utils, logger)
- **Outbound** (what this depends on):
  - `universe/compiled.py` (CompiledUniverse) - checkpoint_utils validates universe metadata (config_hash, drive_hash, observation_dim, action_dim, meter_count, observation_field_uuids) against checkpoints
  - PyTorch - GPU tensor operations (torch.Tensor, torch.device, torch.randint for sampling, torch.stack for batching), checkpoint serialization (torch.save/load), weights_only safety, SummaryWriter for TensorBoard
  - Pydantic - Cold path validation (BaseModel, Field, ConfigDict for CurriculumDecision/PopulationCheckpoint/ExplorationConfig with frozen=True immutability)
  - NumPy - PrioritizedReplayBuffer priorities array (np.float32), importance sampling weights, random.choice for sampling
  - Standard library - hashlib (SHA256 checkpoint digests), pathlib (file operations), logging (checkpoint warnings)

**Patterns Observed:**
- **Hot Path / Cold Path Separation**: BatchedAgentState uses __slots__ + GPU tensors for training loop performance (no validation overhead), Pydantic models (CurriculumDecision/PopulationCheckpoint) for serialization/config with validation (frozen=True immutability)
- **Replay Buffer Strategy Pattern**: Three implementations (ReplayBuffer for feedforward uniform sampling, SequentialReplayBuffer for LSTM temporal structure, PrioritizedReplayBuffer for TD-error importance sampling) with common interface (push, sample, serialize, load_from_serialized, __len__)
- **Dual Reward Storage**: Buffers store rewards_extrinsic and rewards_intrinsic separately, combine at sample time with configurable intrinsic_weight to avoid double-counting (DAC already includes intrinsic in env.step() rewards, so population stores zeros for intrinsic_rewards in some paths)
- **Checkpoint Provenance Tracking**: attach_universe_metadata embeds config_hash (SHA256 of config files), drive_hash (SHA256 of drive_as_code.yaml), observation_field_uuids (VFS field identifiers), meter_count/names, obs_dim, action_dim for reproducibility verification and transfer learning compatibility checks
- **SHA256 Digest Verification**: persist_checkpoint_digest computes hash in 1MiB chunks (memory-bounded for large checkpoints), stores as .pt.sha256 sidecar file, verify_checkpoint_digest checks integrity with optional required=True for strict validation
- **Safe Torch Loading**: safe_torch_load uses weights_only=True by default (prevents arbitrary code execution), allowlists numpy types (np.dtype, numpy.core.multiarray.scalar) for PyTorch 2.6+ compatibility, falls back to weights_only=False for trusted test checkpoints
- **Post-Terminal Masking**: SequentialReplayBuffer creates mask tensor [batch_size, seq_len] with True up to terminal transition, False after (lines 167-178) to prevent LSTM gradients from post-terminal garbage observations (P2.2 pattern for variable-length episodes)
- **Episode Accumulation for Recurrent**: SequentialReplayBuffer stores complete episodes (dict with observations/actions/rewards/dones tensors [seq_len, ...]), samples consecutive sequences to maintain LSTM hidden state context (no mid-episode random sampling)
- **Prioritized Replay with Annealing**: PrioritizedReplayBuffer anneals beta importance sampling exponent from 0.4 to 1.0 over training (corrects for non-uniform sampling bias increasingly over time), updates priorities after each batch with |TD_error| + 1e-6 (epsilon prevents zero priorities)
- **TensorBoard Hierarchical Logging**: Logger uses hierarchical tags (agent_id/Category/Metric) for multi-agent scenarios, logs episode metrics (survival/rewards/curriculum), training metrics (TD_error/loss/Q_values), network stats (weights/gradients/learning_rate), curriculum transitions (stage changes with rationale)
- **Lazy Tensor Initialization**: ReplayBuffer initializes storage tensors on first push (lines 54-60) to infer obs_dim from observations shape, avoids pre-allocation without knowing dimensions
- **Circular Buffer FIFO**: ReplayBuffer uses position % capacity for circular indexing (lines 80-90), SequentialReplayBuffer evicts oldest episodes when num_transitions > capacity (lines 92-94)
- **Serialization for Checkpointing**: All buffers implement serialize() → dict (tensors moved to CPU) and load_from_serialized(dict) → restore to device for training resumption (P1.1 pattern)
- **Context Manager Support**: TensorBoardLogger implements __enter__/__exit__ for automatic flush/close (lines 333-339), enables with TensorBoardLogger(...) as logger: pattern

**Concerns:**
- **PER CPU Storage Inefficiency**: PrioritizedReplayBuffer uses Python lists for transitions (lines 42-46) and NumPy array for priorities (line 49) with CPU storage, not GPU-native like ReplayBuffer/SequentialReplayBuffer - frequent CPU↔GPU transfers during push (lines 81-93 .cpu() calls) and sample (lines 121-125 .to(device) calls), potential bottleneck for large batches
- **PER Beta Annealing Hardcoded Range**: anneal_beta uses hardcoded 0.4→1.0 range (line 155), no configuration via ExplorationConfig or brain.yaml, prevents experimentation with different annealing schedules
- **Sequential Buffer Memory Scaling**: SequentialReplayBuffer stores full episodes in memory (list[Episode]), potential memory pressure for long episodes (e.g., max_steps_per_episode=1000 × num_agents=64 × capacity=10000 = ~640 episodes × 1000 steps = 640K transitions), no episode length limit
- **Checkpoint Digest Optional by Default**: verify_checkpoint_digest has required=False default (line 88), only logs warning if digest missing, should be required=True for production training
- **No Checkpoint Retention Policy**: checkpoint_utils has no automatic cleanup of old checkpoints, DemoRunner saves every CHECKPOINT_INTERVAL=100 episodes, long training runs accumulate unbounded checkpoints
- **TensorBoard Multi-Agent Logging Untested**: Logger has agent_id parameter throughout for multi-agent prefixing, but HAMLET is currently single-agent, multi-agent logging patterns not validated in integration tests
- **Dual Reward Coordination Fragility**: ReplayBuffer/SequentialReplayBuffer store rewards_extrinsic and rewards_intrinsic separately, but DAC engine already combines them in env.step(), population must store zeros for intrinsic_rewards to avoid double-counting - fragile implicit contract
- **ExplorationConfig in Wrong Module**: ExplorationConfig defined in training/state.py (lines 32-51) but logically belongs in exploration/config.py, violates module cohesion
- **SequentialReplayBuffer Episode Length Validation**: sample_sequences raises ValueError if no episodes >= seq_len with error message hinting test workarounds (lines 133-135) rather than addressing root cause
- **PER Max Priority Unbounded Growth**: max_priority updated with max(self.max_priority, priorities.max()) (line 144), never decays, could grow unbounded if early training has spurious high TD-errors
- **BatchedAgentState No Device Validation**: __init__ accepts tensors and device separately (lines 94-106), no validation that all tensors are actually on the specified device
- **Checkpoint Dimension Mismatch Inconsistent Handling**: assert_checkpoint_dimensions raises ValueError for obs_dim/action_dim mismatch, but config_hash_warning only returns warning string - inconsistent error handling (assert vs warn)
- **SequentialReplayBuffer Non-Uniform Sampling**: random.choice samples episodes uniformly (line 143), then random.randint samples start position (line 148), but this doesn't give uniform sampling over all sequences - longer episodes oversampled
- **TensorBoard Flush Timing Unpredictable**: Auto-flush triggers every flush_every episodes (lines 120-122), but if training crashes mid-interval, recent metrics lost, no time-based backup flush

**Confidence:** High - Comprehensive analysis of all 7 files (1335 total lines), dependencies mapped via grep across 38 files mentioning replay buffers, checkpoint utils usage traced in runner.py/population/vectorized.py, replay buffer strategy pattern verified across all 3 implementations, checkpoint provenance tracking (config_hash/drive_hash/observation_field_uuids) examined, post-terminal masking pattern for LSTM training understood, PER algorithm (alpha/beta/priorities/importance sampling) analyzed from Schaul et al. 2016 reference, TensorBoard integration verified in runner.py and test files, serialization support confirmed in all buffers, concerns identified from code inspection (PER CPU storage, no checkpoint cleanup, dual reward coordination fragility, SequentialReplayBuffer non-uniform sampling).

---


## Demo & Inference

**Location:** `src/townlet/demo/`

**Responsibility:** Orchestrates multi-day training with live inference server, providing step-by-step WebSocket visualization, checkpoint management, SQLite metrics persistence, and optional episode recording/replay for frontend real-time monitoring.

**Key Components:**
- `unified_server.py` (543 lines) - UnifiedServer class coordinating training thread (DemoRunner.run) and inference thread (LiveInferenceServer with uvicorn) with graceful shutdown via threading coordination, config snapshot provenance, file logging to run directory
- `runner.py` (951 lines) - DemoRunner class providing multi-day training orchestration with checkpoint save/load (version 3 format with universe_metadata/brain_hash/drive_hash/agent_ids/affordance_layout), SQLite persistence (episode metrics, affordance transitions), TensorBoard logging (hyperparameters, episode metrics, training metrics, meter dynamics, affordance usage), episode recording integration (optional EpisodeRecorder with msgpack+lz4 compression), context manager cleanup (__enter__/__exit__ for database/TensorBoard/recorder)
- `live_inference.py` (1188 lines) - LiveInferenceServer class running FastAPI WebSocket server for real-time agent visualization with auto-checkpoint loading (polls checkpoint_dir for new .pt files), dual mode support ("inference" runs trained checkpoint with epsilon-greedy action selection, "replay" plays back recorded episodes from database+msgpack files), substrate-agnostic rendering (_build_substrate_metadata returns type-specific metadata for Grid2D/Grid3D/GridND/Continuous/Aspatial dispatch), telemetry broadcasting (per-agent runtime registry snapshots with survival_time/epsilon/intrinsic_weight/curriculum_stage), replay controls (play/pause/step/seek commands)
- `database.py` (408 lines) - DemoDatabase class providing SQLite backend with thread-safe WAL mode (check_same_thread=False for concurrent readers) for 5 tables: episodes (survival_time, rewards, curriculum_stage, epsilon), affordance_visits (transition counts from_affordance→to_affordance), position_heatmap (visit counts + novelty values), system_state (key-value pairs for last_checkpoint/training_status), episode_recordings (file_path, metadata, compression stats), context manager pattern (__enter__/__exit__ for connection cleanup)

**Dependencies:**
- **Inbound** (who depends on this):
  - `scripts/run_demo.py` - Main entry point instantiates UnifiedServer with CLI args, registers signal handlers for graceful shutdown
  - `recording/replay.py` (ReplayManager) - Uses DemoDatabase for episode query and recording file lookup
  - `recording/video_export.py` - Uses DemoDatabase for episode metadata during video generation
  - Tests: 11 test files across unit/demo/ and integration/

- **Outbound** (what this depends on):
  - Universe compiler: `universe/compiler.py` (UniverseCompiler.compile), `universe/compiled.py` (CompiledUniverse.to_runtime)
  - Environment: `environment/vectorized_env.py` (VectorizedHamletEnv.from_universe)
  - Population: `population/vectorized.py` (VectorizedPopulation)
  - Curriculum: `curriculum/adversarial.py` (AdversarialCurriculum)
  - Exploration: `exploration/adaptive_intrinsic.py` (AdaptiveIntrinsicExploration)
  - Training infrastructure: `training/{checkpoint_utils,state,tensorboard_logger}.py`
  - Agent: `agent/brain_config.py` (load_brain_config, compute_brain_hash)
  - Config DTOs: `config/hamlet.py` (HamletConfig.load)
  - Recording: `recording/{recorder,replay,data_structures}.py`
  - Substrate: `substrate/{base,grid2d,grid3d,gridnd,continuous,aspatial}.py`
  - FastAPI/Uvicorn: WebSocket server (0.0.0.0:{port}/ws endpoint)
  - SQLite: Database persistence with WAL mode
  - PyYAML/PyTorch/TensorBoard: Config loading, checkpoints, metrics logging

**Patterns Observed:**
- **Thread-Based Orchestration**: UnifiedServer spawns training thread (DemoRunner.run) and inference thread (LiveInferenceServer with uvicorn) with coordinated shutdown via shutdown_requested flag and threading.Lock
- **Context Manager Cleanup**: DemoRunner implements __enter__/__exit__ for guaranteed resource cleanup (database, TensorBoard, recorder) even on exceptions
- **Checkpoint Provenance**: Embeds universe_metadata, brain_hash, drive_hash, agent_ids, affordance_layout for reproducibility verification
- **Auto-Checkpoint Loading**: LiveInferenceServer polls checkpoint_dir for new .pt files, broadcasts "model_loaded" to WebSocket clients
- **Dual Mode Inference Server**: Supports "inference" (runs trained checkpoint) and "replay" (plays back recorded episodes) modes with same WebSocket protocol
- **Substrate-Agnostic Rendering**: _build_substrate_metadata() returns type-specific metadata for Grid2D/Grid3D/GridND/Continuous/Aspatial dispatch
- **Config Snapshot Provenance**: _persist_config_snapshot() copies entire config pack to run_root/config_snapshot for audit trail
- **Heartbeat + Summary Logging**: HEARTBEAT_INTERVAL=10 episodes (survival/reward/epsilon), SUMMARY_INTERVAL=50 episodes (meters/affordances/training metrics)
- **Periodic Checkpointing**: CHECKPOINT_INTERVAL=100 episodes with SHA256 digest for integrity verification
- **Multi-Agent Telemetry**: Population builds telemetry snapshots with per-agent runtime registry state broadcast via WebSocket
- **Graceful Shutdown**: Signal handlers set should_shutdown flag, DemoRunner finishes current episode, saves final checkpoint before exit
- **Pre-Flight Validation**: _validate_checkpoint_compatibility() checks for Version 3 format, raises ValueError for old checkpoints
- **Thread-Safe Database**: check_same_thread=False + WAL mode for concurrent readers and single writer
- **Episode Recording Integration**: Optional EpisodeRecorder (msgpack + lz4 compression) for trajectory capture
- **Dynamic Meter Support**: Reads meter_name_to_index to support variable meter counts across curriculum levels
- **Affordance Transition Tracking**: Records from_affordance→to_affordance transitions for Markov chain analysis

**Concerns:**
- **Frontend Subprocess Code Unused**: _start_frontend() and _stop_frontend() methods exist but never called, dead code should be removed
- **Dual Training Config Paths**: DemoRunner loads config twice (HamletConfig.load + yaml.safe_load), DTO layer should handle all sections
- **Large Monolithic Runner**: DemoRunner.run() spans 566 lines, could extract RunnerTrainingLoop, RunnerMetrics, RunnerCheckpointing classes
- **Training Loop Reward Accounting**: Complex episode_reward accumulation with intrinsic_weight factoring increases double-counting risk
- **Hardcoded Grid Size in Replay**: _send_replay_step() hardcodes width=8, height=8, breaks for L0_0_minimal (3×3), L0_5 (7×7)
- **Q-Value Padding for Legacy Checkpoints**: Pads Q-values with NaN for <6 action checkpoints, should fail loudly per pre-release policy
- **Affordance Randomization at Episode 5000**: Hardcoded generalization test, should be configurable or disabled
- **Context Manager Not Enforced**: DemoRunner implements __enter__/__exit__ but main invocation doesn't use "with" statement
- **Inference Server Dual Config Loading**: LiveInferenceServer._initialize_components() recompiles universe, should accept CompiledUniverse
- **No Checkpoint Episode Validation**: Doesn't validate checkpoint_episode < total_episodes, could show incorrect progress
- **Telemetry Schema Version Unused**: TELEMETRY_SCHEMA_VERSION defined but not validated, no version negotiation
- **Replay Mode Hardcoded Assumptions**: Assumes episode_id, affordances dict structure, no version field in recordings
- **WebSocket Message Size Unbounded**: Sends full telemetry snapshot every step, could exceed limits for large num_agents
- **Shutdown Timeout Hardcoded**: 10s for inference, 30s for training, may not finish checkpoint save for long episodes
- **Database Close Not Enforced**: db.close() in try/except suppresses errors if already closed
- **Epsilon Calculation Inconsistency**: Loads checkpoint["epsilon"] if present, falls back to linear decay estimate
- **Temporal Mechanics Always Present**: Sends temporal fields even if enable_temporal_mechanics=False
- **Action Label Padding**: Pads action_masks to 6 actions assuming legacy configs, should read action_dim
- **No Rate Limiting**: WebSocket broadcast in tight asyncio loop, no backpressure mechanism

**Confidence:** High - Comprehensive analysis of 4 core files (3090 total lines), entry point script analyzed, dependencies mapped via grep across 40 import sites, integration patterns understood, database schema with 5 tables examined, checkpoint provenance traced, WebSocket protocol verified, test coverage confirmed, concerns identified from hardcoded values and code comments.

---

## Configuration DTO Layer

**Location:** `src/townlet/config/` (primary), `src/townlet/substrate/config.py`, `src/townlet/environment/action_config.py`

**Responsibility:** Enforces the "no-defaults principle" by providing Pydantic DTO schemas that validate all behavioral parameters from YAML configs, ensuring operator accountability and reproducible training configurations.

**Key Components:**
- `hamlet.py` (252 lines) - HamletConfig master DTO composing all section configs with cross-config validation (batch_size vs replay_buffer_capacity, network_type vs partial_observability, grid_capacity warnings)
- `training.py` (289 lines) - TrainingConfig for Q-learning hyperparameters (device, max_episodes, train_frequency, batch_size, epsilon-greedy params, sequence_length), rejects brain-managed fields (target_update_frequency, use_double_dqn) when brain.yaml exists
- `environment.py` (130 lines) - TrainingEnvironmentConfig for observability (partial_observability, vision_range), affordances (enabled_affordances, randomize_affordances), energy costs (move/wait/interact depletion)
- `population.py` (151 lines) - PopulationConfig for agent count, network architecture (simple/recurrent/structured), Q-learning params, rejects brain-managed fields (learning_rate, gamma, replay_buffer_capacity) when brain.yaml exists
- `curriculum.py` (70 lines) - CurriculumConfig for adversarial difficulty progression (max_steps_per_episode, survival thresholds, entropy_gate, min_steps_at_stage)
- `exploration.py` (76 lines) - ExplorationConfig for RND (embed_dim) and intrinsic motivation annealing (initial_intrinsic_weight, variance_threshold, min_survival_fraction, survival_window)
- `bar.py` (110 lines) - BarConfig for meter definitions (name, index, tier, range, initial, base_depletion) with structural validation
- `cascade.py` - CascadeConfig for meter relationship definitions (source/target meters, rates, conditions)
- `affordance.py` (132 lines) - AffordanceConfig for affordance structure (id, name, costs, costs_per_tick, duration_ticks, operating_hours, modes, availability, capabilities, effect_pipeline, position)
- `drive_as_code.py` (678 lines) - DriveAsCodeConfig and 18 sub-DTOs for declarative reward functions: RangeConfig/ModifierConfig (contextual adjustment), BarBonusConfig/VariableBonusConfig/ExtrinsicStrategyConfig (9 extrinsic types), IntrinsicStrategyConfig (5 intrinsic types), 11 shaping bonus configs (ApproachRewardConfig, CompletionBonusConfig, EfficiencyBonusConfig, StateAchievementConfig, VFSVariableBonusConfig, StreakBonusConfig, DiversityBonusConfig, TimingBonusConfig, EconomicEfficiencyConfig, BalanceBonusConfig, CrisisAvoidanceConfig), CompositionConfig (normalize, clip, log_components)
- `base.py` (92 lines) - Common utilities: load_yaml_section (YAML file loading with helpful errors), format_validation_error (wraps Pydantic errors with operator-friendly context)
- `cues.py` - CuesConfig for UI metadata validation (SimpleCueConfig, CompoundCueConfig, VisualCueConfig)
- `capability_config.py` - CapabilityConfig for affordance advanced behaviors
- `effect_pipeline.py` - EffectPipeline for cascading affordance effects
- `affordance_masking.py` - BarConstraint/ModeConfig for affordance availability rules
- `substrate/config.py` (407 lines) - SubstrateConfig for spatial topology with 4 substrate types: GridConfig (square/cubic, width/height/depth, boundary, distance_metric, observation_encoding), GridNDConfig (dimension_sizes 4-100D, hypercube topology), ContinuousConfig (dimensions 1-100D, bounds, movement_delta, interaction_radius), AspatialConfig (no positioning)
- `environment/action_config.py` (140 lines) - ActionConfig for composable action space (id, name, type, costs, effects, delta, teleport_to, enabled, VFS reads/writes), ActionSpaceConfig collection wrapper

**Dependencies:**
- **Inbound** (who uses Config DTOs):
  - `universe/compiler.py` (UniverseCompiler) - Consumes all DTOs via RawConfigs loader for 7-stage compilation pipeline
  - `universe/compiler_inputs.py` (RawConfigs) - Aggregates DTOs (bars, cascades, affordances, DAC, substrate, cues) for Stage 1 parsing
  - `demo/runner.py` (DemoRunner) - Loads HamletConfig.load(config_dir) for training orchestration
  - `universe/symbol_table.py` (UniverseSymbolTable) - Registers entities from DTOs for cross-validation
  - `universe/cues_compiler.py` (CuesCompiler) - Validates CuesConfig in Stage 4
  - All subsystems consume DTOs indirectly via CompiledUniverse artifact (environment, population, exploration, curriculum)
- **Outbound** (what DTOs depend on):
  - Pydantic - Schema validation (BaseModel, Field, model_validator, field_validator, ValidationError, ConfigDict, PrivateAttr)
  - PyYAML - Config file parsing (yaml.safe_load in load_yaml_section)
  - VFS schemas - `vfs/schema.py` (VariableDef, WriteSpec) for variable integration in ActionConfig
  - Environment schemas - `environment/cascade_config.py` (BarsConfig), `environment/affordance_config.py` (AffordanceConfigCollection) for legacy compatibility type aliases

**Patterns Observed:**
- **No-Defaults Principle Enforcement**: All behavioral fields required with explicit values; only metadata (description, teaching_note) and computed values (observation_dim) exempted; operators must consciously choose every parameter
- **Permissive Validation Philosophy**: Structural errors fail (missing fields, wrong types, constraint violations), semantic oddities warn (epsilon_decay too slow/fast, POMDP vision_range=0, unusual network_type for observability mode) without blocking
- **Cross-Config Validation**: HamletConfig.validate_batch_size_vs_buffer loads brain.yaml to check batch_size <= replay_buffer_capacity; validate_recurrent_network_consistency warns if network_type mismatches partial_observability; validate_grid_capacity warns if agents+affordances > grid cells
- **Brain As Code Integration**: TrainingConfig.reject_brain_managed_fields and PopulationConfig.reject_brain_managed_fields raise ValueError if brain-managed fields (learning_rate, gamma, target_update_frequency, use_double_dqn, replay_buffer_capacity) present when brain.yaml exists, enforcing single source of truth
- **Config-Dir Sentinel Pattern**: Pass _config_dir as sentinel in data dict for brain.yaml existence checks in @model_validator(mode="before"), removed before Pydantic validation to avoid "extra field" errors
- **Loader Functions**: Each DTO has load_*_config(config_dir) function handling FileNotFoundError (with helpful path info) and ValidationError (with formatted context via format_validation_error)
- **Error Message Formatting**: format_validation_error wraps Pydantic errors with operator guidance ("All parameters must be explicitly specified", "See configs/templates/ for annotated examples")
- **Nested DTO Composition**: HamletConfig.load orchestrates loading 10 section configs (training, environment, population, curriculum, exploration, bars, cascades, affordances, substrate, cues) with tuple aggregation for collections
- **Private Attributes**: Use Pydantic PrivateAttr (e.g., _config_dir in HamletConfig) for validation context without exposing in config schema or JSON serialization
- **Field Validators**: @field_validator for single-field checks (range order, position format, operating_hours bounds), @model_validator for multi-field constraints (epsilon_start >= epsilon_min, advance_threshold > retreat_threshold, source field mutually exclusive)
- **Extra="forbid"**: model_config = ConfigDict(extra="forbid") rejects unknown YAML fields to catch typos and prevent silent config errors
- **Range Validation**: Field(gt=0), Field(ge=0.0, le=1.0), Field(min_length=1) for numeric/string constraints enforced at schema level
- **Nullable Fields Pattern**: Use field | None with default=None for optional fields; requires explicit None in direct construction (no implicit defaults)
- **Manual Validator Invocation**: HamletConfig._validate_batch_size_vs_buffer manually called after __init__ because _config_dir not set during normal Pydantic validation (workaround for context-dependent validation)
- **Validation Error Context**: All ValidationError catches re-raise as ValueError with format_validation_error adding context ("training.yaml", "environment section (training.yaml)", "bars.yaml")
- **Collection DTOs**: Bars/Cascades/Affordances loaded as lists then converted to tuples in HamletConfig for immutability
- **Substrate Polymorphism**: SubstrateConfig uses Union types with discriminator field (type: grid/gridnd/continuous/continuousnd/aspatial) + nested configs (GridConfig, GridNDConfig, ContinuousConfig, etc.)

**Concerns:**
- **Large DAC Config File**: drive_as_code.py at 678 lines with 18+ nested DTOs - could split into drive_as_code/{modifiers,extrinsic,intrinsic,shaping}.py for better maintainability
- **Dual DTO Hierarchies**: VFS has ObservationField (vfs/schema.py), compiler has ObservationSpec (universe/dto/observation_spec.py), adapters translate between them (duplication risk if schemas diverge)
- **Config-Dir Sentinel Hack**: Passing _config_dir via data dict feels fragile; could use context manager or explicit brain_yaml_path parameter instead
- **Brain.yaml Rejection Logic Duplicated**: TrainingConfig and PopulationConfig both implement reject_brain_managed_fields with near-identical logic (could extract to base class method or shared validator)
- **Validation Order Dependencies**: HamletConfig._validate_batch_size_vs_buffer manually called after __init__ because _config_dir not set during normal validation - confusing control flow, hard to reason about validation order
- **Scattered Substrate Configs**: SubstrateConfig in substrate/config.py separate from main config/ directory (inconsistent organization, breaks "one subsystem per directory" pattern)
- **Action Configs Also Scattered**: ActionConfig in environment/action_config.py not in config/ directory (inconsistent organization)
- **DTO Import Cycles Risk**: config/hamlet.py imports from 10+ other config modules; config/affordance.py imports from environment/affordance_config.py; config/bar.py imports from environment/cascade_config.py (circular dependency risk if not careful)
- **Legacy Compat Aliases**: BarConfig imports BarsConfig from environment/cascade_config.py for type alias; AffordanceConfig imports AffordanceConfigCollection (technical debt from pre-DTO era)
- **No Schema Version**: DTOs lack version field; breaking changes to YAML schema have no migration path, version detection, or backwards compatibility mechanism (all configs implicitly "current version")
- **Field Description Duplication**: Many Field(description="...") strings repeated across DTOs (e.g., "Compute device", "Meter name") - could extract to constants
- **Hardcoded Error Messages**: Validation error strings embedded in model_validator methods, not centralized; changing operator guidance requires grepping for error messages
- **Missing Docstring Examples**: Some complex validators (ModifierConfig.validate_ranges_coverage, TrainingEnvironmentConfig.validate_pomdp_vision_range) lack usage examples in docstrings showing valid/invalid inputs
- **Optional Field Inconsistency**: Some optional fields use Field(default=None), others use field | None = None, others use field | None with no default (inconsistent patterns for nullable fields)
- **Validator Warning Suppression**: logger.warning calls in validators can be suppressed by logging config; operators might miss important warnings about unusual configs
- **No Validator Testing Isolation**: Most validator logic tested via full config loading; hard to unit test individual validators (e.g., validate_epsilon_decay_speed logic not directly testable)
- **Private Method Validators**: HamletConfig._validate_batch_size_vs_buffer is private but manually invoked - breaks encapsulation expectations

**Confidence:** High - Comprehensive analysis of 17 config DTO files (estimated 2500+ total lines excluding drive_as_code.py), all major DTOs examined (HamletConfig, TrainingConfig, TrainingEnvironmentConfig, PopulationConfig, CurriculumConfig, ExplorationConfig, BarConfig, CascadeConfig, AffordanceConfig, DriveAsCodeConfig, SubstrateConfig, ActionConfig), patterns identified from validators and cross-config checks, dependencies mapped via grep across 20 import sites, Brain As Code integration understood, no-defaults principle enforcement verified through field definitions and validators, scattered config locations documented (substrate/config.py, environment/action_config.py).

---

## Drive As Code (DAC) Engine

**Location:** `src/townlet/environment/dac_engine.py`, `src/townlet/config/drive_as_code.py`

**Responsibility:** Compiles declarative reward function specifications from YAML into GPU-native computation graphs for runtime reward calculation, replacing all hardcoded Python reward strategies with operator-configurable DAC specs.

**Key Components:**
- `dac_engine.py` (917 lines) - DACEngine runtime class: _compile_modifiers (range-based multipliers with torch.where GPU optimization), _compile_extrinsic (9 strategy types: multiplicative, constant_base_with_shaped_bonus, additive_unweighted, weighted_sum, polynomial, threshold_based, aggregation, vfs_variable, hybrid), _compile_shaping (11 bonus types: approach_reward, completion_bonus, efficiency_bonus, state_achievement, streak_bonus, diversity_bonus, timing_bonus, economic_efficiency, balance_bonus, crisis_avoidance, vfs_variable), calculate_rewards (vectorized reward composition: extrinsic + intrinsic×modifiers + shaping)
- `drive_as_code.py` (682 lines) - Pydantic DTO schemas: RangeConfig/ModifierConfig (contextual adjustment via range-based multipliers), BarBonusConfig/VariableBonusConfig/ExtrinsicStrategyConfig (9 extrinsic strategy schemas), IntrinsicStrategyConfig (5 intrinsic types: rnd, icm, count_based, adaptive_rnd, none), 11 shaping bonus configs (ApproachRewardConfig, CompletionBonusConfig, EfficiencyBonusConfig, StateAchievementConfig, VFSVariableBonusConfig, StreakBonusConfig, DiversityBonusConfig, TimingBonusConfig, EconomicEfficiencyConfig, BalanceBonusConfig, CrisisAvoidanceConfig, VfsVariableConfig), CompositionConfig (normalize, clip, logging), DriveAsCodeConfig (top-level schema with modifier reference validation), load_drive_as_code_config (YAML loader with validation error formatting)

**Dependencies:**
- **Inbound** (who uses DAC):
  - `environment/vectorized_env.py` (VectorizedHamletEnv) - Main consumer: step() calls dac_engine.calculate_rewards() with meters/dones/intrinsic_raw, passes kwargs (agent_positions, affordance_positions, last_action_affordance, affordance_streak, unique_affordances_used, current_hour) for shaping bonuses
  - `universe/compiler.py` (UniverseCompiler) - Stage 1 loads drive_as_code.yaml via load_drive_as_code_config(), Stage 5 computes drive_hash (SHA256 of DAC config) for checkpoint provenance tracking
  - `universe/compiled.py` (CompiledUniverse) - Stores dac_config and drive_hash as compilation artifacts for environment initialization
  - `training/checkpoint_utils.py` - Validates drive_hash matches between checkpoint and current config for reproducibility enforcement
  - `demo/runner.py` (DemoRunner) - Logs drive_hash to checkpoint metadata, ensures checkpoint/config compatibility
- **Outbound** (what DAC uses):
  - `vfs/registry.py` (VariableRegistry) - Reads VFS variables as reward sources (ModifierConfig.variable, ExtrinsicStrategyConfig.variable_bonuses, VfsVariableConfig shaping)
  - `config/drive_as_code.py` - Schema DTOs for YAML validation (DriveAsCodeConfig and 18 sub-DTOs)
  - PyTorch - GPU operations: torch.where (range lookups), torch.zeros/ones/full (initialization), torch.stack (bar aggregation), torch.norm (distance calculation), torch.pow (polynomial terms), torch.clamp (bonus clamping), torch.min/max (aggregation operations)
  - `config/base.py` (load_yaml_section, format_validation_error) - YAML loading utilities

**Patterns Observed:**
- **Compiler Pattern**: YAML → DTO validation → closure compilation → GPU execution (separates spec from implementation, enables A/B testing without code changes)
- **Closure Factory Pattern**: _compile_extrinsic, _compile_shaping return closures capturing config state, enabling strategy-specific optimizations without polymorphism overhead
- **GPU-Native Vectorization**: All operations batched across agents (modifiers use nested torch.where for range evaluation, shaping bonuses vectorize conditions with tensor masks)
- **Escape Hatch Design**: vfs_variable strategy type in both extrinsic and shaping allows custom reward logic via VFS-computed variables when declarative specs insufficient
- **Pedagogical Bug Demonstration**: L0_0_minimal demonstrates "Low Energy Delirium" bug (multiplicative reward + high intrinsic → agents exploit low bars for exploration), L0_5_dual_resource fixes with constant_base_with_shaped_bonus (teaching moment for reward structure importance)
- **Provenance Tracking**: drive_hash (SHA256 of DAC config YAML) embedded in checkpoints ensures reward function reproducibility, prevents checkpoint/config mismatches
- **Crisis Suppression Pattern**: Modifiers apply context-sensitive multipliers (e.g., energy_crisis modifier: energy<0.2 → intrinsic_weight×0.0 disables exploration when survival critical)
- **Composition Formula**: total_reward = extrinsic + (intrinsic × base_weight × modifier₁ × modifier₂...) + shaping (explicit decomposition for component logging)
- **Range Coverage Validation**: ModifierConfig.validate_ranges_coverage ensures ranges span [0.0, 1.0] with no gaps/overlaps (prevents undefined behavior at range boundaries)
- **Modifier Reference Validation**: DriveAsCodeConfig.validate_modifier_references checks extrinsic/intrinsic apply_modifiers lists reference defined modifiers (fails early on typos)
- **Bar Index Indirection**: DACEngine._get_bar_index maps bar_id → meters tensor index via bar_index_map from universe metadata (decouples DAC from meter ordering assumptions)
- **VFS Reader Identity**: DAC reads VFS variables as reader="engine" (not "agent"), enforcing access control separation between agent observations and reward computation
- **Dead Agent Zeroing**: All reward components (extrinsic, intrinsic, shaping) use torch.where(dones, zeros, rewards) to zero rewards for dead agents (prevents post-mortem reward accumulation)

**Concerns:**
- **Large Monolithic Files**: dac_engine.py (917 lines) with 9 extrinsic strategies and 11 shaping bonuses could split into dac_engine/{extrinsic,shaping,modifiers}.py for maintainability; drive_as_code.py (682 lines) with 18 DTOs could split into drive_as_code/{modifiers,extrinsic,intrinsic,shaping}.py
- **String Comparison in Shaping**: completion_bonus, streak_bonus, timing_bonus use list comprehension with string comparison for affordance matching ([1.0 if aff == target else 0.0 for aff in last_action_affordance]) - can't fully vectorize, partial GPU inefficiency for large populations
- **Shaping Bonus Kwargs Dependency**: Shaping functions require specific kwargs (agent_positions, affordance_positions, last_action_affordance, affordance_streak, unique_affordances_used, current_hour) passed from environment - fragile implicit contract, no schema validation for kwargs structure
- **Null Kwarg Handling Inconsistency**: Most shaping bonuses return zeros on missing kwargs (defensive), vfs_variable bonus raises KeyError (fail-fast) - inconsistent error handling philosophy
- **Placeholder Extrinsic Strategy**: _compile_extrinsic returns zeros placeholder for unimplemented strategy types instead of failing (silently breaks reward calculation if operator typos strategy type)
- **Hardcoded Aggregation Operation**: aggregation strategy always uses min(), full implementation would need operation field (min/max/mean/product) in ExtrinsicStrategyConfig schema
- **Time Range Wrap-Around Logic**: timing_bonus handles 22-6 (night) with conditional (start_hour <= end_hour vs wrap-around) - logic duplicated in shaping compilation and environment temporal mechanics, could centralize
- **No Modifier Caching**: Modifiers recomputed every step even if source values unchanged (could cache modifier results for bar-based modifiers with change detection)
- **Hybrid Strategy Underdefined**: hybrid extrinsic strategy uses simplified implementation (weighted bars with optional centering) - comment says "Full implementation would compose multiple sub-strategies" but schema doesn't support composition
- **Polynomial Strategy Overloads center**: polynomial uses BarBonusConfig.center as exponent (semantic overload of field intended for shaped bonus neutral point - confusing API)
- **No Modifier Chaining Validation**: apply_modifiers is unordered list, modifier application order affects result (multiplication non-commutative for side effects), no validation or deterministic ordering
- **Crisis Threshold Semantics**: crisis_avoidance uses > (strictly above) vs efficiency_bonus uses >= (at or above) - subtle difference not documented in field descriptions, operators might not notice
- **Bar Index Map Runtime Dependency**: _get_bar_index raises KeyError if bar_id not in metadata map (environment initialization error manifests as DAC runtime error - obscure error source)
- **Intrinsic Weight Components Dict**: calculate_rewards returns intrinsic_weights [num_agents] but components dict doesn't include per-modifier breakdown (logging loses visibility into which modifier suppressed intrinsic)
- **No DAC Version Checking**: drive_as_code.yaml has version field but DACEngine doesn't validate it (future schema changes could break old configs silently)
- **Shaping Bonus Registration**: Shaping bonuses compiled into list in config order, no ability to disable individual bonuses without removing from config (operators can't A/B test by toggling bonuses on/off)
- **Missing Reward Clipping**: CompositionConfig.clip defined but not implemented in calculate_rewards() (operators might expect clipping but rewards unbounded)
- **Missing Reward Normalization**: CompositionConfig.normalize defined but not implemented in calculate_rewards() (tanh normalization to [-1,1] not applied)

**Confidence:** High - Comprehensive analysis of 2 core files (1599 total lines): dac_engine.py (917 lines) with 3 compilation methods and 20+ strategy implementations examined, drive_as_code.py (682 lines) with 18 DTOs analyzed, dependencies mapped via grep across 28 DAC-related files, integration with VectorizedHamletEnv verified via calculate_rewards() call site, checkpoint provenance flow traced (compiler → drive_hash → checkpoint metadata), pedagogical pattern confirmed in L0_0_minimal vs L0_5_dual_resource configs, GPU vectorization patterns understood, 17 concerns identified from code comments and unimplemented features (CompositionConfig.clip/normalize).

---
