## Curriculum

**Location:** `src/townlet/curriculum/`

**Responsibility:** Controls environment difficulty progression by adapting depletion rates, active meters, and reward modes based on agent performance metrics.

**Key Components:**
- `base.py` - Abstract CurriculumManager interface defining curriculum contract (73 lines)
- `adversarial.py` - AdversarialCurriculum with 5-stage progression based on survival rate, learning progress, and policy entropy (484 lines)
- `static.py` - StaticCurriculum with fixed difficulty for baseline experiments (95 lines)
- `__init__.py` - Module initialization (1 line)

**Dependencies:**
- Inbound: Population (VectorizedPopulation), Demo System (DemoRunner, live_inference), Config DTOs (HamletConfig)
- Outbound: Training State (BatchedAgentState, CurriculumDecision DTOs)

**Patterns Observed:**
- Strategy Pattern - Pluggable curriculum implementations (Static, Adversarial) via abstract base class
- DTO-driven communication - Immutable CurriculumDecision objects with Pydantic validation
- GPU-native state tracking - PerformanceTracker stores agent stages and metrics as PyTorch tensors
- Multi-signal decision logic - Advancement requires high survival (>70%), positive learning, and low entropy (<0.5)
- Hysteresis band - Separate advance/retreat thresholds prevent oscillation between stages
- PyTorch-style checkpointing - state_dict() and load_state_dict() aliases for API consistency

**Concerns:**
- None observed

**Confidence:** High - Clear interface with two concrete implementations, comprehensive test coverage (869 lines), well-documented multi-signal decision logic, GPU tensor state management verified

---

## Exploration

**Location:** `/home/john/hamlet/src/townlet/exploration/`

**Responsibility:** Implements pluggable exploration strategies for action selection and intrinsic motivation, providing epsilon-greedy baseline, Random Network Distillation (RND) novelty detection, and adaptive annealing based on performance consistency.

**Key Components:**
- `base.py` - ExplorationStrategy abstract interface defining select_actions(), compute_intrinsic_rewards(), update(), checkpoint_state(), and load_state() contracts (117 lines)
- `epsilon_greedy.py` - EpsilonGreedyExploration baseline strategy with exponential epsilon decay, no intrinsic motivation (117 lines)
- `rnd.py` - RNDExploration implementing Random Network Distillation with fixed/predictor networks, prediction error as novelty signal, Welford's online running statistics for normalization, and GPU-native novelty map generation (343 lines)
- `adaptive_intrinsic.py` - AdaptiveIntrinsicExploration wrapper composing RNDExploration with variance-based intrinsic weight annealing, survival tracking window, and dual-threshold logic (mean survival + variance) to prevent premature annealing (228 lines)
- `action_selection.py` - Shared epsilon_greedy_action_selection utility with vectorized action masking, multinomial sampling from valid actions, and GPU-optimized exploration/exploitation selection (78 lines)

**Dependencies:**
- Inbound: Population (vectorized.py), Demo System (runner.py, live_inference.py)
- Outbound: Training (BatchedAgentState), PyTorch (nn.Module, tensors, optimizers), NumPy (running statistics)

**Patterns Observed:**
- Strategy pattern with abstract base class and three concrete implementations (EpsilonGreedyExploration, RNDExploration, AdaptiveIntrinsicExploration)
- Composition over inheritance (AdaptiveIntrinsicExploration wraps RNDExploration rather than extending it)
- Shared utility pattern for action selection (single epsilon_greedy_action_selection function used by all strategies)
- GPU-native vectorization (all operations on batched tensors, no Python loops)
- Hot path optimization (select_actions runs every step with minimal overhead, update() can be slow)
- Running statistics pattern using Welford's online algorithm for numerical stability
- Defensive annealing logic requiring both low variance AND high mean survival (prevents "stable failure" triggering)
- Intrinsic weight applied once in replay buffer sampling (not in compute_intrinsic_rewards to avoid double-weighting)
- RND dual-network architecture (fixed random network frozen, predictor network trained via MSE loss)
- Observation masking for handling padding dimensions in variable-length observations
- Checkpoint state includes network weights, optimizer state, epsilon, normalization statistics, and survival history

**Concerns:**
- None observed

**Confidence:** High - All 5 files read completely, strategy pattern verified, RND implementation matches OpenAI/CleanRL references documented in comments, adaptive annealing logic traced with dual-threshold validation, dependencies confirmed via imports and grep, comprehensive test coverage observed (unit/integration/property tests), integration with Population verified

---
## Substrates

**Location:** `/home/john/hamlet/src/townlet/substrate/`

**Responsibility:** Defines spatial representation abstractions for agent positioning, movement, distance calculation, and observation encoding across multiple substrate types (Grid2D, Grid3D, GridND, Continuous1D/2D/3D, ContinuousND, Aspatial).

**Key Components:**
- `base.py` - SpatialSubstrate abstract interface defining position_dim, position_dtype, movement, distance, observation encoding contracts (316 lines)
- `factory.py` - SubstrateFactory for building substrate instances from SubstrateConfig DTOs with type-based dispatching (152 lines)
- `config.py` - Pydantic DTOs for substrate configuration (GridConfig, GridNDConfig, ContinuousConfig, AspatialSubstrateConfig, SubstrateConfig, ActionLabelConfig) with validation logic (392 lines)
- `grid2d.py` - Grid2DSubstrate implementation for 2D square grids with configurable boundaries, distance metrics, observation encoding modes (542 lines)
- `grid3d.py` - Grid3DSubstrate implementation for 3D cubic grids with POMDP support via 3D local windows (554 lines)
- `gridnd.py` - GridNDSubstrate implementation for 4D-100D hypercube grids for high-dimensional research (521 lines)
- `continuous.py` - Continuous1D/2D/3DSubstrate implementations with float-based positioning, proximity-based interaction, configurable action discretization (916 lines)
- `continuousnd.py` - ContinuousNDSubstrate implementation for 4D-100D continuous spaces (500 lines)
- `aspatial.py` - AspatialSubstrate implementation with zero-dimensional positioning for pure resource management (172 lines)

**Dependencies:**
- Inbound: Universe Compiler (compiler.py, runtime.py), Environment (vectorized_env.py, substrate_action_validator.py, action_builder.py), Demo System (live_inference.py), Config DTOs (hamlet.py)
- Outbound: Environment (ActionConfig from action_config.py, iter_affordance_positions from affordance_layout.py)

**Patterns Observed:**
- Abstract base class pattern (SpatialSubstrate defines interface, concrete implementations override all abstract methods)
- Factory pattern (SubstrateFactory.build() creates instances from config DTOs via type field)
- Strategy pattern for boundary handling (clamp, wrap, bounce, sticky modes implemented per substrate)
- Strategy pattern for distance metrics (manhattan, euclidean, chebyshev computed per substrate)
- Strategy pattern for observation encoding (relative [0,1] normalization, scaled with range metadata, absolute raw coordinates)
- Canonical action ordering contract (movement actions → INTERACT → WAIT enables position-based meta-action identification)
- Polymorphic position representation (torch.long for discrete grids, torch.float32 for continuous spaces)
- Device-agnostic tensor operations (device parameter passed to methods, not stored at construction)
- Conceptual agnosticism principle (no assumptions about 2D, Euclidean, or grid-based structure in base interface)
- Fixed affordance vocabulary (14 affordances constant across grid sizes enables transfer learning)
- POMDP support via encode_partial_observation() for local window extraction (Grid2D 5x5 window, Grid3D 5x5x5 window)

**Concerns:**
- None observed

**Confidence:** High - Clear base interface (SpatialSubstrate), factory pattern verified, all 7 substrate types examined (Grid2D, Grid3D, GridND, Continuous1D/2D/3D, ContinuousND, Aspatial), configuration DTOs comprehensive with validation, dependencies bidirectional (inbound from Environment/Universe Compiler, outbound to Environment for ActionConfig), patterns consistent with discovery findings

---

## Recording

**Location:** `/home/john/hamlet/src/townlet/recording/`

**Responsibility:** Captures, stores, and replays training episodes for debugging and video export, providing non-blocking GPU-to-disk serialization with LZ4 compression, multi-criteria recording selection, frame-by-frame replay with seek controls, and matplotlib-based video rendering with ffmpeg encoding.

**Key Components:**
- `recorder.py` - EpisodeRecorder with async queue + background writer thread for non-blocking episode capture, RecordingWriter for criteria evaluation and LZ4-compressed MessagePack serialization (307 lines)
- `replay.py` - ReplayManager for loading/decompressing recorded episodes with step-by-step playback controls (seek, reset, next) and database query integration (222 lines)
- `video_export.py` - Export functions for single and batch MP4 generation from recordings using temporary frame directories and ffmpeg H.264 encoding with YouTube-optimized settings (252 lines)
- `video_renderer.py` - EpisodeVideoRenderer using matplotlib for high-quality frame rendering with 16:9 layout, grid view, meter bars, Q-value charts, and dark/light themes (340 lines)
- `criteria.py` - RecordingCriteria evaluator supporting periodic, stage_transitions, performance percentile, and stage_boundaries criteria with OR logic (224 lines)
- `data_structures.py` - Frozen dataclasses for RecordedStep and EpisodeMetadata optimized for msgpack serialization with ~100-150 bytes per step (108 lines)
- `__main__.py` - CLI for single episode export and batch export with filtering by stage, reason, and reward thresholds (147 lines)

**Dependencies:**
- Inbound: Demo System (runner.py for EpisodeRecorder integration, live_inference.py for ReplayManager replay mode)
- Outbound: Demo System (DemoDatabase for recording metadata storage and queries), LZ4 (lz4.frame for compression), MessagePack (msgpack for serialization), matplotlib (frame rendering), ffmpeg (video encoding subprocess), PIL (PNG frame writing)

**Patterns Observed:**
- Producer-consumer pattern with bounded queue (maxsize=1000) for non-blocking recording
- Background writer thread (daemon=True) with graceful shutdown via stop flag
- Frozen dataclasses with __slots__ for memory efficiency (~100-150 bytes per step)
- Two-phase recording (buffer steps → evaluate criteria at episode end → write if matched)
- OR logic for criteria (any enabled criterion triggers recording)
- LZ4 fast compression (compression_level=0) prioritizing speed over ratio
- MessagePack for binary serialization with tuple preservation via deserialize helpers
- Temporary directory pattern for frame rendering (tempfile.TemporaryDirectory)
- FFmpeg subprocess integration with YouTube-optimized settings (H.264, yuv420p, CRF 18)
- Matplotlib non-interactive backend (Agg) with FigureCanvasAgg for array rendering
- Database indexing for replay queries with filtering (stage, reason, reward range)
- Queue overflow handling with graceful degradation (drop frames with warning)
- Stateful replay with current_step_index and playing flag for VCR-style controls

**Concerns:**
- RecordingWriter._should_record_episode implements only periodic criterion, ignoring full RecordingCriteria evaluator (phase 2 placeholder comment but basic implementation present)
- Criteria evaluator maintains in-memory history (deque maxlen=100) without persistence across restarts
- Video export auto-detects grid size from affordance positions, may fail if positions are edge coordinates
- FFmpeg availability checked at runtime with stdout suppression but no fallback renderer
- Recorder queue full condition drops frames with warning but provides no backpressure to training loop

**Confidence:** High - All 8 files read (1601 total lines), threading model understood from recorder.py, compression pipeline traced (tensor → msgpack → LZ4), replay flow verified (load → decompress → seek → render), video export pipeline clear (frames → ffmpeg), criteria OR logic confirmed, dependencies traced via imports and grep

---

## Config DTOs

**Location:** `/home/john/hamlet/src/townlet/config/`

**Responsibility:** Defines Pydantic data transfer objects for configuration validation across all Townlet subsystems, enforcing no-defaults principle where all behavioral parameters must be explicitly specified in YAML configs for operator accountability and reproducibility.

**Key Components:**
- `__init__.py` - Module interface exposing all DTO classes and loader functions with CONFIG_SCHEMA_VERSION tracking (67 lines)
- `base.py` - Common utilities (load_yaml_section, format_validation_error) for YAML loading and Pydantic error formatting with helpful operator guidance (92 lines)
- `hamlet.py` - HamletConfig master DTO composing all section configs with cross-config validation (batch_size vs buffer, network vs observability, grid capacity warnings) (215 lines)
- `training.py` - TrainingConfig for Q-learning hyperparameters (device, episodes, batch_size, epsilon decay, Double DQN flag, reward strategy, enabled_actions list) with epsilon decay speed warnings (244 lines)
- `environment.py` - TrainingEnvironmentConfig for observability (POMDP/full), vision_range, temporal mechanics, enabled affordances, randomize_affordances, energy costs per action type (130 lines)
- `population.py` - PopulationConfig for agent count, learning_rate, gamma, replay_buffer_capacity, network_type (simple/recurrent/structured), mask_unused_obs flag (74 lines)
- `curriculum.py` - CurriculumConfig for episode length, advancement/retreat thresholds, entropy gate, min_steps_at_stage with hysteresis validation (70 lines)
- `exploration.py` - ExplorationConfig for RND embed_dim, initial_intrinsic_weight, variance_threshold, min_survival_fraction, survival_window (76 lines)
- `bar.py` - BarConfig for meter definitions (name, index, tier, range, initial, base_depletion) with range/initial validation, aliases BarsConfig from cascade_config (107 lines)
- `cascade.py` - CascadeConfig for meter relationship rules (source, target, threshold, strength) with self-cascade prevention, aliases CascadesConfig from cascade_config (95 lines)
- `affordance.py` - AffordanceConfig for interaction definitions (id, name, costs, costs_per_tick, duration_ticks, operating_hours, modes, availability, capabilities, effect_pipeline, position) with position format validation (129 lines)
- `cues.py` - CuesConfig for theory-of-mind cue definitions (simple_cues, compound_cues, visual_cues with range mappings) for Stage 5 cues pipeline (129 lines)
- `capability_config.py` - CapabilityConfig union type of 6 capability DTOs (MultiTickCapability, CooldownCapability, MeterGatedCapability, SkillScalingCapability, ProbabilisticCapability, PrerequisiteCapability) for advanced affordance behaviors (113 lines)
- `effect_pipeline.py` - EffectPipeline DTO for multi-stage affordance effects (on_start, per_tick, on_completion, on_early_exit, on_failure) with has_effects() utility (40 lines)
- `affordance_masking.py` - BarConstraint and ModeConfig DTOs for meter-based affordance availability and operating mode configuration (48 lines)
- `drive_as_code.py` - Drive As Code (DAC) DTOs for declarative reward functions (RangeConfig, ModifierConfig, ExtrinsicStrategyConfig, IntrinsicStrategyConfig, TriggerCondition, shaping bonuses, CompositionConfig, DriveAsCodeConfig) with range coverage validation (16KB file, ~400 lines estimated)

**Dependencies:**
- Inbound: Universe Compiler (compiler.py, compiler_inputs.py, symbol_table.py, compiled.py, cues_compiler.py, runtime.py), Environment (dac_engine.py), Demo System (runner.py), all config subsystem files cross-import each other
- Outbound: Substrate Config (SubstrateConfig from substrate/config.py), Environment Config (BarsConfig, CascadesConfig, AffordanceConfigCollection from environment/cascade_config.py, environment/affordance_config.py), Pydantic (BaseModel, Field, validators), PyYAML (yaml.safe_load)

**Patterns Observed:**
- No-defaults principle enforced via required fields and explicit Field() declarations without defaults for all behavioral parameters
- Pydantic strict validation with extra="forbid" to reject unknown fields and prevent typos
- Helpful error formatting via format_validation_error() wrapping Pydantic errors with context, fix suggestions, and template references
- Cross-config validation in HamletConfig (batch_size <= buffer_capacity, network type vs observability consistency, grid capacity warnings)
- Permissive semantic warnings (epsilon_decay speed, POMDP vision_range, network/observability mismatch) that warn operators without blocking unusual experiments
- Type aliases to richer DTOs (BarsConfig, CascadesConfig, AffordanceConfigCollection from environment layer provide additional methods beyond basic list validation)
- Loader functions follow consistent pattern (load_*_config(config_dir: Path) -> *Config) with ValidationError wrapping
- Master config loader HamletConfig.load() composes all section loaders with optional training_config_path override for CLI flexibility
- Union types for polymorphic configs (CapabilityConfig = MultiTick | Cooldown | MeterGated | SkillScaling | Probabilistic | Prerequisite)
- Nested validation with @field_validator and @model_validator decorators for complex constraints (range order, threshold hysteresis, position format validation)
- Config schema versioning via CONFIG_SCHEMA_VERSION constant for migration tracking

**Concerns:**
- drive_as_code.py is large (16KB) and may deserve subsystem decomposition (RangeConfig, ModifierConfig, strategy configs, shaping bonus configs could be separate files)
- Type aliases (BarsConfig, CascadesConfig, AffordanceConfigCollection) create bidirectional dependency with environment layer (config imports from environment, environment imports from config)
- Cross-config validation warnings in HamletConfig use logging which may not be visible in all execution contexts (CLI vs library usage)
- enabled_affordances validation occurs in TrainingEnvironmentConfig but affordance names not validated against affordances.yaml until compiler stage (early validation could improve error messages)

**Confidence:** High - All 16 files read (2054 total lines), no-defaults principle verified across all DTOs, validation patterns consistent (field validators, model validators, extra="forbid"), dependencies traced via imports and grep (17 inbound consumers including Universe Compiler, Environment, Demo System), loader functions follow consistent pattern, HamletConfig master composition verified, supporting DTOs (capabilities, effect pipeline, affordance masking) examined

---

## Universe Compiler

**Location:** `/home/john/hamlet/src/townlet/universe/`

**Responsibility:** Compiles YAML configuration packs into validated, optimized CompiledUniverse artifacts through a seven-stage pipeline (parse, symbol table, resolve, validate, metadata, optimize, emit) with MessagePack caching, ensuring reproducibility via config hashing and mtime tracking.

**Key Components:**
- `compiler.py` - UniverseCompiler orchestrating 7-stage pipeline with cache management, YAML syntax validation, auto-generated standard variables, and DoS protection limits (2385 lines)
- `compiled.py` - CompiledUniverse immutable artifact with MessagePack serialization, checkpoint compatibility validation, runtime environment creation, and to_runtime() conversion (310 lines)
- `compiler_inputs.py` - RawConfigs container aggregating all parsed DTOs with convenience accessors and action space composition logic (306 lines)
- `symbol_table.py` - UniverseSymbolTable registry for meters, variables, actions, cascades, affordances, cues with duplicate detection and cross-reference resolution (102 lines)
- `runtime.py` - RuntimeUniverse read-only view with frozen model proxies, clone helpers for runtime systems, and meter/affordance lookup (138 lines)
- `errors.py` - CompilationError exception, CompilationMessage structured diagnostic, CompilationErrorCollector batch error accumulator with hints and warnings (107 lines)
- `optimization.py` - OptimizationData dataclass for pre-computed runtime tensors (base_depletions, cascade_data, action_mask_table, affordance_position_map) (23 lines)
- `source_map.py` - SourceMap YAML line number tracking via custom PyYAML loader for precise error reporting with file:line locations (101 lines)
- `cues_compiler.py` - CuesCompiler sub-compiler for cues.yaml validation with meter reference checks and visual cue range coverage validation (143 lines)
- `adapters/vfs_adapter.py` - VFSAdapter converting VFS observation fields to compiler ObservationSpec DTOs with UUID generation and ObservationActivity masking (175 lines)
- `dto/__init__.py` - DTO module interface exposing ActionMetadata, ObservationSpec, UniverseMetadata, MeterMetadata, AffordanceMetadata (29 lines)
- `dto/universe_metadata.py` - UniverseMetadata frozen dataclass with substrate, meter, affordance, action, observation dimensions, economic balance, provenance tracking (72 lines)
- `dto/observation_spec.py` - ObservationSpec and ObservationField frozen dataclasses with UUID generation, semantic type grouping, field lookup by name/type (99 lines)
- `dto/observation_activity.py` - ObservationActivity frozen dataclass for curriculum masking with active_mask, group_slices, active_field_uuids (45 lines)
- `dto/action_metadata.py` - ActionMetadata and ActionSpaceMetadata frozen dataclasses for action vocabulary with enabled flags, costs, effects (56 lines)
- `dto/affordance_metadata.py` - AffordanceInfo and AffordanceMetadata frozen dataclasses for affordance catalog with costs, effects, duration, operating hours (43 lines)
- `dto/meter_metadata.py` - MeterInfo and MeterMetadata frozen dataclasses for meter catalog with range, tier, base_depletion (34 lines)

**Dependencies:**
- Inbound: Config DTOs (HamletConfig, all section configs), VFS (VariableDef, VFSObservationSpecBuilder), Substrate (SubstrateFactory, SubstrateConfig), Environment (EnvironmentConfig, ActionSpaceConfig), Demo System (runner.py, live_inference.py), Training (checkpoint_utils.py), Agent Networks (networks.py)
- Outbound: Config DTOs (all loaders), VFS (observation_builder.py, registry.py, schema.py), Substrate (factory.py), Environment (cascade_config.py, action_config.py), MessagePack (msgpack serialization), PyYAML (yaml.safe_load), PyTorch (tensors for optimization data)

**Patterns Observed:**
- Seven-stage compiler pipeline (Phase 0: YAML syntax → Stage 1: Parse → Stage 2: Symbol table → Stage 3: Resolve references → Stage 4: Cross-validate → Stage 5: Metadata → Stage 6: Optimize → Stage 7: Emit)
- Two-tier caching strategy (mtime check for fast path, config hash for content equality verification)
- Auto-generation pattern (standard variables for grid, position, meters, affordances, temporal state auto-generated from substrate/bars/affordances configs)
- DoS protection via MAX_* constants (MAX_METERS=100, MAX_AFFORDANCES=100, MAX_CASCADES=500, MAX_ACTIONS=300, MAX_VARIABLES=200, MAX_GRID_CELLS=10000, MAX_CACHE_FILE_SIZE=10MB)
- Structured error collection with CompilationErrorCollector batch accumulator, error codes, hints, warnings, source map line number tracking
- Immutable artifact pattern (CompiledUniverse frozen dataclass with MessagePack serialization, RuntimeUniverse read-only view with frozen model proxies)
- Cache fingerprinting (SHA256 hash of all YAML content + modification time tracking for cache invalidation)
- Symbol table pattern (central registry for meters, variables, actions, cascades, affordances, cues with duplicate detection)
- VFS adapter pattern (converting VFS observation fields to compiler DTOs with UUID generation for checkpoint compatibility)
- Sub-compiler delegation (CuesCompiler encapsulates cues.yaml validation logic)
- DTO-driven metadata enrichment (Stage 5 builds ActionSpaceMetadata, MeterMetadata, AffordanceMetadata, UniverseMetadata, ObservationSpec)
- Observation field UUID deterministic generation (SHA256 hash of scope|name|description|dims|semantic_type for checkpoint compatibility)
- Runtime vs compile-time separation (CompiledUniverse for storage/caching, RuntimeUniverse for training systems with read-only views)

**Concerns:**
- Auto-generated variables override user-defined variables silently (user variables registered after auto-generated, but no warning if user redefines standard variable)
- Cache bomb protection uses MAX_CACHE_FILE_SIZE=10MB limit but no incremental reading (entire cache loaded into memory before size check on deserialization)
- Source map line number tracking requires custom PyYAML loader which may drift from yaml.safe_load behavior
- DAC config currently optional (Phase 5 comment indicates will become required but no timeline/migration plan)
- Action space composition in compiler_inputs.py silently trims meter references absent from bars.yaml (hint added but no error on invalid meter references in global_actions.yaml)

**Confidence:** High - Complete 7-stage pipeline traced (compiler.py 2385 lines), all 17 component files read (4173 total lines), symbol table resolution verified, cache invalidation logic (mtime + hash) confirmed, DTO serialization (MessagePack) examined, VFS adapter integration traced, inbound/outbound dependencies verified via grep (12 consumers), DoS protection limits documented, error collection pattern validated, no-defaults principle enforced via Pydantic required fields

---

## Population

**Location:** `/home/john/hamlet/src/townlet/population/`

**Responsibility:** Manages batched Q-learning training for multiple parallel agents through vectorized population coordination, handling Q-network forward passes, replay buffer sampling, gradient updates, target network synchronization, and runtime telemetry tracking.

**Key Components:**
- `vectorized.py` - VectorizedPopulation implementation coordinating shared Q-network training across num_agents parallel agents with dual replay buffer system (sequential for LSTM, standard for feedforward), Double DQN support, gradient clipping, periodic target network updates, and episode lifecycle management (933 lines)
- `base.py` - PopulationManager abstract interface defining step_population() for training coordination and get_checkpoint() for serialization (69 lines)
- `runtime_registry.py` - AgentRuntimeRegistry maintaining per-agent GPU tensors (survival_time, curriculum_stage, epsilon, intrinsic_weight) with JSON-safe snapshot API for telemetry (134 lines)
- `__init__.py` - Module initialization (1 line)

**Dependencies:**
- Inbound: Demo System (runner.py, live_inference.py), Curriculum (adversarial.py for Q-value entropy calculation)
- Outbound: Agent Networks (SimpleQNetwork, RecurrentSpatialQNetwork, StructuredQNetwork), Exploration (ExplorationStrategy, AdaptiveIntrinsicExploration, RNDExploration), Curriculum (CurriculumManager), Training (ReplayBuffer, SequentialReplayBuffer, BatchedAgentState, PopulationCheckpoint), Environment (VectorizedHamletEnv)

**Patterns Observed:**
- Abstract base class pattern (PopulationManager interface, VectorizedPopulation concrete implementation)
- Dual replay buffer strategy (SequentialReplayBuffer for recurrent networks with episode sequences, ReplayBuffer for feedforward networks with individual transitions)
- Network architecture polymorphism (SimpleQNetwork for feedforward, RecurrentSpatialQNetwork for LSTM, StructuredQNetwork for group-based observations)
- Target network stabilization (periodic hard updates every target_update_frequency steps, separate eval-mode target network)
- Double DQN algorithm support (configurable via use_double_dqn flag, online network for action selection, target network for Q-value evaluation)
- Vanilla DQN fallback (target network for both selection and evaluation when use_double_dqn=False)
- Gradient clipping (max_norm=10.0 prevents exploding gradients in deep networks)
- Episode lifecycle management (accumulate transitions → store on done → reset hidden state → update intrinsic annealing)
- LSTM hidden state lifecycle (reset at episode start, persist during rollout, reset per batch in training, separate hidden states for episode vs training batches)
- Post-terminal masking for recurrent training (P2.2 pattern prevents gradients from garbage timesteps after terminal state)
- Runtime registry telemetry (GPU tensors for hot path access, JSON snapshots for cold path serialization)
- Checkpoint compatibility validation (universe_metadata in checkpoint validates meter_count, obs_dim, action_dim match current environment)
- Action masking integration (epsilon_greedy_action_selection respects environment action masks to prevent invalid actions)
- Intrinsic reward composition (RND novelty + adaptive annealing via exploration strategy, weighted in replay buffer sampling)
- TensorBoard histogram logging (network weights and gradients logged every 100 training steps)

**Concerns:**
- Network architecture hardcoded (SimpleQNetwork hidden_dim=128, RecurrentSpatialQNetwork hidden_dim=256, StructuredQNetwork embed_dim=32, q_head_dim=128 have TODO(BRAIN_AS_CODE) comments indicating future config-driven architecture)
- Action dimension defaults to env.action_dim if not specified (TASK-002B Phase 4.1 comment suggests recent change, may have backward compatibility implications)
- Recurrent training requires 3 forward passes for Double DQN (online prediction, online selection, target evaluation) vs 2 for vanilla DQN (noted in code but no optimization)
- Checkpoint loading has soft validation for obs_dim mismatch (warnings.warn instead of ValueError, may cause silent failures on grid size changes)
- Episode flush mechanism (flush_episode) only implemented for recurrent mode (feedforward transitions already in buffer, but asymmetry could confuse future maintainers)

**Confidence:** High - All 4 files read (1137 total lines), training step pipeline traced (Q-values → curriculum → action selection → environment step → intrinsic rewards → replay buffer → gradient update → target sync), Double DQN implementation verified (action selection decoupling for both feedforward and recurrent), LSTM sequence handling confirmed (episode accumulation, post-terminal masking, hidden state lifecycle), checkpoint serialization examined (universe_metadata validation), dependencies verified via grep (4 inbound consumers, 8 outbound dependencies), test coverage observed (1436 lines across 5 test files), runtime registry telemetry integration traced

---

## Agent Networks

**Location:** `/home/john/hamlet/src/townlet/agent/`

**Responsibility:** Provides neural network architectures for Q-value approximation supporting full observability (SimpleQNetwork MLP), partial observability (RecurrentSpatialQNetwork with LSTM and CNN encoders), and structured observation encoding (StructuredQNetwork with semantic group encoders).

**Key Components:**
- `networks.py` - Three Q-network architectures with PyTorch nn.Module implementations (388 lines)
- `__init__.py` - Module interface exposing SimpleQNetwork, RecurrentSpatialQNetwork, StructuredQNetwork (1 line)

**Dependencies:**
- Inbound: Population (vectorized.py creates and trains networks), Exploration (rnd.py architecture reference for RNDEmbeddingNet), Universe Compiler (networks.py import for checkpoint validation)
- Outbound: PyTorch (nn.Module, nn.Linear, nn.LSTM, nn.Conv2d, LayerNorm), Universe Compiler DTOs (ObservationActivity for StructuredQNetwork group slices)

**Patterns Observed:**
- Strategy pattern via network_type config selection (simple, recurrent, structured instantiated polymorphically in Population)
- Encoder composition pattern in RecurrentSpatialQNetwork (vision CNN, position MLP, meter MLP, affordance MLP combined before LSTM)
- Semantic grouping pattern in StructuredQNetwork (ModuleDict with per-group encoders leveraging ObservationActivity metadata)
- Conditional architecture in RecurrentSpatialQNetwork (position_encoder=None for Aspatial substrates with position_dim=0)
- LSTM hidden state lifecycle (reset at episode start, persist during rollout, reset per transition in batch training)
- Device-agnostic initialization (device parameter passed to methods, not stored at construction)
- LayerNorm stabilization throughout all architectures (prevents gradient explosion with deep networks)
- No-defaults principle (PDR-002) enforced via explicit hidden_dim parameters with TODO comments for future BRAIN_AS_CODE config
- Dual network pattern in Population (online network for action selection, target network for Q-target stability)
- Dynamic architecture sizing (RecurrentSpatialQNetwork adjusts lstm_input_dim based on position_dim, window_size, num_meters, num_affordance_types)
- Observation masking support in StructuredQNetwork (skips empty groups from group_slices with size <= 0)

**Concerns:**
- Network architecture parameters hardcoded in Population (hidden_dim=128 for SimpleQNetwork, hidden_dim=256 for RecurrentSpatialQNetwork) with TODO comments but no BRAIN_AS_CODE implementation timeline
- RecurrentSpatialQNetwork ignores temporal features after extraction (sin/cos time encoding present in observations but not processed separately, line 213-215 comment acknowledges this)
- StructuredQNetwork group_embed_dim and q_head_hidden_dim have defaults (32, 128) violating no-defaults principle, though documented as exemption for infrastructure
- LSTM hidden state managed via mutable instance variable (self.hidden_state) creates potential thread-safety issues if networks shared across threads
- No explicit parameter count validation against memory constraints (RecurrentSpatialQNetwork ~650K params could cause GPU OOM with large batch sizes)

**Confidence:** High - Both source files read completely (389 total lines), three network architectures examined with architectural details verified, dependencies traced via grep showing Population instantiation and Exploration reference, integration with ObservationActivity DTO confirmed, LSTM lifecycle verified in Population training loop, test coverage observed (1353 lines across unit and integration tests including network selection, recurrent networks, structured networks), parameter counts documented in CLAUDE.md matched to architectures

---
## Environment

**Location:** `/home/john/hamlet/src/townlet/environment/`

**Responsibility:** Orchestrates GPU-native vectorized environment simulation by composing dynamics engines (affordance interactions, meter cascades, reward computation) with VFS integration, action execution pipeline, and observation construction for parallel agent training.

**Key Components:**
- `vectorized_env.py` - VectorizedHamletEnv main environment class coordinating substrate, VFS registry, dynamics engines, action execution, observation construction, and step lifecycle (1379 lines)
- `affordance_engine.py` - AffordanceEngine processing affordance interactions (instant, multi-tick, continuous) with operating hours validation, cost/effect application, and GPU-vectorized execution (506 lines)
- `cascade_engine.py` - CascadeEngine applying meter dynamics (base depletions, modulations, threshold cascades, terminal conditions) from YAML config with GPU-optimized tensor operations (331 lines)
- `meter_dynamics.py` - MeterDynamics tensor-driven meter updates (deplete, cascades, terminal checks) using compiler-provided optimization data (187 lines)
- `dac_engine.py` - DACEngine runtime reward computation from declarative Drive As Code specs with modifier evaluation, VFS integration, intrinsic weight modulation (849 lines)
- `reward_strategy.py` - RewardStrategy (multiplicative health×energy) and AdaptiveRewardStrategy (additive with crisis suppression) for interoception-aware survival rewards (234 lines)
- `action_builder.py` - ComposedActionSpace for action vocabulary management (substrate + custom + affordance) with enabled/disabled masking and action lookup (229 lines)
- `action_config.py` - ActionConfig and ActionSpaceConfig Pydantic DTOs for action definitions with VFS integration (reads/writes), cost/effect specs, movement delta validation (140 lines)
- `affordance_config.py` - AffordanceConfig and AffordanceConfigCollection DTOs for affordance definitions with multi-tick validation, operating hours, cost/effect schemas (254 lines)
- `cascade_config.py` - BarsConfig, CascadesConfig, TerminalCondition DTOs for meter dynamics with contiguous index validation, self-cascade prevention (318 lines)
- `substrate_action_validator.py` - SubstrateActionValidator ensuring action space compatibility with substrate topology (grid square/cubic/hex, aspatial) and movement delta requirements (95 lines)
- `action_labels.py` - Action label presets (gaming, 6dof, cardinal, math) and custom label support for UI rendering (359 lines)
- `affordance_layout.py` - Affordance position randomization and layout helpers for grid-based substrates (73 lines)

**Dependencies:**
- Inbound: Universe Compiler (compiled.py, runtime.py for CompiledUniverse consumption), Population (vectorized.py for training integration), Demo System (runner.py, live_inference.py for environment instantiation), VFS (registry.py for observation construction), Substrates (all substrate types for spatial representation)
- Outbound: Substrates (base.py SpatialSubstrate interface, factory.py for substrate creation), VFS (registry.py VariableRegistry for state storage, schema.py for WriteSpec), Config DTOs (all environment config DTOs consumed by Universe Compiler), PyTorch (tensors for all state)

**Patterns Observed:**
- Composition over inheritance (VectorizedHamletEnv composes AffordanceEngine, MeterDynamics, RewardStrategy, VariableRegistry rather than inheriting from base environment)
- Engine-based architecture (AffordanceEngine, CascadeEngine, MeterDynamics, DACEngine encapsulate domain logic with independent lifecycle)
- GPU-native vectorization (all state as PyTorch tensors with [num_agents, ...] batch dimension, no Python loops in hot paths)
- VFS integration pattern (environment writes to VariableRegistry with writer="engine", agents read with reader="agent")
- Action pipeline pattern (action validation → custom action dispatch → movement application → affordance interaction → meter updates → cascades → terminal checks → reward computation)
- Observation construction pipeline (VFS registry update → field extraction → concatenation → masking for curriculum)
- Device-agnostic operations (device parameter threaded through all tensor operations, supports CPU/CUDA/MPS)
- Fixed vocabulary principle (observation dim constant across curriculum levels for transfer learning via disabled affordance masking)
- POMDP support (partial observability via local window encoding, full observability via complete grid encoding)
- Modern affordance pipeline (effect_pipeline with on_start/per_tick/on_completion/on_early_exit/on_failure stages)
- Temporal mechanics (24-tick day/night cycle, operating hours validation, interaction progress tracking)
- Retirement mechanics (agent lifespan tracking with retirement bonus reward)
- DTO-driven configuration (all behavioral parameters from CompiledUniverse, no hardcoded defaults except infrastructure parameters)

**Concerns:**
- Bidirectional dependency between config DTOs and environment (config imports BarsConfig/CascadesConfig/AffordanceConfigCollection from environment, environment imports config DTOs)
- VectorizedHamletEnv.__init__ is large (470+ lines including helper methods) with complex initialization sequence (substrate, VFS, engines, dynamics, affordance layout)
- Custom action dispatch uses Python loop over custom_agent_indices (vectorization possible but requires more complex tensor indexing)
- Affordance position randomization regenerates every reset when randomize_affordances=true (deterministic seeding not exposed to environment consumers)
- DAC integration optional but presence checked at initialization (will become required per compiler.py comment)

**Confidence:** High - All 14 files read (4955 total lines), VectorizedHamletEnv step lifecycle traced (action execution → meter updates → cascades → terminal checks → rewards → observations), engine composition verified (AffordanceEngine, MeterDynamics, DACEngine), VFS integration examined (VariableRegistry reads/writes), POMDP vs full observability branches confirmed, dependencies bidirectional (inbound from Universe Compiler/Population/Demo, outbound to Substrates/VFS), GPU vectorization patterns consistent, temporal mechanics understood

---
## VFS (Variable & Feature System)

**Location:** `/home/john/hamlet/src/townlet/vfs/`

**Responsibility:** Provides declarative state space configuration through YAML-defined variables with scope semantics, access control, and observation field specifications, enabling the Universe Compiler to generate observation specs and the Environment to construct runtime observations from GPU tensor storage.

**Key Components:**
- `schema.py` - Pydantic schemas for VFS configuration: VariableDef (variable definitions with scope/type/lifetime/access control), ObservationField (observation field specs with normalization), NormalizationSpec (minmax/zscore normalization), WriteSpec (action variable updates), load_variables_reference_config loader (306 lines)
- `registry.py` - VariableRegistry runtime storage managing GPU tensors for all variables with access control enforcement, scope-based shape computation (global/agent/agent_private), and defensive cloning to prevent aliasing (279 lines)
- `observation_builder.py` - VFSObservationSpecBuilder compile-time spec generator converting variable definitions to ObservationField specs with shape inference, normalization validation, and exposure configuration (201 lines)
- `__init__.py` - Module interface exposing VariableDef, ObservationField, NormalizationSpec, WriteSpec, VariableRegistry, VFSObservationSpecBuilder (26 lines)

**Dependencies:**
- Inbound: Universe Compiler (compiler.py loads variables_reference.yaml via load_variables_reference_config, compiler_inputs.py aggregates VariableDef list, adapters/vfs_adapter.py converts ObservationField to compiler DTOs), Environment (vectorized_env.py uses VariableRegistry for runtime state, dac_engine.py reads variables for DAC reward computation)
- Outbound: Pydantic (BaseModel for schema validation with extra="forbid"), PyYAML (yaml.safe_load for YAML parsing), PyTorch (torch.Tensor for GPU-native storage, torch.device for device placement)

**Patterns Observed:**
- Declarative configuration pattern with YAML as single source of truth for state space (variables_reference.yaml required in all config packs)
- Three-scope model for variable visibility: global (shared single value), agent (per-agent public), agent_private (per-agent hidden from agent observations)
- Access control via reader/writer lists (agent, engine, acs, bac, actions) enforced at registry get/set with PermissionError on violations
- Compile-time vs runtime separation (VFSObservationSpecBuilder generates specs during compilation, VariableRegistry manages tensors during training)
- Type system with vector support: scalar, bool, vec2i, vec3i, vecNi, vecNf (Phase 1 types, Phase 2 planned for derivation graphs and expressions)
- Shape inference from variable type (scalar → [], vec2i → [2], vecNf with dims=N → [N]) with scope-based batching (global → [dims], agent → [num_agents, dims])
- Normalization specification with validation ensuring parameter shape matches observation shape (scalar params for scalar obs, list params for vector obs)
- Defensive copying pattern (get/set operations clone tensors to prevent external mutation of registry state)
- Auto-generation integration (Universe Compiler auto-generates standard variables for position, meters, affordances, temporal features from other configs)
- Observation field semantic typing (bars, spatial, affordance, temporal, custom) for structured encoders and curriculum masking
- Phase 1 vs Phase 2 architecture (Phase 1: basic types + observation specs, Phase 2: derivation graphs + expression evaluation)

**Concerns:**
- WriteSpec expression field currently stored as string with no parsing or validation (Phase 2 planned for AST parsing and execution)
- No validation that source_variable in ObservationField exists in VariableDef list until runtime (could fail during environment initialization)
- Variable scope semantics for agent_private unclear when reader="agent" (registry raises PermissionError but ObservationField exposure to agent not validated at compile-time)
- Normalization parameter validation allows single-element list for scalar observations (dims=1 and len(values)=1 special case may be confusing)

**Confidence:** High - All 4 files read completely (812 total lines), YAML config flow traced (variables_reference.yaml → load_variables_reference_config → VariableDef list → compiler symbol table → VFSAdapter → ObservationSpec DTOs), runtime integration verified (VariableRegistry used by vectorized_env.py and dac_engine.py), access control enforcement examined (readable_by/writable_by checked in get/set), scope semantics confirmed (global/agent/agent_private shape computation), dependencies bidirectional (inbound from Universe Compiler and Environment, outbound to Pydantic/PyYAML/PyTorch), sample config examined (aspatial_test/variables_reference.yaml with 4 meters + 5 affordances + 4 temporal = 13 dims)

---
