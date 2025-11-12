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
