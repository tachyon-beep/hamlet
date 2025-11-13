# Subsystem Catalog

**Project:** HAMLET Deep Reinforcement Learning Environment (Townlet System)
**Analysis Date:** 2025-11-13
**Workspace:** `docs/arch-analysis-2025-11-13-2013/`

---

## Overview

This catalog documents the major subsystems of the HAMLET/Townlet architecture, their responsibilities, dependencies, and architectural patterns. Each entry follows the standard 8-section format for consistency and downstream tool compatibility.

**Total Subsystems:** 10 core subsystems

**Analysis Approach:** Parallel subagent analysis with validation gates

---

# Core Subsystems

## Population

**Location:** `src/townlet/population/`

**Responsibility:** Coordinates batched training of parallel agents with shared curriculum and exploration strategies, managing Q-networks, target networks, and replay buffers for both feedforward and recurrent architectures.

**Key Components:**
- **VectorizedPopulation**: Main class managing num_agents parallel agents with vectorized transitions and batch training
- **q_network/target_network**: Online and target Q-networks (SimpleQNetwork, RecurrentSpatialQNetwork, StructuredQNetwork) for DQN/Double DQN
- **replay_buffer**: ReplayBuffer, SequentialReplayBuffer, or PrioritizedReplayBuffer depending on network type and BrainConfig
- **optimizer/scheduler**: Adam optimizer with optional learning rate scheduling (configurable via BrainConfig)
- **runtime_registry**: AgentRuntimeRegistry for agent telemetry, epsilon, curriculum stage tracking
- **curriculum/exploration**: Pluggable CurriculumManager and ExplorationStrategy for difficulty modulation and action selection

**Dependencies:**
- **Inbound:** DemoRunner, RunScript, inference systems load checkpoints and call step_population() for training/rollout
- **Outbound:** VectorizedHamletEnv (step actions, get masks), CurriculumManager (get_batch_decisions), ExplorationStrategy (select_actions), NetworkFactory/OptimizerFactory/LossFactory (BrainConfig), ReplayBuffer variants (store/sample), RNDExploration (intrinsic rewards)

**Patterns Observed:**
- **Batched Vectorization**: All agents processed in tensors [num_agents]; transitions stored in [num_agents, ...] shape tensors for parallel sampling
- **Dual Network Stability**: Online network (training) and target network (evaluation) with periodic hard sync every N training steps (configurable via target_update_frequency)
- **Recurrent Episode Accumulation**: For LSTM networks, transitions accumulated per-agent then stored as full episodes to SequentialReplayBuffer; for feedforward, immediate per-transition push to ReplayBuffer
- **Curriculum Integration**: Batch curriculum decisions computed before action selection, depletion multiplier applied to environment rewards for difficulty modulation
- **Double DQN/Vanilla DQN Toggle**: Algorithm variant configurable via use_double_dqn flag; Double DQN requires 3 forward passes (online selection, target evaluation, online for action), vanilla requires 2
- **Prioritized Replay (Feedforward Only)**: Optional PER via PrioritizedReplayBuffer with TD-error-based priorities and importance sampling weights (Phase 3 incomplete for recurrent)
- **BrainConfig Integration**: Network architecture, optimizer, loss function, replay capacity configurable via brain.yaml; falls back to legacy constructor parameters when brain_config=None
- **RND Predictor Co-Training**: RNDExploration/AdaptiveIntrinsicExploration trained asynchronously in step_population (line 694); intrinsic rewards computed for logging only (DAC already composes them)

**Concerns:**
- PrioritizedReplayBuffer not yet supported for recurrent networks (TASK-005 Phase 3 incomplete, raises NotImplementedError at line 318)
- BrainConfig architecture types (feedforward/recurrent/dueling) still require explicit if/elif chains rather than plugin system
- Network hidden dimensions (hidden_dim=256 for recurrent, 32 for group_embed) have TODO(BRAIN_AS_CODE) comments indicating incomplete config-driven design
- Episode containers for recurrent mode are pure Python lists with CPU tensors; GPU transfer occurs at store time (potential memory inefficiency)
- Double DQN in recurrent mode requires redundant forward passes through online network (lines 740-748) in addition to target network unroll

**Confidence:** High - VectorizedPopulation is production-critical path for all training; batched patterns verified across 800+ lines; concerns are documented as incomplete TASK-005 phases, not architectural flaws

---

## Variable & Feature System (VFS)

**Location:** `src/townlet/vfs/`

**Responsibility:** Declarative state space configuration providing compile-time observation spec generation and runtime access-controlled storage for VFS variables with GPU tensor backends.

**Key Components:**
- VariableDef: Variable definition with scope (global, agent, agent_private), type system, lifetime, and access control constraints
- VariableRegistry: Runtime GPU-backed tensor storage with permission enforcement for get/set operations
- VFSObservationSpecBuilder: Compile-time schema generation of ObservationField specs from variable definitions
- Access Control: Reader/writer permission lists enforced at registration and access time
- Type System: Scalar, vectors (vec2i, vec3i, vecNi, vecNf), bool with automatic shape computation

**Dependencies:**
- **Inbound:** Universe Compiler (UAC for observation spec generation), Environment (runtime variable access), BAC (consumes ObservationField specs for network input heads)
- **Outbound:** Pydantic (schema validation), PyTorch (GPU tensor storage/device management), YAML (variables_reference.yaml parsing)

**Patterns Observed:**
- Schema-First Validation: Pydantic models enforce all constraints (VariableDef, NormalizationSpec, ObservationField)
- Compile-Time/Runtime Separation: VFSObservationSpecBuilder generates schemas; VariableRegistry handles runtime tensors
- Defensive Copying: registry.get() and registry.set() clone tensors to prevent aliasing bugs
- Scope-Aware Shaping: Registry auto-computes tensor shapes (global: [], agent: [num_agents], etc.)
- Zero-Defaults Principle: All VariableDef fields required; no silent defaults (enforces reproducible configs)
- Semantic Grouping: ObservationField.semantic_type categorizes observations (bars, spatial, affordance, temporal, custom)
- Curriculum Masking: curriculum_active flag enables structured encoders to ignore inactive fields

**Concerns:** None observed

**Confidence:** High - Clean separation of concerns with explicit access control, GPU-native design, and no implicit defaults or hidden behaviors.

---

## Universe Compiler (UAC)

**Location:** `src/townlet/universe/`

**Responsibility:** Seven-stage configuration compilation pipeline that transforms YAML config packs into CompiledUniverse artifacts with validation, metadata generation, and runtime optimization.

**Key Components:**
- UniverseCompiler: Main entry point orchestrating seven-stage pipeline (parse, symbol table, resolve, validate, metadata, optimize, emit)
- UniverseSymbolTable: Central registry for resolving references across bars, affordances, cascades, and variables
- CuesCompiler: UI metadata extraction and compilation for affordances
- CompiledUniverse: Immutable frozen dataclass artifact containing all compiled state
- RawConfigs: Intermediate representation of parsed YAML configurations
- OptimizationData: Runtime performance optimization metadata
- DTOs (UniverseMetadata, ObservationSpec, ActionSpaceMetadata, MeterMetadata, AffordanceMetadata): Strongly-typed metadata structures

**Dependencies:**
- **Inbound:** DemoRunner (checkpoint loading), training pipeline, environment construction, model inference, CLI tools
- **Outbound:** Config system (SubstrateConfig, BarConfig, CascadeConfig, AffordanceConfig, DriveAsCodeConfig), VFS system (VariableRegistry, VFSObservationSpecBuilder, VariableDef), environment schemas (EnvironmentConfig), action system (SubstrateActionValidator, ActionSpaceConfig)

**Patterns Observed:**
- Seven-stage pipeline: YAML → Parse → Symbol Table → Resolve → Cross-Validate → Metadata → Optimize → Emit (immutable artifact)
- Two-level cache validation: mtime check (fast) followed by content hash comparison (accurate)
- Comprehensive error collection with staged reporting (Phase 0 syntax, Stage 3 references, Stage 4 cross-validation)
- Auto-generation of standard variables (spatial, meters, affordances, temporal) supplementing custom variables from variables_reference.yaml
- Immutable artifact pattern: CompiledUniverse is frozen dataclass with comprehensive post-init validation
- Provenance tracking: drive_hash, compiler_version, git_sha for reproducible checkpoints

**Concerns:**
- File size and complexity: 2600+ lines with intricate validation logic spread across 25+ validation methods
- Cache invalidation requires dual checks (mtime + hash) for correctness, adding complexity
- Economic balance validation is computationally expensive and may miss edge cases in complex cascade graphs
- DAC (Drive As Code) integration is mandatory with no fallback—old reward strategy system completely removed
- Symbol resolution for cascades relies on linear graph traversal (O(n²) potential in heavily cascaded systems)

**Confidence:** High - Codebase is well-structured with clear stage separation, comprehensive error handling, and extensive validation logic. Pipeline architecture is mature and documented in CLAUDE.md.

---

## Agent Networks

**Location:** `src/townlet/agent/`

**Responsibility:** Design and implement Q-network architectures for value function approximation across different observability conditions and curriculum levels.

**Key Components:**
- SimpleQNetwork: Stateless MLP (obs_dim → hidden → hidden → action_dim) with LayerNorm for full observability
- RecurrentSpatialQNetwork: LSTM-based recurrent network with encoder pipeline (vision CNN, position, meter, affordance MLPs) for partial observability (POMDP)
- DuelingQNetwork: Value/advantage stream decomposition (Wang et al. 2016) with flexible layer configuration and dropout support
- StructuredQNetwork: Semantic group-based encoders leveraging ObservationActivity for structured observation processing

**Dependencies:**
- **Inbound:** NetworkFactory (builds networks from BrainConfig), VectorizedPopulation training loop, checkpoint loading/inference systems, DemoRunner
- **Outbound:** torch.nn (PyTorch), TYPE_CHECKING import from townlet.universe.dto.ObservationActivity

**Patterns Observed:**
- Encoder-based architecture: Vision, position, meter, affordance encoders isolate concerns and compose into unified representation
- Multi-path processing: Different encoders converge before LSTM or Q-head for representation fusion
- State management lifecycle: LSTM hidden state resets at episode start, persists during rollout, managed via reset/set/get methods
- Flexible substrate support: Conditional position encoder handles Grid2D (2D), Grid3D (3D), and Aspatial (0D) position_dims
- Consistent normalization: LayerNorm applied universally across all network types for training stability
- Explicit parameterization: All architecture parameters specified in constructor (PDR-002 compliance); no hidden defaults

**Concerns:**
- Temporal features extracted in RecurrentSpatialQNetwork but documented as "ignored" (code comment line 213) - potential future signal loss
- DuelingQNetwork exists but not referenced in factory or training code; unclear if actively used in curriculum
- StructuredQNetwork creates tight coupling to ObservationActivity schema; changes to observation structure require network changes
- Hidden state initialization defaults to CPU device (line 271-272) despite network potentially on GPU - device mismatch risk if not handled by caller

**Confidence:** High - Code is well-documented with comprehensive docstrings, architecture patterns are explicit and consistent, dependencies are straightforward and traceable through NetworkFactory and BrainConfig system.

---

## Vectorized Environment

**Location:** `src/townlet/environment/`

**Responsibility:** Execute GPU-native batched simulation of multiple independent Hamlet environments with centralized reward computation and state management.

**Key Components:**
- VectorizedHamletEnv: Main orchestrator managing agent positions, meters, observations, action execution, and temporal mechanics for batched environments
- DACEngine: Declarative reward computation with extrinsic/intrinsic/shaping composition and modifier-based crisis suppression
- ComposedActionSpace: Composable action vocabulary (substrate + custom + affordance) with masking for disabled actions
- AffordanceEngine: Interaction handling with multi-tick temporal mechanics, costs, and effects
- MeterDynamics: Cascading meter depletion and terminal condition checking
- VFS Integration: Variable/Feature System registry for observation building and runtime state access

**Dependencies:**
- **Inbound:** VectorizedPopulation (training loop), DemoRunner (inference), live_inference (visualization), CompiledUniverse (initialization)
- **Outbound:** substrate (Grid2D/3D/Continuous factories), universe.compiled (metadata/optimization tensors), vfs.registry (observation state), exploration.base (intrinsic reward), affordance_engine, meter_dynamics, cascade_engine, action_builder

**Patterns Observed:**
- GPU-native batching: All state (positions, meters, observations, rewards) stored as PyTorch tensors with [num_agents, ...] batch dimension
- Lazy evaluation: DACEngine compiled once at __init__, reused for all reward calculations without recompilation
- Substrate-aware observation encoding: POMDP local window vs full-grid encoding determined at init with comprehensive validation
- State machine for temporal mechanics: interaction_progress tracks multi-tick affordance interactions with position-dependent reset logic
- VFS registry as centralized state bus: All observation features written each step (position, meters, affordances, velocity) before observation construction
- Action masking for efficiency: Invalid movements blocked before Q-network evaluation to preserve exploration budget
- Affordance tracking for shaping rewards: Per-agent tracking of _last_affordances, _affordance_streaks, _affordances_seen for diversity/streak bonuses

**Concerns:**
- POMDP validation (partial observability + Grid3D vision_range, GridND dimension explosion) is comprehensive but catches errors at initialization, not config time
- Multiple parallel affordance tracking mechanisms (_last_affordances, _affordance_streaks, _affordances_seen) create coupling risk if streak/unique logic diverges
- Observation dimension computed dynamically from VFS spec but validated against metadata—mismatch suggests inconsistent compiler output
- Complex closure factories in DACEngine._compile_modifiers/shaping may mask variable capture bugs; tight coupling between config structure and function generation

**Confidence:** High - Vectorized environment is the architectural core, thoroughly instrumented with GPU tensors, comprehensive substrate validation, and explicit observation spec validation. DACEngine integration verified through reward component logging and intrinsic weight modulation.

---

## Drive As Code (DAC)

**Location:** `src/townlet/environment/dac_engine.py` + `configs/*/drive_as_code.yaml`

**Responsibility:** Compiles declarative YAML reward configurations into GPU-native computation graphs for runtime reward calculation with extrinsic/intrinsic/shaping composition and context-sensitive modifier chaining.

**Key Components:**
- **DACEngine**: Main class orchestrating reward compilation and vectorized computation across num_agents
- **Modifiers**: Range-based multiplier functions using torch.where for GPU-efficient lookups (e.g., energy_crisis suppresses intrinsic when energy < 0.2)
- **Extrinsic Strategies**: 9 types for base reward computation (multiplicative, constant_base_with_shaped_bonus, additive_unweighted, weighted_sum, polynomial, threshold_based, aggregation, vfs_variable, hybrid)
- **Intrinsic Strategies**: 5 exploration drives (rnd, icm, count_based, adaptive_rnd, none) with base_weight and modifier application
- **Shaping Bonuses**: 11 behavioral incentive types (approach_reward, completion_bonus, efficiency_bonus, state_achievement, streak_bonus, diversity_bonus, timing_bonus, economic_efficiency, balance_bonus, crisis_avoidance, vfs_variable)
- **Composition Pipeline**: Total reward = extrinsic + (intrinsic × effective_intrinsic_weight) + shaping with dead-agent zeroing

**Dependencies:**
- **Inbound:** VectorizedHamletEnv.step() calls calculate_rewards() each timestep for reward composition
- **Outbound:** DriveAsCodeConfig (Pydantic schema from YAML), VariableRegistry (VFS for variable-based modifiers/bonuses), torch (GPU tensors [num_agents] batch dimension)

**Patterns Observed:**
- **Closure Factories**: Modifier and shaping bonus compilation uses factory functions (create_modifier_fn, create_*_bonus_fn) to capture config state and return closures; ensures correct variable capture for vectorized execution
- **GPU Vectorization**: All reward computations leverage torch.where for conditional logic and tensor broadcasting to [num_agents] shape; torch.stack/torch.clamp for aggregation
- **Modifier Chaining**: Multipliers composed multiplicatively (line 889: intrinsic_weight = intrinsic_weight × modifier_i); supports arbitrary number of context-sensitive suppressions
- **Dead Agent Handling**: torch.where(dones, zeros) zeros extrinsic/intrinsic/shaping for agents that died in current step (lines 169, 207, 235, etc.)
- **Component Logging**: calculate_rewards() returns dict of component tensors (extrinsic, intrinsic, shaping) for per-step reward analysis and debugging
- **VFS Integration**: Modifiers and bonuses read from VariableRegistry with reader="engine" access control; supports bar-based (meters[:, bar_idx]) and variable-based (vfs_registry.get()) sources

**Concerns:**
None observed

**Confidence:** High - DACEngine design is clean with explicit separation of compile-time (init) and runtime (calculate_rewards) phases, consistent closure factory patterns across 11 shaping bonus types, comprehensive vectorization using torch primitives, and clear integration point (single calculate_rewards call per env step). Formula is well-documented and modifiers provide pedagogically valuable reward hacking suppression without hardcoded thresholds.

---

## Curriculum

**Location:** `src/townlet/curriculum/`

**Responsibility:** Provides configurable difficulty progression strategies through per-agent environment adaptation, enabling both static baselines and adversarial auto-tuning based on performance metrics.

**Key Components:**
- **CurriculumManager** (base.py): Abstract interface defining get_batch_decisions(), checkpoint_state(), load_state() contract called once per episode
- **StaticCurriculum** (static.py): Fixed difficulty implementation returning identical decisions for all agents; parameters: difficulty_level, reward_mode, active_meters, depletion_multiplier
- **AdversarialCurriculum** (adversarial.py): Auto-tuning manager advancing/retreating agents through 5 stages based on survival rate, learning progress, and entropy thresholds
- **PerformanceTracker** (adversarial.py): GPU-backed metrics tracking per-agent survival, reward, learning progress, and stage tenure with tensor-based state for checkpoint serialization
- **StageConfig** (adversarial.py): Pydantic model defining stage specification (active_meters, depletion_multiplier, reward_mode, description)
- **STAGE_CONFIGS** (adversarial.py): Pre-configured 5-stage progression from basic needs + shaped rewards (Stage 1: 20% depletion) to full complexity + sparse rewards (Stage 5)

**Dependencies:**
- **Inbound:** VectorizedPopulation (calls get_batch_decisions once per episode before action selection, passes Q-values for entropy), DemoRunner (checkpoint loading/saving), training loop
- **Outbound:** training.state.BatchedAgentState/CurriculumDecision (DTOs), torch (GPU tensor operations for PerformanceTracker), yaml (loading curriculum config), pydantic (StageConfig validation)

**Patterns Observed:**
- **Strategy Pattern**: CurriculumManager interface with pluggable Static/Adversarial implementations for baseline vs adaptive curriculum
- **Episodic Decision Frequency**: get_batch_decisions called once per episode (not per step) to amortize computation; curriculum state persists across steps within episode
- **GPU-Native Tracking**: PerformanceTracker uses torch tensors [num_agents] for efficient batched metrics; device-aware storage enables checkpoint serialization
- **Multi-Signal Advancement Logic**: Adversarial requires conjunction of three signals (high survival AND positive learning AND low entropy) for advancement; OR logic for retreat (low survival OR negative learning)
- **Stage-Based Configuration**: STAGE_CONFIGS array maps integer stages (1-5) to difficulty parameters; difficulty_level normalized to [0.0, 1.0] output range
- **Transition Event Recording**: AdversarialCurriculum logs stage transitions with full context (agent_id, from/to stage, metrics, reason) for telemetry and curriculum analysis
- **Minimum Tenure Gate**: min_steps_at_stage prevents thrashing (default 1000 steps before stage transition allowed)
- **Adaptive Entropy Gating**: Entropy calculated from Q-value distributions (softmax → categorical entropy → normalized by log(num_actions)) for policy convergence detection

**Concerns:**
- PerformanceTracker.update_step() comment (lines 88-89) indicates semantic ambiguity: rewards parameter actually receives step counts not rewards, creating confusion for future maintainers
- STAGE_CONFIGS is hardcoded module-level constant; no per-pack curriculum configuration (Stage definitions not parameterizable via config YAML unlike environment/training parameters)
- Entropy calculation in _calculate_action_entropy() always normalizes by log(num_actions); assumes fixed action space—may fail if action space size varies across substrates
- Retreat logic (line 256) uses OR condition; single failed metric triggers retreat, while advancement requires conjunction—asymmetric criteria may cause stage cycling
- AdversarialCurriculum.get_batch_decisions() (lines 365-379) is test stub with dummy Q-values; production relies on get_batch_decisions_with_qvalues()—risk of calling wrong method

**Confidence:** High - Curriculum system has clear separation between static baseline and adaptive strategies, well-documented state management with checkpoint support, and explicit tensor-based GPU integration. Strategy pattern enables future implementations (e.g., teacher curriculum, hierarchical progression). Minor concerns are documentation ambiguities and stage configuration inflexibility, not architectural flaws.

---

## Training State

**Location:** `src/townlet/training/`

**Responsibility:** Provide cold path DTOs for curriculum/exploration/checkpoint configuration and hot path GPU tensor representations for batched agent state and experience storage during training.

**Key Components:**
- **CurriculumDecision**: Pydantic DTO (cold path) for curriculum configuration with difficulty, active meters, depletion multiplier, reward mode
- **ExplorationConfig**: Pydantic DTO (cold path) for exploration strategy parameters (epsilon-greedy, RND, adaptive intrinsic)
- **PopulationCheckpoint**: Pydantic DTO (cold path) for serializing full population state including generation, agent IDs, per-agent curriculum/exploration states, Pareto frontier
- **BatchedAgentState**: Hot path class with __slots__ for GPU tensor batching [batch, ...] with observations, actions, rewards, dones, curriculum difficulties, intrinsic rewards
- **ReplayBuffer**: Circular buffer with dual reward tracking (extrinsic + intrinsic), lazy tensor initialization, random mini-batch sampling with combined rewards, serialize/deserialize for checkpointing

**Dependencies:**
- **Inbound:** VectorizedPopulation (training loop creates/manages BatchedAgentState, stores/samples ReplayBuffer), DemoRunner (loads PopulationCheckpoint for inference, serializes for saving)
- **Outbound:** torch (GPU tensors, device management), numpy (telemetry extraction), pydantic (schema validation)

**Patterns Observed:**
- **Cold/Hot Path Separation**: Pydantic DTOs (frozen=True) for configuration/checkpoints (cold path); lightweight PyTorch tensors for training loop (hot path)
- **Dual Reward Composition**: ReplayBuffer separates extrinsic and intrinsic rewards; sampling applies intrinsic_weight multiplier for flexible reward composition
- **Circular Buffer FIFO**: ReplayBuffer uses position % capacity for efficient memory reuse; lazy tensor initialization on first push reduces startup overhead
- **Batch Tensor Representation**: BatchedAgentState uses __slots__ for memory efficiency; all data is [batch_size, ...] shaped for GPU vectorization
- **Serialization for Checkpointing**: Both PopulationCheckpoint and ReplayBuffer implement serialize/load_from_serialized for reproducible checkpoint restoration
- **Device Portability**: Explicit device parameter in ReplayBuffer and BatchedAgentState.to() method for CPU/CUDA flexibility
- **No-Validation Hot Path**: BatchedAgentState.__init__ skips validation for performance; DemoRunner validates CurriculumDecision/ExplorationConfig at construction

**Concerns:**
- ReplayBuffer lazy initialization (storage tensors created on first push) may cause device mismatches if buffer created on CPU but first push on GPU
- PopulationCheckpoint uses dict[str, Any] for curriculum_states/exploration_states—loose typing sacrifices schema enforcement for extensibility
- BatchedAgentState.to() creates new instance but doesn't validate device consistency of info dict (could contain CPU-only data)
- No prioritized replay variants in state.py (variants handled elsewhere in Population subsystem)

**Confidence:** High - Code is clearly separated into cold/hot paths with well-defined responsibilities, comprehensive docstrings, explicit device handling, and straightforward integration points with Population and DemoRunner.

---

## Frontend Visualization

**Location:** `frontend/src/components/`

**Responsibility:** Render real-time agent simulation state (spatial grids, meters, affordances, action history) with substrate-aware layouts and WebSocket-based streaming from inference server.

**Key Components:**
- **Grid.vue**: SVG-based 2D grid rendering with agent positions, affordances, heat map overlays (RND novelty), and agent movement trails with fading opacity
- **AspatialView.vue**: Non-spatial resource management dashboard for aspatial substrates with three sections (large meters, affordance grid, action history)
- **MeterPanel.vue**: Hierarchical meter display (Primary: survival-critical energy/health/money, Secondary: modifiers mood/satiation/fitness, Tertiary: accelerators hygiene/social) with cascade relationship annotations and special animations (pulse critical, strobe for loneliness)

**Dependencies:**
- **Inbound:** Pinia store (useSimulationStore) supplies substrate metadata, agent positions, meter values, heat map data; WebSocket streaming from live_inference server on port 8766
- **Outbound:** Vue 3 (setup script composition API), Pinia (reactive state management), WebSocket API (real-time updates), utils/constants.js (CELL_SIZE, AFFORDANCE_ICONS), utils/formatting.js (capitalize, formatMeterValue, getMeterPercentage), EmptyState.vue (shared fallback component)

**Patterns Observed:**
- **Props-first data flow**: All components receive data via props, never directly access store (enables testing and parent composition)
- **Substrate-aware rendering**: Grid.vue for spatial (grid2d/grid3d), AspatialView.vue for non-spatial substrates (selector based on substratePositionDim)
- **SVG vectorization**: Grid.vue uses SVG <g> groups for layers (cells, heat map, affordances, trails, agents) with responsive viewBox scaling
- **Design token system**: Color variables (--color-meter-*, --tier-color), spacing, border-radius, transitions centralized in CSS custom properties
- **Accessibility-first**: ARIA roles (progressbar, listitem, status), aria-live regions for meter updates, semantic HTML (<section>, <h3>), keyboard shortcuts (H to toggle heat map)
- **Multi-layer heat map**: Intensity gradient mapped to color ramp (blue → cyan → green → yellow → red) with 0.6 opacity overlay
- **Meter tier hierarchy**: Primary (critical, full-width bars) → Secondary (modifiers, with relationship text) → Tertiary (accelerators, flagged with ⚡) reflecting cascade dependencies
- **Responsive grid layouts**: Mobile single-column → tablet+ 2-column for AspatialView; Grid.vue max-height scales (400px mobile, 500px tablet, max-grid-size desktop)
- **Animation repertoire**: Pulse (critical meters), strobe-slow/strobe-fast (mood when lonely), fade transitions (agent trails), hover lift (affordance cards)

**Concerns:**
- Heat map color gradient hardcoded in getHeatColor() function (lines 253-272) - no externalization to utils/constants, making changes requires component edit
- Action names hardcoded as array in AspatialView.vue (line 93: ['Up', 'Down', 'Left', 'Right', 'Interact', 'Wait']) - should derive from universe substrate metadata to support custom action vocabularies
- Meter relationships and tier classifications hardcoded in MeterPanel.vue (lines 208-221, 213-221) - should be computed from universe cascade graph at init time, risks divergence from engine configuration
- Color values inconsistent: MeterPanel.vue mixes CSS variables (--color-meter-*) with hardcoded hex (#10b981, #8b5cf6, #991b1b) - consolidation needed
- No error handling for malformed WebSocket messages or missing/null props - should add graceful fallbacks and console warnings
- Keyboard event handler in Grid.vue (line 221) only excludes INPUT/TEXTAREA from event handling - should extend to other form controls (select, button, contenteditable)
- Agent trail tracking in Grid.vue uses client-side state (agentTrails ref) rather than server-provided history - potential desync if WebSocket messages arrive out of order

**Confidence:** Medium - Components are well-structured with strong accessibility and design patterns, but contain hardcoded configuration values that should be derived from CompiledUniverse metadata. WebSocket streaming from backend verified in simulation.js. Concerns around hardcoded meter hierarchies and action vocabularies suggest tight coupling to specific game configurations rather than generic infrastructure.

---

## Exploration

**Location:** `src/townlet/exploration/`

**Responsibility:** Implement action selection strategies (epsilon-greedy, RND, adaptive intrinsic) and compute intrinsic motivation rewards for exploration-exploitation trade-off in batched vectorized training.

**Key Components:**
- **ExplorationStrategy**: Abstract base class defining select_actions, compute_intrinsic_rewards, update, checkpoint_state, load_state interface
- **EpsilonGreedyExploration**: Simple baseline with decaying epsilon, no intrinsic motivation
- **RNDExploration**: Random Network Distillation using fixed + predictor network pair; prediction error as novelty signal with Welford's online normalization
- **AdaptiveIntrinsicExploration**: RND with variance-based annealing; wraps RNDExploration and reduces intrinsic weight when survival variance drops below threshold
- **epsilon_greedy_action_selection**: Shared vectorized utility for action selection with action masking and per-agent epsilon values
- **RunningMeanStd**: Welford's online algorithm tracker for numerically stable intrinsic reward normalization

**Dependencies:**
- **Inbound:** VectorizedPopulation (select_actions for training step, compute_intrinsic_rewards for logging), checkpoint loading/restoration
- **Outbound:** training.state.BatchedAgentState, torch (GPU tensors), numpy (RunningMeanStd calculations)

**Patterns Observed:**
- **Strategy pattern**: ExplorationStrategy interface enables pluggable implementations (epsilon-greedy, RND, adaptive)
- **Composition pattern**: AdaptiveIntrinsicExploration wraps RNDExploration instance for code reuse and annealing logic
- **Vectorized GPU operations**: All action selection uses batch tensors; multinomial sampling 10-100× faster than Python loops
- **Welford online statistics**: RunningMeanStd avoids storing full history, enabling memory-efficient running variance tracking
- **Epsilon annealing schedule**: Exponential decay (epsilon *= decay_rate) with floor (epsilon_min) to prevent premature convergence to pure greedy
- **Stateful survival tracking**: AdaptiveIntrinsicExploration maintains survival_history window for variance computation; incremental window checking allows early annealing feedback
- **Checkpoint/restore lifecycle**: All strategies support serialization (weights, optimizer state, epsilon, intrinsic weight, normalization stats) for resumable training

**Concerns:**
- AdaptiveIntrinsicExploration annealing logic relies on variance + mean survival thresholds; threshold values (variance_threshold=100.0, min_survival_fraction=0.4) are config parameters but not exposed in curriculum configs, requiring code edits to tune
- RND normalization uses running statistics across episodes; stats reset may cause reward scale discontinuities if checkpoint loaded mid-training
- Action masking in epsilon_greedy_action_selection adds 1e-8 to probabilities defensively but may introduce subtle numerical bias in multinomial sampling with many invalid actions
- No formal convergence guarantee for adaptive annealing; "consistently succeeding" threshold may be curriculum-dependent, requiring empirical tuning

**Confidence:** High - All exploration implementations follow clear interface contracts, vectorization patterns are GPU-optimized and well-tested, dependency chains are straightforward (strategy ← population ← training), and checkpoint mechanism is comprehensive with state restoration parity.

---

