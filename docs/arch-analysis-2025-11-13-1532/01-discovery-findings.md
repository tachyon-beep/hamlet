# Discovery Findings - HAMLET Architecture

**Scan Date**: November 13, 2025  
**Scope**: Codebase architecture mapping (Medium thoroughness)  
**Focus**: `src/townlet/` (production code) with supporting systems analysis  

## Directory Structure

HAMLET follows a **layer-based + domain-based hybrid** organization:

```
src/townlet/                        # Production code (active)
├── agent/                          # Neural network architectures & Q-learning
├── compiler/                       # CLI entry point for Universe Compiler
├── config/                         # Configuration DTO layer (YAML→Python)
├── curriculum/                     # Curriculum strategies (adversarial/static)
├── demo/                          # Training runner & inference server
├── environment/                    # Core RL loop & engine layer
├── exploration/                    # Intrinsic motivation (RND, ICM, epsilon-greedy)
├── population/                     # Batched agent training orchestration
├── recording/                      # Trajectory recording & replay
├── substrate/                      # Spatial topology implementations (Grid2D, Grid3D, Continuous, etc.)
├── training/                       # Training utilities (replay buffers, checkpointing)
├── universe/                       # Configuration compiler (7-stage pipeline)
└── vfs/                           # Variable & Feature System (observation spec generation)

configs/                            # Configuration packs (curriculum levels)
├── L0_0_minimal/                  # Pedagogical level 0: temporal credit assignment
├── L0_5_dual_resource/            # Pedagogical level 0.5: multiple resources
├── L1_full_observability/         # Pedagogical level 1: full observability baseline
├── L1_3D_house/                   # Grid3D variant of L1
├── L1_continuous_1D/              # Continuous substrate variant
├── L2_partial_observability/      # POMDP with LSTM + local vision window
├── L3_temporal_mechanics/         # Time-based dynamics (day/night cycles)
└── experiments/                   # A/B test configurations

frontend/                           # Vue.js real-time visualization
├── src/
│   ├── components/                # Grid.vue (spatial), AspatialView.vue (meters-only)
│   ├── stores/                    # Pinia state management
│   ├── styles/                    # Meter colors, grid styling
│   └── utils/                     # Affordance icons, WebSocket communication

tests/                              # Pytest suite (GPU/CPU/integration)
├── test_townlet/
│   ├── unit/                      # Isolated component tests
│   ├── integration/               # Full training loop tests
│   └── e2e/                       # End-to-end training runs
```

### Configuration Pack Structure (Per Level)

Each curriculum level contains 9 required files:

```
configs/L{N}/
├── substrate.yaml                 # Spatial topology (Grid2D size, boundaries, etc.)
├── bars.yaml                      # Meter definitions (energy, health, satiation, etc.)
├── cascades.yaml                  # Meter relationships (e.g., low energy → low health)
├── affordances.yaml               # Interaction definitions with effects
├── cues.yaml                      # UI metadata (colors, symbols, tooltips)
├── training.yaml                  # Hyperparameters (learning_rate, batch_size, etc.)
├── drive_as_code.yaml            # REQUIRED: Reward function declarative spec
├── brain.yaml                     # Network architecture & optimizer config
└── variables_reference.yaml       # REQUIRED: Variable declarations for VFS
```

## Entry Points

### Training Entry
- **File**: `/home/user/hamlet/scripts/run_demo.py`
- **Usage**: `python run_demo.py --config configs/L1_full_observability --episodes 10000`
- **Flow**: 
  1. Parses CLI args (config path, episode count, checkpoint dir)
  2. Creates `UnifiedServer` instance
  3. Starts training + inference server (frontend runs separately)
  4. Broadcasts agent observations/rewards via WebSocket to `localhost:8766`

### Inference Server Entry
- **File**: `src/townlet/demo/live_inference.py`
- **Class**: `LiveInference`
- **Usage**: Spawned by `UnifiedServer` to serve live agent states for visualization
- **WebSocket**: Broadcasts on `localhost:8766` (configurable)

### Compiler CLI Entry
- **File**: `src/townlet/compiler/__main__.py`
- **Usage**: `python -m townlet.compiler {compile,inspect,validate} <config_dir>`
- **Commands**:
  - `compile`: Seven-stage UAC pipeline, generates cache artifacts
  - `inspect`: Diagnostic tool for debugging config packs
  - `validate`: Pre-flight validation before training

### Configuration Entry Points (Declarative)
- **Config Pack**: Directory path to `configs/L{N}/`
- **Parser**: `UniverseCompiler` class (2542 lines)
- **Pipeline**: YAML files → Pydantic DTOs → CompiledUniverse dataclass

### Database Entry (Persistence)
- **File**: `src/townlet/demo/database.py`
- **Class**: `DemoDatabase` (SQLite backend)
- **Usage**: Stores training episodes, checkpoints, metrics for resumption

## Technology Stack

### Core ML/RL Framework
- **PyTorch** 2.9.0+: Neural network training, GPU-native tensors
- **Gymnasium** 1.0.0+: RL environment interface compliance
- **PettingZoo** 1.24.0+: Multi-agent RL utilities (curriculum integration)
- **TensorFlow/TensorFlow[cuda]** 2.20.0+: Recording & analysis pipeline

### Configuration & Serialization
- **PyYAML** 6.0+: Configuration file parsing
- **Pydantic** 2.0+: Schema validation (DTO layer enforces all fields explicit)
- **CloudPickle** 3.0+: Cross-process checkpoint serialization
- **msgpack + lz4**: Trajectory recording compression

### Server & Communication
- **FastAPI** 0.100.0+: Async REST endpoints (inference API)
- **Uvicorn** 0.23.0+: ASGI server
- **Flask** 3.0.0+ (secondary): Legacy compatibility
- **WebSockets** 11.0+: Real-time agent state streaming to frontend
- **python-dotenv** 1.0.0+: Environment variable configuration

### Frontend
- **Vue.js** 3.x: Component framework
- **Vite**: Build tool & dev server (HMR support)
- **WebSocket client**: Browser native WebSocket API

### Data Analysis & Logging
- **MLflow** 2.9.0+: Experiment tracking & checkpoint versioning
- **Pandas** 2.0.0+: Time-series analysis (episode metrics)
- **NumPy** 1.24.0+: Numeric computation
- **scikit-learn** 1.3.0+: Metric utilities
- **Rich** 13.0.0+: Terminal output formatting

### Development & Testing
- **pytest** 7.4.0+: Test runner
- **pytest-cov**: Coverage measurement
- **black**: Code formatting
- **ruff**: Linting (E, F, I, N, W, UP rules)
- **mypy** 1.4.0+: Type checking
- **hypothesis** 6.100.0+: Property-based testing

### Versioning & Analysis
- **GitPython** 3.1.0+: Programmatic Git access (commit provenance)
- **Requests** 2.31.0+: HTTP client for API calls

### Language & Version Constraints
- **Python**: 3.13+ (target-version in ruff/black)
- **Type Hints**: Gradual typing (mypy lenient by default, can tighten)
- **Code Style**: 140-character line length (ruff/black)

## Subsystem Identification

### 1. **Universe Compiler (UAC)** - 7-Stage Configuration Pipeline
**Responsibility**: Transform YAML config packs into GPU-native computation graphs

**Key Files**:
- `universe/compiler.py` (2542 lines): Main pipeline orchestration
- `universe/compiled.py`: Output dataclass (immutable artifact)
- `universe/symbol_table.py`: Symbol resolution across config files
- `universe/cues_compiler.py`: UI metadata compilation

**Pipeline Stages**:
1. Validate YAML syntax
2. Build symbol table (all entity names)
3. Resolve references (affordances→cascades→bars)
4. Cross-validate constraints (meter balance, affordance effects)
5. Generate observation specs + metadata
6. Optimize GPU compute graphs
7. Cache/emit compiled artifacts

**Coupling**: Central hub—integrates config DTOs, VFS specs, DAC engine, substrate validators

**Status**: ✅ Production (TASK-002C, TASK-003C, TASK-004C complete)

---

### 2. **Drive As Code (DAC) Engine** - Declarative Reward System
**Responsibility**: Compile YAML reward specifications into GPU-native reward computation

**Key Files**:
- `config/drive_as_code.py`: DAC schema DTOs
- `environment/dac_engine.py` (40K lines): Runtime reward computation

**Features**:
- **9 extrinsic strategies**: multiplicative, constant_base_with_shaped_bonus, additive, weighted_sum, polynomial, threshold_based, aggregation, vfs_variable, hybrid
- **5 intrinsic strategies**: RND (Random Network Distillation), ICM (Intrinsic Curiosity), count_based, adaptive_rnd, none
- **11 shaping bonuses**: approach_reward, completion_bonus, efficiency_bonus, state_achievement, streak_bonus, diversity_bonus, timing_bonus, economic_efficiency, balance_bonus, crisis_avoidance, vfs_variable
- **Context-sensitive modifiers**: Range-based multipliers (crisis suppression pattern)
- **Provenance tracking**: SHA256 `drive_hash` embedded in checkpoints

**Pedagogical Design**: L0_0_minimal demonstrates "Low Energy Delirium" bug; L0_5_dual_resource fixes it (teaches reward design importance)

**Status**: ✅ Production (DAC implementation complete; runtime integration complete)

---

### 3. **Vectorized Training Loop (Population)** - Batched Agent Orchestration
**Responsibility**: Batch training for multiple agents with curriculum progression

**Key Files**:
- `population/vectorized.py` (55K lines): VectorizedPopulation orchestrator
- `population/runtime_registry.py`: GPU tensor registry (global state)

**Features**:
- **Batch size**: Configurable (1-1024 agents typically)
- **GPU tensors**: All state on GPU for vectorized operations
- **Curriculum integration**: Static or adversarial difficulty progression
- **Checkpoint management**: Episode/transition granularity recovery
- **Metrics aggregation**: Per-agent + population-wide stats

**Coupling**: Tight integration with VectorizedHamletEnv, ExplorationStrategy, AdversarialCurriculum

**Status**: ✅ Production

---

### 4. **Vectorized Environment (Core RL Loop)** - Engine Layer
**Responsibility**: GPU-native vectorized HAMLET environment implementation

**Key Files**:
- `environment/vectorized_env.py` (72K lines): Core RL environment
- `environment/action_builder.py`: Action space construction
- `environment/affordance_engine.py`: Affordance effect application

**Features**:
- **Multi-substrate support**: Grid2D, Grid3D, GridND, Continuous, ContinuousND, Aspatial
- **Vectorized ops**: Batch reset/step/render on GPU
- **8 affordances max per action**: Multiplexed effects
- **Cascade system**: Meter relationships (energy drain → health drain)
- **Action validation**: Per-substrate constraint enforcement

**Integration Points**:
- DAC engine for rewards
- Substrate implementations for spatial operations
- VFS registry for observation construction

**Status**: ✅ Production

---

### 5. **Variable & Feature System (VFS)** - Observation Spec Generation
**Responsibility**: Declarative state space configuration + access control

**Key Files**:
- `vfs/schema.py`: VariableDef, ObservationField, NormalizationSpec Pydantic models
- `vfs/registry.py`: Runtime GPU tensor storage with access control
- `vfs/observation_builder.py`: Compile-time spec generator

**Features**:
- **Variable scopes**: global (shared), agent (per-agent), agent_private (hidden)
- **Access control**: Readers (agent, engine, acs, bac), Writers (engine, actions, bac)
- **Normalization modes**: relative [0,1], scaled (with metadata), absolute (raw)
- **Integration**: `variables_reference.yaml` REQUIRED in all config packs

**Observation Dimensions** (Grid2D examples):
- **L0/L0.5/L1/L3**: 29 dims (2 coords + 8 meters + 15 affordances + 4 temporal)
- **L2 (POMDP)**: 54 dims (25 local window + 2 coords + 8 meters + 15 affordances + 4 temporal)

**Key Insight**: Constant observation dimension across grid sizes enables transfer learning

**Status**: ✅ Production (TASK-002C complete)

---

### 6. **Agent Networks & Q-Learning** - Neural Architecture Selection
**Responsibility**: Train and deploy action-value networks with configurable algorithms

**Key Files**:
- `agent/networks.py`: SimpleQNetwork (MLP), RecurrentSpatialQNetwork (CNN+LSTM)
- `agent/network_factory.py`: Network instantiation from config
- `agent/loss_factory.py`: Loss function selection (MSE vs Huber)
- `agent/optimizer_factory.py`: Optimizer creation (Adam, SGD variants)
- `agent/brain_config.py`: Network architecture DTOs

**Architecture Options**:

1. **SimpleQNetwork** (Full observability - L0, L0.5, L1):
   - MLP: obs_dim → 256 → 128 → action_dim
   - All Grid2D configs: 29→8 (enables checkpoint transfer)
   - ~26K params

2. **RecurrentSpatialQNetwork** (Partial observability - L2, L3):
   - Vision encoder: 5×5 window → CNN → 128 features
   - Position encoder: (x,y) → MLP → 32 features
   - Meter encoder: 8 meters → MLP → 32 features
   - LSTM: 192 input → 256 hidden
   - Q-head: 256 → 128 → action_dim
   - ~650K params
   - LSTM hidden state: resets at episode start, persists during rollout

**Algorithm Variants**:
- **Vanilla DQN** (Mnih et al. 2015): Q-target = r + γ × max_a Q_target(s', a)
- **Double DQN** (van Hasselt et al. 2016): Q-target = r + γ × Q_target(s', argmax_a Q_online(s', a))
- Configurable via `training.yaml: use_double_dqn`

**Regularization**:
- Gradient clipping: max_norm=10.0
- Experience replay: Batch size configurable
- Target network sync: Configurable frequency

**Status**: ✅ Production

---

### 7. **Substrate Implementations** - Spatial Topology Abstraction
**Responsibility**: Pluggable spatial topology implementations

**Key Files**:
- `substrate/base.py`: Abstract Substrate interface
- `substrate/grid2d.py`: 2D discrete grid (Manhattan/Euclidean/Chebyshev distance)
- `substrate/grid3d.py`: 3D discrete grid
- `substrate/gridnd.py`: 4D-100D discrete hypercubes
- `substrate/continuous.py`: 1D/2D/3D continuous space
- `substrate/continuousnd.py`: 4D-100D continuous space
- `substrate/aspatial.py`: No positioning (pure resource management)
- `substrate/config.py`: SubstrateConfig (Pydantic validation)
- `substrate/factory.py`: Substrate instantiation

**Features**:
- **Boundary modes**: clamp (hard walls), wrap (toroidal), bounce (elastic), sticky
- **Distance metrics**: manhattan (L1), euclidean (L2), chebyshev (L∞)
- **Coordinate systems**: Discretized continuous (via action labels)
- **POMDP support**: ✅ Grid2D, Grid3D (vision_range ≤ 2), Aspatial | ❌ Continuous, GridND (N≥4)

**Status**: ✅ Production

---

### 8. **Exploration Strategies** - Intrinsic Motivation
**Responsibility**: Drive emergent behavior through novelty-seeking

**Key Files**:
- `exploration/base.py`: ExplorationStrategy interface
- `exploration/rnd.py`: Random Network Distillation (novelty bonuses)
- `exploration/adaptive_intrinsic.py`: RND + performance-based annealing
- `exploration/epsilon_greedy.py`: ε-greedy exploration
- `exploration/action_selection.py`: Tie-breaking (max ties)
- `config/exploration.py`: Exploration config DTOs

**Strategies**:
- **RND**: Forward network predicts next latent state; prediction error = novelty signal
- **Adaptive RND**: Anneals intrinsic weight based on mean episode survival (threshold=100.0)
- **Epsilon-greedy**: Decay schedule (linear, exponential, polynomial)

**Note**: All strategies are GPU-native with vectorized batch operations

**Status**: ✅ Production

---

### 9. **Curriculum Strategies** - Difficulty Progression
**Responsibility**: Adapt environment difficulty based on agent performance

**Key Files**:
- `curriculum/base.py`: CurriculumStrategy interface
- `curriculum/adversarial.py`: Adversarial curriculum (adjust difficulty per-batch)
- `curriculum/static.py`: Static curriculum (fixed difficulty)
- `config/curriculum.py`: Curriculum config DTOs

**Adversarial Curriculum**:
- Monitors batch-level survival metrics
- Adjusts affordance availability or meter drain rates
- Goal: Keep learning signal balanced (avoid overly easy/hard)

**Status**: ✅ Production

---

### 10. **Recording & Replay System** - Trajectory Capture
**Responsibility**: Capture & export training trajectories

**Key Files**:
- `recording/recorder.py`: Trajectory recording orchestrator
- `recording/criteria.py`: Filter conditions (e.g., record only interesting episodes)
- `recording/video_renderer.py`: Trajectory → video export
- `recording/video_export.py`: FFmpeg integration
- `recording/replay.py`: Trajectory playback utilities

**Features**:
- **Compression**: msgpack + lz4 for trajectory storage
- **Filtering**: Record only episodes meeting criteria (e.g., high reward)
- **Export**: Generate MP4/GIF animations for documentation

**Status**: ✅ Production (TASK-005A partial - recording infrastructure exists)

---

### 11. **Training Infrastructure** - Checkpointing & Optimization
**Responsibility**: Stateful training management, replay buffers, logging

**Key Files**:
- `training/checkpoint_utils.py`: Safe checkpoint save/load with verification
- `training/replay_buffer.py`: Standard experience replay
- `training/prioritized_replay_buffer.py`: Prioritized replay (importance sampling)
- `training/sequential_replay_buffer.py`: Time-ordered replay (for LSTM)
- `training/state.py`: BatchedAgentState dataclass (GPU tensor aggregation)
- `training/tensorboard_logger.py`: TensorBoard integration

**Features**:
- **Checkpoint digest**: SHA256 verification + provenance metadata
- **Resume logic**: Load checkpoint, resume from episode N
- **Replay strategies**: Uniform, prioritized by TD-error, sequential
- **Metrics logging**: Per-agent + population aggregates

**Status**: ✅ Production

---

### 12. **Demo & Inference** - Live Visualization & Server
**Responsibility**: Training orchestration, inference serving, WebSocket broadcast

**Key Files**:
- `demo/unified_server.py`: Main orchestrator (training + inference)
- `demo/runner.py` (DemoRunner class): Multi-day training with checkpointing
- `demo/live_inference.py`: Inference server (broadcasts agent states)
- `demo/database.py`: SQLite checkpoint/metadata persistence

**Architecture**:
- **Unified Server** spawns:
  1. Training loop (DemoRunner + VectorizedPopulation)
  2. Inference server (LiveInference on port 8766)
  3. Broadcasts observations/rewards every N steps
- **Frontend**: Connects via WebSocket, renders agent positions/meters/affordances

**Entry Point**: `scripts/run_demo.py` → `UnifiedServer.start()`

**Status**: ✅ Production

---

## Initial Observations

### Architectural Patterns

1. **Seven-Stage Compilation Pipeline (UAC)**
   - Separates configuration concerns from runtime
   - Enables caching (cache-bomb protection with 10MB limit)
   - Error collection allows batch diagnostics instead of fail-fast

2. **Vectorized Everything**
   - All operations batch-agnostic (1 agent or 1000)
   - GPU tensors throughout (no CPU <→ GPU transfer bottlenecks)
   - Enables efficient curriculum experiments

3. **Declarative Configuration (No-Defaults Principle)**
   - **CRITICAL**: All behavioral parameters explicitly specified in YAML
   - Pydantic DTO layer enforces all fields present (no hidden defaults)
   - Reason: Changing code defaults breaks old configs silently
   - Exemptions only: Metadata (descriptions) + computed values (observation_dim)

4. **Fixed Affordance Vocabulary**
   - All curriculum levels observe 14 affordances (even if not deployed)
   - Enables checkpoint transfer without retraining observation encoder

5. **Pre-Release Freedom (Anti-Patterns Removed)**
   - **ZERO backwards compatibility** → immediate breaking changes
   - `hasattr()` checks for old fields: DELETED
   - `try/except` for legacy config formats: DELETED
   - Version checks or feature flags: DELETED
   - All obsolete code: DELETED from `src/hamlet/` (marked legacy)

6. **Tight Pedagogical Integration**
   - "Low Energy Delirium" bug demonstrated intentionally in L0_0_minimal
   - Students learn why L0_5_dual_resource fixes it (reward design matters)
   - Unusual but effective teaching tool

### Subsystem Dependencies (Coupling Analysis)

**Tight Coupling**:
- VectorizedHamletEnv ↔ VectorizedPopulation (core training loop)
- DAC engine ↔ VectorizedHamletEnv (reward computation)
- UniverseCompiler ↔ VFS registry (observation spec generation)

**Loose Coupling**:
- Substrate implementations ↔ Everything else (pluggable via factory)
- Exploration strategies ↔ Training loop (swappable reward modifiers)
- Recording system ↔ Training (optional, no performance impact if disabled)

**Hub Pattern**:
- UniverseCompiler acts as integration hub
- Aggregates: substrate validators, VFS specs, DAC configs, action metadata

### Code Quality Metrics

- **Total Production Code**: ~26.6K lines (src/townlet/)
- **Largest files**: 
  - `universe/compiler.py`: 2542 lines (UAC pipeline)
  - `environment/vectorized_env.py`: 2544 lines (core RL loop)
  - `population/vectorized.py`: 1875 lines (training orchestration)
- **Test coverage target**: 70%+ (via pytest --cov)
- **Type hints**: Gradual (lenient mypy settings, can tighten)

### Known Limitations

1. **POMDP Support**: ❌ GridND (N≥4), Continuous substrates
   - Vision windows too large for memory
   - Only Grid2D, Grid3D (with vision_range ≤ 2), Aspatial supported

2. **Substrate Scaling**: Grid size constrained to 10K cells (100×100 max)
   - DoS protection against adversarial configs

3. **Action Space**: Max 300 actions (increased for discretized continuous, 32×7=195)
   - Handles typical curriculum levels comfortably

4. **Meter Count**: Max 100 meters (sufficient for pedagogical use)

### Interesting Architectural Decisions

1. **Aspatial Mode + Custom Frontend Component**
   - AspatialView.vue renders as meters-only dashboard (not fake grid)
   - Pedagogically correct—students learn aspatial = no positioning
   - Unusual but justified

2. **Global Action Vocabulary**
   - `configs/global_actions.yaml`: All levels share action labels
   - Enables checkpoint transfer without action encoder retraining

3. **Drive Hash Provenance**
   - SHA256(drive_as_code.yaml) embedded in checkpoints
   - Ensures reward function reproducibility across resumed training

4. **Grammar-Based YAML Validation**
   - Phase 0 of compiler validates YAML syntax before semantic analysis
   - Prevents cryptic errors from malformed YAML

## Recommended Analysis Strategy

### Sequencing Recommendation: **PARALLEL with Critical Path Focus**

**Rationale**:
- Subsystems 1, 2 are critical path (config → reward)
- Subsystems 3-7 depend on 1-2 (can run in parallel)
- Subsystems 8-12 are orthogonal (can run independently)
- No circular dependencies detected

### Suggested Parallel Analysis Groups

**Group A (Critical Infrastructure)** - Analyze First:
- Universe Compiler (UAC) - Subsystem 1
- Drive As Code Engine - Subsystem 2
- Config DTO Layer - Interconnects A

**Group B (Core Training Loop)** - Then Parallel:
- Vectorized Training Loop - Subsystem 3
- Vectorized Environment - Subsystem 4
- Variable & Feature System - Subsystem 5

**Group C (Supporting Subsystems)** - Then Parallel:
- Agent Networks & Q-Learning - Subsystem 6
- Substrate Implementations - Subsystem 7
- Exploration Strategies - Subsystem 8
- Curriculum Strategies - Subsystem 9

**Group D (Peripheral Systems)** - Finally Parallel:
- Recording & Replay - Subsystem 10
- Training Infrastructure - Subsystem 11
- Demo & Inference - Subsystem 12

### Analysis Depth Priorities

**High Priority** (Deep analysis first):
1. UniverseCompiler (2542 lines, complex state machine)
2. VectorizedHamletEnv (2544 lines, core RL loop)
3. DAC engine (40K lines, pedagogical reward logic)

**Medium Priority** (Moderate analysis depth):
4. VectorizedPopulation (curriculum integration)
5. Substrate implementations (pluggable architecture pattern)
6. Agent networks (algorithm selection + LSTM design)

**Lower Priority** (Surface analysis sufficient):
7. Recording/replay system (orthogonal to training)
8. Demo server (orchestration wrapper)
9. Exploration strategies (modular reward add-ons)

### Data Points for Analysis

**Critical Dependency Chain**:
```
YAML Config Packs
  ↓
UniverseCompiler (7-stage pipeline)
  ├→ VFS Registry + Observation Specs
  ├→ DAC Engine (reward compilation)
  ├→ Substrate Validators
  └→ CompiledUniverse (artifact)
       ↓
VectorizedHamletEnv (accepts artifact)
  ├→ Action builder
  ├→ Affordance engine
  ├→ Cascade engine
  └→ DAC runtime execution
       ↓
VectorizedPopulation (training orchestration)
  ├→ Agent networks
  ├→ Exploration strategies
  ├→ Curriculum strategies
  └→ Checkpoint management
```

**Orthogonal Systems**:
- Recording/replay: Can be toggled independently
- Frontend visualization: Consumes WebSocket stream, doesn't affect training
- CLI tools: Diagnostic only

### Deliverables per Group

**Group A Analysis Output**:
1. UAC compilation pipeline flow diagram
2. DAC strategy taxonomy + examples
3. Config DTOs validation rules
4. Error collection/reporting mechanism

**Group B Analysis Output**:
1. Training loop state machine
2. VFS access control enforcement
3. Observation dimension math (why 29/54 dims)
4. Vectorization strategy (batch ops on GPU)

**Group C Analysis Output**:
1. Network architecture decision tree
2. Substrate interface + implementations catalog
3. Exploration reward formula + annealing logic
4. Curriculum difficulty adjustment algorithm

**Group D Analysis Output**:
1. Recording format + compression details
2. Checkpoint persistence + recovery
3. Server architecture + WebSocket protocol
4. Frontend-backend communication spec

---

## Next Steps

1. **Immediate**: Use this discovery as map for parallel deep-dive analysis
2. **Group A First**: Understand config compilation pipeline (foundation for all other analysis)
3. **Document as You Go**: Maintain `02-subsystem-details.md`, `03-dataflow-diagrams.md`, etc.
4. **Cross-Reference**: Link analysis outputs to source code files (absolute paths)
5. **Decision Points**: Flag architectural trade-offs, anti-patterns, potential refactoring opportunities

---

**Total Scan Time**: ~20 minutes  
**Analysis Thoroughness**: Medium (enough to identify subsystem boundaries; ready for deep dives)  
**Next Phase**: Parallel deep analysis of Groups A-D per recommended sequencing
