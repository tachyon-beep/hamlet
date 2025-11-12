# Discovery Findings - Townlet Architecture
**Date:** 2025-11-12
**Archaeologist:** Claude Code (System Archaeologist)
**Scope:** `src/townlet/` + Configuration System

---

## Executive Summary

**Townlet** is a GPU-native, vectorized Deep Reinforcement Learning (DRL) training system for pedagogical research. The codebase implements a complete DRL pipeline from configuration compilation through training to inference, with a focus on curriculum learning and intrinsic exploration.

**Scale:** ~23,600 LOC (Python 3.13)
**Complexity:** High (multi-stage compiler, GPU-native vectorization, configurable substrates)
**Architecture:** Layered with clear separation of concerns
**Coupling:** Moderate - subsystems communicate through well-defined DTOs and interfaces

---

## 1. Directory Structure Analysis

### Organization Pattern: **Domain-Driven + Layered Architecture**

The codebase follows a **hybrid architecture**:
- **Domain-driven modules** (agent, environment, population, exploration)
- **Layered compilation pipeline** (config → compiler → runtime)
- **Infrastructure modules** (vfs, substrate, universe)

```
src/townlet/
├── config/          # [LAYER] Configuration DTOs (data contracts)
├── universe/        # [LAYER] Compiler infrastructure (7-stage pipeline)
├── vfs/             # [LAYER] Variable & Feature System (state space config)
├── substrate/       # [INFRA] Spatial substrate implementations
├── environment/     # [DOMAIN] Vectorized environment + dynamics
├── agent/           # [DOMAIN] Neural network architectures
├── population/      # [DOMAIN] Population management + training
├── exploration/     # [DOMAIN] Exploration strategies (RND, ε-greedy)
├── training/        # [DOMAIN] Training loop + replay buffer
├── curriculum/      # [DOMAIN] Curriculum strategies
├── demo/            # [INTERFACE] Inference server + demo runner
└── recording/       # [INTERFACE] Episode recording + replay
```

**Key Observations:**
- **Clean layering:** Config → Compiler → Runtime → Training
- **Strong encapsulation:** Each module has clear responsibilities
- **Plugin architecture:** Substrate types are interchangeable
- **DTO-driven communication:** Heavy use of Pydantic models for contracts

---

## 2. Entry Points

### Primary CLI Entry Points

1. **Universe Compiler CLI** (`python -m townlet.compiler`)
   - Location: `src/townlet/compiler/__main__.py`
   - Commands: `compile`, `inspect`, `validate`
   - Purpose: Compile YAML config packs → MessagePack artifacts
   - Used by: CI/CD pipelines, development validation

2. **Recording System CLI** (`python -m townlet.recording`)
   - Location: `src/townlet/recording/__main__.py`
   - Purpose: Episode recording/replay management
   - Used by: Video export, debugging

### Secondary Entry Points

3. **Demo Runner** (`scripts/run_demo.py`)
   - Location: `src/townlet/demo/runner.py` (DemoRunner context manager)
   - Purpose: Training orchestration + checkpointing
   - Coordinates: Environment, Population, Curriculum, Exploration

4. **Live Inference Server** (`python -m townlet.demo.live_inference`)
   - Location: `src/townlet/demo/live_inference.py`
   - Purpose: WebSocket server for frontend visualization
   - Protocol: JSON messages over WS (localhost:8766)

### Configuration Entry Points

5. **Config Packs** (YAML → Compiled Universe)
   - Location: `configs/L*_*/` directories
   - Required files: `substrate.yaml`, `bars.yaml`, `cascades.yaml`, `affordances.yaml`, `training.yaml`, `variables_reference.yaml`
   - Compilation: UAC 7-stage pipeline → `.compiled/universe.msgpack`

---

## 3. Technology Stack

### Core Technologies

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Language** | Python 3.13 | Primary implementation language |
| **Deep Learning** | PyTorch 2.9+ | Neural networks, GPU acceleration |
| **Validation** | Pydantic 2.0+ | Configuration validation, DTOs |
| **Serialization** | MessagePack, YAML | Config persistence, artifact caching |
| **Compression** | LZ4 | Episode recording compression |
| **Logging** | TensorBoard (TensorFlow) | Training metrics visualization |

### Development Tools

- **Testing:** pytest + hypothesis (property-based testing)
- **Type Checking:** mypy (strict mode capable)
- **Formatting:** black (line-length=140)
- **Linting:** ruff (modern Python linter)

### Framework Dependencies

- **Gymnasium 1.0+** - RL environment interface (PettingZoo compatible)
- **FastAPI + uvicorn** - REST API for demo server
- **WebSockets 11+** - Live inference streaming
- **Rich** - CLI output formatting
- **MLflow** - Experiment tracking (optional)

---

## 4. Subsystem Identification

### Primary Subsystems (13 identified)

| Subsystem | Location | Responsibility | Complexity |
|-----------|----------|----------------|------------|
| **1. Config DTOs** | `config/` | Data contracts for all configurations | Medium |
| **2. Universe Compiler** | `universe/` | 7-stage compilation pipeline (parse → emit) | High |
| **3. VFS** | `vfs/` | Variable & Feature System (state space) | Medium |
| **4. Substrates** | `substrate/` | Spatial representations (Grid2D, Grid3D, GridND, Continuous, Aspatial) | High |
| **5. Environment** | `environment/` | Vectorized environment + dynamics engines | High |
| **6. Agent Networks** | `agent/` | Q-network architectures (MLP, LSTM) | Medium |
| **7. Population** | `population/` | Batched training + gradient updates | High |
| **8. Exploration** | `exploration/` | RND, ε-greedy, adaptive intrinsic | Medium |
| **9. Training** | `training/` | Training loop, replay buffer, checkpointing | Medium |
| **10. Curriculum** | `curriculum/` | Adversarial + static curriculum strategies | Low |
| **11. Demo System** | `demo/` | Inference server + demo runner | Medium |
| **12. Recording** | `recording/` | Episode recording + video export | Low |
| **13. Compiler Adapters** | `universe/adapters/` | VFS integration, DTO transformations | Low |

### Subsystem Dependencies (High-Level)

```
Config DTOs ──→ Universe Compiler ──→ CompiledUniverse
                        ↓
                  VFS Adapter
                        ↓
                Variable Registry ──→ Observation Builder
                                              ↓
Substrates ──────────────────────→ Environment ←──── Affordance Engine
                                        ↓
                                   Population
                                   ↙       ↘
                          Agent Networks  Exploration
                                   ↘       ↙
                                   Training
                                        ↓
                               Curriculum Strategies
                                        ↓
                                  Checkpointing
                                        ↓
                               Demo/Recording
```

---

## 5. Architectural Patterns

### Pattern 1: **Configuration as Code (Universe As Code - UAC)**

**Implementation:** 7-stage compiler pipeline
- **Stage 0:** YAML syntax validation
- **Stage 1:** Parse + Symbol Table construction
- **Stage 2:** Cross-reference resolution
- **Stage 3:** Semantic validation
- **Stage 4:** Metadata enrichment
- **Stage 5:** Optimization (dead code elimination, constant folding)
- **Stage 6:** Emit + Cache (MessagePack serialization)

**Benefits:**
- Declarative configuration (no Python required)
- Compile-time validation (fail fast)
- Artifact caching (instant load from cache)
- Reproducibility (config hash tracking)

### Pattern 2: **DTO-Driven Communication**

**Implementation:** Pydantic BaseModel for all data contracts
- All subsystems communicate through immutable DTOs
- Strict validation at boundaries (no defaults principle)
- Type safety enforced at runtime + compile time

**Example DTOs:**
- `UniverseMetadata`, `ObservationSpec`, `ActionSpaceMetadata`
- `MeterMetadata`, `AffordanceMetadata`
- `BarConfig`, `AffordanceConfig`, `SubstrateConfig`

### Pattern 3: **GPU-Native Vectorization**

**Implementation:** All state stored as PyTorch tensors
- Batched operations: `[num_agents, ...]` tensors
- Device-agnostic code (`device` parameter throughout)
- Vectorized affordance interactions (no Python loops)
- Parallel environment steps (single GPU kernel)

**Performance Impact:**
- ~100x faster than Python loops
- Scales to thousands of agents

### Pattern 4: **Strategy Pattern for Exploration**

**Implementation:** Pluggable exploration strategies
- Base interface: `ExplorationStrategy`
- Concrete strategies: `EpsilonGreedy`, `RND`, `AdaptiveIntrinsicExploration`
- Runtime selection via config

### Pattern 5: **Factory Pattern for Substrates**

**Implementation:** `SubstrateFactory` creates substrate instances
- Dynamic substrate selection via `type` field
- Polymorphic substrate interface
- Supports 7 substrate types (Grid2D, Grid3D, GridND, Continuous, ContinuousND, Aspatial, + future types)

### Pattern 6: **Observer Pattern for Logging**

**Implementation:** TensorBoard logger observes training events
- Decoupled from training loop
- Async metric emission
- Extensible to MLflow, W&B

### Pattern 7: **No-Defaults Principle**

**Implementation:** All behavioral parameters explicit in config
- DTOs reject missing required fields
- No hidden defaults in code
- Reproducibility enforced at schema level

**Rationale:** Hidden defaults create non-reproducible configs

---

## 6. Data Flow Overview

### Primary Data Flows

#### Flow 1: **Configuration → Compiled Universe**

```
YAML Configs (bars, affordances, substrate, training)
    ↓
UniverseCompiler.compile()
    ↓
Phase 0: YAML syntax validation
Phase 1: Parse + build symbol table
Phase 2: Resolve cross-references
Phase 3: Validate semantics
Phase 4: Enrich metadata
Phase 5: Optimize
Phase 6: Emit MessagePack artifact
    ↓
CompiledUniverse (cached in .compiled/universe.msgpack)
```

#### Flow 2: **Training Episode Lifecycle**

```
DemoRunner.run()
    ↓
Initialize: Environment, Population, Curriculum, Exploration
    ↓
FOR each episode:
    1. Environment.reset() → initial observations [num_agents, obs_dim]
    2. FOR each step:
        a. Exploration.select_action(obs, q_values) → actions [num_agents]
        b. Environment.step(actions) → (next_obs, rewards, dones, info)
        c. Population.observe(transition) → store in replay buffer
        d. IF train_frequency:
            Population.train_step() → batch sample + gradient update
    3. Curriculum.should_adjust_difficulty() → adjust if needed
    4. Population.maybe_update_target_network()
    ↓
Checkpointing.save() → weights + metadata
```

#### Flow 3: **Observation Construction (VFS Pipeline)**

```
variables_reference.yaml
    ↓
VFSObservationSpecBuilder.build()
    ↓
ObservationSpec (fields, normalization, access control)
    ↓
VariableRegistry (runtime storage)
    ↓
Environment.get_observations() → [num_agents, obs_dim] tensor
    ↓
Agent Q-Network forward pass
```

#### Flow 4: **Affordance Interaction**

```
Agent executes INTERACT action at position (x, y)
    ↓
AffordanceEngine.execute_affordance(agent_idx, affordance_id)
    ↓
1. Check interaction_type (instant, duration, delayed)
2. Apply costs (energy, money, etc.)
3. IF duration: track active_interactions state
4. Apply effects (meter changes)
5. Update Variable Registry
    ↓
MeterDynamics.apply_cascades() → propagate meter changes
    ↓
RewardStrategy.compute_reward() → scalar reward
```

---

## 7. Configuration System Architecture

### Config Pack Structure

Each curriculum level is a directory containing 7 YAML files:

| File | Purpose | Validation |
|------|---------|-----------|
| `substrate.yaml` | Grid size, topology, boundaries | `SubstrateConfig` DTO |
| `bars.yaml` | Meter definitions (energy, health, etc.) | `BarConfig` DTO |
| `cascades.yaml` | Meter relationships (e.g., low energy → health decay) | `CascadeConfig` DTO |
| `affordances.yaml` | Interaction definitions (fridge, bed, shower, etc.) | `AffordanceConfig` DTO |
| `training.yaml` | Hyperparameters (lr, gamma, epsilon, etc.) | `TrainingConfig` DTO |
| `cues.yaml` | UI metadata (colors, icons, labels) | `CuesConfig` DTO |
| `variables_reference.yaml` | VFS configuration (state space) | `VariableDef` schema |

### Reference Config: L1_full_observability

**Characteristics:**
- **Grid:** 8×8 square grid, Manhattan distance, clamp boundaries
- **Affordances:** 14 deployed (all available)
- **Network:** SimpleQNetwork (MLP, ~26K params)
- **Observability:** Full (no POMDP)
- **Observation Dim:** 29 (2 coords + 8 meters + 15 affordances + 4 temporal)
- **Action Space:** 8 actions (6 substrate movements + INTERACT + WAIT)
- **Curriculum:** Adversarial (5-stage difficulty progression)
- **Exploration:** RND + adaptive intrinsic annealing
- **Expected Performance:** 1000-2000 episodes to learn, 250-350 step survival

---

## 8. Key Architectural Insights

### Insight 1: **Transfer Learning by Design**

All Grid2D configs share identical observation dimensions (29) and action space (8), enabling:
- Checkpoint reuse across grid sizes
- Curriculum progression without architecture changes
- Zero-shot transfer to new environments

**Implementation:** Fixed affordance vocabulary (14 types), constant observation encoding

### Insight 2: **Compile-Time vs Runtime Separation**

**Compile-time (UAC):**
- Config validation
- Symbol resolution
- Optimization
- Artifact generation

**Runtime:**
- Fast artifact loading (MessagePack)
- GPU tensor operations
- Training loop execution

**Benefit:** Development iteration speed (compile once, train many times)

### Insight 3: **Pedagogical Design Philosophy**

The system is explicitly designed to produce "interesting failures" as teaching moments:
- Reward hacking (e.g., "Low Energy Delirium" bug preserved as lesson)
- Curriculum levels (L0 → L6) for progressive difficulty
- No proximity shaping (agents must explore to learn)

**Target Audience:** Graduate RL students learning through experimentation

### Insight 4: **GPU-Native from Ground Up**

Not a retrofitted GPU implementation:
- All state is PyTorch tensors (no NumPy conversions)
- Vectorized affordance interactions (batch operations)
- Device-agnostic code (CPU/CUDA/MPS)
- Parallel environment steps

### Insight 5: **Pre-Release Breaking Changes**

Codebase enforces **zero backwards compatibility**:
- Breaking changes are encouraged (no users to support)
- Old code paths deleted immediately
- No fallback mechanisms or migration paths
- Clean breaks = simpler codebase at launch

**Evidence:** VFS integration required breaking all old configs, obsolete code deleted

---

## 9. Technology Choices Rationale

| Choice | Rationale |
|--------|-----------|
| **Python 3.13** | Latest features (structural pattern matching, type improvements) |
| **PyTorch** | Superior GPU support, dynamic computation graphs |
| **Pydantic 2.0** | Runtime validation, excellent error messages, strict mode |
| **MessagePack** | Fast binary serialization, smaller than JSON |
| **YAML** | Human-readable configs, comments supported |
| **LZ4** | Fast compression for episode recording |
| **TensorBoard** | Standard RL logging tool, GPU-accelerated |

---

## 10. Identified Risks & Technical Debt

### Risk 1: **Complexity Scaling**

- 13 subsystems with moderate coupling
- Adding new substrate types requires changes across 5+ modules
- **Mitigation:** Strong DTO contracts, comprehensive tests

### Risk 2: **Cache Invalidation**

- MessagePack cache must invalidate on config changes
- Cache hash collision could cause subtle bugs
- **Mitigation:** Config hash includes all YAML content, size limits enforced

### Risk 3: **GPU Memory Management**

- Large batches (num_agents > 1000) could OOM
- **Mitigation:** Configurable batch sizes, gradient accumulation

### Technical Debt Observations

1. **Network architecture hardcoded** - Future: BRAIN_AS_CODE (network config YAML)
2. **Limited substrate types** - No graph substrates, no continuous action spaces
3. **Single-agent focus** - Multi-agent support planned (L5, L6) but not implemented

---

## 11. Recommendations for Further Analysis

### High Priority

1. **Universe Compiler Deep-Dive** - Analyze all 7 stages in detail
2. **VFS Integration Analysis** - How variables flow from config → observations
3. **Environment Vectorization** - GPU optimization patterns
4. **Training Loop** - Replay buffer management, gradient updates

### Medium Priority

5. **Substrate Implementations** - Compare Grid2D vs Grid3D vs Aspatial
6. **Exploration Strategies** - RND implementation, annealing logic
7. **Curriculum Strategies** - Adversarial difficulty adjustment

### Low Priority

8. **Recording System** - Video export pipeline
9. **Demo Server** - WebSocket protocol
10. **Cues Compiler** - UI metadata processing

---

## 12. Next Steps

**For Parallel Analysis:**

1. Create 13 subagent task specifications (one per subsystem)
2. Spawn parallel subagents to produce subsystem catalog entries
3. Each subagent reads this discovery document + explores assigned subsystem
4. Validation gate after subsystem catalog completion

**Expected Output:**

- `02-subsystem-catalog.md` with detailed component inventories
- Interfaces, dependencies, design patterns per subsystem
- Confidence levels marked for each entry

---

## Appendix A: File Statistics

```
Total Python LOC: ~23,600
Total Modules: 91 Python files
Total Subsystems: 13 major, 3 minor
Total Config DTOs: ~15 Pydantic models
Total Entry Points: 4 (compiler CLI, recording CLI, demo runner, live inference)
```

## Appendix B: Key Terminology

- **UAC** - Universe As Code (configuration compilation system)
- **VFS** - Variable & Feature System (state space configuration)
- **DTO** - Data Transfer Object (Pydantic models)
- **POMDP** - Partially Observable Markov Decision Process
- **RND** - Random Network Distillation (intrinsic exploration)
- **CompiledUniverse** - Cached MessagePack artifact from UAC
- **Substrate** - Spatial representation (Grid2D, Aspatial, etc.)
- **Affordance** - Interaction point (fridge, bed, shower, job, etc.)
- **Meter** - Resource bar (energy, health, hygiene, etc.)
- **Cascade** - Meter relationship (e.g., low energy → health decay)
