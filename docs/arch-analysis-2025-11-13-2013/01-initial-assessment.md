# HAMLET Architecture: Initial Assessment
**Phase 2 - System Archeologist Workflow**  
**Date:** 2025-11-13  
**Codebase Size:** 27,138 lines of Python (src/townlet/); 644+ passing tests (70% coverage)  
**Status:** Pre-Release (Zero Backwards Compatibility)

---

## 1. Directory Structure Map

### Top-Level Organization

```
/home/user/hamlet/
├── src/                          # Python source code (ACTIVE)
│   └── townlet/                  # Main implementation (15 subsystems)
├── configs/                      # Configuration packs (YAML)
├── frontend/                     # Vue 3 visualization
├── tests/                        # Test suite (unit/integration/slow)
├── docs/                         # Architecture & guides
├── scripts/                      # Entry point scripts
└── [build/CI/metadata]           # pyproject.toml, .github/, etc.
```

### Active vs Obsolete Status

| Directory | Status | Purpose | Lines |
|-----------|--------|---------|-------|
| `src/townlet/` | **ACTIVE** | Main DRL environment | 27,138 |
| `src/hamlet/` | OBSOLETE | Legacy code (marked for deletion per CLAUDE.md) | 0 (removed) |
| `configs/` | **ACTIVE** | 9 curriculum levels + experiments | YAML-based |
| `frontend/` | **ACTIVE** | Vue 3 + Vite live visualization | ~3,000 lines JS |
| `tests/` | **ACTIVE** | 644+ tests (unit/integration/slow) | ~8,500 lines |
| `docs/` | **ACTIVE** | Guides, architecture, decisions | 50+ markdown files |

### src/townlet/ Subsystem Structure

```
src/townlet/                        Total: 27,138 LOC
├── agent/                          1,389 LOC (Neural networks)
│   ├── networks.py                 SimpleQNetwork, RecurrentSpatialQNetwork
│   └── [optimizer configs]
├── environment/                    4,940 LOC (Vectorized execution)
│   ├── vectorized_env.py           GPU-native environment
│   ├── dac_engine.py               Drive As Code runtime
│   ├── cascade_engine.py           Meter relationship execution
│   ├── affordance_engine.py        Interaction system
│   └── [action, substrate support]
├── population/                     1,358 LOC (Batched training)
│   ├── vectorized.py               VectorizedPopulation (main trainer)
│   └── [replay buffer, state management]
├── universe/                       4,349 LOC (UAC - Seven-stage compiler)
│   ├── compiler.py                 Main pipeline
│   ├── symbol_table.py             Resolution
│   ├── dto/                        Compiled spec objects
│   ├── adapters/                   YAML → DTO translation
│   └── [cues_compiler, optimization, runtime]
├── vfs/                            ~1,200 LOC (Variable & Feature System)
│   ├── schema.py                   Pydantic models (VariableDef, ObservationField)
│   ├── registry.py                 Runtime storage + access control
│   └── observation_builder.py      Compile-time spec generation
├── training/                       ~900 LOC (State/replay management)
├── curriculum/                     ~600 LOC (Static/adversarial strategies)
├── exploration/                    ~1,500 LOC (RND, epsilon-greedy, ICM)
├── config/                         ~3,000 LOC (Pydantic config DTOs)
├── substrate/                      ~1,200 LOC (Grid2D, Grid3D, Aspatial, etc.)
├── demo/                           ~1,100 LOC (Entry points)
├── compiler/                       ~200 LOC (CLI wrapper)
├── recording/                      ~800 LOC (Episode capture/replay)
└── [package metadata]
```

### Configuration Pack Structure

Each curriculum level (L0, L0.5, L1, L2, L3) is a directory with **9 required YAML files**:

```
configs/L0_0_minimal/
├── substrate.yaml              # Spatial substrate (Grid2D, size, topology)
├── bars.yaml                   # Meters (energy, health, satiation, etc.)
├── cascades.yaml               # Meter relationships (e.g., low energy → fatigue)
├── affordances.yaml            # Interaction definitions (REST, EAT, SOCIALIZE)
├── cues.yaml                   # UI metadata (colors, icons, descriptions)
├── training.yaml               # Hyperparameters (lr, batch_size, episodes)
├── drive_as_code.yaml          # Declarative reward configuration (REQUIRED)
├── variables_reference.yaml    # VFS variable definitions (REQUIRED)
└── brain.yaml                  # Network architecture config (optional override)
```

**Active Config Packs:**
- `L0_0_minimal` - Temporal credit assignment (3×3 grid, 1 affordance)
- `L0_5_dual_resource` - Multiple resources (7×7 grid, 4 affordances)
- `L1_full_observability` - Full observability baseline (8×8 grid, 14 affordances)
- `L2_partial_observability` - POMDP with LSTM (8×8 grid, 5×5 vision window)
- `L3_temporal_mechanics` - Time-based dynamics (24-tick day/night cycle)
- `L1_3D_house` - 3D grid variant
- `L1_continuous_*` - Continuous space variants (1D/2D/3D)
- `aspatial_test` - Pure resource management (no position)
- Experiments directory with A/B test configurations

### Frontend Structure

```
frontend/
├── src/
│   ├── components/              Vue 3 components
│   │   ├── Grid.vue             2D grid rendering (SVG)
│   │   ├── AspatialView.vue     Meter-only dashboard
│   │   ├── MeterBar.vue         Individual meter visualization
│   │   └── [action log, controls, etc.]
│   ├── stores/                  State management (Pinia)
│   ├── styles/                  CSS tokens (colors, dimensions)
│   └── utils/                   Constants, WebSocket client
├── vite.config.js               Build configuration (Vite)
├── index.html                   SPA entry point
└── demo.html                    Static demo HTML (fallback)
```

### Tests Organization (644+ Tests, 70% Coverage)

```
tests/test_townlet/
├── unit/                        Fast, isolated tests
│   ├── agent/                   Network tests
│   ├── environment/             Vectorized ops, POMDP validation
│   ├── universe/                Compiler stages, symbol resolution
│   ├── vfs/                     Variable system, access control
│   ├── population/              Batched training, replay buffer
│   ├── exploration/             RND novelty, annealing
│   ├── curriculum/              Strategy logic
│   └── [substrate, config, demo, training, recording]
├── integration/                 Multi-component tests
│   ├── vfs/                     End-to-end variable system
│   └── config/                  Config loading + validation
├── slow/                        Long-running tests (nightly)
│   └── [full training runs, checkpointing]
├── fixtures/                    Test data (config packs)
├── conftest.py                  Pytest configuration
└── [helpers, utils, properties]
```

### Docs Organization (50+ Markdown Files)

```
docs/
├── architecture/                 Design documents
│   ├── COMPILER_ARCHITECTURE.md  UAC seven-stage pipeline
│   ├── [vfs-design, dac-design, etc.]
├── config-schemas/               Configuration documentation
│   ├── substrate.md              Substrate types & options
│   ├── drive_as_code.md          DAC reward specification
│   ├── variables.md              VFS variable definitions
│   └── [bars, affordances, training, etc.]
├── guides/                       How-to documentation
│   ├── dac-migration.md          Reward function migration
│   ├── vfs-integration-guide.md  Variable system integration
│   └── [no-defaults-lint guide, etc.]
├── plans/                        Implementation plans
│   ├── 2025-11-12-dac-implementation.md
│   ├── 2025-11-12-dac-runtime-integration.md
│   └── [other planning docs]
└── [testing/, decisions/, diagrams/, research/, etc.]
```

---

## 2. Entry Points: How the System Runs

### Training Entry Point

**Primary Command:**
```bash
python scripts/run_demo.py --config configs/L1_full_observability --episodes 10000
```

**Script Path:** `/home/user/hamlet/scripts/run_demo.py`  
**Handler:** `townlet.demo.unified_server.UnifiedServer`  
**Functionality:**
1. Parses YAML config pack
2. Launches `UnifiedServer` which:
   - Starts training loop in background thread (GPU)
   - Saves checkpoints every 100 episodes to `runs/LX_timestamp/checkpoints/`
   - Monitors checkpoint directory for new weights
3. Starts WebSocket inference server on port 8766 (configurable)
4. Waits for frontend connection (separate process)

**Key Arguments:**
- `--config` (required): Path to config directory or training.yaml
- `--episodes`: Override max_episodes from config
- `--checkpoint-dir`: Custom checkpoint location (auto-generated if not provided)
- `--inference-port`: WebSocket server port (default: 8766)
- `--debug`: Enable debug logging

### Inference Entry Point

**Direct Command:**
```bash
python -m townlet.demo.live_inference \
    <checkpoint_dir> 8766 0.2 10000 <config_path>
```

**Handler:** `townlet.demo.live_inference.LiveInference`  
**Functionality:**
- Loads checkpoint weights
- Runs episodes on demand (fast, CPU-capable inference)
- Broadcasts state via WebSocket to frontend
- Supports multiple episodes in sequence

**Arguments:** checkpoint_dir, port, speed (0.2 = 5x slower), total_episodes, config_path

### Frontend Development Server

**Command:**
```bash
cd frontend && npm run dev
```

**Configuration:** `frontend/vite.config.js` (Vite)  
**Port:** 5173 (default)  
**Architecture:** Vue 3 + Pinia (state management)  
**WebSocket:** Connects to localhost:8766 (live inference server)  
**Rendering Modes:**
- **Spatial** (Grid2D/3D): SVG-based `Grid.vue` with heat maps, trails
- **Aspatial**: Meter-only dashboard `AspatialView.vue`

### Compiler CLI Entry Point

**Command:**
```bash
python -m townlet.compiler {compile,inspect,validate} <config_dir>
```

**Handler:** `src/townlet/compiler/__main__.py`  
**Functionality:**
- **compile**: Load config pack → run UAC pipeline → emit compiled specs
- **inspect**: Print symbol table, metadata, optimization details
- **validate**: Check for errors, cross-references, missing affordances

**Integration:** Wired into GitHub Actions (`.github/workflows/config-validation.yml`)

### Test Runner

**Command:**
```bash
uv run pytest tests/
```

**Configuration:** `pyproject.toml` [tool.pytest.ini_options]  
**Default Behavior:**
- Runs all tests except `@pytest.mark.slow`
- Coverage report (target: 70%+)
- Markers: `slow`, `gpu`, `integration`, `e2e`

**Full Suite (Nightly):**
```bash
uv run pytest tests/ -m "slow or not slow"
```

### Linting & Code Quality

**Black (Formatting):**
```bash
uv run black src/townlet tests/
```

**Ruff (Linting):**
```bash
uv run ruff check src/townlet
```

**Mypy (Type Checking):**
```bash
uv run mypy src/townlet
```

**No-Defaults Lint (Custom):**
```bash
python scripts/no_defaults_lint.py --check src/townlet
```
(Enforces CLAUDE.md "No-Defaults Principle" - all config fields must be explicit)

---

## 3. Technology Stack

### Backend (Python)

| Layer | Technology | Version | Purpose |
|-------|-----------|---------|---------|
| **Core DRL** | PyTorch | ≥2.9.0 | GPU-native tensor operations, Q-network training |
| **Environment** | Gymnasium | ≥1.0.0 | Standard RL environment interface |
| **Data** | NumPy | ≥1.24 | Vectorized array operations |
| **Config** | Pydantic | ≥2.0.0 | Type-safe configuration DTOs |
| **Serialization** | PyYAML | ≥6.0.0 | Config file parsing |
| **Inference API** | FastAPI | ≥0.100.0 | REST + WebSocket server |
| **ASGI Server** | Uvicorn | ≥0.23.0 | Async HTTP/WebSocket handler |
| **Monitoring** | MLflow | ≥2.9.0 | Experiment tracking |
| **Data Pipes** | pandas | ≥2.0.0 | Episode analysis, statistics |
| **ML Utility** | scikit-learn | ≥1.3.0 | Metrics, validation |
| **Recording** | ffmpeg-python | ≥0.2.0 | MP4 export (optional) |
| **Version Control** | GitPython | ≥3.1.0 | Checkpoint versioning |

**Runtime:**
- Python 3.13+ (strict)
- CUDA-compatible GPU (recommended, CPU fallback supported)
- TensorFlow/CUDA optional (listed but may be unused legacy)

### Frontend (JavaScript/Vue 3)

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Framework** | Vue 3 | Component-based reactive UI |
| **Build Tool** | Vite | Fast dev server, production bundling |
| **State Management** | Pinia | Centralized store (episode data, metrics) |
| **Styling** | CSS 3 | Grid rendering, responsive design |
| **Rendering** | SVG (Grid.vue) | 2D spatial visualization |
| **Networking** | WebSocket (native) | Real-time state push from server |

### Configuration Format

**Format:** YAML (human-readable, versionable)  
**Validation:** Pydantic DTOs (`src/townlet/config/*.py`)  
**Compilation:** Universe Compiler (UAC) seven-stage pipeline

### Testing Infrastructure

| Tool | Purpose | Integration |
|------|---------|-----------|
| pytest 7.4+ | Test runner | `pyproject.toml`, GitHub Actions |
| pytest-cov 4.1+ | Coverage tracking | Target 70%+ |
| pytest-asyncio | Async test support | VFS registry, server tests |
| hypothesis | Property-based testing | Exploration strategies |
| vulture 2.0+ | Dead code detection | CI linting |

### Build & Packaging

| Tool | Purpose |
|------|---------|
| UV | Python package manager (faster than pip) |
| Hatchling | Build backend (PEP 517) |
| GitHub Actions | CI/CD pipeline |
| Pre-commit | Local hook framework |

---

## 4. Candidate Subsystems (10 Major + Utilities)

### Core Subsystems (10)

#### 1. **Vectorized Environment** (`src/townlet/environment/`)
   - **LOC:** 4,940
   - **Core Classes:** `VectorizedHamletEnv` (GPU-native batched environment)
   - **Purpose:** Execute multiple episodes in parallel on GPU tensors
   - **Key Features:**
     - Batched observation generation (all agents simultaneously)
     - Meter state management (energy, health, satiation, mood, fitness, health, social, money)
     - Affordance interaction resolution
     - Cascade engine (meter relationships)
     - DAC engine (declarative reward computation)
   - **Key Files:**
     - `vectorized_env.py` - Main environment loop
     - `dac_engine.py` - Drive As Code reward engine
     - `cascade_engine.py` - Meter cascades (e.g., low energy → fatigue)
     - `affordance_engine.py` - Interaction handler
     - `action_config.py`, `action_builder.py`, `action_labels.py` - Action space
   - **Integration Points:** Receives actions from population, produces observations/rewards

#### 2. **Agent (Neural Networks)** (`src/townlet/agent/`)
   - **LOC:** 1,389
   - **Core Classes:** `SimpleQNetwork`, `RecurrentSpatialQNetwork`
   - **Purpose:** Q-value estimation networks for action selection
   - **Architectures:**
     - **SimpleQNetwork:** MLP (full observability) - 29→256→128→action_dim
     - **RecurrentSpatialQNetwork:** Vision encoder + LSTM (POMDP) - ~650K params
   - **Training Integration:** Backprop via DQN/DoubleDQN loss
   - **Checkpoint Compatibility:** Network weights transferable across curriculum levels

#### 3. **Population (Batched Training)** (`src/townlet/population/`)
   - **LOC:** 1,358
   - **Core Classes:** `VectorizedPopulation`
   - **Purpose:** Coordinate batched training across multiple episodes
   - **Responsibilities:**
     - Collect experiences from vectorized environment
     - Manage replay buffer (prioritized or uniform)
     - Update Q-network via DQN/DoubleDQN
     - Handle epsilon-greedy action selection
     - Checkpoint save/load
   - **Integration:** Links environment ↔ agent ↔ curriculum

#### 4. **Universe Compiler (UAC)** (`src/townlet/universe/`)
   - **LOC:** 4,349
   - **Purpose:** Seven-stage declarative pipeline: YAML → Compiled Specs
   - **Seven Stages:**
     1. **Parse** - YAML → Python objects
     2. **Symbol Table** - Build resolution context
     3. **Resolve** - Cross-reference affordances, cascades, variables
     4. **Cross-Validate** - Check constraints, dependencies
     5. **Metadata** - Compute observation dims, action space size
     6. **Optimize** - Simplify expressions, pre-compute constants
     7. **Emit/Cache** - Output compiled specs, store in `_compiler_cache.pkl`
   - **Key Files:**
     - `compiler.py` - Main orchestrator
     - `symbol_table.py` - Reference resolution
     - `dto/` - Compiled spec Pydantic models
     - `adapters/` - YAML → DTO translation
     - `cues_compiler.py` - UI metadata compilation
   - **Output:** `CompiledSubstrate`, `CompiledUniverseSpec` (immutable, hashable)
   - **Validation:** Wired into CI (`.github/workflows/config-validation.yml`)

#### 5. **Variable & Feature System (VFS)** (`src/townlet/vfs/`)
   - **LOC:** ~1,200
   - **Purpose:** Declarative state space configuration + access control
   - **Components:**
     - `schema.py` - Pydantic models (VariableDef, ObservationField, NormalizationSpec)
     - `registry.py` - Runtime GPU tensor storage + access enforcement
     - `observation_builder.py` - Compile-time spec generation
   - **Features:**
     - Variable scopes: `global`, `agent`, `agent_private`
     - Access control: Readers (agent, engine, acs, bac), Writers (engine, actions, bac)
     - Normalization specs: Linear, exponential, categorical
     - Write specs: Accumulation, replacement, delta
   - **Integration:** Integrated into config loading (TASK-002C complete)
   - **Requirements:** All config packs **must** include `variables_reference.yaml`

#### 6. **Drive As Code (DAC)** (`src/townlet/environment/dac_engine.py` + `config/drive_as_code.py`)
   - **LOC:** ~1,500 (spread across environment + config)
   - **Purpose:** Declarative, GPU-native reward computation system
   - **Features:**
     - **Extrinsic strategies** (9 types): multiplicative, additive, threshold, polynomial, etc.
     - **Intrinsic strategies** (5 types): RND, ICM, count-based, adaptive RND, none
     - **Shaping bonuses** (11 types): approach, completion, efficiency, balance, crisis, etc.
     - **Modifiers**: Context-sensitive multipliers (e.g., energy crisis suppression)
   - **Architecture:** YAML config → UAC compilation → DACEngine execution
   - **Provenance:** `drive_hash` (SHA256 of DAC spec) in checkpoints for reproducibility
   - **Status:** ✅ Production (TASK-004C complete)

#### 7. **Curriculum** (`src/townlet/curriculum/`)
   - **LOC:** ~600
   - **Strategies:**
     - **StaticCurriculum** - Fixed progression (e.g., deterministic reward structure)
     - **AdversarialCurriculum** - Adaptive difficulty (increases affordance positions, reward complexity)
   - **Purpose:** Progressive training from simple to complex
   - **Progression:** 5 stages (L0 → L0.5 → L1 → L2 → L3)
   - **Configuration:** Specified in `training.yaml`

#### 8. **Exploration** (`src/townlet/exploration/`)
   - **LOC:** ~1,500
   - **Strategies:**
     - **RandomNetworkDistillation (RND)** - Novelty-seeking via prediction error
     - **AdaptiveIntrinsicMotivation** - RND with variance-based annealing
     - **EpsilonGreedy** - Random action selection (configurable decay)
     - **IntrinsicCuriosityModule (ICM)** - Forward model-based exploration
   - **Purpose:** Discover reward-sparse regions of state space
   - **Integration:** Configurable via DAC (intrinsic strategy selection)

#### 9. **Training State & Replay** (`src/townlet/training/`)
   - **LOC:** ~900
   - **Components:**
     - State tracking (episode counters, episode rewards, win rates)
     - Replay buffer (uniform sampling or prioritized)
     - Gradient accumulation
   - **Purpose:** Manage training loop state across checkpoints
   - **Persistence:** Checkpoint save/load integration

#### 10. **Demo/Inference** (`src/townlet/demo/`)
   - **LOC:** ~1,100
   - **Components:**
     - `runner.py` - `DemoRunner` context manager (training + checkpoint management)
     - `unified_server.py` - `UnifiedServer` (training + inference server)
     - `live_inference.py` - `LiveInference` (inference-only mode)
     - `database.py` - Episode storage (SQLite)
   - **Purpose:** User-facing training/inference interface
   - **Entry Points:** `scripts/run_demo.py` → `UnifiedServer`

### Supporting Subsystems (5)

#### **Configuration System** (`src/townlet/config/`)
   - **LOC:** ~3,000
   - **Purpose:** Parse YAML configs → type-safe Pydantic DTOs
   - **No-Defaults Principle:** All fields required, no implicit defaults
   - **DTOs:** TrainingConfig, EnvironmentConfig, PopulationConfig, CurriculumConfig, etc.

#### **Substrate** (`src/townlet/substrate/`)
   - **LOC:** ~1,200
   - **Types:** Grid2D, Grid3D, GridND, Continuous, Aspatial
   - **Purpose:** Define spatial/coordinate system and topology
   - **Features:** Boundary modes (clamp, wrap, bounce), distance metrics

#### **Recording** (`src/townlet/recording/`)
   - **LOC:** ~800
   - **Purpose:** Episode capture, serialization, MP4 export
   - **Uses:** msgpack (serialization), lz4 (compression), ffmpeg (video)

#### **Compiler CLI** (`src/townlet/compiler/`)
   - **LOC:** ~200
   - **Purpose:** Command-line interface to UAC
   - **Commands:** compile, inspect, validate

#### **Frontend** (`frontend/src/`)
   - **LOC:** ~3,000 (Vue + JS)
   - **Purpose:** Real-time visualization + control interface
   - **Components:** Grid renderer, meter display, action log, episode info

---

## 5. Architectural Peculiarities & Design Decisions

### A. GPU-Native Vectorization (Extreme Performance Focus)

**Design Pattern:**
- All operations expressed as **batched PyTorch tensors**
- Vectorized environment executes N episodes in parallel on GPU
- Observation generation: single tensor operation for all agents
- Reward computation: DACEngine (DAC compiled to GPU operations)

**Impact:** 100x+ speedup vs sequential Python loops on GPU hardware

**Trade-off:** Requires careful batching discipline; errors propagate across batch

### B. Declarative Configuration System (Separation of Concerns)

**Pattern:**
```
YAML Config → UAC Pipeline → Compiled Specs → Runtime Registry
```

**Key Principle:** **No-Defaults Principle** (CLAUDE.md)
- All behavioral parameters **explicitly specified** in YAML
- No implicit defaults from code
- Changing code defaults does **not** affect old configs
- Validation via `no_defaults_lint.py` in CI

**Benefit:** Non-reproducible configurations fail loudly, not silently

**Examples:**
- `variables_reference.yaml` required for all packs (VFS integration)
- `drive_as_code.yaml` required for all packs (DAC reward spec)
- All reward functions removed from Python code (pure YAML)

### C. Pre-Release Zero Backwards Compatibility (CRITICAL)

**From CLAUDE.md:**

> THIS PROJECT IS PRE-RELEASE WITH ZERO USERS AND ZERO DOWNLOADS.
> NO backwards compatibility arrangements - Delete old code paths immediately

**Implications:**
- **Breaking changes are free** - no migration required
- **Version checks removed** - `if hasattr()` checks are antipatterns
- **Old code deleted** - not deprecated, not supported, removed
- **Silent fallbacks forbidden** - errors must fail loudly

**Examples:**
- Old `reward_strategy` field → **DELETED** (replaced by DAC)
- `src/hamlet/` → **MARKED OBSOLETE** (work only in `src/townlet/`)
- Legacy observation code → **REMOVED** (VFS required everywhere)

**Philosophy:**
> Dead code confuses future developers. Technical debt for zero users is inexcusable.

### D. Pedagogical Mission: "Interesting Failures as Teaching Moments"

**Design Goal:**
> "Trick students into learning graduate-level RL by making them think they're just playing The Sims"

**Consequence:** System **preserves "failures"** as learning opportunities:

**Example 1: Low Energy Delirium Bug**
- **Bug:** Multiplicative reward (energy × health) + high intrinsic weight
- **Result:** Agents exploit low bars for exploration
- **Curriculum Response:**
  - L0_0_minimal: Demonstrates bug (multiplicative, no suppression)
  - L0_5_dual_resource: Fixes bug (constant_base_with_shaped_bonus + modifiers)
  - **Lesson:** Students learn importance of reward structure design

**Example 2: Q-Value Overestimation**
- **Feature:** Vanilla DQN available (not Double DQN)
- **Student Discovery:** Overestimation bias limits performance
- **Learning:** Understanding Double DQN improvement becomes intuitive

**Implications:**
- Reward hacking **not fixed immediately** - becomes teaching case study
- Emergent behaviors **documented as phenomena** - not bugs
- Test suite validates system still works, **not** that strategies are "optimal"

### E. Observation Encoding as Transfer Learning Foundation

**Key Insight:** Same observation for all curriculum levels enables checkpoint transfer

**Grid2D Encoding (Relative, Default):**
- **Dimensions:** 29 (constant across grid sizes)
  - 2 coords (normalized to [0,1])
  - 8 meters (energy, health, satiation, mood, fitness, social, money, hygiene)
  - 15 affordances (binary: in range or not)
  - 4 temporal (time-of-day, day-of-week, season, tick)
- **Benefit:** Network weights transfer across 3×3 → 7×7 → 8×8 grids
- **Why Relative:** Normalized coordinates scale with grid, not fixed grid

**POMDP Support (L2):**
- 5×5 local window observation (54 dims total)
- Paired with LSTM for hidden state memory
- Enables partial observability learning

**Aspatial Edge Case:**
- No position concept → fake grid would be pedagogically harmful
- Renders as meter dashboard only

### F. Action Space as Global Vocabulary

**Design:**
- **Fixed set** of actions across all levels
- **Global definitions** in `configs/global_actions.yaml`
- **Custom actions** per pack (REST, MEDITATE, etc.)
- **Configurable labels** (gaming, 6dof, cardinal, math presets)

**Benefit:** Checkpoint weights (policy) transfer between curriculum levels

### G. DAC: Declarative Reward Functions (No Python Hardcoding)

**Before:**
- Reward functions written in Python
- Changes required code modifications + recompilation

**After (Drive As Code):**
```yaml
drive_as_code:
  extrinsic:
    type: constant_base_with_shaped_bonus
    base: 1.0
  intrinsic:
    strategy: rnd
    base_weight: 0.1
  modifiers:
    energy_crisis:
      variable: energy
      range: [0, 0.2]
      multiplier: 0.0
```

**Benefits:**
- A/B test reward structures without code changes
- Operators (non-programmers) can adjust rewards
- Checkpoint provenance via `drive_hash`

### H. Universe Compiler: Multi-Stage Declarative Pipeline

**Why Seven Stages?**

1. **Parse** - Simple YAML loading
2. **Symbol Table** - Build reference context (which affordances exist?)
3. **Resolve** - Cross-references (which cascade triggers which affordance?)
4. **Cross-Validate** - Constraint checking (meter ranges valid? cycles exist?)
5. **Metadata** - Compute observation dims, action space size
6. **Optimize** - Simplify expressions, pre-compute constants
7. **Emit** - Output compiled specs, cache to pickle

**Benefit:** Complex multi-file configs compile → single immutable `CompiledSubstrate` object

### I. Variable & Feature System: Structured State Access

**Pattern:**
```
Config → Variable Definitions → Access Control → Runtime Registry → Observations
```

**Features:**
- **Scopes:** Global (shared), agent (per-agent), agent_private (hidden from others)
- **Access Control:** Enforce who can read/write each variable
- **Normalization:** Linear, exponential, categorical
- **GPU Native:** All variables stored as tensors

**Use Case:** Prevent agents from reading "private" variables they shouldn't see (learning false features)

### J. Test-Driven Validation (644+ Tests, 70% Coverage)

**Test Organization:**
- **Unit:** Isolated component tests (fast)
- **Integration:** Multi-component flow tests
- **Slow:** Full training runs (nightly only)
- **Properties:** Hypothesis-based generative tests
- **Fixtures:** Pre-built config packs for testing

**Coverage Targets:**
- Core logic: 80%+
- Utilities: 70%+
- Entry points: 60%+ (harder to mock external systems)

**No Mocking Philosophy:**
- Tests use real configs, real data
- Environment executes actual YAML compilation
- Catches integration bugs missed by unit tests

### K. Multi-Substrate Support (Extensible Positioning)

**Supported Types:**
- **Grid2D** - 2D discrete grid (primary)
- **Grid3D** - 3D discrete grid (house environments)
- **GridND** - 4D-100D (high-dimensional exploration)
- **Continuous1D/2D/3D** - Continuous space
- **Aspatial** - Pure resource management (no position)

**Rationale:** Pedagogical diversity - students learn different RL problems

**Constraint:** POMDP (vision-based) only works with Grid2D/3D (limited vision_range ≤ 2)

### L. Live Inference Server + Frontend Visualization

**Architecture:**
```
Training Loop (GPU) → Checkpoint writes to disk → LiveInference watches → WebSocket broadcasts → Frontend renders
```

**Benefits:**
- Training/visualization decoupled (independent processes)
- Frontend can restart without stopping training
- Low-latency state updates (WebSocket push)
- Vue HMR for frontend development

**Limitations:**
- Single agent visualization (no multi-agent yet)
- Requires manual frontend start (not automatic)

---

## Summary: Architecture at a Glance

| Aspect | Characteristic |
|--------|-----------------|
| **Total LOC** | 27,138 (Python) + 3,000 (JS) + 40K+ (tests) |
| **Primary Technology** | PyTorch (DRL), Vue 3 (frontend), YAML (config) |
| **Execution Model** | Batched GPU-native (vectorized environment) |
| **Configuration** | Declarative YAML → 7-stage compiler → immutable specs |
| **Reward Functions** | Pure YAML (DAC), no Python hardcoding |
| **Testing** | 644+ tests, 70% coverage, strict validation |
| **Status** | Pre-release, zero users, breaking changes encouraged |
| **Pedagogical Focus** | Teaching moments over perfect agents |
| **Entry Points** | 4 main (training, inference, frontend, compiler) |
| **Key Subsystems** | 10 core + 5 supporting (environment, agent, population, UAC, VFS, DAC, curriculum, exploration, training, demo) |

---

## Next Steps (Phase 3+)

- **Phase 3 (Detailed Dependency Analysis):** Map cross-subsystem dependencies, identify coupling
- **Phase 4 (Risk Assessment):** Identify fragile components, technical debt, refactoring opportunities
- **Phase 5 (Modernization Roadmap):** Plan improvements (modularity, testability, performance)

---

**Document Version:** 1.0  
**Generated:** 2025-11-13  
**Status:** Ready for Review
