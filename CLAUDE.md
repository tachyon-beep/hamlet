# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HAMLET is a pedagogical Deep Reinforcement Learning (DRL) environment where agents learn to survive by managing multiple competing needs (energy, hygiene, satiation, money, health, fitness, mood, social). The primary mission is to "trick students into learning graduate-level RL by making them think they're just playing The Sims."

**Current Implementation**: **Townlet** - GPU-native vectorized training system with adversarial curriculum and intrinsic exploration.

**Key insight**: The project deliberately produces "interesting failures" (like reward hacking) as teaching moments rather than bugs to fix.

## AI-Friendly Documentation

**All documentation files in `docs/` use structured frontmatter to help AI assistants understand content before reading the entire file.**

This pattern saves tokens and improves context efficiency:

**Architecture Documents** (`docs/architecture/`):

- Include YAML frontmatter with: Document Type, Status, Version, Audience, Technical Level
- Provide **AI-Friendly Summary** section with:
  - What the document describes (1-2 sentences)
  - Why it exists (problem it solves)
  - Who should read it (Must/Should/Optional by role)
  - Reading strategy (Quick scan / Partial read / Deep study)
- Define scope boundaries (In scope / Out of scope / Boundaries)
- List related documents and reading order

**Example frontmatter structure**:

```markdown
# 1. Executive Summary

**Document Type**: Overview (Executive Summary)
**Status**: Draft
**Version**: 2.5
**Audience**: Researchers, Educators, Policy Analysts, Engineers
**Technical Level**: Executive to Intermediate

## AI-Friendly Summary (Skim This First!)

**What This Document Describes**: High-level overview of Townlet v2.5 architecture...
**Why This Document Exists**: Provides entry point for understanding the system...
**Who Should Read This**:
- **Must Read**: Anyone new to the project
- **Should Read**: Engineers implementing features
- **Optional**: Advanced researchers focusing on specific modules
```

**Best Practice for AI Assistants**:

1. **Read frontmatter FIRST** before diving into full content
2. Use the "AI-Friendly Summary" to decide relevance
3. Follow the "Reading Strategy" guidance for token efficiency
4. Check "Related Documents" for prerequisites

This approach helps you avoid reading 2000+ line files when only specific sections are relevant.

### Universe Compiler (UAC) quick reference
- **Source**: `src/townlet/universe/compiler.py` orchestrates the seven-stage pipeline (parse → symbol table → resolve → cross-validate → metadata → optimization → emit/cache).
- **Helper modules**: `src/townlet/universe/{cues_compiler,compiled,compiler_inputs,optimization}.py` hold cue validation, artifact DTOs, stage 1 loader, and tensor prep logic, respectively.
- **Docs**: Start with `docs/UNIVERSE-COMPILER.md` for the condensed implementation guide, then fall back to `docs/architecture/COMPILER_ARCHITECTURE.md` for the broader rationale.
- **Tests to run**: `uv run pytest tests/test_townlet/unit/universe/` exercises each stage (Stage 1–7, cues, runtime). Use `UV_CACHE_DIR=.uv-cache` when sandboxed so uv can write its cache locally.

## Development Commands

### Setup

```bash
# Install dependencies
uv sync

# Install with development tools
uv sync --extra dev
```

### Training (Townlet System)

**Note:** Phase 5 (TASK-002A) changed checkpoint format. If you have old checkpoints, delete them before training: `rm -rf checkpoints_*`

```bash
# Set PYTHONPATH to include src directory
export PYTHONPATH=$(pwd)/src:$PYTHONPATH

# Training command pattern:
# python -m townlet.demo.runner --config <config_pack_dir>

# L0: Learn temporal credit assignment (single affordance)
uv run scripts/run_demo.py --config configs/L0_0_minimal

# L0.5: Multiple resource management (4 affordances)
uv run scripts/run_demo.py --config configs/L0_5_dual_resource

# L1: Full observability baseline (all 14 affordances)
uv run scripts/run_demo.py --config configs/L1_full_observability

# L2: Partial observability with LSTM (POMDP)
uv run scripts/run_demo.py --config configs/L2_partial_observability

# L3: Temporal mechanics (time-based affordances)
uv run scripts/run_demo.py --config configs/L3_temporal_mechanics

# Note: Each config pack is a directory containing:
#   bars.yaml, cascades.yaml, affordances.yaml, cues.yaml, training.yaml
```

### Inference Server (Live Visualization)

```bash
# IMPORTANT: Run inference server from worktree (where checkpoints are),
# but run frontend from main repo (where package.json is)

# Terminal 1: Start inference server (from worktree)
cd /home/john/hamlet/.worktrees/temporal-mechanics  # or your worktree path
export PYTHONPATH=$(pwd)/src:$PYTHONPATH

# For Level 2 POMDP:
python -m townlet.demo.live_inference checkpoints_level2 8766 0.2 10000 configs/townlet_level_2_pomdp.yaml

# For Level 2.5 Temporal Mechanics:
python -m townlet.demo.live_inference checkpoints_level2_5 8766 0.2 10000 configs/townlet_level_2_5_temporal.yaml

# Args: <checkpoint_dir> <port> <speed> <total_episodes> <config_path>
# - checkpoint_dir: Where checkpoints are saved
# - port: WebSocket port (default 8766)
# - speed: Simulation speed multiplier (0.2 = slower, 2.0 = faster)
# - total_episodes: Number of episodes to run
# - config_path: Training config YAML (REQUIRED for POMDP/temporal to load correct network type)

# Terminal 2: Start frontend (from main repo)
cd /home/john/hamlet/frontend  # Main repo, not worktree!
npm run dev
# Open http://localhost:5173

# Terminal 3 (optional): Monitor training progress (from worktree)
cd /home/john/hamlet/.worktrees/temporal-mechanics
watch -n 5 'sqlite3 demo_level2_5.db "SELECT episode, survival_steps, total_reward, stage FROM training_history ORDER BY episode DESC LIMIT 10"'
```

### Testing

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_townlet/test_integration.py

# Run with coverage
uv run pytest --cov=townlet --cov-report=term-missing
```

### Code Quality

```bash
# Format code
uv run black src/townlet tests/

# Lint
uv run ruff check src/townlet

# Type checking
uv run mypy src/townlet
```

### Using DemoRunner for Checkpoint Operations

When creating `DemoRunner` for checkpoint operations **without running full training**, use the context manager pattern for automatic resource cleanup:

```python
# ✅ GOOD: Guaranteed cleanup with context manager
from townlet.demo.runner import DemoRunner

with DemoRunner(
    config_dir=config_path,
    db_path=db_path,
    checkpoint_dir=checkpoint_dir,
    max_episodes=1,
    training_config_path=config_yaml_path,
) as runner:
    # Load checkpoint for analysis
    runner.load_checkpoint()

    # Access checkpoint state
    network_weights = runner.population.q_network.state_dict()

    # Resources automatically cleaned up on exit

# ❌ BAD: Resources leak if run() not called
runner = DemoRunner(...)
runner.load_checkpoint()
# Database connection and TensorBoard logger stay open indefinitely!
```

**Why this matters:**

- `DemoRunner.__init__()` opens database connections and TensorBoard writers
- These resources are **only** closed in `run()`'s finally block
- If you never call `run()`, resources leak until garbage collection
- Context manager ensures cleanup happens deterministically

**When to use each pattern:**

- **Context manager** (`with`): Checkpoint loading, analysis, testing
- **Direct run()**: Full training sessions

## Architecture

### Townlet System (ACTIVE)

```
src/townlet/
├── agent/
│   └── networks.py          # SimpleQNetwork, RecurrentSpatialQNetwork (LSTM)
├── environment/
│   └── vectorized_env.py    # VectorizedHamletEnv (GPU-native batched environments)
├── population/
│   └── vectorized.py        # VectorizedPopulation (batched agent training)
├── curriculum/
│   ├── adversarial.py       # AdversarialCurriculum (adaptive difficulty)
│   └── static.py            # StaticCurriculum (fixed difficulty)
├── exploration/
│   ├── adaptive_intrinsic.py # AdaptiveIntrinsicExploration (RND + annealing)
│   ├── rnd.py               # RNDExploration (novelty rewards)
│   └── epsilon_greedy.py    # EpsilonGreedyExploration
├── training/
│   ├── state.py             # BatchedAgentState, PopulationCheckpoint
│   ├── replay_buffer.py     # ReplayBuffer (experience replay)
│   └── (runner.py coming)   # DemoRunner (main entry point - currently in hamlet/demo/)
└── vfs/                     # Variable & Feature System (Phase 1)
    ├── schema.py            # Pydantic schemas (VariableDef, ObservationField, etc.)
    ├── registry.py          # VariableRegistry (runtime storage with access control)
    └── observation_builder.py # VFSObservationSpecBuilder (compile-time spec generation)
```

### State Representation

**Fixed Affordance Vocabulary**: All curriculum levels observe the same 14 affordances (for transfer learning and observation stability), even if not all are deployed.

**Observation Encoding Modes** (Phase 6):

Substrates support three configurable observation encoding modes via `observation_encoding` in substrate.yaml:

1. **"relative"** (default): Normalized coordinates [0, 1]
   - Best for: Transfer learning across grid sizes, POMDP (required)
   - Grid2D: 2 dims (normalized x, y)
   - Grid3D: 3 dims (normalized x, y, z)
   - Continuous: N dims (normalized to bounds)

2. **"scaled"**: Normalized coordinates + range metadata
   - Best for: Size-aware strategies (agent learns grid size matters)
   - Grid2D: 4 dims (normalized x, y, width, height)
   - Grid3D: 6 dims (normalized x, y, z, width, height, depth)
   - Continuous: 2N dims (normalized coords + bounds)

3. **"absolute"**: Raw unnormalized coordinates
   - Best for: Physical simulation, coordinate-space reasoning
   - Grid2D: 2 dims (raw x, y as floats)
   - Continuous: N dims (raw coordinates)

**Full Observability**:

- Position encoding: substrate-specific (2 dims for Grid2D "relative", 4 dims for "scaled")
- Meters: 8 normalized values (energy, health, satiation, money, mood, social, fitness, hygiene)
- Affordance at position: 15 one-hot (14 affordances + "none")
- Temporal extras: 4 values (time_sin, time_cos, interaction_progress, lifetime_progress)

**Observation dimensions by level** (with "relative" encoding):

- **L0_0_minimal**: 29 dims (2 coords + 8 meters + 15 affordances + 4 temporal)
- **L0_5_dual_resource**: 29 dims (same - all use Grid2D with relative encoding)
- **L1_full_observability**: 29 dims (same - grid size doesn't affect coords)
- **L3_temporal_mechanics**: 29 dims (same)

**Key insight**: With coordinate encoding, observation dim is **constant** across all grid sizes. This enables true transfer learning - a model trained on 3×3 grid works on 8×8 grid without architecture changes.

**Action space dimensions** (UPDATED - includes custom actions):

- **Grid2D configs**: **8 actions** (6 substrate + 2 custom)
- **Grid3D configs**: **10 actions** (8 substrate + 2 custom)
- **GridND (7D) configs**: **16 actions** (14 substrate + 2 custom)
- **Aspatial configs**: **4 actions** (2 substrate + 2 custom)

**Key insight**: With global action vocabulary, action_dim is **constant** across all Grid2D configs (L0, L0.5, L1, L3). This enables checkpoint transfer - a Q-network trained on L0_0_minimal works on L1_full_observability without architecture changes!

**Partial Observability (Level 2 POMDP)**:

- Local grid: 5×5 window (25 dims) - agent only sees local region
- Position: normalized (x, y) (2 dims) - substrate.normalize_positions()
- Meters: 8 normalized values (8 dims)
- Affordance at position: 15 one-hot (15 dims)
- Temporal extras: 4 values (4 dims)
- **Total**: 54 dimensions
- **Requirement**: POMDP always uses "relative" encoding (normalized positions)

**POMDP Limitations** (Partial Observability):

| Substrate | Support Status | Notes | Test Coverage |
|-----------|---------------|-------|---------------|
| Grid2D | ✅ **SUPPORTED** | Local 5×5 window (25 cells) | ✅ Unit + Integration |
| Grid3D | ✅ **SUPPORTED** | vision_range ≤ 2 (5×5×5 = 125 cells max) | ✅ Unit (2 tests) + Validation (3 tests) |
| Continuous1D/2D/3D | ❌ **NOT SUPPORTED** | Raises NotImplementedError (no discrete grid) | ✅ Unit (1 test) |
| GridND (N≥4) | ❌ **NOT SUPPORTED** | Environment rejects POMDP (window too large) | ✅ Validation (1 test) |
| ContinuousND (N≥4) | ❌ **NOT SUPPORTED** | Raises NotImplementedError | ✅ Unit (1 test) |
| Aspatial | ✅ **SPECIAL CASE** | Returns empty tensor (no position) | ✅ Unit (1 test) |

**Key Requirements**:

- Must use `observation_encoding="relative"` (normalized positions)
- Grid3D POMDP enforces vision_range ≤ 2 (larger windows rejected)
- Continuous substrates: Use full observability instead (POMDP conceptually invalid)

**Test Files**:

- Unit tests: `tests/test_townlet/unit/substrate/test_interface.py`
- Integration tests: `tests/test_townlet/integration/test_substrate_observations.py`
- Validation tests: `tests/test_townlet/unit/environment/test_pomdp_validation.py`

---

### Variable & Feature System (VFS) - Phase 1

**Status**: ✅ INTEGRATED INTO PRODUCTION (TASK-002C Complete + Integration)
**Purpose**: Declarative state space configuration for observation specs, access control, and action dependencies

**Integration Status**:

- ✅ VFS integrated into VectorizedHamletEnv (replaces legacy ObservationBuilder)
- ✅ All config packs now use `variables_reference.yaml` for observation generation
- ✅ Training validated (L0_0_minimal tested successfully)
- ✅ Checkpoints save/load correctly
- ✅ Dimension compatibility maintained (L0_0: 38, L0_5: 78, L1: 93, L2: 54, L3: 93)

**Architecture**:

```
VFS Pipeline: YAML Config → Schema Validation → Observation Spec → Runtime Registry → Observations
                                                                                    ↓
                                                                    VectorizedHamletEnv._get_observations()
```

**Key Components**:

1. **Schema Definitions** (`src/townlet/vfs/schema.py`)
   - `VariableDef`: Variable definitions with scope, type, lifetime, access control
   - `ObservationField`: Observation specs with shape and normalization
   - `NormalizationSpec`: Minmax/zscore normalization specifications
   - `WriteSpec`: Action write specifications (variable_id + expression)

2. **Variable Registry** (`src/townlet/vfs/registry.py`)
   - Runtime storage for VFS variables with GPU tensor backing
   - Access control enforcement (reader/writer permissions)
   - Scope semantics: global, agent, agent_private
   - Shape management: `[]` (global scalar) to `[num_agents, dims]` (agent vector)

3. **Observation Spec Builder** (`src/townlet/vfs/observation_builder.py`)
   - Compile-time observation spec generation from variable definitions
   - Dimension calculation and validation
   - Normalization configuration
   - Ensures checkpoint compatibility

**Variable Scope Semantics**:

| Scope | Shape (Scalar) | Shape (Vector) | Use Case |
|-------|----------------|----------------|----------|
| `global` | `[]` | `[dims]` | Time, weather, global state |
| `agent` | `[num_agents]` | `[num_agents, dims]` | Meters, position, visible state |
| `agent_private` | `[num_agents]` | `[num_agents, dims]` | Hidden state, internal rewards |

**Type System**:

- **Scalar types**: `scalar` (float), `bool`
- **Vector types**: `vec2i`, `vec3i`, `vecNi` (int), `vecNf` (float)
- **Storage**: PyTorch tensors (CPU/CUDA) with automatic device management

**Access Control**:

- **Readers**: agent, engine, acs (Adversarial Curriculum), bac (Behavioral Action Compiler)
- **Writers**: engine, actions, bac
- **Enforcement**: `registry.get(var_id, reader="agent")` raises `PermissionError` if denied

**Configuration Example** (`configs/*/variables_reference.yaml`):

```yaml
version: "1.0"

variables:
  - id: "energy"
    scope: "agent"
    type: "scalar"
    lifetime: "episode"  # persistent across ticks
    readable_by: ["agent", "engine", "acs"]
    writable_by: ["actions", "engine"]
    default: 1.0
    description: "Energy level [0.0-1.0]"

  - id: "position"
    scope: "agent"
    type: "vecNf"
    dims: 2
    lifetime: "episode"
    readable_by: ["agent", "engine", "acs"]
    writable_by: ["actions", "engine"]
    default: [0.0, 0.0]
    description: "Normalized agent position"

exposed_observations:
  - id: "obs_energy"
    source_variable: "energy"
    exposed_to: ["agent"]
    shape: []
    normalization:
      kind: "minmax"
      min: 0.0
      max: 1.0
```

**ActionConfig Integration** (Phase 1):

Actions declare variable dependencies via `reads` and `writes` fields:

```python
action = ActionConfig(
    id=0,
    name="MOVE_UP",
    type="movement",
    delta=[0, -1],
    costs={"energy": 0.005},
    reads=["position", "energy"],  # VFS: Dependencies
    writes=[WriteSpec(variable_id="position", expression="position + delta")],
)
```

**Observation Dimension Validation** (CRITICAL):

VFS maintains **exact compatibility** with current hardcoded dimensions to preserve checkpoints:

| Config Level | Expected Dims | VFS Calculated | Status |
|--------------|---------------|----------------|--------|
| L0_0_minimal | 38 | 38 | ✅ Validated |
| L0_5_dual_resource | 78 | 78 | ✅ Validated |
| L1_full_observability | 93 | 93 | ✅ Validated |
| L2_partial_observability | 54 | 54 | ✅ Validated |
| L3_temporal_mechanics | 93 | 93 | ✅ Validated |

**Regression tests** ensure dimension compatibility: `tests/test_townlet/unit/vfs/test_observation_dimension_regression.py`

**Test Coverage**:

- **88 tests total** (76 unit + 12 integration) - ALL PASSING ✅
- Schema validation: 23 tests (93% coverage)
- Registry operations: 25 tests (83% coverage)
- Observation builder: 22 tests (92% coverage)
- Dimension regression: 6 tests (5 configs validated)
- End-to-end integration: 12 tests (YAML → observations)

**Documentation**:

- Configuration guide: `docs/config-schemas/variables.md`
- Design document: `docs/plans/2025-11-06-variables-and-features-system.md`
- Implementation plan: `docs/tasks/TASK-002-variables-and-features-system.md`

**Phase 1 Status**: ✅ COMPLETE & INTEGRATED
**Phase 2 (Future)**: Behavioral Action Compiler (BAC) integration

**BREAKING CHANGE** (Post-Integration):

All config packs **MUST** include `variables_reference.yaml`. Legacy configs will fail with:

```
FileNotFoundError: variables_reference.yaml is required but not found in configs/your_config/.

Quick fix:
  1. Copy reference: cp configs/L1_full_observability/variables_reference.yaml configs/your_config/
  2. Edit to match your configuration
  3. See docs/config-schemas/variables.md for schema details
```

**Migration**: See `docs/vfs-integration-guide.md` for complete migration instructions.

**Removed Files**:

- `src/townlet/environment/observation_builder.py` (212 lines) - Replaced by VFS

---

### Action Space (Composable - TASK-002B)

**Architecture**: Action Space = Substrate Actions + Custom Actions

**Global Vocabulary** (`configs/global_actions.yaml`):

- All curriculum levels share the **same action vocabulary**
- Enables checkpoint transfer (same action_dim across configs)
- Custom actions: REST, MEDITATE (substrate-agnostic)

**Per-Config Enabled Actions** (future - `training.yaml`):

- Each config can specify which actions are **enabled**
- Disabled actions masked at runtime (cannot be selected)
- Currently: All actions enabled by default

**Action Counts by Substrate**:

- Grid2D: 6 substrate + 2 custom = **8 actions total**
- Grid3D: 8 substrate + 2 custom = **10 actions total**
- GridND (7D): 14 substrate + 2 custom = **16 actions total**
- Aspatial: 2 substrate + 2 custom = **4 actions total**

**Example - Grid2D Action Space**:

| ID | Name | Type | Source | Delta | Costs |
|----|------|------|--------|-------|-------|
| 0 | UP | movement | substrate | [0, -1] | energy: 0.005 |
| 1 | DOWN | movement | substrate | [0, 1] | energy: 0.005 |
| 2 | LEFT | movement | substrate | [-1, 0] | energy: 0.005 |
| 3 | RIGHT | movement | substrate | [1, 0] | energy: 0.005 |
| 4 | INTERACT | interaction | substrate | - | energy: 0.003 |
| 5 | WAIT | passive | substrate | - | energy: 0.004 |
| 6 | REST | passive | custom | - | energy: -0.002 (restore) |
| 7 | MEDITATE | passive | custom | - | energy: 0.001, mood: +0.02 |

**Custom Actions**:

- **REST**: Passive recovery (restores energy + mood), available anywhere
- **MEDITATE**: Mental health action (costs energy, restores mood)

**Checkpoint Transfer**:

All Grid2D configs have action_dim = 8, enabling checkpoint transfer:

- L0_0_minimal: 8 actions (3×3 grid, 1 affordance)
- L0_5_dual_resource: 8 actions (7×7 grid, 4 affordances)
- L1_full_observability: 8 actions (8×8 grid, 14 affordances)

A Q-network trained on L0 can be transferred to L1 without architecture changes!

**Implementation Details**:

- **ActionConfig**: Pydantic schema defining action properties
- **ComposedActionSpace**: Container tracking all actions with metadata
- **ActionSpaceBuilder**: Composes substrate + custom actions from YAML
- **Canonical Ordering**: [substrate actions...], INTERACT, WAIT come last
- **Dynamic Deltas**: Movement deltas loaded from ActionConfig (no hardcoding)

**Action Labels** (`src/townlet/environment/action_labels.py`):

- Configurable domain-specific terminology for actions
- **Presets**: `gaming` (LEFT/RIGHT/UP/DOWN), `6dof` (SWAY/HEAVE/SURGE), `cardinal` (NORTH/SOUTH), `math` (X_NEG/X_POS)
- **Custom labels**: User-defined terminology for specialized domains
- **N-dimensional support** (N≥4): All presets use dimension index notation (D0_NEG, D0_POS, etc.)
  - No canonical "gaming" or "cardinal" directions exist in 4D+ space
  - Custom labels supported for domain-specific N-dimensional applications
- **Example**: 7D GridND uses D0_NEG, D1_NEG, ..., D6_NEG, D0_POS, ..., D6_POS, INTERACT, WAIT (16 actions total)

**See Also**:

- `docs/config-schemas/enabled_actions.md` - Detailed enabled_actions pattern
- `configs/global_actions.yaml` - Global action vocabulary
- `/home/john/hamlet/docs/plans/2025-11-06-composable-action-space.md` - Implementation plan

### Reward Structure

**Framework Capability**: The Townlet Framework allows operators to define custom reward functions using any combination of bars (meters) and mathematical operations via UAC configuration. This enables domain-specific reward shaping without code changes.

**Reference Implementation (Townlet Town)**:

- **Per-step reward**: `r_t = bars[energy] * bars[health]` (multiplicative)
- **Dead agents**: 0.0 reward (episode ends when any bar reaches 0)
- **NO proximity shaping**: Agents must explore and interact to survive (no reward for being near affordances)
- **Sparse reward design**: Multiplicative formula forces long-horizon credit assignment
- **Rationale**: Product of critical bars creates sparse reward signal requiring agents to maintain multiple needs simultaneously

**Intrinsic Rewards** (Framework Feature):

- RND (Random Network Distillation) for novelty-seeking
- Adaptive annealing based on performance consistency
- Combined with extrinsic rewards: `total = extrinsic + intrinsic * weight`

---

## Configuration System

Training is controlled via YAML configs in `configs/`. Each config pack is a directory containing multiple YAML files that define the universe.

### Active Config Packs (Curriculum Levels)

**L0_0_minimal** - Pedagogical: Learn temporal credit assignment

- Single affordance (Bed) on 3×3 grid
- Teaches spacing behavior (don't spam bed, space out usage)
- Fast learning: epsilon_decay=0.99

**L0_5_dual_resource** - Pedagogical: Multiple resource management

- Four affordances (Bed, Hospital, HomeMeal, Job) on 7×7 grid
- Teaches balancing energy + health cycles
- Moderate learning: epsilon_decay=0.995

**L1_full_observability** - Full observability baseline

- All 14 affordances on 8×8 grid
- Agent sees complete grid (no POMDP)
- Standard MLP Q-network (~26K params)

**L2_partial_observability** - POMDP with LSTM

- All 14 affordances on 8×8 grid
- Agent sees only 5×5 local window (must build mental map)
- Recurrent spatial Q-network with LSTM (~600K params)

**L3_temporal_mechanics** - Time-based affordances + multi-tick interactions

- Operating hours (Job 9am-5pm, Bar 6pm-2am, etc.)
- Multi-tick interactions (75% linear + 25% completion bonus)
- 24-tick day/night cycle

### Config Pack Structure (UNIVERSE_AS_CODE)

Each config pack directory contains:

```
configs/L0_0_minimal/
├── substrate.yaml    # Spatial substrate (grid size, topology, boundaries, distance metrics)
├── bars.yaml         # Meter definitions (energy, health, money, etc.)
├── cascades.yaml     # Meter relationships (low satiation → drains energy)
├── affordances.yaml  # Interaction definitions (Bed, Hospital, Job, etc.)
├── cues.yaml         # UI metadata for visualization
└── training.yaml     # Hyperparameters and enabled affordances
```

**Key principle**: Everything configurable via YAML (UNIVERSE_AS_CODE). The system loads and validates these files at startup.

**Important**: All curriculum levels use the **same affordance vocabulary** (14 affordances) for observation stability. Only deployment varies via `enabled_affordances` in training.yaml.

### Substrate Configuration (substrate.yaml)

Defines the spatial substrate (coordinate system, topology, boundaries, distance metrics).

**Available Substrate Types**:

- `grid`: 2D discrete grid (Grid2DSubstrate)
- `grid3d`: 3D discrete grid (Grid3DSubstrate)
- `gridnd`: 4D-100D discrete grid (GridNDSubstrate)
- `continuous`: 1D/2D/3D continuous space (Continuous1D/2D/3DSubstrate)
- `continuousnd`: 4D-100D continuous space (ContinuousNDSubstrate)
- `aspatial`: No positioning, pure resource management

**Example (Standard 2D Grid)**:

```yaml
version: "1.0"
description: "8×8 square grid with hard boundaries"
type: "grid"

grid:
  topology: "square"       # 2D Cartesian grid
  width: 8                 # Number of columns
  height: 8                # Number of rows
  boundary: "clamp"        # Hard walls (clamp, wrap, bounce, sticky)
  distance_metric: "manhattan"  # L1 norm (manhattan, euclidean, chebyshev)
  observation_encoding: "relative"  # Position encoding (relative, scaled, absolute)
```

**Example (7D Hypercube Grid)**:

```yaml
version: "1.0"
description: "7D hypercube grid for high-dimensional RL research"
type: "gridnd"

gridnd:
  dimension_sizes: [5, 5, 5, 5, 5, 5, 5]  # 7D grid, 5 cells per dimension
  boundary: "clamp"
  distance_metric: "manhattan"
  observation_encoding: "relative"  # 7 dims (normalized coords)
```

**Example (4D Continuous Space)**:

```yaml
version: "1.0"
description: "4D continuous space for abstract state experiments"
type: "continuousnd"

continuousnd:
  bounds:
    - [0.0, 10.0]  # Dimension 0: [0, 10]
    - [0.0, 10.0]  # Dimension 1: [0, 10]
    - [0.0, 10.0]  # Dimension 2: [0, 10]
    - [0.0, 10.0]  # Dimension 3: [0, 10]
  boundary: "clamp"
  movement_delta: 0.5
  interaction_radius: 0.8
  distance_metric: "euclidean"
  observation_encoding: "relative"  # 4 dims (normalized coords)
```

**Boundary Modes**:

- `clamp`: Hard walls (agent position clamped to edges)
- `wrap`: Toroidal wraparound (Pac-Man style)
- `bounce`: Elastic reflection (agent bounces back from boundary)
- `sticky`: Sticky walls (agent stays in place when hitting boundary)

**Distance Metrics**:

- `manhattan`: L1 norm, |x1-x2| + |y1-y2| (matches 4-directional movement)
- `euclidean`: L2 norm, sqrt((x1-x2)² + (y1-y2)²) (straight-line distance)
- `chebyshev`: L∞ norm, max(|x1-x2|, |y1-y2|) (8-directional movement)

**Observation Encoding Modes** (Phase 5C):

- `relative`: Normalized coordinates [0,1] per dimension (N dims)
  - Grid-size independent, enables transfer learning
  - Example: 7D grid → 7 observation dims
- `scaled`: Normalized + dimension sizes (2N dims)
  - Includes metadata about grid/space dimensions
  - Example: 7D grid → 14 observation dims (7 normalized + 7 sizes)
- `absolute`: Raw unnormalized coordinates (N dims)
  - Network learns size-specific patterns
  - Example: 7D grid → 7 observation dims (raw coords)

**Action Space Formula**:

- **N-dimensional substrates**: 2N + 1 actions
  - GridND: 2N movement actions (±1 per dimension) + INTERACT
  - ContinuousND: 2N movement actions (±movement_delta per dimension) + INTERACT
  - Example: 7D grid → 15 actions (14 movement + 1 interact)
- **Low-dimensional substrates**: Varies by dimensionality
  - 2D grid: 6 actions (UP, DOWN, LEFT, RIGHT, INTERACT, WAIT)
  - 3D grid: 8 actions (add FORWARD, BACKWARD)
  - 1D continuous: 4 actions (LEFT, RIGHT, INTERACT, WAIT)

**High-Dimensional Substrate Warnings**:

- N≥10 dimensions triggers warning (action space ≥21 grows large)
- Partial observability (POMDP) not supported for N≥4
  - 4D with vision_range=2: 5⁴ = 625 cells
  - 7D with vision_range=2: 5⁷ = 78,125 cells (impractical)
  - Use full observability with coordinate encoding instead

**Aspatial Mode** (no positioning):

```yaml
version: "1.0"
description: "Aspatial universe (pure resource management)"
type: "aspatial"
aspatial: {}
```

**Template Configs**:

- `configs/templates/substrate.yaml` - Standard 2D grid
- `configs/templates/substrate_continuous_1d.yaml` - 1D continuous line
- `configs/templates/substrate_continuous_2d.yaml` - 2D continuous plane
- `configs/templates/substrate_continuous_3d.yaml` - 3D continuous volume
- `configs/templates/substrate_gridnd.yaml` - 4D-100D discrete grids
- `configs/templates/substrate_continuousnd.yaml` - 4D-100D continuous spaces

### Action Labels Configuration (action_labels.yaml) - OPTIONAL

Defines domain-specific terminology for actions. If not specified, defaults to "gaming" preset.

**Key Concept**: Action labels separate **canonical action semantics** (what substrates interpret) from **user-facing labels** (what students see).

**Example (Gaming Preset - Default)**:

```yaml
preset: "gaming"
# Automatically provides: UP, DOWN, LEFT, RIGHT, INTERACT, WAIT, FORWARD, BACKWARD
```

**Example (Custom Submarine Labels)**:

```yaml
custom:
  0: "PORT"       # Move left
  1: "STARBOARD"  # Move right
  2: "AFT"        # Move backward
  3: "FORE"       # Move forward
  4: "INTERACT"   # Interact with affordance
  5: "WAIT"       # Hold position
  6: "SURFACE"    # Move up (3D only)
  7: "DIVE"       # Move down (3D only)
```

**Available Presets**:

- `gaming`: Standard gaming controls (LEFT/RIGHT/UP/DOWN/FORWARD/BACKWARD)
- `6dof`: Robotics 6-DoF terminology (SWAY_LEFT/RIGHT, HEAVE_UP/DOWN, SURGE_FORWARD/BACKWARD)
- `cardinal`: Compass directions (NORTH/SOUTH/EAST/WEST/ASCEND/DESCEND)
- `math`: Explicit axis notation (X_NEG/X_POS, Y_NEG/Y_POS, Z_POS/Z_NEG)

**Dimensionality Filtering**:
Labels are automatically filtered to match substrate dimensionality:

- **Aspatial (0D)**: INTERACT, WAIT (2 actions)
- **1D**: LEFT, RIGHT, INTERACT, WAIT (4 actions)
- **2D**: UP, DOWN, LEFT, RIGHT, INTERACT, WAIT (6 actions)
- **3D**: UP, DOWN, LEFT, RIGHT, INTERACT, WAIT, FORWARD, BACKWARD (8 actions)

**Pedagogical Value**:

- Demonstrates that labels are arbitrary, semantics matter
- Enables domain-appropriate learning (robotics, marine, aviation, gaming)
- Reveals how different communities label identical mathematical transformations

See `configs/templates/action_labels_*.yaml` for preset examples and custom label guidelines.

### Config Structure (training.yaml)

```yaml
environment:
  grid_size: 8
  partial_observability: false  # true for POMDP
  vision_range: 2  # 5×5 window when partial_observability=true
  enabled_affordances: ["Bed", "Hospital", "HomeMeal", "Job"]  # Which to deploy

population:
  num_agents: 1
  learning_rate: 0.00025  # 0.0001 for recurrent networks
  gamma: 0.99
  replay_buffer_capacity: 10000
  network_type: simple  # or 'recurrent' for LSTM

curriculum:
  max_steps_per_episode: 500
  survival_advance_threshold: 0.7
  survival_retreat_threshold: 0.3
  entropy_gate: 0.5
  min_steps_at_stage: 1000

exploration:
  embed_dim: 128
  initial_intrinsic_weight: 1.0
  variance_threshold: 100.0
  survival_window: 100

training:
  device: cuda
  max_episodes: 5000
  train_frequency: 4
  target_update_frequency: 100
  batch_size: 64
  max_grad_norm: 10.0
  epsilon_start: 1.0
  epsilon_decay: 0.995
  epsilon_min: 0.01
```

---

## Progressive Complexity Levels (Curriculum)

The curriculum progresses from simple pedagogical tasks to complex POMDP challenges:

**L0_0_minimal** (✅ Implemented): **Temporal Credit Assignment**

- 3×3 grid, 1 affordance (Bed only)
- Teaches spacing behavior (don't spam, space out usage)
- Fast learning: epsilon_decay=0.99, ~500 episodes
- Pedagogical goal: Learn that immediate actions have delayed consequences

**L0_5_dual_resource** (✅ Implemented): **Multiple Resource Management**

- 7×7 grid, 4 affordances (Bed, Hospital, HomeMeal, Job)
- Teaches balancing energy + health cycles
- Moderate learning: epsilon_decay=0.995
- Introduces economic loop (Job → money → affordances)

**L1_full_observability** (✅ Implemented): **Full Observability Baseline**

- 8×8 grid, all 14 affordances
- Agent sees complete grid (no POMDP)
- Standard MLP Q-network (~26K params)
- NO proximity shaping - must interact to survive
- Baseline for comparing POMDP performance

**L2_partial_observability** (✅ Implemented): **POMDP with LSTM Memory**

- 8×8 grid, all 14 affordances
- Agent sees only 5×5 local window (must build mental map)
- Recurrent spatial Q-network with LSTM (~600K params)
- Teaches spatial memory and exploration under uncertainty

**L3_temporal_mechanics** (✅ Implemented): **Time-Based Dynamics**

- 24-tick day/night cycle with operating hours (Job 9am-5pm, Bar 6pm-2am)
- Multi-tick interactions: 75% linear rewards + 25% completion bonus
- Time-based action masking (closed affordances unavailable)
- Early exit mechanics (agents keep accumulated benefits)
- Per-tick costs prevent "free sampling"
- Agent learns temporal planning and opportunity cost
- **Visualization**: Time-of-day gradient bar, interaction progress ring

**L4_multi_zone** (🎯 Future): Multi-zone environment with hierarchical RL
**L5_multi_agent** (🎯 Future): Multi-agent competition with theory of mind
**L6_communication** (🎯 Future): Family communication and emergent language

---

## Important Implementation Details

### Network Architecture Selection

- **SimpleQNetwork**: Full observability (L0, L0.5, L1)
  - MLP: obs_dim → 256 → 128 → action_dim (8 for Grid2D)
  - L0: 29 input dims → 8 output dims (~26K params)
  - L0.5: 29 input dims → 8 output dims (~26K params)
  - L1: 29 input dims → 8 output dims (~26K params)
  - **Note**: All Grid2D configs have same architecture (29→8), enabling checkpoint transfer!

- **RecurrentSpatialQNetwork**: Partial observability (L2, L3)
  - Vision encoder: 5×5 local window → CNN → 128 features
  - Position encoder: (x, y) → MLP → 32 features
  - Meter encoder: 8 meters → MLP → 32 features
  - LSTM: 192 input → 256 hidden (memory for POMDP)
  - Q-head: 256 → 128 → action_dim (8 for Grid2D)
  - Total: ~650K params
  - LSTM hidden state resets at episode start
  - Hidden state persists during episode rollout (memory)
  - Hidden state resets per transition during batch training (simplified approach)

### Gradient Clipping

Q-network uses gradient clipping (`max_norm=10.0`) to prevent exploding gradients.

### Economic Balance

The environment is intentionally balanced for sustainability:

- Full cycle cost varies by affordance choices
- Job payment = $22.5
- Sustainable with proper cycles

### Intrinsic Weight Annealing

**Fixed bug (2025-10-30)**: Premature annealing caused by low variance threshold (10.0)

- **New threshold**: 100.0
- **New requirement**: Mean survival >50 steps before annealing
- Prevents "consistently failing" from triggering "consistent performance" annealing

---

## Testing Strategy

Tests focus on:

- Environment mechanics (vectorized operations, GPU tensors)
- Population training (batched updates, curriculum progression)
- Exploration (RND novelty, annealing logic)
- Integration (full training loop)

**Do NOT test for "correct" strategies** - emergent behaviors are valuable even if unexpected.

---

### No-Defaults Principle & DTOs

**All behavioral parameters must be explicitly specified in config files.** No implicit defaults allowed. The new DTO layer (TASK-003) enforces this across:

- `townlet.config.training.TrainingConfig`
- `townlet.config.environment.EnvironmentConfig`
- `townlet.config.population.PopulationConfig`
- `townlet.config.curriculum.CurriculumConfig`
- `townlet.config.bar.BarConfig`
- `townlet.config.cascade.CascadeConfig`
- `townlet.config.affordance.AffordanceConfig`
- `townlet.config.hamlet.HamletConfig` (master)

Existing DTOs already cover Substrate (`townlet.substrate.config.SubstrateConfig`) and Actions (`townlet.environment.action_config.ActionConfig`).

**Why**: Hidden defaults create non-reproducible configs, operators don’t know what values are being used, and changing code defaults silently breaks old configs.

**Exemptions**: Only metadata (descriptions) and computed values (e.g., observation_dim). All behavioral configs—including `cues.yaml`—are mandatory.

**Enforcement**: DTOs are used at load time (runner.py instantiates `HamletConfig`). Missing/invalid fields raise descriptive validation errors; CI will run `scripts/validate_configs.py` to catch inconsistent packs.

---

## Frontend Visualization (Multi-Substrate Support)

**Location**: `frontend/src/components/`

The frontend supports **two rendering modes** based on substrate type:

### Spatial Mode (Grid2D Substrates)

**Component**: `Grid.vue` (SVG-based 2D grid)

**Features**:

- Grid cells rendered as SVG rectangles
- Agents positioned at (x, y) coordinates
- Affordances displayed as icons at grid positions
- Heat map overlay (position visit frequency)
- Agent trails (last 3 positions)
- Novelty heatmap (RND exploration)

**WebSocket Contract**:

```json
{
  "substrate": {
    "type": "grid2d",
    "position_dim": 2,
    "topology": "square",
    "width": 8,
    "height": 8,
    "boundary": "clamp",
    "distance_metric": "manhattan"
  },
  "grid": {
    "agents": [{"id": "agent_0", "x": 3, "y": 5}],
    "affordances": [{"type": "Bed", "x": 2, "y": 1}]
  }
}
```

**Topology Field** (Grid Substrates Only):

- **Grid2D**: `topology: "square"` (4-connected 2D grid)
- **Grid3D**: `topology: "cubic"` (6-connected 3D grid)
- **GridND**: `topology: "hypercube"` (2N-connected ND grid)
- **Continuous/Aspatial**: Topology field omitted (not applicable)

**Rationale**: Topology describes discrete connectivity pattern. Continuous spaces have no discrete cells, so topology is meaningless and omitted from metadata.

---

### Aspatial Mode (Aspatial Substrates)

**Component**: `AspatialView.vue` (Meters-only dashboard)

**Features**:

- Large meter bars with color coding (critical/warning/healthy)
- Affordance list (no positions, just availability)
- Recent actions log (temporal context without spatial context)
- No heat map (position-based feature)
- No agent trails (position-based feature)

**WebSocket Contract**:

```json
{
  "substrate": {
    "type": "aspatial",
    "position_dim": 0
  },
  "grid": {
    "agents": [{"id": "agent_0"}],
    "affordances": [{"type": "Bed"}]
  }
}
```

**Rationale**:
Aspatial universes have no concept of "position" - rendering a fake grid would:

1. Be pedagogically harmful (teaches incorrect intuitions)
2. Mislead operators about agent behavior
3. Imply spatial relationships that don't exist

Instead, aspatial mode focuses on **meters** (primary learning signal) and **interaction history** (temporal, not spatial).

---

### Substrate Detection

**Logic** (in `App.vue`):

```vue
<Grid v-if="store.substrateType === 'grid2d'" ... />
<AspatialView v-else-if="store.substrateType === 'aspatial'" ... />
```

**Fallback** (for legacy checkpoints without substrate metadata):

```javascript
const substrate = message.substrate || {
  type: "grid2d",  // Assume legacy spatial behavior
  position_dim: 2,
  width: message.grid.width,
  height: message.grid.height
}
```

---

### Running Frontend

**Prerequisites**:

1. Live inference server running (provides WebSocket endpoint)
2. Node.js and npm installed

**Commands**:

```bash
# Terminal 1: Start inference server (from worktree with checkpoints)
cd /home/john/hamlet/.worktrees/substrate-abstraction
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
python -m townlet.demo.live_inference checkpoints 8766 0.2 10000 configs/L1_full_observability

# Terminal 2: Start frontend (from main repo)
cd /home/john/hamlet/frontend
npm run dev

# Open http://localhost:5173
```

**Port Configuration**:

- Frontend dev server: `localhost:5173` (Vite default)
- WebSocket endpoint: `localhost:8766` (live_inference default)
- Frontend auto-connects to WebSocket on component mount

---

### Customizing Visualization

**Affordance Icons** (`frontend/src/utils/constants.js`):

```javascript
export const AFFORDANCE_ICONS = {
  Bed: '🛏️',
  Hospital: '🏥',
  Job: '💼',
  // ... add custom icons here
}
```

**Meter Colors** (`frontend/src/styles/tokens.js`):

```javascript
'--color-success': '#22c55e',  // Healthy meters
'--color-warning': '#f59e0b',  // Warning meters
'--color-error': '#ef4444',    // Critical meters
```

**Grid Cell Size** (`frontend/src/utils/constants.js`):

```javascript
export const CELL_SIZE = 75  // Pixels per grid cell
```

---

## Development Philosophy

> "Trick students into learning graduate-level RL by making them think they're just playing The Sims."

When in doubt:

- Prioritize pedagogical value over technical purity
- Preserve "interesting failures" as teaching moments
- Document unexpected behaviors rather than immediately fixing them
- Remember: The goal is to teach RL intuitively, not build production-ready agents
- **Work only in `src/townlet/`** - hamlet is obsolete legacy code
