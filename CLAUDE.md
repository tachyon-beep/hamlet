# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HAMLET is a pedagogical Deep Reinforcement Learning (DRL) environment where agents learn to survive by managing multiple competing needs (energy, hygiene, satiation, money, health, fitness, mood, social). The primary mission is to "trick students into learning graduate-level RL by making them think they're just playing The Sims."

**Current Implementation**: **Townlet** - GPU-native vectorized training system with adversarial curriculum and intrinsic exploration.

**Key insight**: The project deliberately produces "interesting failures" (like reward hacking) as teaching moments rather than bugs to fix.

## CRITICAL: Pre-Release Status - ZERO Backwards Compatibility Required

**THIS PROJECT IS PRE-RELEASE WITH ZERO USERS AND ZERO DOWNLOADS.**

**ABSOLUTE RULES:**

1. **NO backwards compatibility arrangements** - Delete old code paths immediately
2. **NO fallback mechanisms** - Breaking changes are free and encouraged
3. **NO deprecation warnings** - Just break things and update references
4. **NO migration paths** - Old configs/code should fail loudly, not be supported
5. **NO "support both old and new"** - Technical debt for a non-existent user base is inexcusable

**Why this matters:**

- Every fallback path is technical debt that serves zero users
- Every "support both" pattern doubles maintenance burden for no benefit
- Every deprecation warning delays inevitable breaking changes
- Clean breaks now = simpler codebase at launch

**When you see:**
- "Let's support the old way too" → **NO. Delete it.**
- "We should maintain backwards compatibility" → **NO. We have zero users.**
- "Let's add a fallback for old configs" → **NO. Break them and update the templates.**
- "What if someone was using the old API?" → **They don't exist. Break it.**

**Examples of correct behavior:**
- VFS integration: Deleted all old observation code, required `variables_reference.yaml` for all packs
- reward_strategy field: Made it REQUIRED, broke old configs, updated all test fixtures
- Obsolete code: `src/hamlet/` marked obsolete, work exclusively in `src/townlet/`

**ANTIPATTERNS - These are WRONG and should be removed immediately:**

❌ **ANTIPATTERN**: `if hasattr(obj, 'old_field')` checks for old vs new attributes
- Why it's wrong: Maintaining dual code paths for zero users
- Fix: Delete the old code path, update all references to new field

❌ **ANTIPATTERN**: `try/except` blocks catching old config formats
- Why it's wrong: Silent fallbacks hide breaking changes that should fail loudly
- Fix: Let it raise, update the config, delete the try/except

❌ **ANTIPATTERN**: Version checks or feature flags for "legacy support"
- Why it's wrong: We have no legacy users to support
- Fix: Delete version checks, delete old code paths

❌ **ANTIPATTERN**: Making fields "Optional" when they should be required
- Why it's wrong: Implicit defaults create non-reproducible configs
- Fix: Make fields required, update all configs with explicit values

❌ **ANTIPATTERN**: Code comments saying "for backwards compatibility"
- Why it's wrong: If you're writing this comment, you're doing it wrong
- Fix: Delete the backwards compatibility code and the comment

❌ **ANTIPATTERN**: Keeping obsolete code "just in case"
- Why it's wrong: Dead code confuses future developers and bloats the codebase
- Fix: Delete it. Git history preserves it if you really need it later

**The rule is simple: If it's old and not in use, DELETE IT. Don't maintain it, don't support it, don't document it. Pre-release means freedom to break everything without consequence. Backwards compatibility patterns are ANTIPATTERNS at this stage.**

## AI-Friendly Documentation Pattern

**All documentation files in `docs/` use structured frontmatter** to help AI assistants understand content before reading the entire file. This saves tokens and improves context efficiency.

**Best Practice**: Read frontmatter FIRST, use the "AI-Friendly Summary" to decide relevance, then follow the "Reading Strategy" guidance. This helps avoid reading 2000+ line files when only specific sections are relevant.

### Universe Compiler (UAC) Quick Reference

- **Source**: `src/townlet/universe/compiler.py` - seven-stage pipeline (parse → symbol table → resolve → cross-validate → metadata → optimization → emit/cache)
- **Docs**: Start with `docs/UNIVERSE-COMPILER.md`, then `docs/architecture/COMPILER_ARCHITECTURE.md`
- **Tests**: `uv run pytest tests/test_townlet/unit/universe/` (use `UV_CACHE_DIR=.uv-cache` in sandboxed environments)
- **CLI**: `python -m townlet.compiler {compile,inspect,validate}` - wired into CI via `.github/workflows/config-validation.yml`

## Development Commands

### Setup

```bash
uv sync                  # Install dependencies
uv sync --extra dev      # Install with development tools
```

### Training (Townlet System)

```bash
export PYTHONPATH=$(pwd)/src:$PYTHONPATH

# Curriculum levels (pedagogical progression):
uv run scripts/run_demo.py --config configs/L0_0_minimal           # Temporal credit assignment
uv run scripts/run_demo.py --config configs/L0_5_dual_resource     # Multiple resources
uv run scripts/run_demo.py --config configs/L1_full_observability  # All affordances
uv run scripts/run_demo.py --config configs/L2_partial_observability  # POMDP + LSTM
uv run scripts/run_demo.py --config configs/L3_temporal_mechanics  # Time-based dynamics
```

### Inference Server (Live Visualization)

```bash
# Terminal 1: Start inference server (from directory with checkpoints)
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
python -m townlet.demo.live_inference <checkpoint_dir> 8766 0.2 10000 <config_path>
# Args: checkpoint_dir, port, speed, total_episodes, config_path

# Terminal 2: Start frontend
cd frontend && npm run dev
# Open http://localhost:5173
```

### Testing & Code Quality

```bash
uv run pytest                                        # Run all tests
uv run pytest tests/test_townlet/test_integration.py # Specific test
uv run pytest --cov=townlet --cov-report=term-missing # With coverage

uv run black src/townlet tests/    # Format
uv run ruff check src/townlet      # Lint
uv run mypy src/townlet            # Type check
```

### DemoRunner Context Manager Pattern

When using `DemoRunner` for checkpoint operations **without running full training**, use context manager for automatic resource cleanup:

```python
# ✅ GOOD: Guaranteed cleanup
with DemoRunner(config_dir=..., db_path=..., checkpoint_dir=...) as runner:
    runner.load_checkpoint()
    network_weights = runner.population.q_network.state_dict()

# ❌ BAD: Resources leak if run() not called
runner = DemoRunner(...)  # Opens DB connections and TensorBoard writers
runner.load_checkpoint()   # Resources stay open indefinitely!
```

## Architecture Overview

### Townlet System (ACTIVE)

```
src/townlet/
├── agent/networks.py           # SimpleQNetwork, RecurrentSpatialQNetwork (LSTM)
├── environment/vectorized_env.py  # VectorizedHamletEnv (GPU-native)
├── population/vectorized.py    # VectorizedPopulation (batched training)
├── curriculum/{adversarial,static}.py  # Curriculum strategies
├── exploration/{adaptive_intrinsic,rnd,epsilon_greedy}.py
├── training/{state,replay_buffer}.py
├── universe/compiler.py        # UAC seven-stage pipeline
└── vfs/                        # Variable & Feature System
    ├── schema.py               # Pydantic schemas
    ├── registry.py             # Runtime storage with access control
    └── observation_builder.py  # Compile-time spec generation
```

**Note**: Work only in `src/townlet/` - hamlet is obsolete legacy code.

### State Representation

**Fixed Affordance Vocabulary**: All curriculum levels observe the same 14 affordances (for transfer learning), even if not all are deployed.

**Observation Encoding Modes** (configurable via `substrate.yaml`):

- **relative** (default): Normalized [0,1] coordinates - best for transfer learning, required for POMDP
- **scaled**: Normalized + range metadata - network learns grid size matters
- **absolute**: Raw unnormalized coordinates - for physical simulation

**Observation Dimensions** (Grid2D with "relative" encoding):

- **L0/L0.5/L1/L3**: 29 dims (2 coords + 8 meters + 15 affordances + 4 temporal)
- **L2 (POMDP)**: 54 dims (25 local window + 2 coords + 8 meters + 15 affordances + 4 temporal)

**Key insight**: Observation dim is **constant** across all Grid2D grid sizes, enabling true transfer learning.

**Action Space** (global vocabulary enables checkpoint transfer):

- Grid2D: 8 actions (6 substrate + INTERACT + WAIT)
- Grid3D: 10 actions
- GridND (7D): 16 actions
- Aspatial: 4 actions

**POMDP Support**:

- ✅ **Supported**: Grid2D, Grid3D (vision_range ≤ 2), Aspatial (special case)
- ❌ **Not Supported**: Continuous substrates, GridND (N≥4) - window too large

See `tests/test_townlet/unit/environment/test_pomdp_validation.py` for validation logic.

### Variable & Feature System (VFS)

**Status**: ✅ INTEGRATED INTO PRODUCTION (TASK-002C Complete)

**Purpose**: Declarative state space configuration for observation specs, access control, and action dependencies.

**Pipeline**: `YAML Config → Schema Validation → Observation Spec → Runtime Registry → Observations`

**Key Components**:

- `schema.py`: VariableDef, ObservationField, NormalizationSpec, WriteSpec
- `registry.py`: Runtime storage with GPU tensors, access control enforcement
- `observation_builder.py`: Compile-time spec generation, dimension validation

**Variable Scopes**: `global` (shared), `agent` (per-agent), `agent_private` (hidden state)

**Access Control**: Readers (agent, engine, acs, bac), Writers (engine, actions, bac)

**Breaking Change**: All config packs **MUST** include `variables_reference.yaml`. See `docs/vfs-integration-guide.md` for migration.

**Documentation**:

- Configuration guide: `docs/config-schemas/variables.md`
- Design document: `docs/plans/2025-11-06-variables-and-features-system.md`

### Action Space (Composable)

**Architecture**: Action Space = Substrate Actions + Custom Actions

- **Global Vocabulary** (`configs/global_actions.yaml`): All levels share same action vocabulary
- **Custom Actions**: REST (energy recovery), MEDITATE (mood boost)
- **Action Labels**: Configurable terminology (gaming, 6dof, cardinal, math presets)

See `docs/config-schemas/enabled_actions.md` for details.

## Drive As Code (DAC)

**Status**: PRODUCTION (TASK-004C Complete)

**Purpose**: Declarative reward function system for HAMLET environments

Drive As Code (DAC) is a declarative reward function compiler that extracts all reward logic from Python into composable YAML configurations. Operators can A/B test reward structures without code changes. DAC compiles YAML specs into GPU-native computation graphs with provenance tracking.

### Key Components

**Files**: Each config pack requires `drive_as_code.yaml`:
```
configs/<level>/
├── substrate.yaml
├── bars.yaml
├── drive_as_code.yaml    # DAC reward specification (REQUIRED)
├── training.yaml
└── variables_reference.yaml
```

**Architecture**: YAML Config → Pydantic DTOs → Compiler Validation → Runtime Execution

**Formula**:
```
total_reward = extrinsic + (intrinsic × effective_intrinsic_weight) + shaping

where:
    effective_intrinsic_weight = base_weight × modifier₁ × modifier₂ × ...
```

### Components

**Modifiers** (Context-Sensitive Adjustment):
- Range-based multipliers for bars or VFS variables
- Use case: Crisis suppression (disable intrinsic when resources low)
- Example: `energy_crisis` modifier suppresses intrinsic when energy < 0.2

**Extrinsic Strategies** (Base Rewards - 9 types):
- `multiplicative`: reward = base × bar₁ × bar₂ (compound survival)
- `constant_base_with_shaped_bonus`: reward = base + Σ(bonuses) (fixes "Low Energy Delirium")
- `additive_unweighted`: reward = base + Σ(bars)
- `weighted_sum`, `polynomial`, `threshold_based`, `aggregation`, `vfs_variable`, `hybrid`

**Intrinsic Strategies** (Exploration Drives - 5 types):
- `rnd`: Random Network Distillation (novelty-seeking)
- `icm`: Intrinsic Curiosity Module
- `count_based`: Pseudo-count bonuses
- `adaptive_rnd`: RND with performance-based annealing
- `none`: Pure extrinsic (ablation studies)

**Shaping Bonuses** (Behavioral Incentives - 11 types):
- `approach_reward`, `completion_bonus`, `efficiency_bonus`, `state_achievement`
- `streak_bonus`, `diversity_bonus`, `timing_bonus`, `economic_efficiency`
- `balance_bonus`, `crisis_avoidance`, `vfs_variable`

### Pedagogical Pattern: "Low Energy Delirium" Bug

**Bug**: Multiplicative reward (energy × health) + high intrinsic weight → agents exploit low bars for exploration

**Curriculum**:
- **L0_0_minimal**: Demonstrates bug (multiplicative, no suppression)
- **L0_5_dual_resource**: Fixes bug (constant_base_with_shaped_bonus)
- **Comparison**: Students learn importance of reward structure design

### Breaking Change

**Old System** (DELETED):
- `training.yaml: reward_strategy` field → REMOVED
- `src/townlet/environment/reward_strategy.py` → DELETED
- Hardcoded Python reward classes → REPLACED

**New System** (REQUIRED):
- `drive_as_code.yaml` required for all config packs
- DACEngine compiles YAML → GPU computation graphs
- Checkpoint provenance via `drive_hash` (SHA256 of DAC config)

**Migration**: See `docs/guides/dac-migration.md`

### Example: L0_0_minimal

```yaml
drive_as_code:
  version: "1.0"

  modifiers: {}

  extrinsic:
    type: multiplicative
    base: 1.0
    bars: [energy]

  intrinsic:
    strategy: rnd
    base_weight: 0.1
    apply_modifiers: []

  shaping: []

  composition:
    normalize: false
    clip: null
    log_components: true
    log_modifiers: true
```

### Documentation

- **Config Reference**: `docs/config-schemas/drive_as_code.md`
- **Migration Guide**: `docs/guides/dac-migration.md`
- **Implementation Plan**: `docs/plans/2025-11-12-drive-as-code-implementation.md`

### Q-Learning Algorithm Variants

**Configurable via** `training.yaml: use_double_dqn`

**Vanilla DQN** (Mnih et al. 2015):
- Q-target: `r + γ * max_a Q_target(s', a)`
- Uses target network for both action selection and evaluation
- Susceptible to Q-value overestimation (max operator bias)

**Double DQN** (van Hasselt et al. 2016):
- Q-target: `r + γ * Q_target(s', argmax_a Q_online(s', a))`
- Decouples action selection (online network) from evaluation (target network)
- Reduces overestimation, typically converges faster

**Implementation Notes**:
- Feedforward: Minimal overhead (<1%)
- Recurrent: 3 forward passes required (online prediction, online selection, target evaluation) vs 2 for vanilla
- All curriculum levels support both algorithms
- Checkpoints persist `use_double_dqn` flag for reproducibility

**Usage Recommendation**: Run baseline with `use_double_dqn: false`, then compare against `use_double_dqn: true` to quantify improvement.

**Documentation**: See `docs/config-schemas/training.md` for detailed configuration guide.

## Configuration System

Training controlled via YAML configs in `configs/`. Each config pack is a directory containing:

```
configs/<level>/
├── substrate.yaml       # Spatial substrate (grid size, topology, boundaries)
├── bars.yaml            # Meter definitions (energy, health, etc.)
├── cascades.yaml        # Meter relationships
├── affordances.yaml     # Interaction definitions
├── cues.yaml            # UI metadata
├── training.yaml        # Hyperparameters
└── variables_reference.yaml  # VFS configuration (REQUIRED)
```

### Active Config Packs (Curriculum)

**L0_0_minimal**: Temporal credit assignment (3×3 grid, 1 affordance)
**L0_5_dual_resource**: Multiple resources (7×7 grid, 4 affordances)
**L1_full_observability**: Full observability baseline (8×8 grid, 14 affordances)
**L2_partial_observability**: POMDP with LSTM (8×8 grid, 5×5 vision window)
**L3_temporal_mechanics**: Time-based dynamics (24-tick day/night cycle)

**Future**: L4 (multi-zone), L5 (multi-agent), L6 (communication)

### Substrate Types

- `grid`: 2D discrete grid (Grid2DSubstrate)
- `grid3d`: 3D discrete grid (Grid3DSubstrate)
- `gridnd`: 4D-100D discrete grid (GridNDSubstrate)
- `continuous`: 1D/2D/3D continuous space
- `continuousnd`: 4D-100D continuous space
- `aspatial`: No positioning, pure resource management

**Boundary Modes**: clamp (hard walls), wrap (toroidal), bounce (elastic), sticky

**Distance Metrics**: manhattan (L1), euclidean (L2), chebyshev (L∞)

See `configs/templates/` for substrate configuration examples and `docs/config-schemas/` for detailed schema documentation.

### No-Defaults Principle

**All behavioral parameters must be explicitly specified in config files.** The DTO layer enforces this:

- `townlet.config.{training,environment,population,curriculum,bar,cascade,affordance,hamlet}.{*}Config`
- Existing: `townlet.substrate.config.SubstrateConfig`, `townlet.environment.action_config.ActionConfig`

**Why**: Hidden defaults create non-reproducible configs. Changing code defaults silently breaks old configs.

**Exemptions**: Only metadata (descriptions) and computed values (e.g., observation_dim).

## Network Architecture Selection

**SimpleQNetwork** (Full observability - L0, L0.5, L1):

- MLP: obs_dim → 256 → 128 → action_dim
- All Grid2D configs: 29→8 architecture (enables checkpoint transfer!)
- ~26K params

**RecurrentSpatialQNetwork** (Partial observability - L2, L3):

- Vision encoder: 5×5 local window → CNN → 128 features
- Position encoder: (x,y) → MLP → 32 features
- Meter encoder: 8 meters → MLP → 32 features
- LSTM: 192 input → 256 hidden (memory for POMDP)
- Q-head: 256 → 128 → action_dim
- ~650K params
- LSTM hidden state: resets at episode start, persists during rollout, resets per transition in batch training

**Training Details**:

- Gradient clipping: `max_norm=10.0` (prevents exploding gradients)
- Economic balance: Job payment = $22.5, sustainable with proper cycles
- Intrinsic weight annealing: threshold=100.0, requires mean survival >50 steps

## Frontend Visualization

**Location**: `frontend/src/components/`

**Rendering Modes** (substrate-specific):

### Spatial Mode (Grid2D/3D Substrates)

Component: `Grid.vue` (SVG-based 2D grid)

Features: Grid cells, agent positions, affordances, heat map overlay, agent trails, novelty heatmap (RND)

### Aspatial Mode (Aspatial Substrates)

Component: `AspatialView.vue` (meters-only dashboard)

Features: Meter bars with color coding, affordance list, action log

**Rationale**: Aspatial universes have no position concept - rendering a fake grid would be pedagogically harmful.

**WebSocket**: Server broadcasts on `localhost:8766`, frontend connects on mount

**Customization**:

- Affordance icons: `frontend/src/utils/constants.js`
- Meter colors: `frontend/src/styles/tokens.js`
- Grid cell size: `frontend/src/utils/constants.js`

## Testing Strategy

Tests focus on:

- Environment mechanics (vectorized operations, GPU tensors)
- Population training (batched updates, curriculum progression)
- Exploration (RND novelty, annealing logic)
- Integration (full training loop)

**Do NOT test for "correct" strategies** - emergent behaviors are valuable even if unexpected.

## Development Philosophy

> "Trick students into learning graduate-level RL by making them think they're just playing The Sims."

When in doubt:

- Prioritize pedagogical value over technical purity
- Preserve "interesting failures" as teaching moments
- Document unexpected behaviors rather than immediately fixing them
- Remember: The goal is to teach RL intuitively, not build production-ready agents
- **Work only in `src/townlet/`** - hamlet is obsolete legacy code
