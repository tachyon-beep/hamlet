# HAMLET Architecture Analysis Report

**Project:** HAMLET - Pedagogical Deep Reinforcement Learning Environment
**System:** Townlet (GPU-Native Vectorized Training System)
**Analysis Date:** 2025-11-13
**Version:** Pre-Release (Zero Backwards Compatibility)
**Analyst:** Claude (axiom-system-archaeologist workflow)
**Workspace:** `docs/arch-analysis-2025-11-13-2013/`

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Overview](#system-overview)
   - [Mission & Philosophy](#mission--philosophy)
   - [Technical Foundation](#technical-foundation)
   - [Scale & Maturity](#scale--maturity)
3. [Architecture Diagrams](#architecture-diagrams)
   - [Context Diagram](#context-diagram)
   - [Container Diagram](#container-diagram)
   - [Component Diagrams](#component-diagrams)
4. [Four-Layer Architecture](#four-layer-architecture)
   - [Configuration Layer](#configuration-layer)
   - [Execution Layer](#execution-layer)
   - [Intelligence Layer](#intelligence-layer)
   - [Presentation Layer](#presentation-layer)
5. [Subsystem Deep Dives](#subsystem-deep-dives)
   - [Universe Compiler (UAC)](#universe-compiler-uac)
   - [Variable & Feature System (VFS)](#variable--feature-system-vfs)
   - [Drive As Code (DAC)](#drive-as-code-dac)
   - [Vectorized Environment](#vectorized-environment)
   - [Agent Networks](#agent-networks)
   - [Population](#population)
   - [Exploration](#exploration)
   - [Curriculum](#curriculum)
   - [Training State](#training-state)
   - [Frontend Visualization](#frontend-visualization)
6. [Key Architectural Patterns](#key-architectural-patterns)
7. [Concerns & Recommendations](#concerns--recommendations)
8. [Appendices](#appendices)
   - [A: Methodology](#appendix-a-methodology)
   - [B: Assumptions & Limitations](#appendix-b-assumptions--limitations)
   - [C: Cross-References](#appendix-c-cross-references)

---

## Executive Summary

**HAMLET (Townlet)** is a pedagogical Deep Reinforcement Learning environment designed to "trick students into learning graduate-level RL by making them think they're just playing The Sims." The system implements a **GPU-native vectorized training architecture** with **100% declarative configuration** (YAML → compiled specs) that enables researchers to conduct reproducible experiments across curriculum levels while students learn RL concepts through hands-on experimentation with reward hacking and emergent behaviors.

The architecture comprises **10 core subsystems** organized into **4 architectural layers**: (1) **Configuration Layer** (UAC, VFS, DAC) compiles YAML into immutable specs, (2) **Execution Layer** (Environment, Networks, Population) runs batched GPU training, (3) **Intelligence Layer** (Exploration, Training State, Curriculum) provides adaptive algorithms, and (4) **Presentation Layer** (Frontend) renders real-time visualization. The system processes **27,138 lines of production Python**, maintains **644+ passing tests** with 70% coverage, and supports **9 curriculum levels** from minimal temporal credit assignment (L0) to full POMDP with LSTM (L2).

**Key innovations** include: (1) **Drive As Code (DAC)** - 100% of reward logic in YAML with zero Python reward classes, (2) **Seven-stage Universe Compiler** - parse → symbol → resolve → validate → metadata → optimize → emit with provenance tracking, (3) **Transfer Learning Foundation** - constant 29-dim observation space across curriculum enables checkpoint reuse, (4) **Pre-Release Freedom** - zero backwards compatibility enables aggressive refactoring without technical debt. The system is production-ready for research and pedagogy, with minor concerns around config hardcoding in the frontend and incomplete prioritized replay for recurrent networks.

---

## System Overview

### Mission & Philosophy

HAMLET's **pedagogical mission** fundamentally shapes its architecture. The system deliberately preserves "interesting failures" (like the "Low Energy Delirium" bug where multiplicative rewards + high intrinsic weight cause reward hacking) as teaching moments rather than immediately patching them. This philosophy manifests in:

- **Curriculum progression** (L0→L3) that gradually introduces RL complexity (temporal credit assignment → multi-resource management → POMDP)
- **Declarative reward functions** (Drive As Code) that students can A/B test without writing Python
- **Real-time visualization** showing emergent behaviors (agents getting stuck in local optima, exploration-exploitation trade-offs)
- **Explicit observation specs** via VFS that make state space design decisions visible and modifiable

The **"No-Defaults Principle"** enforces that all behavioral parameters must be explicitly specified in YAML configs. This eliminates hidden assumptions, improves reproducibility, and teaches students that every hyperparameter decision matters.

### Technical Foundation

**Core Technologies:**
- **PyTorch 2.9+** - GPU-native tensor operations, neural network modules (LSTM, CNN, MLP)
- **Pydantic** - Config validation and DTO layer with frozen dataclasses
- **YAML** - Human-readable declarative configuration (9 files per curriculum level)
- **Vue 3 + Vite** - Frontend with Pinia state management and WebSocket streaming
- **pytest** - 644+ tests (unit/integration/slow) with 70% coverage

**Vectorization Strategy:**
All operations use batched tensors with `[num_agents, ...]` batch dimensions. Training 100 agents in parallel on a single GPU is **100x faster** than sequential CPU execution. State (positions, meters, observations, rewards, Q-values) flows through PyTorch tensors exclusively - no Python loops in hot paths.

**Configuration Pipeline:**
```
YAML Configs → UAC (7 stages) → CompiledUniverse (immutable) → Runtime Execution
```

The Universe Compiler (UAC) transforms human-authored YAML into frozen dataclass artifacts with comprehensive validation, metadata generation (observation dimensions, action spaces), and optimization data (lookup tensors for meter/affordance indices). Checkpoints include provenance hashes (`drive_hash`, `git_sha`, `compiler_version`) to prevent config drift.

### Scale & Maturity

**Codebase Metrics:**
- **27,138 lines** of Python across 10 subsystems in `src/townlet/`
- **644+ tests** passing with 70% code coverage
- **9 curriculum levels** (L0_0_minimal through L3_temporal_mechanics + experimental variants)
- **3,000 lines** of Vue.js frontend components
- **50+ documentation files** in `docs/` (architecture, guides, config schemas)

**Pre-Release Status:**
The project has **zero users and zero downloads**, enabling aggressive breaking changes without backwards compatibility. Legacy `src/hamlet/` code has been fully removed. All development focuses exclusively on `src/townlet/`. This freedom eliminates technical debt from deprecation patterns and dual code paths.

**Production Readiness:**
Despite pre-release status, the system demonstrates production-quality engineering:
- Comprehensive validation at compile-time (UAC) and runtime (Pydantic DTOs)
- GPU memory management with explicit device handling
- Checkpoint serialization/deserialization with state restoration parity
- CI/CD integration via `.github/workflows/config-validation.yml`
- Type hints throughout with mypy compliance

---

## Architecture Diagrams

### Context Diagram

**External Actors:**
- **Researchers** - Run training scripts, load checkpoints, analyze telemetry
- **Students** - Experiment with reward functions, observe emergent behaviors via frontend
- **Operators** - Author YAML config packs, validate compilations, manage checkpoints

**External Systems:**
- **PyTorch** - GPU tensor operations and neural network training (torch.nn/optim)
- **WebSocket Client** - Frontend visualization consuming real-time simulation state at 5-10 Hz

See [03-diagrams.md#context-diagram](./03-diagrams.md#context-diagram-c4-level-1) for full C4 Context diagram.

### Container Diagram

The system decomposes into **10 containers** organized across **4 architectural layers**:

#### Configuration Layer (Declarative Compilation)
- **Universe Compiler (UAC)** - 7-stage YAML→CompiledUniverse pipeline
- **Variable & Feature System (VFS)** - State space configuration with access control
- **Drive As Code (DAC)** - Reward function compiler (extrinsic/intrinsic/shaping)

#### Execution Layer (GPU-Native Training)
- **Vectorized Environment** - Batched simulation with DACEngine/AffordanceEngine/MeterDynamics
- **Agent Networks** - Q-network architectures (SimpleQNetwork, RecurrentSpatialQNetwork)
- **Population** - Training coordinator managing networks, replay buffer, optimizer

#### Intelligence Layer (Adaptive Algorithms)
- **Exploration** - Action selection strategies (epsilon-greedy, RND, adaptive)
- **Training State** - DTOs and GPU tensors (BatchedAgentState, ReplayBuffer)
- **Curriculum** - Difficulty progression (static/adversarial with PerformanceTracker)

#### Presentation Layer (Visualization)
- **Frontend Visualization** - Vue 3 components (Grid.vue, AspatialView.vue, MeterPanel.vue)

**Cross-Cutting Concerns:**
- **GPU Tensor Flow** - All subsystems use `[num_agents, ...]` batch dimensions
- **Checkpoint Provenance** - `drive_hash`, `git_sha`, `compiler_version` tracking
- **No-Defaults Principle** - All parameters explicit in YAML configs

See [03-diagrams.md#container-diagram](./03-diagrams.md#container-diagram-c4-level-2) for full C4 Container diagram with dependency arrows.

### Component Diagrams

Three subsystems receive detailed component-level analysis:

**UAC (Universe Compiler):** 7-stage pipeline (Parse → Symbol Table → Resolve → Cross-Validate → Metadata → Optimize → Emit) with cache manager, VFS integration, and cues compiler. Demonstrates **compiler pattern** for declarative configuration.

**Vectorized Environment:** 8-phase execution flow (Substrate → AffordanceEngine → MeterDynamics → CascadeEngine → VFS Registry → ObservationBuilder → DACEngine → Orchestrator). Demonstrates **GPU vectorization** and **state machine** patterns.

**DAC (Drive As Code):** Two-phase architecture (Compile-time: YAML → closure factories; Runtime: execute closures → rewards). Demonstrates **closure factory** pattern for variable capture and **modifier chaining** for context-sensitive reward modulation.

See [03-diagrams.md#component-diagrams](./03-diagrams.md#component-diagrams-c4-level-3) for full C4 Component diagrams.

---

## Four-Layer Architecture

### Configuration Layer

**Purpose:** Transform human-authored YAML into machine-optimized immutable specifications with comprehensive validation.

**Subsystems:**
- **Universe Compiler (UAC)** - 7-stage pipeline ensuring config correctness
- **Variable & Feature System (VFS)** - State space schema with access control
- **Drive As Code (DAC)** - Pure YAML reward functions with GPU compilation

**Key Pattern: Compile-Time Validation**
All configuration errors surface at compile time, not runtime. The UAC's cross-validation stage checks:
- Affordance costs reference valid meters
- Cascade graph has no cycles
- Action space matches substrate capabilities
- Observation dimensions consistent with VFS spec
- Economic balance (job payment sustains survival costs)

**Integration Points:**
```
YAML Configs → UAC.compile() → CompiledUniverse
                ↓
          VFS schemas, DAC specs, action metadata
                ↓
          Environment initialization
```

**Benefits:**
- Config errors fail fast with actionable error messages
- No runtime surprises from typos or invalid references
- Reproducible experiments via provenance hashing
- A/B testing without code changes (edit YAML, recompile, run)

### Execution Layer

**Purpose:** Run GPU-accelerated batched training with vectorized environment steps and neural network forward/backward passes.

**Subsystems:**
- **Vectorized Environment** - Orchestrates simulation state machine
- **Agent Networks** - Q-function approximation (MLP for full obs, LSTM for POMDP)
- **Population** - Coordinates training loop (step → store → sample → train → update)

**Key Pattern: Batched Vectorization**
All operations process `[num_agents, ...]` tensors in parallel:
```python
# Environment step (vectorized across num_agents)
observations, rewards, dones, info = env.step(actions)  # [num_agents, obs_dim], [num_agents], [num_agents]

# Network forward pass (vectorized)
q_values = q_network(observations)  # [num_agents, action_dim]

# Action selection (vectorized)
actions = exploration.select_actions(q_values, epsilon, action_masks)  # [num_agents]
```

**Hot Path Optimization:**
- Zero Python loops - all iteration via PyTorch tensor ops
- Lazy compilation (DACEngine compiled once, executed millions of times)
- Target network hard sync (periodic copy avoids per-step overhead)
- Gradient clipping (max_norm=10.0) prevents exploding gradients

**Training Cycle:**
```
Population.step_population():
  1. Curriculum.get_batch_decisions() → difficulty levels
  2. Exploration.select_actions(q_values) → action indices
  3. Environment.step(actions) → next_obs, rewards, dones
  4. ReplayBuffer.push(transition) → experience storage
  5. ReplayBuffer.sample(batch_size) → training batch
  6. Loss computation + backward pass + optimizer step
  7. Target network sync (if global_step % target_update_freq == 0)
```

### Intelligence Layer

**Purpose:** Provide adaptive algorithms for exploration-exploitation balance and curriculum difficulty modulation.

**Subsystems:**
- **Exploration** - Intrinsic motivation and action selection (RND, adaptive, epsilon-greedy)
- **Training State** - Experience storage and checkpoint serialization
- **Curriculum** - Difficulty progression based on performance metrics

**Key Pattern: Strategy Pattern**
All three subsystems use pluggable implementations:
- `ExplorationStrategy` → EpsilonGreedy | RND | AdaptiveIntrinsic
- `CurriculumManager` → Static | Adversarial
- `ReplayBuffer` → Standard | Sequential (for LSTM) | Prioritized (feedforward only)

**Adaptive Mechanisms:**

**1. RND Exploration (Random Network Distillation):**
```
Fixed network (frozen) + Predictor network (trained)
Intrinsic reward = || fixed(obs) - predictor(obs) ||²
High error → novel state → exploration bonus
```

**2. Adaptive Intrinsic Annealing:**
```
If survival_variance < 100.0 AND mean_survival > 50:
    intrinsic_weight *= 0.5  # Reduce exploration, increase exploitation
```

**3. Adversarial Curriculum:**
```
5 stages: Basic (20% depletion) → Full complexity (100% depletion)
Advancement: high_survival AND positive_learning AND low_entropy
Retreat: low_survival OR negative_learning
```

**Coordination:**
Curriculum modulates environment difficulty → affects survival → triggers exploration annealing → changes action selection → impacts learning progress → feeds back to curriculum.

### Presentation Layer

**Purpose:** Real-time visualization of agent behavior for debugging, pedagogy, and analysis.

**Subsystems:**
- **Frontend Visualization** - Vue 3 components with WebSocket streaming

**Key Pattern: Substrate-Aware Rendering**
```
if substrate.position_dim > 0:
    render Grid.vue (SVG spatial grid with heat maps, trails)
else:
    render AspatialView.vue (meter dashboard, affordance grid)
```

**WebSocket Architecture:**
```
Inference Server (port 8766) → broadcasts at 5-10 Hz
    ↓ JSON messages
Frontend Store (Pinia) → reactive state
    ↓ props
Vue Components → SVG/HTML rendering
```

**Visualization Features:**
- **Grid.vue:** Agent positions, affordances, RND novelty heat maps, movement trails
- **MeterPanel.vue:** Hierarchical meters (Primary: survival-critical, Secondary: modifiers, Tertiary: accelerators) with cascade annotations
- **AspatialView.vue:** Pure resource management dashboard for position-free substrates

**Concerns:**
Frontend contains hardcoded configuration (action names, meter relationships, color gradients) that should derive from `CompiledUniverse` metadata. This creates tight coupling to specific game configurations and risks divergence from backend state.

---

## Subsystem Deep Dives

### Universe Compiler (UAC)

**Location:** `src/townlet/universe/` (4,349 LOC)

**Responsibility:** Seven-stage configuration compilation pipeline transforming YAML config packs into `CompiledUniverse` artifacts with validation, metadata generation, and runtime optimization.

**Seven-Stage Pipeline:**

1. **Parse** - Load 9 YAML files (substrate, bars, cascades, affordances, cues, training, drive_as_code, variables_reference, brain) with syntax validation
2. **Symbol Table** - Build central registry resolving references across bars, affordances, cascades, variables
3. **Resolve** - Link symbolic references (e.g., affordance cost "energy" → meter index 0)
4. **Cross-Validate** - Check semantic constraints (cascade cycles, action space compatibility, POMDP validation)
5. **Metadata** - Generate `ObservationSpec`, `ActionSpaceMetadata`, `MeterMetadata` for runtime
6. **Optimize** - Create lookup tensors (meter indices, affordance positions) for GPU efficiency
7. **Emit** - Freeze `CompiledUniverse` dataclass with provenance tracking

**Caching Strategy:**
```
Cache key = config_dir path
Invalidation: mtime check (fast) → content hash (accurate)
Cache hit: Skip stages 1-7, load CompiledUniverse from disk
```

**Validation Examples:**
- **Affordance costs:** "REST: energy=-5" checks energy meter exists and cost is negative (replenishes)
- **Cascade cycles:** Detect A→B→C→A meter dependencies and reject
- **Economic balance:** Verify job payment (e.g., $22.5) sustains survival costs over typical episode
- **POMDP compatibility:** Grid3D with vision_range=2 allowed; GridND (N≥4) rejected (window too large)

**Provenance Tracking:**
```python
CompiledUniverse(
    drive_hash="sha256:abc123...",  # DAC config hash
    compiler_version="1.2.3",        # UAC version
    git_sha="def456...",             # Git commit
    # ... compiled specs
)
```

**Integration:**
Every training run, inference server, and test fixture calls `UniverseCompiler.compile(config_dir)`. Checkpoints embed `drive_hash` to prevent loading weights trained with mismatched reward functions.

**Concerns:**
- **File size:** 2,600+ lines with 25+ validation methods (complexity risk)
- **Cache invalidation:** Dual checks (mtime + hash) add latency
- **Symbol resolution:** Linear graph traversal for cascades (O(n²) in heavily cascaded systems)

**Confidence:** High - Well-structured pipeline with clear stage separation, comprehensive error handling, mature design.

---

### Variable & Feature System (VFS)

**Location:** `src/townlet/vfs/` (~1,200 LOC)

**Responsibility:** Declarative state space configuration providing compile-time observation spec generation and runtime access-controlled storage for VFS variables with GPU tensor backends.

**Core Components:**

**1. VariableDef (Schema):**
```yaml
# variables_reference.yaml
variables:
  - name: energy
    scope: agent          # global | agent | agent_private
    type: scalar
    readers: [agent, engine, acs]
    writers: [engine, actions]
    normalization: {min: 0.0, max: 100.0}
```

**2. VariableRegistry (Runtime):**
```python
registry.set("energy", energy_tensor, writer="engine")  # [num_agents]
energy = registry.get("energy", reader="agent")         # [num_agents]
```

**3. VFSObservationSpecBuilder (Compile-Time):**
```python
# UAC Stage 5: Metadata
spec = VFSObservationSpecBuilder.build(variable_defs)
# → ObservationField(name="energy", semantic_type="bars", dim=1, normalization=...)
```

**Access Control Matrix:**

| Role | Can Read | Can Write |
|------|----------|-----------|
| agent | agent, global | none |
| engine | agent, global, agent_private | agent, global |
| actions | agent, global | agent |
| acs (adversarial curriculum) | agent, global | agent (curriculum_active flag) |
| bac (brain-as-code) | all | none |

**Patterns:**
- **Schema-First Validation:** Pydantic enforces all constraints before runtime
- **Defensive Copying:** `registry.get()` clones tensors to prevent aliasing bugs
- **Scope-Aware Shaping:** Registry auto-computes tensor shapes (global: `[]`, agent: `[num_agents]`)
- **Semantic Grouping:** `ObservationField.semantic_type` categorizes observations (bars, spatial, affordance, temporal, custom)

**Integration:**
UAC Stage 5 calls `VFSObservationSpecBuilder` → generates `ObservationSpec` → Environment uses spec to validate observation dimensions → Runtime registry stores actual tensor values.

**Concerns:** None observed - clean design with explicit access control.

**Confidence:** High

---

### Drive As Code (DAC)

**Location:** `src/townlet/environment/dac_engine.py` + `configs/*/drive_as_code.yaml`

**Responsibility:** Compile declarative YAML reward configurations into GPU-native computation graphs with extrinsic/intrinsic/shaping composition and context-sensitive modifier chaining.

**Reward Formula:**
```
total_reward = extrinsic + (intrinsic × effective_intrinsic_weight) + shaping

where:
    effective_intrinsic_weight = base_weight × modifier₁ × modifier₂ × ...
```

**Strategy Types:**

**Extrinsic (9 types):**
- `multiplicative` - `base × bar₁ × bar₂` (compound survival, prone to "Low Energy Delirium")
- `constant_base_with_shaped_bonus` - `base + Σ(bonuses)` (fixes delirium bug)
- `additive_unweighted`, `weighted_sum`, `polynomial`, `threshold_based`, `aggregation`, `vfs_variable`, `hybrid`

**Intrinsic (5 types):**
- `rnd` - Random Network Distillation (novelty-seeking)
- `icm` - Intrinsic Curiosity Module
- `count_based` - Pseudo-count bonuses
- `adaptive_rnd` - RND with performance-based annealing
- `none` - Pure extrinsic (ablation studies)

**Shaping (11 types):**
- `approach_reward`, `completion_bonus`, `efficiency_bonus`, `state_achievement`
- `streak_bonus`, `diversity_bonus`, `timing_bonus`, `economic_efficiency`
- `balance_bonus`, `crisis_avoidance`, `vfs_variable`

**Modifiers (Context-Sensitive):**
```yaml
modifiers:
  energy_crisis:
    target: intrinsic
    source: meters
    meter: energy
    ranges:
      - {min: 0.0, max: 0.2, multiplier: 0.0}  # Suppress intrinsic when energy < 20%
      - {min: 0.2, max: 1.0, multiplier: 1.0}  # Normal intrinsic otherwise
```

**Compilation Pattern (Closure Factories):**
```python
# Compile-time (DACEngine.__init__)
def create_modifier_fn(config):
    # Capture config variables
    meter_idx = config.meter_idx
    ranges = config.ranges

    def modifier_fn(meters, vfs_registry):
        # Use captured config
        meter_val = meters[:, meter_idx]
        return torch.where(meter_val < 0.2, 0.0, 1.0)  # GPU vectorized

    return modifier_fn

self.modifier_fns = [create_modifier_fn(cfg) for cfg in modifier_configs]

# Runtime (calculate_rewards)
intrinsic_weight = base_weight
for modifier_fn in self.modifier_fns:
    intrinsic_weight = intrinsic_weight * modifier_fn(meters, vfs_registry)
```

**Pedagogical Example - "Low Energy Delirium":**

**Bug Config (L0_0_minimal - demonstrates reward hacking):**
```yaml
extrinsic:
  type: multiplicative
  base: 1.0
  bars: [energy]  # reward = 1.0 × energy

intrinsic:
  base_weight: 0.1
  apply_modifiers: []  # No crisis suppression
```

**Emergent Behavior:** Agents learn to keep energy LOW (0.1-0.3) because:
- Extrinsic reward: 1.0 × 0.2 = 0.2
- Intrinsic reward: high (unexplored low-energy states) × 0.1 = 0.3
- **Total: 0.5** (better than high energy with low novelty)

**Fix Config (L0_5_dual_resource - teaches reward design):**
```yaml
extrinsic:
  type: constant_base_with_shaped_bonus
  base: 0.1  # Constant baseline
  bonuses: [{type: state_achievement, meter: energy, threshold: 0.8}]

modifiers:
  energy_crisis: {ranges: [{min: 0.0, max: 0.2, multiplier: 0.0}]}  # Suppress intrinsic when low

intrinsic:
  apply_modifiers: [energy_crisis]
```

**Result:** Agents learn proper survival (keep energy high to avoid intrinsic suppression).

**Concerns:** None observed - design is clean with explicit compile/runtime separation.

**Confidence:** High

---

### Vectorized Environment

**Location:** `src/townlet/environment/` (4,940 LOC)

**Responsibility:** Execute GPU-native batched simulation of multiple independent Hamlet environments with centralized reward computation and state management.

**Execution Flow (8 Phases per Step):**

1. **Substrate** - Process movement actions with boundary handling (clamp/wrap/bounce)
2. **AffordanceEngine** - Execute INTERACT actions (multi-tick temporal mechanics, costs, effects)
3. **MeterDynamics** - Apply meter depletion (configurable per tick, day/night modulation)
4. **CascadeEngine** - Propagate meter relationships (low energy → fatigue)
5. **VFS Registry** - Write state to VFS (position, meters, affordances, velocity, temporal)
6. **ObservationBuilder** - Construct observations from VFS spec (29-dim for Grid2D, 54-dim for POMDP)
7. **DACEngine** - Calculate rewards (extrinsic + intrinsic + shaping)
8. **Orchestrator** - Return (obs, rewards, dones, info) to Population

**Key Tensors (All [num_agents, ...]):**
```python
self.positions: [num_agents, position_dim]  # Agent positions
self.meters: [num_agents, num_meters]       # Energy, health, etc.
self.observations: [num_agents, obs_dim]    # Network input
self.rewards: [num_agents]                  # Total reward
self.dones: [num_agents]                    # Terminal flags
```

**POMDP Support (Partial Observability):**
```python
# Full observability (L1)
obs_dim = 29  # 2 coords + 8 meters + 15 affordances + 4 temporal

# POMDP (L2) - 5×5 local window
obs_dim = 54  # 25 local cells + 2 coords + 8 meters + 15 affordances + 4 temporal
```

**Validation at Init:**
- POMDP only supported for Grid2D, Grid3D (vision_range ≤ 2), Aspatial
- Continuous substrates: No POMDP (window concept undefined)
- GridND (N≥4): No POMDP (window explodes: 5⁴ = 625 cells, 5⁷ = 78,125 cells)

**Integration:**
```python
# Population training loop
observations, rewards, dones, info = env.step(actions)
```

**Concerns:**
- POMDP validation at init (not config time) - errors surface late
- Multiple affordance tracking mechanisms (coupling risk if logic diverges)
- Observation dim validation against metadata (mismatch suggests compiler inconsistency)

**Confidence:** High - Core architectural component, thoroughly instrumented.

---

### Agent Networks

**Location:** `src/townlet/agent/` (1,389 LOC)

**Responsibility:** Q-network architectures for value function approximation across observability conditions.

**Network Types:**

**1. SimpleQNetwork (Full Observability - L0, L0.5, L1):**
```python
MLP: obs_dim → 256 → 128 → action_dim
Parameters: ~26K
Architecture: Linear → LayerNorm → ReLU → Linear → LayerNorm → ReLU → Linear
```

**2. RecurrentSpatialQNetwork (Partial Observability - L2, L3):**
```python
Vision Encoder:  5×5 local → Conv2D(16) → Conv2D(32) → Flatten → Linear(128)
Position Encoder: (x, y) → Linear(32)
Meter Encoder: 8 meters → Linear(32)
Affordance Encoder: 15 affordances → Linear(32)
LSTM: 192 input → 256 hidden (memory for POMDP)
Q-head: 256 → 128 → action_dim
Parameters: ~650K
```

**3. DuelingQNetwork (Unused in current curriculum):**
```python
Value stream: shared → value head
Advantage stream: shared → advantage head
Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))
```

**4. StructuredQNetwork (Semantic Grouping):**
```python
Groups observations by semantic_type (bars, spatial, affordance, temporal)
Separate encoders per group → concatenate → Q-head
```

**LSTM State Lifecycle:**
```python
# Episode start
network.reset_hidden_state(batch_size)  # Zero LSTM state

# Rollout
for step in episode:
    q_values = network(obs)  # Hidden state persists

# Batch training
for batch in replay_buffer:
    network.reset_hidden_state()  # Fresh state per transition
```

**Transfer Learning Foundation:**
All Grid2D configs use **identical 29→8 architecture** regardless of grid size (3×3 vs 8×8). Observation encoding normalizes to [0,1], making dimension constant. This enables:
- Checkpoint trained on L0 (3×3) → load on L1 (8×8)
- Curriculum progression without retraining from scratch

**Concerns:**
- Temporal features "ignored" (comment in code) - potential signal loss
- DuelingQNetwork exists but unused (unclear if intended for future curriculum)
- StructuredQNetwork tight coupling to ObservationActivity schema
- LSTM hidden state defaults to CPU device (device mismatch risk)

**Confidence:** High - Well-documented, explicit patterns, traceable dependencies.

---

### Population

**Location:** `src/townlet/population/` (1,358 LOC)

**Responsibility:** Coordinate batched training of parallel agents with shared curriculum and exploration strategies, managing Q-networks, target networks, and replay buffers.

**Training Loop (step_population):**

```python
def step_population(self, num_steps):
    for _ in range(num_steps):
        # 1. Curriculum decisions (once per episode)
        if episode_start:
            curriculum_decisions = self.curriculum.get_batch_decisions_with_qvalues(
                agent_state, q_values
            )

        # 2. Action selection (exploration strategy)
        with torch.no_grad():
            q_values = self.q_network(observations)  # [num_agents, action_dim]
        actions = self.exploration.select_actions(
            q_values, epsilon, action_masks
        )  # [num_agents]

        # 3. Environment step
        next_obs, rewards, dones, info = self.env.step(actions)

        # 4. Store transition
        if recurrent:
            episode_containers[agent_id].append(transition)  # Accumulate
            if done:
                self.replay_buffer.push_episode(episode_containers[agent_id])
        else:
            self.replay_buffer.push(transition)  # Immediate

        # 5. Training (if buffer ready)
        if len(replay_buffer) >= batch_size:
            batch = replay_buffer.sample(batch_size)

            # Double DQN or Vanilla DQN
            if use_double_dqn:
                # Action selection: online network
                next_actions = self.q_network(next_obs).argmax(dim=1)
                # Action evaluation: target network
                next_q = self.target_network(next_obs).gather(1, next_actions)
            else:
                # Both selection and evaluation: target network
                next_q = self.target_network(next_obs).max(dim=1)[0]

            # TD target
            target = rewards + gamma * next_q * (1 - dones)

            # Loss + backprop
            loss = F.mse_loss(q_values, target)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
            self.optimizer.step()

        # 6. Target network sync (periodic hard copy)
        if global_step % target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # 7. RND predictor training (if adaptive exploration)
        if isinstance(exploration, RNDExploration):
            exploration.update(batch)
```

**Dual Network Stability:**
- **Online network** - Trained every step, used for action selection (if Double DQN)
- **Target network** - Frozen for N steps, used for TD target evaluation
- **Hard sync** - Periodic copy (every `target_update_frequency` steps, typically 1000)

**Replay Buffer Variants:**
- **ReplayBuffer** - Feedforward networks, per-transition storage, random sampling
- **SequentialReplayBuffer** - Recurrent networks, full episode storage, maintains temporal order
- **PrioritizedReplayBuffer** - TD-error prioritization, importance sampling (feedforward only)

**BrainConfig Integration:**
```yaml
# brain.yaml
architecture:
  type: recurrent  # feedforward | recurrent | dueling
  hidden_dim: 256
network:
  learning_rate: 0.0001
  optimizer: adam
replay:
  capacity: 100000
  prioritized: false  # Not yet supported for recurrent
```

**Concerns:**
- Prioritized replay not supported for recurrent networks (NotImplementedError)
- BrainConfig architecture types require if/elif chains (not plugin system)
- Hidden dimensions have TODO(BRAIN_AS_CODE) comments (incomplete config-driven design)
- Episode containers (recurrent mode) are CPU Python lists (GPU transfer at store time)

**Confidence:** High - Production-critical path, batched patterns verified, concerns are documented as incomplete phases.

---

### Exploration

**Location:** `src/townlet/exploration/` (~1,500 LOC)

**Responsibility:** Implement action selection strategies and compute intrinsic motivation rewards for exploration-exploitation trade-off.

**Strategy Types:**

**1. EpsilonGreedyExploration:**
```python
if random() < epsilon:
    action = uniform_random(valid_actions)  # Explore
else:
    action = argmax(q_values)  # Exploit

epsilon *= decay_rate  # Exponential annealing
epsilon = max(epsilon, epsilon_min)  # Floor
```

**2. RNDExploration (Random Network Distillation):**
```python
# Two networks
fixed_network = MLP(obs_dim → 128)  # Frozen
predictor_network = MLP(obs_dim → 128)  # Trained

# Intrinsic reward = prediction error
intrinsic = || fixed_network(obs) - predictor_network(obs) ||²

# Welford online normalization
intrinsic_normalized = (intrinsic - running_mean) / sqrt(running_variance + 1e-8)
```

**3. AdaptiveIntrinsicExploration (RND + Annealing):**
```python
# Wrap RNDExploration
rnd = RNDExploration(...)

# Track survival variance
survival_history.append(mean_survival)
variance = var(survival_history)

# Anneal if "consistently succeeding"
if variance < 100.0 AND mean_survival > 50.0:
    intrinsic_weight *= 0.5  # Reduce exploration
```

**Action Selection (Vectorized):**
```python
def epsilon_greedy_action_selection(q_values, epsilon, action_masks):
    # q_values: [num_agents, action_dim]
    # epsilon: [num_agents] (per-agent epsilon)
    # action_masks: [num_agents, action_dim] (1=valid, 0=invalid)

    # Random actions
    probs = action_masks.float() / action_masks.sum(dim=1, keepdim=True)
    random_actions = torch.multinomial(probs, num_samples=1).squeeze()

    # Greedy actions (mask invalid)
    masked_q = q_values.clone()
    masked_q[~action_masks] = -inf
    greedy_actions = masked_q.argmax(dim=1)

    # Mix per-agent
    explore_mask = torch.rand(num_agents) < epsilon
    actions = torch.where(explore_mask, random_actions, greedy_actions)

    return actions  # [num_agents]
```

**Checkpoint State:**
```python
{
    "predictor_state_dict": predictor.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "epsilon": 0.05,
    "intrinsic_weight": 0.025,
    "running_mean": 1.23,
    "running_variance": 0.45,
    "count": 50000
}
```

**Concerns:**
- Adaptive annealing thresholds (variance_threshold=100.0, min_survival_fraction=0.4) not exposed in curriculum configs
- RND normalization stats reset may cause reward scale discontinuities
- Action masking adds 1e-8 defensively (subtle numerical bias risk)
- No formal convergence guarantee for adaptive annealing

**Confidence:** High - Clear interface contracts, GPU-optimized vectorization, comprehensive checkpointing.

---

### Curriculum

**Location:** `src/townlet/curriculum/` (~600 LOC)

**Responsibility:** Configurable difficulty progression strategies through per-agent environment adaptation.

**Strategy Types:**

**1. StaticCurriculum:**
```python
# Fixed difficulty, identical for all agents
decisions = CurriculumDecision(
    difficulty_level=0.8,
    active_meters=["energy", "health", "mood"],
    depletion_multiplier=1.0,
    reward_mode="sparse"
)
```

**2. AdversarialCurriculum:**
```python
# 5-stage progression
STAGE_CONFIGS = [
    Stage 1: {active_meters: [energy, health], depletion: 0.2, reward: shaped},
    Stage 2: {active_meters: [energy, health, mood], depletion: 0.4, reward: shaped},
    Stage 3: {active_meters: [energy, health, mood, satiation], depletion: 0.6, reward: mixed},
    Stage 4: {active_meters: all except social, depletion: 0.8, reward: mixed},
    Stage 5: {active_meters: all, depletion: 1.0, reward: sparse}
]

# Advancement logic (once per episode)
if survival_rate > 0.7 AND learning_progress > 0.0 AND entropy < 0.5:
    stage[agent] += 1  # Advance to harder stage
elif survival_rate < 0.3 OR learning_progress < -0.1:
    stage[agent] -= 1  # Retreat to easier stage
```

**Performance Tracking (GPU Tensors):**
```python
class PerformanceTracker:
    survival_counts: [num_agents]  # Consecutive survival episodes
    total_rewards: [num_agents]    # Cumulative rewards
    learning_progress: [num_agents]  # Reward delta vs baseline
    stage_tenure: [num_agents]     # Steps at current stage
```

**Entropy Calculation (Policy Convergence):**
```python
# From Q-values
probs = softmax(q_values)  # [num_agents, action_dim]
entropy = -sum(probs * log(probs))  # [num_agents]
entropy_normalized = entropy / log(num_actions)  # [0, 1]

# Low entropy → converged policy → ready for harder stage
```

**Integration:**
```python
# Population calls once per episode
curriculum_decisions = curriculum.get_batch_decisions_with_qvalues(
    agent_state, q_values
)
# → [num_agents] CurriculumDecision objects

# Environment applies difficulty
env.apply_curriculum_decisions(curriculum_decisions)
# → Modulates depletion_multiplier, activates/deactivates meters
```

**Concerns:**
- `update_step()` parameter named "rewards" but receives step counts (semantic confusion)
- STAGE_CONFIGS hardcoded (not parameterizable via YAML)
- Entropy calculation assumes fixed action space (may fail with substrate variations)
- Retreat uses OR (single failure triggers), advancement uses AND (all metrics required) - asymmetric

**Confidence:** High - Clear separation between static/adversarial, documented state management, explicit tensor-based GPU integration.

---

### Training State

**Location:** `src/townlet/training/` (~900 LOC)

**Responsibility:** Cold path DTOs for configuration and hot path GPU tensors for batched agent state and experience storage.

**Cold Path (Configuration):**
```python
@dataclass(frozen=True)
class CurriculumDecision:
    difficulty_level: float  # [0.0, 1.0]
    active_meters: list[str]
    depletion_multiplier: float
    reward_mode: str  # "shaped" | "mixed" | "sparse"

@dataclass(frozen=True)
class PopulationCheckpoint:
    generation: int
    agent_ids: list[str]
    curriculum_states: dict[str, Any]  # Loose typing for extensibility
    exploration_states: dict[str, Any]
    pareto_frontier: list[dict]
```

**Hot Path (GPU Tensors):**
```python
class BatchedAgentState:
    __slots__ = ['observations', 'actions', 'rewards', 'dones', 'info']

    observations: Tensor  # [batch, obs_dim]
    actions: Tensor       # [batch]
    rewards: Tensor       # [batch]
    dones: Tensor         # [batch]
    info: dict[str, Tensor]  # Additional data

class ReplayBuffer:
    storage: dict[str, Tensor]  # Lazy init on first push
    capacity: int
    position: int  # Circular buffer pointer

    def push(self, obs, action, reward, next_obs, done):
        idx = self.position % self.capacity
        self.storage['observations'][idx] = obs
        # ... store other tensors
        self.position += 1

    def sample(self, batch_size):
        indices = randint(0, len(self), size=batch_size)
        batch = {k: v[indices] for k, v in self.storage.items()}
        return batch
```

**Dual Reward Tracking:**
```python
# ReplayBuffer stores separately
self.storage['extrinsic_rewards']: [capacity]
self.storage['intrinsic_rewards']: [capacity]

# Sampling applies intrinsic weight
def sample(self, batch_size, intrinsic_weight=0.1):
    batch['rewards'] = (
        batch['extrinsic_rewards'] +
        intrinsic_weight * batch['intrinsic_rewards']
    )
    return batch
```

**Serialization (Checkpointing):**
```python
# Save
checkpoint_data = {
    'storage': {k: v.cpu().numpy() for k, v in buffer.storage.items()},
    'position': buffer.position,
    'capacity': buffer.capacity
}
torch.save(checkpoint_data, path)

# Load
data = torch.load(path)
buffer.storage = {k: torch.from_numpy(v).to(device) for k, v in data['storage'].items()}
buffer.position = data['position']
```

**Concerns:**
- Lazy initialization may cause device mismatches (buffer on CPU, first push on GPU)
- `PopulationCheckpoint` uses `dict[str, Any]` (loose typing sacrifices validation)
- `BatchedAgentState.to()` doesn't validate device consistency of info dict
- No prioritized replay variants in state.py (handled elsewhere in Population)

**Confidence:** High - Clear cold/hot path separation, comprehensive docstrings, explicit device handling.

---

### Frontend Visualization

**Location:** `frontend/src/components/` (~3,000 LOC JS)

**Responsibility:** Render real-time agent simulation state with substrate-aware layouts and WebSocket streaming.

**Component Architecture:**

**1. Grid.vue (Spatial Substrates):**
```vue
<template>
  <svg :viewBox="`0 0 ${width} ${height}`">
    <!-- Layer 1: Grid cells -->
    <g class="cells">
      <rect v-for="(cell, i) in cells" :x="cell.x" :y="cell.y" />
    </g>

    <!-- Layer 2: Heat map (RND novelty) -->
    <g v-if="showHeatMap" class="heatmap">
      <rect v-for="(cell, i) in heatData" :fill="getHeatColor(cell.intensity)" />
    </g>

    <!-- Layer 3: Affordances -->
    <g class="affordances">
      <circle v-for="aff in affordances" :cx="aff.x" :cy="aff.y" />
      <text>{{ AFFORDANCE_ICONS[aff.type] }}</text>
    </g>

    <!-- Layer 4: Agent trails (fading opacity) -->
    <g class="trails">
      <line v-for="(trail, i) in agentTrails" :opacity="1.0 - i/maxTrailLength" />
    </g>

    <!-- Layer 5: Agents -->
    <g class="agents">
      <circle :cx="agent.x" :cy="agent.y" :fill="getAgentColor(agent)" />
    </g>
  </svg>
</template>

<script setup>
const props = defineProps({
  gridSize: Number,
  agents: Array,
  affordances: Array,
  heatMap: Array
})

// Keyboard shortcut: H toggles heat map
onMounted(() => {
  window.addEventListener('keydown', (e) => {
    if (e.key === 'h' && !isInputFocused()) showHeatMap.value = !showHeatMap.value
  })
})
</script>
```

**2. AspatialView.vue (Non-Spatial Substrates):**
```vue
<template>
  <div class="aspatial-container">
    <!-- Primary meters (survival-critical) -->
    <section class="meters-primary">
      <MeterDisplay v-for="meter in primaryMeters" :meter="meter" size="large" />
    </section>

    <!-- Affordance grid -->
    <section class="affordances-grid">
      <div v-for="aff in affordances" class="affordance-card">
        <span class="icon">{{ AFFORDANCE_ICONS[aff.type] }}</span>
        <span class="name">{{ aff.name }}</span>
        <span class="cost">{{ formatCost(aff.cost) }}</span>
      </div>
    </section>

    <!-- Action history -->
    <section class="action-log">
      <div v-for="action in recentActions" class="action-item">
        <span class="timestamp">{{ action.tick }}</span>
        <span class="action">{{ ACTION_NAMES[action.type] }}</span>
      </div>
    </section>
  </div>
</template>
```

**3. MeterPanel.vue (Hierarchical Meter Display):**
```vue
<template>
  <div class="meter-panel">
    <!-- Primary tier: Survival-critical -->
    <div class="tier-primary">
      <MeterBar v-for="meter in ['energy', 'health', 'money']"
                :value="meters[meter]"
                :class="{critical: meters[meter] < 0.2, pulse: meters[meter] < 0.2}" />
    </div>

    <!-- Secondary tier: Modifiers -->
    <div class="tier-secondary">
      <MeterBar v-for="meter in ['mood', 'satiation', 'fitness']"
                :value="meters[meter]">
        <span class="relationship">
          {{ getCascadeRelationship(meter) }}  <!-- e.g., "Low mood → loneliness" -->
        </span>
      </MeterBar>
    </div>

    <!-- Tertiary tier: Accelerators -->
    <div class="tier-tertiary">
      <MeterBar v-for="meter in ['hygiene', 'social']"
                :value="meters[meter]"
                :icon="⚡" />
    </div>
  </div>
</template>
```

**WebSocket Integration:**
```javascript
// stores/simulation.js (Pinia)
export const useSimulationStore = defineStore('simulation', {
  state: () => ({
    agents: [],
    meters: {},
    heatMap: [],
    substrateType: 'grid2d'
  }),

  actions: {
    connectWebSocket(url = 'ws://localhost:8766') {
      this.ws = new WebSocket(url)

      this.ws.onmessage = (event) => {
        const data = JSON.parse(event.data)
        this.agents = data.agents
        this.meters = data.meters
        this.heatMap = data.heatMap
      }
    }
  }
})
```

**Design Tokens:**
```css
:root {
  /* Meter colors */
  --color-meter-energy: #10b981;
  --color-meter-health: #ef4444;
  --color-meter-mood: #8b5cf6;

  /* Tier colors */
  --tier-primary: #1e40af;
  --tier-secondary: #7c3aed;
  --tier-tertiary: #059669;

  /* Animations */
  --transition-fast: 150ms;
  --transition-medium: 300ms;

  /* Spacing */
  --spacing-sm: 0.5rem;
  --spacing-md: 1rem;
  --spacing-lg: 2rem;
}
```

**Concerns:**
- **Heat map colors** hardcoded in `getHeatColor()` (should externalize to constants)
- **Action names** hardcoded as array (should derive from CompiledUniverse metadata)
- **Meter relationships** hardcoded in MeterPanel (should compute from cascade graph)
- **Color inconsistency** - mixing CSS variables with hardcoded hex values
- **No error handling** for malformed WebSocket messages or missing props
- **Keyboard shortcuts** only exclude INPUT/TEXTAREA (should extend to all form controls)
- **Agent trails** tracked client-side (potential desync with out-of-order WebSocket messages)

**Confidence:** Medium - Well-structured components with strong UX patterns, but hardcoded configuration creates tight coupling to specific game configs rather than generic infrastructure.

---

## Key Architectural Patterns

### 1. GPU-Native Vectorization

**Pattern:** All operations process batched tensors with `[num_agents, ...]` dimensions. Zero Python loops in hot paths.

**Benefits:**
- **100x speedup** - Training 100 agents in parallel on GPU faster than 1 agent on CPU
- **Memory efficiency** - Contiguous tensor storage reduces allocation overhead
- **Gradient flow** - Batched backprop leverages CUDA parallelism

**Implementation Examples:**

**Environment Step:**
```python
# Bad (Python loop)
for i in range(num_agents):
    obs[i], reward[i], done[i] = single_env_step(actions[i])

# Good (Vectorized)
observations, rewards, dones = vectorized_env.step(actions)  # All [num_agents]
```

**Action Selection:**
```python
# Bad (Python loop)
actions = []
for i in range(num_agents):
    if random() < epsilon[i]:
        actions.append(random_action())
    else:
        actions.append(q_values[i].argmax())

# Good (Vectorized)
random_actions = torch.multinomial(action_probs, 1).squeeze()  # [num_agents]
greedy_actions = q_values.argmax(dim=1)  # [num_agents]
explore_mask = torch.rand(num_agents) < epsilon  # [num_agents]
actions = torch.where(explore_mask, random_actions, greedy_actions)  # [num_agents]
```

**Reward Calculation (DAC):**
```python
# Extrinsic (multiplicative)
extrinsic = base * meters[:, energy_idx] * meters[:, health_idx]  # [num_agents]

# Intrinsic (RND)
intrinsic = (fixed(obs) - predictor(obs)).pow(2).sum(dim=1)  # [num_agents]

# Modifiers (crisis suppression)
energy_vals = meters[:, energy_idx]  # [num_agents]
multiplier = torch.where(energy_vals < 0.2, 0.0, 1.0)  # [num_agents]
intrinsic = intrinsic * multiplier  # [num_agents]
```

**Adoption Across Subsystems:**
- **Environment:** Positions, meters, observations all [num_agents, ...]
- **Networks:** Batch forward pass [num_agents, obs_dim] → [num_agents, action_dim]
- **Population:** Batched transitions, replay sampling, gradient updates
- **Exploration:** Vectorized action selection, intrinsic reward computation
- **Curriculum:** PerformanceTracker uses [num_agents] tensors for metrics

### 2. Declarative Configuration (YAML → Compiled Specs)

**Pattern:** All behavioral parameters authored in YAML, compiled by UAC into immutable frozen dataclasses, executed without runtime interpretation.

**Benefits:**
- **No code changes** - A/B test reward functions by editing YAML
- **Reproducibility** - Provenance hashes (drive_hash, git_sha) prevent config drift
- **Validation at compile time** - Errors surface before training starts
- **No-Defaults Principle** - All parameters explicit, no hidden assumptions

**Pipeline:**
```
YAML Files (9 per config pack)
    ↓ UAC Stage 1: Parse
RawConfigs (intermediate Python objects)
    ↓ UAC Stage 2-3: Symbol Table + Resolve
Resolved References (meter IDs, affordance indices)
    ↓ UAC Stage 4: Cross-Validate
Validated Constraints (cascade cycles, POMDP compatibility)
    ↓ UAC Stage 5-6: Metadata + Optimize
ObservationSpec, ActionSpaceMetadata, lookup tensors
    ↓ UAC Stage 7: Emit
CompiledUniverse (frozen dataclass, immutable)
    ↓ Runtime
Environment/Population/DAC consume specs (no YAML access)
```

**Example Compilation Flow (Affordance):**

**YAML:**
```yaml
# affordances.yaml
affordances:
  - name: REST
    cost:
      energy: -5  # Negative = replenish
    effect:
      fatigue: -10
    duration_ticks: 1
```

**Stage 2-3 (Symbol Resolution):**
```python
# Resolve "energy" → meter index 0
# Resolve "fatigue" → meter index 5
```

**Stage 4 (Cross-Validation):**
```python
# Check: energy meter exists? ✓
# Check: cost is negative (replenishes)? ✓
# Check: fatigue meter exists? ✓
# Check: duration_ticks > 0? ✓
```

**Stage 5-6 (Metadata + Optimization):**
```python
# Create lookup tensor
affordance_costs = torch.tensor([
    [0, 0, 0, 0, 0, -5, 0, 0],  # REST: energy=-5
    [3, 0, 0, 0, 0, 0, 0, 0],   # WORK: energy=3
    # ...
])  # [num_affordances, num_meters]
```

**Stage 7 (Emit):**
```python
@dataclass(frozen=True)
class CompiledUniverse:
    affordances: list[AffordanceMetadata]
    affordance_costs_tensor: Tensor  # GPU lookup table
    drive_hash: str  # "sha256:abc123..."
    # ... other specs
```

**Runtime (Environment):**
```python
# No YAML access - only compiled specs
costs = compiled_universe.affordance_costs_tensor[affordance_idx, :]  # [num_meters]
meters -= costs  # Apply cost
```

**Adoption:**
- **UAC:** Compiles substrate, bars, cascades, affordances, cues, training, drive_as_code, variables_reference, brain
- **VFS:** Generates ObservationSpec from variables_reference.yaml
- **DAC:** Compiles drive_as_code.yaml into closure factories
- **Action System:** Validates enabled_actions.yaml against substrate capabilities

### 3. Strategy Pattern (Pluggable Algorithms)

**Pattern:** Abstract base classes with pluggable implementations enable baseline comparisons and algorithm research.

**Implementations:**

**Exploration Strategies:**
```python
class ExplorationStrategy(ABC):
    @abstractmethod
    def select_actions(self, q_values, epsilon, action_masks): pass

    @abstractmethod
    def compute_intrinsic_rewards(self, observations): pass

# Concrete strategies
EpsilonGreedyExploration(ExplorationStrategy)  # Baseline
RNDExploration(ExplorationStrategy)            # Novelty-seeking
AdaptiveIntrinsicExploration(ExplorationStrategy)  # RND + annealing
```

**Curriculum Strategies:**
```python
class CurriculumManager(ABC):
    @abstractmethod
    def get_batch_decisions(self, agent_state): pass

# Concrete strategies
StaticCurriculum(CurriculumManager)       # Fixed difficulty
AdversarialCurriculum(CurriculumManager)  # Auto-tuning
```

**Replay Buffer Variants:**
```python
# Base interface implied
class ReplayBuffer:
    def push(self, transition): pass
    def sample(self, batch_size): pass

# Variants
ReplayBuffer                  # Standard random sampling
SequentialReplayBuffer        # Full episodes for LSTM
PrioritizedReplayBuffer       # TD-error prioritization
```

**Benefits:**
- **Baseline comparisons** - Run same config with epsilon-greedy vs RND
- **Ablation studies** - Isolate algorithm impact from environment changes
- **Research flexibility** - Add new strategies without modifying Population
- **Config-driven selection** - `exploration.strategy: "rnd"` in YAML

**Population Integration:**
```python
class VectorizedPopulation:
    def __init__(self, exploration_config, curriculum_config):
        # Factory pattern selects implementation
        self.exploration = create_exploration_strategy(exploration_config)
        self.curriculum = create_curriculum_manager(curriculum_config)

    def step_population(self):
        # Polymorphic calls
        actions = self.exploration.select_actions(q_values, ...)
        decisions = self.curriculum.get_batch_decisions(agent_state)
```

### 4. Closure Factories (Compile-Time Variable Capture)

**Pattern:** Factory functions capture configuration at compile time, return closures for runtime execution. Ensures correct variable capture in GPU-vectorized loops.

**DAC Modifier Example:**

**YAML Config:**
```yaml
modifiers:
  energy_crisis:
    target: intrinsic
    source: meters
    meter: energy
    ranges:
      - {min: 0.0, max: 0.2, multiplier: 0.0}
      - {min: 0.2, max: 1.0, multiplier: 1.0}
```

**Factory Function:**
```python
def create_modifier_fn(config: ModifierConfig):
    # Capture config at compile time
    meter_idx = config.meter_idx  # Resolved to 0 (energy index)
    ranges = config.ranges        # [(0.0, 0.2, 0.0), (0.2, 1.0, 1.0)]

    # Return closure
    def modifier_fn(meters: Tensor, vfs_registry: VariableRegistry) -> Tensor:
        # Use captured variables
        meter_vals = meters[:, meter_idx]  # [num_agents]

        # GPU vectorized range lookup
        multiplier = torch.zeros_like(meter_vals)
        for rmin, rmax, rmult in ranges:
            mask = (meter_vals >= rmin) & (meter_vals < rmax)
            multiplier = torch.where(mask, rmult, multiplier)

        return multiplier  # [num_agents]

    return modifier_fn

# Compile all modifiers once
self.modifier_fns = [create_modifier_fn(cfg) for cfg in modifier_configs]
```

**Runtime Execution:**
```python
# calculate_rewards() called millions of times
intrinsic_weight = base_weight  # 0.1
for modifier_fn in self.modifier_fns:  # Pre-compiled closures
    intrinsic_weight = intrinsic_weight * modifier_fn(meters, vfs_registry)
```

**Why Closures?**
- **Variable capture** - Config values frozen at compile time, not re-read each step
- **Performance** - No dict lookups or YAML parsing in hot path
- **Type safety** - Closures have consistent signature, composable
- **Testability** - Can test individual modifier_fn in isolation

**Other Closure Factory Uses:**
- **Shaping bonuses** - 11 bonus types compiled to closures (approach_reward, streak_bonus, etc.)
- **Extrinsic strategies** - 9 strategy types compiled to closures (multiplicative, weighted_sum, etc.)
- **VFS observation builders** - Field extractors compiled from ObservationField specs

### 5. Checkpoint Provenance Tracking

**Pattern:** Embed configuration hashes and version metadata in checkpoints to prevent training/inference mismatches.

**Checkpoint Structure:**
```python
checkpoint = {
    # Model weights
    'q_network_state_dict': q_network.state_dict(),
    'target_network_state_dict': target_network.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),

    # Training state
    'replay_buffer': replay_buffer.serialize(),
    'exploration_state': exploration.checkpoint_state(),
    'curriculum_state': curriculum.checkpoint_state(),
    'global_step': 1_000_000,

    # Provenance (CRITICAL)
    'drive_hash': 'sha256:abc123...',      # Hash of drive_as_code.yaml
    'compiler_version': '1.2.3',           # UAC version
    'git_sha': 'def456...',                # Git commit
    'config_dir': 'configs/L1_full_observability',
    'timestamp': '2025-11-13T20:13:00Z'
}
```

**Validation on Load:**
```python
def load_checkpoint(path, config_dir):
    checkpoint = torch.load(path)

    # Compile current config
    compiled = UniverseCompiler.compile(config_dir)

    # Validate provenance
    if checkpoint['drive_hash'] != compiled.drive_hash:
        raise ValueError(
            f"Checkpoint trained with different reward function!\n"
            f"Checkpoint drive_hash: {checkpoint['drive_hash']}\n"
            f"Current drive_hash: {compiled.drive_hash}\n"
            f"This mismatch would cause training/inference inconsistency."
        )

    # Load weights
    q_network.load_state_dict(checkpoint['q_network_state_dict'])
    # ...
```

**Provenance Tracking Locations:**
- **CompiledUniverse** - drive_hash, compiler_version, git_sha
- **Population checkpoints** - Embed CompiledUniverse provenance
- **Inference server** - Validate checkpoint matches loaded config
- **CI/CD** - Reject checkpoints without provenance metadata

**Benefits:**
- **Prevent reward hacking** - Can't evaluate policy trained with shaped rewards using sparse rewards
- **Reproducibility** - Git SHA enables exact code reconstruction
- **Version migration** - Compiler version tracks breaking changes
- **Debugging** - Timestamp + config_dir trace experiment history

### 6. Two-Phase Architecture (Compile vs Runtime)

**Pattern:** Strict separation between compile-time (expensive, once) and runtime (cheap, millions of times).

**Examples:**

**DAC (Drive As Code):**
```python
# Compile-time (DACEngine.__init__)
def __init__(self, dac_config, num_agents, device):
    # Parse YAML
    # Resolve meter/variable references
    # Create closure factories
    self.extrinsic_fn = compile_extrinsic_strategy(dac_config.extrinsic)
    self.modifier_fns = [compile_modifier(m) for m in dac_config.modifiers]
    self.shaping_fns = [compile_shaping(s) for s in dac_config.shaping]
    # Total cost: ~10ms

# Runtime (called millions of times)
def calculate_rewards(self, meters, vfs_registry, dones, intrinsic):
    extrinsic = self.extrinsic_fn(meters, vfs_registry)  # Closure
    intrinsic_weight = self.base_weight
    for modifier_fn in self.modifier_fns:  # Pre-compiled
        intrinsic_weight *= modifier_fn(meters, vfs_registry)
    shaping = sum(fn(meters, vfs_registry) for fn in self.shaping_fns)
    total = extrinsic + intrinsic * intrinsic_weight + shaping
    return torch.where(dones, 0.0, total)  # GPU vectorized
    # Total cost: <0.1ms
```

**UAC (Universe Compiler):**
```python
# Compile-time (once per training run)
compiled_universe = UniverseCompiler.compile(config_dir)
# Stages: Parse → Symbol → Resolve → Validate → Metadata → Optimize → Emit
# Total cost: ~500ms (includes YAML I/O, validation, hash computation)

# Runtime (every environment construction)
env = VectorizedHamletEnv(compiled_universe, num_agents)
# Uses pre-computed lookup tensors, no YAML access
# Total cost: ~50ms (GPU tensor allocation)
```

**VFS (Variable & Feature System):**
```python
# Compile-time (UAC Stage 5)
obs_spec = VFSObservationSpecBuilder.build(variable_defs)
# Generates ObservationField list with dimensions, normalization, semantic types
# Total cost: ~10ms

# Runtime (every env.step())
observations = observation_builder.build_observations(vfs_registry, obs_spec)
# Reads from registry, applies normalization, concatenates tensors
# Total cost: <0.5ms
```

**Benefits:**
- **Performance** - Amortize expensive operations (YAML parsing, validation) over millions of steps
- **Error detection** - Fail fast at compile time, not 2 hours into training
- **Debugging** - Easier to debug compiled artifacts (frozen dataclasses) than runtime interpretation
- **Caching** - Compiled universes cached by mtime + content hash

### 7. No-Defaults Principle

**Pattern:** All behavioral parameters must be explicitly specified in configuration. No implicit defaults that hide assumptions.

**Enforcement:**

**Pydantic Required Fields:**
```python
@dataclass
class SubstrateConfig:
    substrate_type: str  # No default - REQUIRED
    grid_size: int | tuple[int, ...]  # No default
    boundary_mode: str  # No default ("clamp" | "wrap" | "bounce")
    distance_metric: str  # No default ("manhattan" | "euclidean")
    position_encoding: str  # No default ("relative" | "scaled" | "absolute")

    # ❌ BAD: substrate_type: str = "grid"  # Hidden assumption
    # ✅ GOOD: substrate_type: str  # Forces explicit choice
```

**Linter Validation:**
```python
# .claudelint.yml (hypothetical)
no_defaults:
  - path: src/townlet/config/*.py
    pattern: '= .*'  # Flag any default values
    whitelist:
      - description  # Metadata OK
      - _computed_field  # Derived values OK
```

**Config Templates:**
```yaml
# configs/templates/substrate.yaml (MUST be copied and edited)
substrate:
  substrate_type: ???  # Placeholder forces replacement
  grid_size: ???
  boundary_mode: ???
  distance_metric: ???
  position_encoding: ???
```

**Benefits:**
- **Reproducibility** - Config files are self-contained documentation
- **No hidden behavior** - Reading config reveals all system behavior
- **Version safety** - Changing code defaults doesn't break old configs
- **Pedagogy** - Students learn that every hyperparameter matters

**Trade-off:**
- **Verbosity** - Config files longer (200+ lines typical)
- **Learning curve** - New users must understand all parameters
- **Boilerplate** - Similar configs have redundant specifications

**Mitigation:**
- **Template system** - `configs/templates/` with documented examples
- **Validation errors** - UAC provides actionable messages ("substrate_type is required")
- **Documentation** - `docs/config-schemas/` explains every field

---

## Concerns & Recommendations

### Critical Issues (Blocking Training/Inference)

**None identified.** The system is production-ready for research and pedagogy.

### Warnings (Quality/Maintainability Issues)

#### 1. Frontend Configuration Hardcoding

**Concern:** Frontend components hardcode game-specific configuration (action names, meter relationships, color gradients) instead of deriving from `CompiledUniverse` metadata.

**Locations:**
- `AspatialView.vue` line 93: `['Up', 'Down', 'Left', 'Right', 'Interact', 'Wait']`
- `MeterPanel.vue` lines 208-221: Hardcoded tier classifications (Primary/Secondary/Tertiary)
- `Grid.vue` lines 253-272: Heat map color gradient function

**Impact:**
- **Tight coupling** - Frontend assumes specific action vocabulary/meter structure
- **Divergence risk** - Backend changes to action names don't propagate to frontend
- **Limited reusability** - Frontend doesn't work with custom config packs without code changes

**Recommendation:**
```javascript
// CURRENT (hardcoded)
const ACTION_NAMES = ['Up', 'Down', 'Left', 'Right', 'Interact', 'Wait']

// PROPOSED (derive from metadata)
const { data: universeMetadata } = await fetch('/api/universe/metadata')
const ACTION_NAMES = universeMetadata.action_space.action_labels

// Backend API
@app.get('/api/universe/metadata')
def get_metadata():
    return compiled_universe.to_dict()  # Includes action labels, meter hierarchy
```

**Priority:** Medium - Doesn't block current use cases, but limits extensibility.

#### 2. Incomplete Prioritized Replay for Recurrent Networks

**Concern:** `PrioritizedReplayBuffer` raises `NotImplementedError` for recurrent networks (TASK-005 Phase 3 incomplete).

**Location:** `src/townlet/population/vectorized.py` line 318

**Impact:**
- **Feature gap** - LSTM networks can't use PER (prioritized experience replay)
- **Research limitation** - Can't compare PER vs uniform sampling for POMDP tasks

**Current Workaround:**
L2 (POMDP curriculum) uses `SequentialReplayBuffer` with uniform sampling.

**Recommendation:**
```python
# Extend PrioritizedReplayBuffer to support episodes
class PrioritizedSequentialReplayBuffer:
    def push_episode(self, episode):
        # Store full episode
        # Compute TD-error for entire trajectory (sum or max)
        # Update priorities
        pass

    def sample_episodes(self, batch_size):
        # Sample episodes proportional to priority
        # Unroll episodes for training
        pass
```

**Priority:** Low - Workaround (uniform sampling) still effective, PER is optimization not requirement.

#### 3. BrainConfig Architecture Selection via if/elif Chains

**Concern:** `VectorizedPopulation.__init__` uses if/elif chains to select network architecture based on `brain_config.architecture.type`.

**Location:** `src/townlet/population/vectorized.py` lines ~150-180 (inferred)

**Code Pattern:**
```python
if brain_config.architecture.type == "feedforward":
    self.q_network = SimpleQNetwork(...)
    self.replay_buffer = ReplayBuffer(...)
elif brain_config.architecture.type == "recurrent":
    self.q_network = RecurrentSpatialQNetwork(...)
    self.replay_buffer = SequentialReplayBuffer(...)
elif brain_config.architecture.type == "dueling":
    self.q_network = DuelingQNetwork(...)
    self.replay_buffer = ReplayBuffer(...)
else:
    raise ValueError(f"Unknown architecture type: {brain_config.architecture.type}")
```

**Impact:**
- **Extensibility** - Adding new network types requires editing Population code
- **Plugin system missing** - Can't register custom architectures without core changes
- **TODO comments** - Hidden dimensions have "TODO(BRAIN_AS_CODE)" indicating incomplete design

**Recommendation:**
```python
# Plugin registry pattern
ARCHITECTURE_REGISTRY = {
    "feedforward": (SimpleQNetwork, ReplayBuffer),
    "recurrent": (RecurrentSpatialQNetwork, SequentialReplayBuffer),
    "dueling": (DuelingQNetwork, ReplayBuffer),
}

def register_architecture(name, network_cls, buffer_cls):
    ARCHITECTURE_REGISTRY[name] = (network_cls, buffer_cls)

# Population uses registry
network_cls, buffer_cls = ARCHITECTURE_REGISTRY[brain_config.architecture.type]
self.q_network = network_cls(obs_dim, action_dim, **brain_config.network)
self.replay_buffer = buffer_cls(capacity, **brain_config.replay)
```

**Priority:** Low - Current system works, plugin architecture nice-to-have for research flexibility.

#### 4. STAGE_CONFIGS Hardcoded in Curriculum

**Concern:** `AdversarialCurriculum` has module-level `STAGE_CONFIGS` array hardcoded, not parameterizable via YAML.

**Location:** `src/townlet/curriculum/adversarial.py` lines ~50-80

**Impact:**
- **Inflexibility** - Can't customize stage progression without code changes
- **Inconsistent design** - Environment/training parameters in YAML, curriculum stages in Python
- **Research limitation** - Can't A/B test different stage progressions easily

**Recommendation:**
```yaml
# curriculum.yaml (new file)
adversarial:
  stages:
    - name: "Basic Survival"
      active_meters: [energy, health]
      depletion_multiplier: 0.2
      reward_mode: shaped
    - name: "Emotional Stability"
      active_meters: [energy, health, mood]
      depletion_multiplier: 0.4
      reward_mode: shaped
    # ... 3 more stages

  advancement_thresholds:
    survival_rate: 0.7
    learning_progress: 0.0
    entropy: 0.5

  retreat_thresholds:
    survival_rate: 0.3
    learning_progress: -0.1
```

```python
# AdversarialCurriculum loads from config
class AdversarialCurriculum:
    def __init__(self, curriculum_config_path):
        config = yaml.safe_load(open(curriculum_config_path))
        self.stages = [StageConfig(**s) for s in config['adversarial']['stages']]
```

**Priority:** Medium - Impacts research flexibility, but hardcoded stages work for current curriculum.

#### 5. Exploration Annealing Thresholds Not in Config

**Concern:** `AdaptiveIntrinsicExploration` has hardcoded thresholds (`variance_threshold=100.0`, `min_survival_fraction=0.4`) not exposed in config YAML.

**Location:** `src/townlet/exploration/adaptive_intrinsic.py` constructor

**Impact:**
- **Requires code edits** - Tuning annealing behavior needs modifying Python
- **Inconsistent** - Other hyperparameters (epsilon, intrinsic_weight) in YAML
- **No-Defaults violation** - Implicit assumptions about "consistently succeeding"

**Recommendation:**
```yaml
# training.yaml
exploration:
  strategy: adaptive_rnd
  rnd:
    predictor_lr: 0.0001
    target_dim: 128
  adaptive:
    variance_threshold: 100.0  # Expose
    min_survival_fraction: 0.4  # Expose
    annealing_rate: 0.5
    min_intrinsic_weight: 0.01
```

**Priority:** Low - Default thresholds work well empirically, but should follow no-defaults principle.

#### 6. Temporal Features Documented as "Ignored"

**Concern:** `RecurrentSpatialQNetwork` extracts temporal features (tick, day_progress, time_of_day, is_night) but documents them as "ignored" in code comment.

**Location:** `src/townlet/agent/networks.py` line 213

**Impact:**
- **Signal loss** - Temporal mechanics (day/night cycles in L3) may provide valuable state information
- **Unclear intent** - Comment suggests features were planned but never utilized
- **Network inefficiency** - Extracting features without using them wastes computation

**Investigation Needed:**
- Why were temporal features added to observation spec if ignored?
- Does curriculum L3 (temporal mechanics) benefit from temporal encoding?
- Should temporal encoder be activated for L3 networks?

**Recommendation:**
```python
# Option 1: Remove temporal features from observation spec if truly unused
# Option 2: Add temporal encoder and conditional activation
if curriculum_level >= 3:  # Temporal mechanics curriculum
    temporal_encoded = self.temporal_encoder(temporal_features)
    combined = torch.cat([vision, position, meter, affordance, temporal], dim=1)
else:
    combined = torch.cat([vision, position, meter, affordance], dim=1)
```

**Priority:** Low - Network trains successfully without temporal encoding, unclear if utilizing temporal would improve performance.

#### 7. Curriculum PerformanceTracker Semantic Ambiguity

**Concern:** `PerformanceTracker.update_step()` parameter named "rewards" but actually receives step counts.

**Location:** `src/townlet/curriculum/adversarial.py` lines 88-89 (comment acknowledges confusion)

**Impact:**
- **Maintainer confusion** - Future developers may misinterpret parameter semantics
- **Bug risk** - Passing actual rewards instead of step counts would break advancement logic

**Recommendation:**
```python
# CURRENT
def update_step(self, agent_idx, rewards, done):  # "rewards" misleading
    self.survival_counts[agent_idx] += rewards  # Actually step counts

# PROPOSED
def update_step(self, agent_idx, step_count, done):
    self.survival_counts[agent_idx] += step_count
```

**Priority:** Low - Code works correctly despite confusing naming, simple refactor fixes.

---

### Recommendations Summary

**High Priority (Implement Soon):**
- None - system is production-ready

**Medium Priority (Improves Extensibility):**
1. Frontend derive config from CompiledUniverse metadata (not hardcoded)
2. Parameterize curriculum stages via YAML (not hardcoded STAGE_CONFIGS)

**Low Priority (Nice-to-Have):**
3. Implement PrioritizedSequentialReplayBuffer for recurrent networks
4. Plugin registry for network architectures (replace if/elif chains)
5. Expose exploration annealing thresholds in config YAML
6. Investigate temporal features (use or remove from observation spec)
7. Rename PerformanceTracker.update_step() parameter for clarity

**Long-Term Research:**
- Evaluate temporal encoder activation for L3 curriculum
- Benchmark PER vs uniform sampling for LSTM networks
- Custom architecture plugin system for research experiments

---

## Appendices

### Appendix A: Methodology

**Analysis Approach:** System Archeologist Workflow (axiom-system-archaeologist skill pack v1.0.2)

**Phases Executed:**

1. **Workspace Setup** - Created `docs/arch-analysis-2025-11-13-2013/` with artifacts/ and temp/ directories
2. **Coordination Planning** - Documented scope, strategy, execution plan in `00-coordination-plan.md`
3. **Holistic Assessment** - Top-down reconnaissance of codebase structure, entry points, technology stack (`01-initial-assessment.md`)
4. **Detailed Subsystem Analysis** - Parallel analysis of 10 core subsystems following exact 8-section catalog format (`02-subsystem-catalog.md`)
5. **Catalog Validation** - Systematic contract compliance verification (APPROVED with 0 critical violations)
6. **Architecture Diagrams** - C4 diagrams at 3 levels: Context, Container, Component (`03-diagrams.md`)
7. **Diagram Validation** - Cross-document consistency checks with catalog dependencies (APPROVED with 100% coverage)
8. **Final Report Synthesis** - Pattern identification, concern extraction, multi-audience documentation (this document)

**Analysis Duration:** ~2 hours wall-clock time (10 parallel subagent analyses)

**Tools Used:**
- Code reading (Read tool) - 100+ file reads across src/townlet/, configs/, frontend/, docs/
- Dependency analysis (Grep tool) - Import statements, function calls, class references
- Structure mapping (Glob tool) - Directory traversal, file enumeration
- Documentation synthesis - Cross-referencing catalog, diagrams, initial assessment

**Validation Gates:**
- Subsystem catalog: 10/10 entries passed contract compliance (8 sections, no extras)
- Architecture diagrams: 5/5 diagrams passed C4 compliance + dependency cross-checks
- Cross-document consistency: 100% catalog dependencies mapped to diagram arrows

**Confidence Assessment:**
- **High confidence (9/10 subsystems):** Complete code coverage, traceable dependencies, comprehensive testing
- **Medium confidence (1/10 subsystems):** Frontend Visualization - hardcoded config creates uncertainty about adaptability to custom packs

### Appendix B: Assumptions & Limitations

**Assumptions:**

1. **Active Codebase:** Analysis focuses exclusively on `src/townlet/` (27,138 LOC). Legacy `src/hamlet/` marked obsolete and excluded.
2. **Production Readiness:** Despite pre-release status, system is architected for production use (comprehensive validation, checkpoint provenance, CI/CD integration).
3. **Curriculum Completeness:** L0-L3 curriculum levels documented in CLAUDE.md are implemented and tested. Higher levels (L4-L6) mentioned as "future" not analyzed.
4. **GPU Availability:** Architecture assumes CUDA-capable GPU for training. CPU fallback exists but not optimized.
5. **Single-Agent Inference:** Live inference server streams single agent state. Multi-agent visualization support unclear from code analysis.
6. **WebSocket Protocol Stability:** Frontend assumes JSON message format from port 8766. Protocol versioning/migration not documented.
7. **Config Pack Completeness:** All 9 YAML files (substrate, bars, cascades, affordances, cues, training, drive_as_code, variables_reference, brain) required per pack. Partial packs rejected by UAC.
8. **No Runtime Config Mutation:** CompiledUniverse assumed immutable after emit. Runtime parameter changes require recompilation.

**Limitations:**

1. **No Runtime Testing:** Analysis based on code reading, not live training runs. Emergent behaviors (like "Low Energy Delirium") understood from documentation and code structure, not observed execution.
2. **No Performance Profiling:** GPU utilization, memory consumption, training throughput not measured. 100x speedup claim from CLAUDE.md taken as documented fact.
3. **Frontend Analysis Depth:** Vue components analyzed structurally (props, events, rendering logic) but not interactively tested. WebSocket message format inferred from backend code, not captured traffic.
4. **Curriculum Progression Not Traced:** L0→L3 transfer learning capability understood architecturally (constant observation dimension) but not validated with actual checkpoint transfers.
5. **DAC Pedagogical Patterns:** "Low Energy Delirium" bug understood from config comparison (L0_0 vs L0_5) but not reproduced in training.
6. **Test Coverage Gaps:** 644+ tests with 70% coverage documented, but specific uncovered code paths not identified. Integration tests vs unit tests ratio unknown.
7. **Dependency Version Constraints:** PyTorch 2.9+ requirement documented, but compatibility with PyTorch 2.10/2.11 not verified. Vue 3 minor version requirements unclear.
8. **Checkpoint Size/Performance:** Provenance tracking overhead (drive_hash, git_sha) on checkpoint size not quantified. Cache invalidation latency (mtime + hash) not measured.

**Gaps in Analysis:**

- **Multi-Agent Coordination (L5):** Future curriculum level mentioned but not implemented. Architecture unclear for agent-agent interactions.
- **Communication (L6):** Future curriculum level mentioned but not implemented. Message passing architecture not designed.
- **Economic Balance Validation:** UAC performs economic balance checks (job payment sustains costs) but validation algorithm complexity/coverage not deeply analyzed.
- **Cascade Cycle Detection:** Linear graph traversal for cycles noted as O(n²) concern but not validated with complex cascade graphs.
- **POMDP Window Explosion:** GridND POMDP rejection (5⁷ = 78K cells) documented but not tested at boundary (N=3: 125 cells feasible?).
- **Frontend Error Handling:** No error handling for malformed WebSocket messages noted as concern, but failure modes (message corruption, out-of-order delivery, reconnection) not analyzed.
- **Checkpoint Migration:** Compiler version tracking enables migration but migration scripts/procedures not documented.
- **CI/CD Coverage:** `.github/workflows/config-validation.yml` mentioned but workflow details (matrix testing, curriculum levels, hardware) not examined.

### Appendix C: Cross-References

**Document Cross-References:**

- [Initial Assessment](./01-initial-assessment.md) - Directory structure, entry points, technology stack
- [Subsystem Catalog](./02-subsystem-catalog.md) - Detailed 10-subsystem analysis with dependencies
- [Architecture Diagrams](./03-diagrams.md) - C4 Context/Container/Component diagrams
- [Coordination Plan](./00-coordination-plan.md) - Analysis scope, strategy, execution timeline

**Validation Reports:**

- [Catalog Validation](./temp/validation-catalog.md) - Contract compliance: APPROVED (0 critical, 0 warnings)
- [Diagram Validation](./temp/validation-diagrams.md) - C4 compliance + cross-document consistency: APPROVED (100% coverage)

**External Documentation:**

- [CLAUDE.md](../../CLAUDE.md) - Project overview, development philosophy, no-defaults principle
- [UNIVERSE-COMPILER.md](../UNIVERSE-COMPILER.md) - UAC seven-stage pipeline deep dive
- [VFS Integration Guide](../vfs-integration-guide.md) - Variable & Feature System migration guide
- [DAC Migration Guide](../guides/dac-migration.md) - Drive As Code system adoption
- [Config Schemas](../config-schemas/) - YAML configuration reference (substrate, bars, training, etc.)

**Code References (Key Entry Points):**

- Training: `scripts/run_demo.py` → `Population.step_population()`
- Inference: `src/townlet/demo/unified_server.py` → WebSocket broadcasting
- Compilation: `src/townlet/universe/compiler.py` → `UniverseCompiler.compile()`
- Environment: `src/townlet/environment/vectorized_env.py` → `VectorizedHamletEnv.step()`
- Rewards: `src/townlet/environment/dac_engine.py` → `DACEngine.calculate_rewards()`

**Dependency Relationships:**

| Subsystem | Primary Dependencies (Outbound) | Dependents (Inbound) |
|-----------|--------------------------------|----------------------|
| UAC | Config DTOs, VFS, Pydantic | Environment, Population, DemoRunner |
| VFS | Pydantic, PyTorch | UAC, Environment, DAC |
| DAC | DriveAsCodeConfig, VFS | Environment |
| Environment | Substrate, UAC, VFS, DAC | Population, DemoRunner, Inference |
| Agent Networks | PyTorch, ObservationActivity | Population, NetworkFactory |
| Population | Environment, Networks, Training State, Exploration, Curriculum | DemoRunner, Training Scripts |
| Exploration | Training State, PyTorch | Population |
| Curriculum | Training State, PyTorch | Population |
| Training State | PyTorch, Pydantic | Population, Exploration, Curriculum |
| Frontend | Pinia, WebSocket, Vue 3 | Inference Server (WebSocket) |

**Pattern Application Map:**

| Pattern | Subsystems Implementing |
|---------|-------------------------|
| GPU-Native Vectorization | Environment, Population, Exploration, Curriculum, DAC |
| Declarative Configuration | UAC, VFS, DAC, Substrate, Action System |
| Strategy Pattern | Exploration (3 strategies), Curriculum (2 strategies), Replay Buffer (3 variants) |
| Closure Factories | DAC (modifiers, extrinsic, shaping), VFS (observation builders) |
| Checkpoint Provenance | UAC, Population, DemoRunner |
| Two-Phase Architecture | UAC (compile/runtime), DAC (compile/runtime), VFS (build/access) |
| No-Defaults Principle | All config DTOs, Pydantic validation, UAC enforcement |

**Concern Severity Map:**

| Severity | Count | Subsystems Affected |
|----------|-------|---------------------|
| Critical | 0 | None |
| Warning | 7 | Frontend (3), Population (2), Curriculum (1), Exploration (1) |
| Observation | 3 | Agent Networks (temporal features), UAC (complexity), Training State (typing) |

---

## Report Metadata

**Total Pages:** ~50 (estimated print)
**Total Words:** ~15,000
**Cross-References:** 40+ internal links
**Code Examples:** 30+ blocks
**Diagrams:** 5 (via reference to 03-diagrams.md)
**Tables:** 12

**Target Audiences:**
- **Executives:** Read Executive Summary only (2-3 paragraphs)
- **Architects:** Read System Overview + Architecture Diagrams + Key Patterns (~10 pages)
- **Engineers:** Read Subsystem Deep Dives + Concerns (~30 pages)
- **Operators:** Read Configuration Layer + Appendix B (Assumptions) (~8 pages)
- **Students:** Read System Overview + "Low Energy Delirium" case study in DAC section (~12 pages)

**Document Status:** FINAL
**Validation Status:** All artifacts APPROVED (0 critical violations)
**Confidence:** HIGH (9/10 subsystems), MEDIUM (1/10 Frontend due to hardcoded config)

---

**End of Report**
