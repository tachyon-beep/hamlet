# STREAM-001: UAC/BAC Foundation Implementation

**Created**: 2025-11-04
**Type**: STREAM (Implementation sequence document)
**Method**: Research → Plan → Review Loop (see `methods/RESEARCH-PLAN-REVIEW-LOOP.md`)
**Analysis**: Parallel subagent dependency extraction and synthesis

**NOTE**: This document describes the implementation order for the UNIVERSE_AS_CODE (UAC) and BRAIN_AS_CODE (BAC) foundation work. Task numbers have been renumbered from the original TASK_IMPLEMENTATION_PLAN.md to align with the standardized task naming convention. See "Task Renumbering" section below for the mapping.

---

## Task Renumbering

This STREAM document reflects the following task renumbering:

| Original | New | Description |
|----------|-----|-------------|
| TASK-000 | TASK-002A | UAC Action Space |
| TASK-001 | TASK-001 | Variable Meters (no change) |
| TASK-002 | obsolete | Original compiler concept (references clarified/removed) |
| TASK-002A | TASK-003 | UAC Core DTOs |
| TASK-002B | TASK-004B | UAC Capability System |
| TASK-003 | TASK-002B | Spatial Substrates |
| TASK-004 | TASK-004A | Universe Compiler |
| TASK-005 | TASK-005 | BRAIN_AS_CODE (no change) |
| TASK-006 | TASK-006 | Substrate-Agnostic Visualization (no change) |

---

## Executive Summary

This document defines the optimal execution order for TASK-001 through TASK-006 (excluding obsolete TASK-002) based on dependency analysis, effort estimates, and architectural principles. The analysis reveals **TASK-001 (Variable Meters)** as the critical starting point, with a total implementation timeline of **139-193 hours** using parallelization.

**Key Finding**: TASK-001 is marked CRITICAL priority with the highest leverage (13-19h effort unlocks entire design space from 4-meter to 32-meter universes).

---

## Optimal Task Execution Order

```
┌─────────────────────────────────────────────────────────────┐
│  PHASE 1: CRITICAL FOUNDATION (13-19h)                      │
│  ✓ TASK-001: Variable Meters                                │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  PHASE 2: SPATIAL FOUNDATION (51-65h)                       │
│  ✓ TASK-002B: Spatial Substrates  │ PARALLEL ↔             │
│  ✓ TASK-002A: Action Space         │                         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  PHASE 3: VALIDATION LAYER (7-12h)                          │
│  ✓ TASK-003: Core DTOs                                      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  PHASE 4: COMPILATION (46-66h)                              │
│  ✓ TASK-004A: Universe Compiler    │ PARALLEL ↔             │
│  ✓ TASK-004B: Capability DTOs      │                         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  PHASE 5: ADVANCED CONFIG (22-31h)                          │
│  ✓ TASK-005: BRAIN_AS_CODE                                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  PHASE 6: VISUALIZATION (DEFERRED)                          │
│  ○ TASK-006: Substrate-Agnostic Visualization               │
└─────────────────────────────────────────────────────────────┘
```

---

## Task Dependency Matrix

| Task | Effort | Priority | Depends On | Provides |
|------|--------|----------|------------|----------|
| **TASK-001** (Variable Meters) | 13-19h | **CRITICAL** | None | Variable meter system (1-32), dynamic tensors, checkpoint metadata |
| **TASK-002B** (Spatial Substrates) | 51-65h | HIGH | None | Substrate abstraction, substrate.yaml schema, text visualization |
| **TASK-002A** (Action Space) | 11-17h | HIGH | None | actions.yaml schema, multi-meter cost system |
| **TASK-003** (Core DTOs) | 7-12h | HIGH | TASK-002B, TASK-001, TASK-002A | Core DTO validation for all configs including SubstrateConfig |
| **TASK-004B** (Capability DTOs) | 8-12h | HIGH | TASK-003 | Capability system DTOs for advanced affordances |
| **TASK-004A** (Compiler) | 46-66h | HIGH | TASK-003 (required), TASK-001 (optional) | 7-stage compiler, cross-validation, caching, CompiledUniverse |
| **TASK-005** (BRAIN_AS_CODE) | 22-31h | MEDIUM | TASK-002B, TASK-003, TASK-004A | brain.yaml, NetworkFactory, BrainCompiler |
| **TASK-006** (Visualization) | 14-64h | **LOW** | TASK-002B, TASK-002A | GUI rendering for hex/3D/graph (deferred) |

---

## Dependency Graph

```
TASK-001 (Variable Meters) ────┐
  13-19h, CRITICAL              │
                                ├──> TASK-004A (Compiler)
TASK-002B (Substrates) ──────┐  │      46-66h, HIGH
  51-65h, HIGH                │  │            │
                              ├──┤            │
TASK-002A (Actions) ─────┐   │  │            ├──> TASK-005 (BRAIN_AS_CODE)
  11-17h, HIGH            │   │  │            │      22-31h, MEDIUM
                          │   │  │            │
                          ├───┴──┴──> TASK-003 (Core DTOs)
                          │             7-12h, HIGH
                          │                   │
                          │                   └──> TASK-004B (Capability DTOs)
                          │                          8-12h, HIGH
                          │
                          └──────────────────> TASK-006 (Visualization)
                                                 14-64h, LOW (DEFERRED)
```

---

## Phase-by-Phase Implementation Plan

### Phase 1: Critical Foundation (13-19h)

**Goal**: Unlock design space by making meter count configurable

#### TASK-001: Variable-Size Meter System
- **Effort**: 13-19h
- **Priority**: CRITICAL
- **Dependencies**: None
- **Deliverables**:
  - Variable meter count (1-32 meters)
  - Dynamic tensor sizing
  - 4-meter and 12-meter example configs
  - Checkpoint metadata for compatibility validation
- **Why First**:
  - Only task marked CRITICAL priority
  - "Highest-leverage infrastructure change"
  - Shortest effort (13-19h) with biggest impact
  - Unblocks 4-meter tutorials → 32-meter complex simulations
  - No dependencies - can start immediately

**Success Criteria**:
- [ ] Config loader accepts `num_meters` parameter
- [ ] Tensors dynamically sized based on meter count
- [ ] Network layers adapt to variable observation dimensions
- [ ] Checkpoints include meter count metadata
- [ ] Tests pass for 4-meter and 12-meter configs

---

### Phase 2: Spatial Foundation (51-65h parallelized)

**Goal**: Define spatial substrate and action space abstractions

#### TASK-002B: Configurable Spatial Substrates
- **Effort**: 51-65h
- **Priority**: HIGH
- **Dependencies**: None
- **Deliverables**:
  - Substrate abstraction layer (`SpatialSubstrate` interface)
  - Pydantic schema for `substrate.yaml`
  - Reference implementations: SquareGrid, CubicGrid, HexGrid, Aspatial
  - Coordinate encoding for 3D grids
  - Text-based visualization for all substrate types
  - Migration of 5 existing configs to substrate.yaml
- **Why Second**:
  - Defines coordinate system and spatial reasoning
  - Enables 2D/3D/hex/graph/aspatial universes
  - Provides substrate.yaml schema for TASK-003 validation
  - Longest task in critical path (51-65h)

**Parallel Execution**: TASK-002B + TASK-002A

#### TASK-002A: UAC Action Space
- **Effort**: 11-17h
- **Priority**: HIGH
- **Dependencies**: None (informs TASK-001 scope)
- **Deliverables**:
  - actions.yaml configuration file
  - Pydantic DTOs for action definitions
  - Multi-meter cost system (not just energy)
  - Action abstraction layer
  - Pattern consistency with affordances.yaml
- **Why Third**:
  - Actions depend on substrate (4-way square, 6-way hex, 0-way aspatial)
  - Shorter task (11-17h) runs parallel to TASK-002B
  - Provides actions.yaml schema for TASK-003 validation
  - Proves UNIVERSE_AS_CODE concept before larger migrations

**Success Criteria (Phase 2)**:
- [ ] substrate.yaml defines grid topology, boundaries, distance metrics
- [ ] actions.yaml defines movement deltas and multi-meter costs
- [ ] Agent can navigate 2D, 3D, hex, and aspatial substrates
- [ ] Text visualization renders all substrate types
- [ ] Action space adapts to substrate type (4-way vs 6-way)

**Time Savings**: Parallelization saves 11-17h vs sequential execution

---

### Phase 3: Validation Layer (7-12h)

**Goal**: Create Pydantic DTOs to validate YAML schemas

#### TASK-003: UAC Core DTOs
- **Effort**: 7-12h
- **Priority**: HIGH
- **Dependencies**: TASK-002B, TASK-001, TASK-002A (schemas must exist first)
- **Deliverables**:
  - Core Pydantic DTOs: TrainingConfig, EnvironmentConfig, CurriculumConfig, PopulationConfig
  - SubstrateConfig DTO (integrates with TASK-002B)
  - ActionConfig DTO (integrates with TASK-002A)
  - MeterConfig DTO (integrates with TASK-001)
  - No-defaults enforcement (all behavioral parameters required)
  - Position validation for multi-dimensional substrates
- **Why Fourth**:
  - Now that schemas are defined (002B/001/002A), create validators
  - DTOs enforce "compilation over interpretation" principle
  - Required by TASK-004A (compiler uses DTOs for validation)
  - Enables fail-fast on config errors at load time

**Success Criteria**:
- [ ] All 10 core DTOs implemented with Pydantic
- [ ] No-defaults principle enforced (missing fields raise clear errors)
- [ ] SubstrateConfig validates topology, boundaries, dimensions
- [ ] ActionConfig validates movement deltas and costs
- [ ] MeterConfig validates meter definitions and ranges

---

### Phase 4: Compilation Infrastructure (46-66h parallelized)

**Goal**: Implement 7-stage universe compiler with cross-validation

#### TASK-004A: Compiler Implementation
- **Effort**: 46-66h
- **Priority**: HIGH
- **Dependencies**: TASK-003 (required), TASK-001 (optional but beneficial)
- **Deliverables**:
  - UniverseCompiler (7-stage pipeline)
  - UniverseSymbolTable (reference resolution)
  - CompilationError and ErrorCollector (error handling)
  - CompiledUniverse (immutable artifact)
  - Metadata computation (obs_dim, available_actions)
  - MessagePack caching (10-100x startup speedup)
  - Cross-file validation:
    - Dangling references (undefined meters, affordances)
    - Spatial feasibility (affordances outside grid bounds)
    - Cascade circularity (meter A → B → A)
    - Capability conflicts (incompatible affordance capabilities)
- **Why Fifth**:
  - Requires TASK-003 DTOs for validation contracts
  - Benefits from TASK-001 variable meters for dynamic sizing
  - Longest task in Phase 4 (46-66h)
  - Blocks TASK-005 (BRAIN_AS_CODE needs compiler metadata)

**Parallel Execution**: TASK-004A + TASK-004B

#### TASK-004B: UAC Capability System
- **Effort**: 8-12h
- **Priority**: HIGH
- **Dependencies**: TASK-003 (core DTOs must exist)
- **Deliverables**:
  - 6 capability DTOs: MultiTickConfig, CooldownConfig, MeterGatedConfig, SkillScalingConfig, ProbabilisticConfig, PrerequisiteConfig
  - EffectPipeline (5-stage lifecycle: immediate, tick, completion, cooldown, prerequisite)
  - BarConstraint (meter-gated access)
  - ModeConfig (state transitions)
- **Why Sixth**:
  - Extends affordance system with advanced features
  - Doesn't block TASK-004A core compiler (only extends it)
  - Enables L4+ curriculum levels (multi-tick interactions, cooldowns, prerequisites)
  - Can run parallel to TASK-004A implementation

**Success Criteria (Phase 4)**:
- [ ] Universe compiler validates all YAML files in 7 stages
- [ ] Symbol table resolves cross-file references
- [ ] Clear error messages with line numbers and examples
- [ ] CompiledUniverse cached with MessagePack (10-100x speedup)
- [ ] Cross-validation catches dangling refs, circular cascades, spatial impossibilities
- [ ] Capability system DTOs validate multi-tick, cooldowns, prerequisites

**Time Savings**: Parallelization saves 8-12h vs sequential execution

---

### Phase 5: Advanced Configuration (22-31h)

**Goal**: Move neural network architecture to YAML configuration

#### TASK-005: BRAIN_AS_CODE
- **Effort**: 22-31h
- **Priority**: MEDIUM
- **Dependencies**: TASK-002B (obs_dim computation), TASK-003 (DTO patterns), TASK-004A (compiler metadata)
- **Deliverables**:
  - Pydantic DTO schemas for brain configuration
  - NetworkFactory (config-driven network construction)
  - BrainCompiler (optimizer/loss/scheduler creation)
  - brain.yaml templates for all curriculum levels (L0-L3)
  - Architecture catalog: MLP, CNN-LSTM, Transformer variants
  - Optimizer selection: Adam, RMSprop, SGD with momentum
  - Learning rate schedules: constant, exponential decay, cosine annealing
- **Why Seventh**:
  - Depends on TASK-002B for substrate metadata (obs_dim calculation)
  - Depends on TASK-003 for DTO validation patterns
  - Depends on TASK-004A for CompiledUniverse metadata
  - Pedagogically valuable for architecture experimentation
  - Enables A/B testing different networks via config

**Success Criteria**:
- [ ] brain.yaml defines network architecture (layers, activations, sizes)
- [ ] NetworkFactory constructs networks from config
- [ ] BrainCompiler creates optimizer and loss function from config
- [ ] Learning rate schedules configurable in YAML
- [ ] All curriculum levels (L0-L3) have brain.yaml templates
- [ ] Agent training runs end-to-end from config files

---

### Phase 6: Visualization (DEFERRED)

**Goal**: GUI rendering for alternative spatial substrates

#### TASK-006: Substrate-Agnostic Visualization
- **Effort**: 14-64h (depending on scope)
- **Priority**: LOW (explicitly deferred)
- **Dependencies**: TASK-002B (substrates), TASK-002A (actions)
- **Deliverables**:
  - Hexagonal grid visualization (8-12h)
  - 3D floor projection UI (6-8h)
  - Graph node-edge rendering with D3.js (10-14h)
  - Optional: Full 3D WebGL with Three.js (20-30h)
- **Why Last (Deferred)**:
  - LOW priority - text visualization from TASK-002B sufficient for training
  - "Nice-to-have" for demos, not critical for learning
  - Students analyze metrics, not graphics (pedagogical focus)
  - Text viz covers validation, debugging, experimentation needs
  - Defer until TASK-001 through TASK-005 complete

**Deferral Rationale**:
- TASK-002B Phase 4 implements text-based visualization (4-6h)
- Text output shows agent position, affordances, meters
- Fast debugging without WebGL setup
- Works in any terminal
- Example: 3D cubic grid floor visualization in text

**When to Revisit**:
- After core UAC infrastructure complete (TASK-001 through TASK-005)
- If impressive demos needed for stakeholders
- If student feedback requests visual learners support
- If budget allows 14-64h GUI implementation

---

## Timeline and Effort Estimates

### Sequential Timeline (No Parallelization)
- **Phase 1**: 13-19h (TASK-001)
- **Phase 2**: 62-82h (TASK-002B + TASK-002A sequential)
- **Phase 3**: 7-12h (TASK-003)
- **Phase 4**: 54-78h (TASK-004A + TASK-004B sequential)
- **Phase 5**: 22-31h (TASK-005)
- **Total**: **158-222h** (excludes TASK-006)

### Parallelized Timeline (Optimal)
- **Phase 1**: TASK-001 = **13-19h**
- **Phase 2**: max(TASK-002B, TASK-002A) = **51-65h** (saves 11-17h)
- **Phase 3**: TASK-003 = **7-12h**
- **Phase 4**: max(TASK-004A, TASK-004B) = **46-66h** (saves 8-12h)
- **Phase 5**: TASK-005 = **22-31h**
- **Total**: **139-193h** (saves 19-29h vs sequential)

### Time Savings from Parallelization
- **Sequential**: 158-222h (≈20-28 working days)
- **Parallelized**: 139-193h (≈17-24 working days)
- **Savings**: 19-29h (12-13% reduction, ≈2-4 working days)

### Critical Path
The critical path (cannot be shortened without reducing scope):

```
TASK-001 → TASK-002B → TASK-003 → TASK-004A → TASK-005
13-19h   →  51-65h   →   7-12h  →  46-66h   →  22-31h
═══════════════════════════════════════════════════════
           Total: 139-193h (parallelized)
```

---

## Rationale by Design Principle

### 1. "Bones First, Content Later"

**Order**: Foundation → Validation → Compilation → Content

- **Foundation** (TASK-001/002B/002A): Define universe structure (meters, spatial, actions)
- **Validation** (TASK-003): Validate that structure with DTOs
- **Compilation** (TASK-004A): Compile and cross-validate everything
- **Content** (TASK-005): Configure AI agents within the universe

**From RESEARCH-PLAN-REVIEW-LOOP.md**:
> "Get the infrastructure right first. No point designing perfect reward functions if the 8-bar constraint limits what universes you can create."

**Applied**:
- Researched spatial substrates (foundation) ✓
- Researched UAC compiler (foundation) ✓
- Deferred reward model research (content) until foundation solid ✓

---

### 2. "Substrate Thinking" (Most Fundamental First)

**HAMLET substrate hierarchy** (from CLAUDE.md):

1. **Meters** (energy, health, money) - true state space → **TASK-001** ✓
2. **Spatial substrate** (2D grid, 3D grid, graph, or none) - optional positioning layer → **TASK-002B** ✓
3. **Actions** (movement, interaction) - defined by substrate → **TASK-002A** ✓
4. **Affordances** (bed, job, hospital) - placed in substrate → *(existing configs)*
5. **Cascades** (hunger → health) - meter dynamics → *(existing configs)*

**Pattern**: When designing systems, identify the substrate (most fundamental abstraction) and make everything else build on it.

**Applied**: Implementation order follows substrate hierarchy exactly (001 → 002B → 002A → 003 → 004A → 005)

---

### 3. "Identify Leverage Points"

**From RESEARCH-PLAN-REVIEW-LOOP.md**:
> "Research reveals highest-leverage changes - where small effort enables huge capability."

**Leverage Analysis**:
- **TASK-001 (Variable meters)**: 13-19h → Unblocks 4-meter to 32-meter universes ✓ **HIGHEST LEVERAGE**
- **TASK-002A (Actions)**: 11-17h → Config-driven action space, multi-meter costs
- **TASK-002B (Substrates)**: 51-65h → Enables 2D/3D/hex/graph/aspatial topologies

vs lower-leverage:
- **TASK-006 (Visualization)**: 14-64h → Only affects GUI, text viz sufficient

**Pattern**: Prioritize enabling capabilities over optimizing content. Capabilities compound, content is linear.

**Applied**: TASK-001 first (highest leverage), TASK-006 deferred (lowest leverage)

---

### 4. "Compilation Over Interpretation"

**No-Defaults Principle**:
> "All behavioral parameters must be explicitly specified in config files. No implicit defaults allowed."

**Why**: Hidden defaults create non-reproducible configs, operator doesn't know what values are being used, and changing code defaults silently breaks old configs.

**Applied**:
- **TASK-003** enforces no-defaults with Pydantic DTOs (required fields)
- **TASK-004A** validates at load time (compilation), not runtime (interpretation)
- **CompiledUniverse** is immutable artifact, computed once, used many times

**7-Stage Compilation Pipeline** (TASK-004A):
1. Parse YAML files
2. Build symbol tables
3. Resolve cross-file references
4. Cross-validate (dangling refs, circularity, spatial feasibility)
5. Compute metadata (obs_dim, available_actions)
6. Optimize (constant folding, precompute lookups)
7. Emit CompiledUniverse artifact

**Benefits**:
- Fail-fast on config errors (catch at load time)
- 10-100x startup speedup with MessagePack caching
- Clear error messages with line numbers and examples
- Operator accountability (no hidden defaults)

---

### 5. "Enable Experimentation" ("Fuck Around and Find Out")

**From RESEARCH-PLAN-REVIEW-LOOP.md**:
> "The bar for this sort of thing should be low if there's any pedagogical value, letting students or hobbyists just 'fuck around and find out' is half the point."

**Pattern**: If feature enables experimentation ("What if the world was a sphere?"), include it even if it's not "serious" or "practical."

**Applied in TASK-002B** (Spatial Substrates):
- ✓ 3D cubic grids (high value, low effort) - "What if there are floors?"
- ✓ Toroidal boundaries (literally change clamp to modulo) - "What if the world wraps?"
- ✓ Aspatial substrates (reveals grid is optional) - "What if there's no space?"
- ✓ Hexagonal grids (teaches coordinate systems) - "What if cells have 6 neighbors?"

**Applied in TASK-001** (Variable Meters):
- ✓ 4-meter tutorials - "What if I only track energy and hunger?"
- ✓ 32-meter complex sims - "What if I model neurotransmitters?"
- ✓ Domain-specific universes - "What if this is a factory, not a Sim?"

---

### 6. "Research Identifies Dependencies"

**From RESEARCH-PLAN-REVIEW-LOOP.md**:
> "Research phase reveals true dependency graph, not just logical task breakdown."

**Example**:
- **Requirement**: "Make action space configurable"
- **Research insight**: "Actions depend on substrate (4-way for square, 6-way for hex, 0-way for aspatial)"
- **Conclusion**: Substrate (TASK-002B) must inform Actions (TASK-002A)

**Applied**: Initially planned TASK-002A first, but research revealed TASK-002B should run parallel (actions informed by substrate, not blocked by it).

**Dependency Discovery** (from subagent analysis):
- TASK-003 notes "SubstrateConfig (TASK-002B integration)" → 002B precedes 003
- TASK-004A "TASK-003 (required)" → 003 precedes 004A
- TASK-005 "depends on TASK-002B (obs_dim)" → 002B precedes 005
- TASK-006 "TASK-002B (Spatial Substrates)" → 002B precedes 006

---

### 7. "Exposed vs Hidden Knobs"

**Test**: "Does changing this parameter change gameplay meaningfully?"
- **Yes** → Should be in YAML (exposed knob)
- **No** → Can stay in code (implementation detail)

**Applied across tasks**:

**TASK-002B (Substrates)**:
- ✓ Exposed: `grid_size`, `topology`, `boundary_behavior`, `distance_metric`
- ✗ Hidden: `tensor_dtype`, `device_placement`

**TASK-002A (Actions)**:
- ✓ Exposed: `move_energy_cost`, `action_deltas`, `multi_meter_costs`
- ✗ Hidden: `action_tensor_indexing`

**TASK-005 (BRAIN_AS_CODE)**:
- ✓ Exposed: `hidden_layer_sizes`, `activation_functions`, `learning_rate`, `optimizer_type`
- ✗ Hidden: `cuda_kernel_selection`, `tensor_memory_layout`

**Pattern**: Expose all parameters that affect the world's physics. Hide only true implementation details.

---

## Risk Analysis

### High-Risk Dependencies

**Risk 1: TASK-004A blocks TASK-005**
- **Issue**: 46-66h compiler must complete before 22-31h BRAIN_AS_CODE
- **Impact**: If TASK-004A slips, TASK-005 delayed
- **Mitigation**:
  - Start TASK-005 planning/design during TASK-004A implementation
  - Create brain.yaml schema early (doesn't require compiler)
  - Implement NetworkFactory stub that works with simple configs

**Risk 2: TASK-002B is longest critical path**
- **Issue**: 51-65h for substrates, longest task in Phase 2
- **Impact**: Delays all subsequent phases
- **Mitigation**:
  - Parallelize with TASK-002A (saves 11-17h)
  - Break TASK-002B into incremental phases (2D first, then 3D/hex)
  - Use text visualization (4-6h) before full implementation (51-65h)

**Risk 3: Schema changes in 002B/002A require DTO revisions**
- **Issue**: TASK-003 DTOs might need updates after 002B/002A implementation
- **Impact**: Rework effort, potential delays
- **Mitigation**:
  - Keep DTOs flexible with Optional fields during development
  - Use Pydantic's schema evolution features (Field aliases, validators)
  - Document schema changes in TASK-002B/002A completion notes

**Risk 4: Checkpoint compatibility breaks**
- **Issue**: TASK-001 changes meter count, breaks existing checkpoints
- **Impact**: Loss of trained agents, curriculum progress
- **Mitigation**:
  - TASK-001 includes checkpoint metadata (num_meters field)
  - Implement checkpoint migration tool (8-meter → variable-meter)
  - Retain old 8-meter configs for legacy compatibility

---

### Low-Risk Opportunities

**Opportunity 1: TASK-001 is quick win**
- **Advantage**: 13-19h, no dependencies, CRITICAL priority
- **Benefit**: Immediate design space unlock, boosts momentum
- **Recommendation**: Start with TASK-001 for fast, high-impact delivery

**Opportunity 2: TASK-004B doesn't block TASK-004A**
- **Advantage**: Capability DTOs can run parallel to compiler
- **Benefit**: Saves 8-12h by overlapping Phase 4
- **Recommendation**: Start TASK-004B during TASK-004A implementation

**Opportunity 3: TASK-006 is explicitly deferred**
- **Advantage**: Text viz from TASK-002B sufficient for training
- **Benefit**: Defer 14-64h GUI work indefinitely if not needed
- **Recommendation**: Only implement if demo/stakeholder requirements emerge

**Opportunity 4: Incremental substrate rollout**
- **Advantage**: TASK-002B phases can deliver value incrementally
- **Benefit**: 2D substrates first (24h), then 3D/hex (27-41h additional)
- **Recommendation**: Break TASK-002B into Phase 1 (2D) and Phase 2 (3D/hex/graph)

---

## Alternative Ordering: Contract-First Approach

If preferring **contract-first design**, TASK-003 could move earlier:

### Alternative Sequence:
1. TASK-001 (Meters) - 13-19h
2. **TASK-003 (Core DTOs)** - 7-12h ← **EARLIER**
3. TASK-002B (Substrates) - 51-65h
4. TASK-002A (Actions) - 11-17h
5. TASK-004B (Capability DTOs) - 8-12h
6. TASK-004A (Compiler) - 46-66h
7. TASK-005 (BRAIN_AS_CODE) - 22-31h

**Pros**:
- Forces validation thinking during design
- DTOs define contracts before implementation
- Fail-fast on schema mismatches

**Cons**:
- TASK-003 needs to know schema structure from 002B/002A
- DTOs might require revision after 002B/002A implementation
- Rework effort if schemas change

**Trade-off**: DTOs document intended schemas, but may need updates when implementation reveals edge cases.

**Recommendation**: **Stick with Option A (Implementation First)** because:
- Subagent noted "SubstrateConfig (TASK-002B integration)" suggesting 002B precedes 003
- TASK-002B/002A are well-specified in existing task docs (schemas already designed)
- DTOs validate known schemas, not exploratory ones
- Implementation-first reduces rework risk

---

## Success Metrics

### Phase Completion Criteria

**Phase 1 Complete When**:
- [ ] Variable meter count (1-32) configurable in YAML
- [ ] 4-meter and 12-meter configs load successfully
- [ ] Agents train end-to-end with variable meters
- [ ] Checkpoints include num_meters metadata

**Phase 2 Complete When**:
- [ ] 2D, 3D, hex, graph, aspatial substrates implemented
- [ ] substrate.yaml loads and validates
- [ ] actions.yaml defines movement and costs
- [ ] Agent navigates all substrate types
- [ ] Text visualization renders all substrates

**Phase 3 Complete When**:
- [ ] All 10 core DTOs implemented
- [ ] No-defaults principle enforced (missing fields raise errors)
- [ ] SubstrateConfig, ActionConfig, MeterConfig validate schemas
- [ ] Clear error messages with examples on validation failure

**Phase 4 Complete When**:
- [ ] 7-stage compilation pipeline implemented
- [ ] Cross-file validation catches dangling refs, circularity, spatial impossibilities
- [ ] CompiledUniverse cached with MessagePack (10-100x speedup)
- [ ] Capability DTOs validate multi-tick, cooldowns, prerequisites

**Phase 5 Complete When**:
- [ ] brain.yaml defines network architecture
- [ ] NetworkFactory constructs networks from config
- [ ] All curriculum levels (L0-L3) have brain.yaml templates
- [ ] Agent training runs end-to-end from config files only

**Phase 6 Deferred** (revisit after Phase 5 complete)

---

## Integration Points

### Between Phases

**Phase 1 → Phase 2**:
- TASK-001 provides variable meter count → TASK-002B uses for obs_dim calculation
- TASK-001 provides meter definitions → TASK-002A uses for multi-meter costs

**Phase 2 → Phase 3**:
- TASK-002B provides substrate.yaml schema → TASK-003 creates SubstrateConfig DTO
- TASK-002A provides actions.yaml schema → TASK-003 creates ActionConfig DTO

**Phase 3 → Phase 4**:
- TASK-003 provides core DTOs → TASK-004A uses for validation contracts
- TASK-003 provides SubstrateConfig → TASK-004A validates spatial feasibility

**Phase 4 → Phase 5**:
- TASK-004A provides CompiledUniverse → TASK-005 reads metadata (obs_dim)
- TASK-004A provides symbol table → TASK-005 references meter names

**Phase 5 → Phase 6** (deferred):
- TASK-005 provides trained agents → TASK-006 visualizes in GUI

---

## Lessons Applied from Research Method

### From RESEARCH-PLAN-REVIEW-LOOP.md:

**Lesson 1: "Bones First, Content Later"** ✓
- Foundation (001/002B/002A) before content (005)

**Lesson 2: Technical Debt vs Design Constraints** ✓
- TASK-001 removes 8-meter constraint (was debt, not design)

**Lesson 3: Substrate Thinking** ✓
- Implementation order follows substrate hierarchy (meters → spatial → actions)

**Lesson 4: "Fuck Around and Find Out" Pedagogy** ✓
- TASK-002B enables topology experimentation (3D, hex, toroidal, aspatial)

**Lesson 5: Research Identifies Leverage Points** ✓
- TASK-001 first (13-19h unlocks entire design space)

**Lesson 6: Dependency Ordering from Research** ✓
- Subagent analysis revealed true dependencies (002B → 003, 003 → 004A, 004A → 005)

**Lesson 7: Exposed vs Hidden Knobs** ✓
- All gameplay parameters moved to YAML (substrate, actions, brain)

**Lesson 8: Research Documents as Future Reference** ✓
- This document captures rationale for task ordering

**Lesson 9: When Research Prevents Rework** ✓
- Dependency analysis prevents implementing tasks out of order

**Lesson 10: Distinguishing Research from Analysis Paralysis** ✓
- Subagent analysis time-boxed (parallel execution, concise extraction)
- Stopping criteria: dependency graph mapped, optimal ordering determined

---

## Recommendations

### Immediate Next Steps

1. **Start TASK-001 (Variable Meters)** - 13-19h
   - Quick win, highest leverage, no dependencies
   - Creates momentum for subsequent tasks

2. **Plan TASK-002B and TASK-002A concurrently**
   - Design substrate.yaml and actions.yaml schemas
   - Identify integration points (action deltas depend on substrate)

3. **Defer TASK-006 indefinitely**
   - Text visualization from TASK-002B is sufficient
   - Revisit only if demo requirements emerge

### Parallelization Strategy

- **Phase 2**: Assign TASK-002B and TASK-002A to separate developers (saves 11-17h)
- **Phase 4**: Assign TASK-004A and TASK-004B to separate developers (saves 8-12h)
- **Total savings**: 19-29h (12-13% reduction)

### Incremental Delivery

Rather than "big bang" delivery after 139-193h:

**Milestone 1** (13-19h): TASK-001 complete
- Variable meter system deployed
- 4-meter and 12-meter configs testable

**Milestone 2** (64-84h): Phase 2 complete
- Substrate and action space configurable
- Students can experiment with 3D/hex/graph topologies

**Milestone 3** (71-96h): Phase 3 complete
- DTO validation enforces schema correctness
- Config errors caught at load time with clear messages

**Milestone 4** (117-162h): Phase 4 complete
- Universe compiler validates cross-file references
- 10-100x startup speedup with caching

**Milestone 5** (139-193h): Phase 5 complete
- BRAIN_AS_CODE enables architecture experimentation
- All 5 curriculum levels configurable via YAML

---

## Conclusion

The optimal task execution order is:

**001 → (002B ∥ 002A) → 003 → (004A ∥ 004B) → 005 → (006 deferred)**

This sequence:
- **Respects dependencies** (no task starts before prerequisites complete)
- **Maximizes parallelization** (saves 19-29h via concurrent execution)
- **Follows substrate hierarchy** (meters → spatial → actions → validation → compilation)
- **Prioritizes leverage** (TASK-001 first: shortest effort, highest impact)
- **Defers low-value work** (TASK-006 text viz sufficient)
- **Enables incremental delivery** (5 milestones instead of single release)

**Total Timeline**: 139-193h parallelized (vs 158-222h sequential)

**Critical Path**: TASK-001 → TASK-002B → TASK-003 → TASK-004A → TASK-005

**Next Action**: Begin TASK-001 (Variable-Size Meter System) for immediate design space unlock.

---

## Appendix: Task Summaries

### TASK-001: Variable-Size Meter System
**GOAL**: Make meter count configurable instead of hardcoded to exactly 8, enabling 4-meter tutorials, 12-meter complex simulations, and domain-specific universes.

**DEPENDS ON**: None (foundational)

**PROVIDES**: Variable-size meter system (1-32 meters), dynamic tensor sizing, 4-meter and 12-meter example configs, checkpoint metadata, foundation for TASK-003 and TASK-004A.

**EFFORT**: 13-19 hours

**PRIORITY**: CRITICAL (highest-leverage infrastructure change)

**TASK FILE**: `docs/tasks/TASK-001-VARIABLE-METERS.md`

---

### TASK-002A: UAC Action Space
**GOAL**: Move hardcoded action space from Python to config-driven YAML (actions.yaml) to achieve full UNIVERSE_AS_CODE compliance.

**DEPENDS ON**: None (foundational, informs TASK-001 scope)

**PROVIDES**: Config-driven action space schema with Pydantic DTOs, multi-meter cost system, abstraction layer enabling alternative universes, pattern consistency with affordances.yaml.

**EFFORT**: 11-17 hours

**PRIORITY**: HIGH (proves UAC concept)

**TASK FILE**: `docs/tasks/TASK-002A-UAC-ACTION-SPACE.md`

---

### TASK-002B: Configurable Spatial Substrates
**GOAL**: Move spatial substrate (grid topology, boundaries, distance metrics) from hardcoded Python to config-driven YAML, enabling 2D/3D/hex/continuous/graph/aspatial universes.

**DEPENDS ON**: None

**PROVIDES**: Substrate abstraction layer (SpatialSubstrate interface), Pydantic schema for substrate.yaml, coordinate encoding for 3D grids, reference implementations (SquareGrid, CubicGrid, Aspatial), text-based visualization.

**EFFORT**: 51-65 hours

**PRIORITY**: HIGH (foundational for TASK-003/002A/006)

**TASK FILE**: `docs/tasks/TASK-002B-SPATIAL-SUBSTRATES.md`

---

### TASK-003: UAC Core DTOs
**GOAL**: Create core DTOs for training, environment, curriculum, population, substrate, and master configuration with no-defaults enforcement.

**DEPENDS ON**: TASK-002B, TASK-001, TASK-002A (schemas must exist first)

**PROVIDES**: 10 foundational DTOs including SubstrateConfig, ActionConfig, MeterConfig, position validation for multi-dimensional substrates.

**EFFORT**: 7-12 hours

**PRIORITY**: HIGH (foundational for entire UAC system)

**TASK FILE**: `docs/tasks/TASK-003-UAC-CORE-DTOS.md`

---

### TASK-004A: Universe Compiler
**GOAL**: Implement 7-stage universe compiler that validates YAML configs, resolves cross-file references, and emits immutable CompiledUniverse artifact with caching.

**DEPENDS ON**: TASK-003 (required), TASK-001 (optional)

**PROVIDES**: UniverseCompiler (7-stage pipeline), UniverseSymbolTable, CompilationError/ErrorCollector, CompiledUniverse artifact, metadata computation, MessagePack caching (10-100x speedup), cross-file validation.

**EFFORT**: 46-66 hours

**PRIORITY**: HIGH (foundational for UAC integrity)

**TASK FILE**: `docs/tasks/TASK-004A-UNIVERSE-COMPILER.md`

---

### TASK-004B: UAC Capability System
**GOAL**: Extend AffordanceConfig with capability composition (multi_tick, cooldown, meter_gated, skill_scaling, probabilistic, prerequisite) and effect pipelines.

**DEPENDS ON**: TASK-003 (core DTOs must exist)

**PROVIDES**: 6 capability DTOs, EffectPipeline (5-stage lifecycle), BarConstraint, ModeConfig for advanced curriculum levels (L4+).

**EFFORT**: 8-12 hours

**PRIORITY**: HIGH (enables advanced features)

**TASK FILE**: `docs/tasks/TASK-004B-UAC-CAPABILITY-SYSTEM.md`

---

### TASK-005: BRAIN_AS_CODE
**GOAL**: Move hardcoded neural network architecture and learning hyperparameters to YAML configuration (brain.yaml) to enable architecture experimentation.

**DEPENDS ON**: TASK-002B (obs_dim), TASK-003 (DTOs), TASK-004A (compiler)

**PROVIDES**: Pydantic brain config DTOs, NetworkFactory, BrainCompiler, brain.yaml templates for L0-L3, ability to A/B test architectures via config.

**EFFORT**: 22-31 hours

**PRIORITY**: MEDIUM (pedagogically valuable)

**TASK FILE**: `docs/tasks/TASK-005-BRAIN-AS-CODE.md`

---

### TASK-006: Substrate-Agnostic Visualization
**GOAL**: Implement substrate-agnostic GUI rendering to visualize alternative spatial substrates (hexagonal, 3D, graph, aspatial).

**DEPENDS ON**: TASK-002B (substrates), TASK-002A (actions)

**PROVIDES**: Hexagonal grid viz (8-12h), 3D floor projection (6-8h), graph rendering (10-14h), optional full 3D WebGL (20-30h).

**EFFORT**: 14-64 hours (depending on scope)

**PRIORITY**: LOW (deferred - text viz from TASK-002B sufficient)

**TASK FILE**: `docs/tasks/TASK-006-SUBSTRATE-VISUALIZATION.md`
