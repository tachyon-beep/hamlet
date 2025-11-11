# Townlet Codebase Dependency Analysis

## Executive Summary

The Townlet codebase (98 Python files across 12 modules) exhibits a **layered architecture with strong separation of concerns** and clear hot/cold path separation. Import patterns reveal a hexagonal/ports-and-adapters design centered on the Universe Compiler module.

### Key Metrics
- **Total Python Files**: 98
- **Major Modules**: 12 (vfs, config, substrate, environment, universe, training, exploration, agent, population, curriculum, demo, recording)
- **Top Central Dependency**: `townlet.vfs.schema` (13 imports)
- **Circular Dependencies Detected**: 2 (1 medium severity, 1 low severity)
- **Architectural Pattern**: Layered + Hexagonal

---

## Top 10 Most Imported Modules

| Rank | Module | Import Count | Role |
|------|--------|--------------|------|
| 1 | `townlet.vfs.schema` | 13 | VFS type definitions (foundation) |
| 2 | `townlet.environment.action_config` | 12 | Action space configuration |
| 3 | `townlet.training.state` | 10 | Training state DTOs |
| 4 | `townlet.environment.cascade_config` | 10 | Meter mechanics configuration |
| 5 | `townlet.config.base` | 9 | YAML loading utilities |
| 6 | `townlet.substrate.config` | 8 | Spatial substrate configuration |
| 7 | `townlet.substrate.base` | 7 | Spatial base class |
| 8 | `townlet.config.affordance` | 5 | Affordance definitions |
| 9 | `townlet.config.bar` | 5 | Meter definitions |
| 10 | `townlet.config.cascade` | 5 | Cascade definitions |

**Insight**: The top 3 dependencies are all foundational types (VFS, actions, training state), indicating well-chosen core abstractions.

---

## Architectural Layers (Discovery Order)

### Layer 0: Foundation (100% isolated)
- **VFS (4 files)**: Pure type definitions with zero external townlet imports
- **Pattern**: Single responsibility - defines all Variable & Feature System schemas

### Layer 1: Configuration (88.6% internal cohesion)
- **Config (15 files)**: YAML loading, validation, config DTOs
- **Substrate.Config (8 imports)**: Spatial configuration schemas
- **Dependencies**: Minimal - mostly self-contained

### Layer 2: Domain Mechanics (66-72% internal cohesion)
- **Substrate (10 files)**: Grid2D, Grid3D, GridND, Continuous, Aspatial implementations
- **Environment (13 files)**: Affordances, cascades, vectorized simulation engine
- **Coupling**: Bidirectional ⚠️ (environment ↔ substrate)

### Layer 3: Universe Compiler (HUB MODULE)
- **18 files**: 7-stage compilation pipeline
- **Imports**: config (30%), environment (15.7%), vfs (15.7%), substrate (7.1%), internal (31.4%)
- **Pattern**: Hexagonal architecture - INBOUND (config) → PROCESSORS → OUTBOUND (compiled)
- **Role**: Configuration validation and compilation

### Layer 4: Training Loop (HOT PATH)
- **Training (6 files)**: State, checkpoints, logging (100% depends on universe only)
- **Exploration (6 files)**: Strategies (60% internal, 40% to training)
- **Agent (2 files)**: Neural network architectures
- **Pattern**: Clean isolation - no circular dependencies
- **Characteristic**: Uses PyTorch tensors, GPU operations

### Layer 5: Orchestration
- **Population (4 files)**: Manages batched agents (33% exploration, 33% training)
- **Curriculum (4 files)**: Difficulty progression (60% training, 40% internal)
- **Demo (5 files)**: Integration hub (depends on 9 modules - intentional orchestrator)
- **Recording (8 files)**: Episode recording and replay (83% internal)
- **Compiler (2 files)**: CLI interface

---

## Circular Dependencies Identified

### Cycle 1: environment ↔ substrate (MEDIUM SEVERITY)
```
townlet.substrate → townlet.environment.action_config (8 imports)
townlet.environment → townlet.substrate.base (3 imports)
```

**Root Cause**: ActionConfig is needed by both layers
- Substrate needs action definitions to validate action space
- Environment needs substrate characteristics (grid size, etc.)

**Impact**: Medium - creates bidirectional coupling but not a runtime issue
**Recommendation**:
- Option A: Extract ActionConfigInterface as abstraction
- Option B: Move action validation to environment layer
- Option C: Accept and document as intentional

### Cycle 2: demo ↔ recording (LOW SEVERITY)
```
townlet.recording → townlet.demo.database (1 import)
townlet.demo → townlet.recording.replay (2 imports)
```

**Root Cause**: Inference server needs replay; replayer needs database
**Impact**: Low - minor coupling in optional inference feature
**Recommendation**: Extract shared database interface to separate module

---

## Key Dependency Patterns

### Pattern 1: Hexagonal Architecture (Universe Compiler)
```
CONFIG (inbound) → [Compiler Pipeline] → COMPILED UNIVERSE (outbound)
                      ↓
                  [Symbol Table]
                      ↓
                  [Cross-reference]
                      ↓
                  [Validation]
                      ↓
                  [DTO Generation]
```

### Pattern 2: Domain-Driven Design
- **Domain Layer**: environment, substrate (business logic)
- **Application Layer**: training, population, curriculum (orchestration)
- **Infrastructure Layer**: config, universe compiler (bootstrap)
- **Interface Layer**: demo, recording (external contracts)

### Pattern 3: Strict Layering (Mostly)
Clear unidirectional dependency flow:
```
Application Layer (demo, compiler)
    ↓
Training Loop (population, curriculum, exploration, agent, training)
    ↓
Universe Compiler (cold path validation)
    ↓
Domain Mechanics (environment, substrate)
    ↓
Configuration & Utilities (config, substrate.config)
    ↓
Foundation (vfs)
```

### Pattern 4: Hub & Spoke Topology
Two critical hubs identified:

**Hub 1: Universe Module (Config Time)**
- Spoke: config (input raw YAML)
- Spoke: environment (compilation target)
- Spoke: vfs (observation specs)
- Spoke: substrate (spatial validation)

**Hub 2: Demo Module (Runtime)**
- Spoke: universe (compiled universe)
- Spoke: training (training loop)
- Spoke: population (agent management)
- Spoke: exploration (strategy selection)
- Spoke: environment (simulation)
- Spoke: substrate (physics)
- Spoke: curriculum (difficulty)
- Spoke: recording (visualization)

### Pattern 5: Hot/Cold Path Separation
Clear separation of initialization from execution:

**Cold Path** (Configuration):
- config → universe compiler → CompiledUniverse
- Pydantic validation everywhere
- Zero PyTorch tensors
- One-time execution

**Hot Path** (Training):
- CompiledUniverse → training loop
- PyTorch tensors throughout
- Vectorized GPU operations
- Repeated execution

---

## Cross-Cutting Concerns

### Config Base Module
- **Role**: Shared YAML loading and validation utilities
- **Usage**: 9 files across config submodules
- **Pattern**: Utility module with no domain logic

### Training State Module
- **Role**: Cold path DTOs (CurriculumDecision, BatchedAgentState, etc.)
- **Usage**: training, population, curriculum modules
- **Pattern**: Pydantic validation for immutable state objects

### VFS Schema Module
- **Role**: Foundation type system for Variable & Feature System
- **Usage**: Most imported module (13 times)
- **Pattern**: Foundational abstractions used everywhere

---

## Module Cohesion Analysis

### High Cohesion (✓ Good)
- **Substrate** (10 files): All spatial types + config
- **Environment** (13 files): All affordances, cascades, vectorized env
- **Config** (15 files): All config loading utilities
- **Universe** (18 files): All compilation stages + DTOs
- **Demo** (5 files): Training runner, inference, database

### Medium Cohesion
- **Training** (6 files): state, replay_buffer, checkpoints, logging
- **Exploration** (6 files): epsilon_greedy, rnd, adaptive_intrinsic

### Low Cohesion (Candidates for Splitting)
- **Recording** (8 files): video_renderer, video_export, replay, criteria
  - Could split: video_generation vs. replay_management

---

## Architectural Health Assessment

### Strengths
1. ✓ **Clear hot/cold separation** - Config loading isolated from training
2. ✓ **VFS fully isolated** - 100% internal cohesion, zero external imports
3. ✓ **No downward violations** - config never imports training
4. ✓ **Universe compiler as clear hub** - well-defined integration point
5. ✓ **Demo orchestration** - successfully glues together 9 modules

### Weaknesses
1. ⚠️ **environment ↔ substrate cycle** - Medium impact bidirectional coupling
2. ⚠️ **ActionConfig bridging layers** - Tight coupling across boundary
3. ⚠️ **Universe compiler fan-in** - 30% of imports from config layer
4. ⚠️ **Demo module thin orchestrator** - Depends on 9 modules (necessary but complex)

### Testing Implications
- **VFS layer**: Unit tests (isolated)
- **Config layer**: Integration tests with universe compiler
- **Environment/Substrate**: Mock one or accept bidirectional dependency in tests
- **Demo**: End-to-end tests (depends on all layers)

---

## Recommendations

### 1. Resolve environment ↔ substrate Cycle (PRIORITY: Medium)
**Current**:
- substrate imports action_config from environment (8 times)
- environment imports substrate.base (3 times)

**Options**:
- **Option A (Recommended)**: Extract ActionConfigInterface abstract base
  - Benefit: Maintains separation of concerns
  - Cost: Small refactoring effort

- **Option B (Fast)**: Accept cycle and document as intentional
  - Benefit: Zero refactoring effort
  - Cost: Architectural documentation becomes critical

### 2. Reduce Universe Compiler Complexity (PRIORITY: Low)
**Current**: 18 files handling 7-stage pipeline
**Suggest**: Split compiler.py into stage modules:
- `parse_stage.py`
- `resolution_stage.py`
- `validation_stage.py`
- `emission_stage.py`

**Benefit**: Easier to navigate compilation pipeline
**Cost**: Moderate refactoring

### 3. Strengthen VFS Module Isolation (PRIORITY: Low)
**Current**: 100% isolated
**Future**: Expand to action derivations (phase 2) while maintaining isolation

### 4. Document Architectural Decisions (PRIORITY: High)
Create Architecture Decision Record (ADR) explaining:
- Why demo depends on 9 modules (intentional orchestration)
- Why circular dependency environment ↔ substrate exists
- Why training has minimal imports (hot path isolation)
- Cold/hot path separation rationale

---

## Conclusion

The Townlet architecture demonstrates **strong foundational design** with clear separation of concerns, proper layering, and intentional hub modules. The primary areas for improvement are:

1. Documenting the intentional environment ↔ substrate coupling
2. Optionally refactoring to use dependency inversion (ActionConfigInterface)
3. Adding architectural decision records for future maintainers

The import patterns reveal a well-thought-out system that balances clean architecture with pragmatic engineering constraints.
