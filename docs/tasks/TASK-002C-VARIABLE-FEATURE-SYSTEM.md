# TASK-002C: Variable & Feature System (VFS) - Phase 1 Foundation

**Status**: Planned
**Priority**: High
**Estimated Effort**: 16-20 hours (2-3 days)
**Dependencies**: TASK-002A (Stratum), TASK-002B (ACS)
**Enables**: TASK-003 (UAC Core DTOs), TASK-005 (Brain as Code), TASK-004A (UAC Compiler)
**Created**: 2025-11-06
**Completed**: [Pending]

**Keywords**: variables, features, observations, typed-contracts, UAC, BAC, ACS, schema, registry, cross-cutting
**Subsystems**: VFS (new), ACS (action schema), BAC (observation spec), UAC (variable definitions), Engine (storage)
**Architecture Impact**: Major (new foundational subsystem)
**Breaking Changes**: No (purely additive in Phase 1)

---

## AI-Friendly Summary (Skim This First!)

**What**: Create a Variable & Feature System (VFS) that defines typed variables (scalars, vectors, booleans) with clear scope/lifetime/access control, exposes selected ones as observations to BAC, and lets ACS actions read/write them through typed contracts.

**Why**: BAC (TASK-003) and UAC (TASK-004) compilers need typed contracts for observations and variables. Without VFS, both will hardcode observation shapes and use ad-hoc variable handling, requiring expensive cross-system refactoring later. VFS also solves BLOCKER 2 (world model + curriculum changes) by making world configuration observable.

**Scope**: Phase 1 establishes minimal foundation (schema + simple registry + observation specs + action annotations). Complex features (derivation graphs, incremental masks, complex types) deferred to Phase 2 post-TASK-004.

**Quick Assessment**:

- **Current Limitation**: Observations hardcoded (29 dims L1, 54 dims L2), variables ad-hoc (meters as arrays)
- **After Implementation**: Typed variable contracts, declarative observation specs, action read/write annotations
- **Unblocks**: BAC compiler (TASK-003) uses observation specs, UAC compiler (TASK-004) generates variable definitions
- **Impact Radius**: 5 subsystems (VFS, ACS, BAC, UAC, Engine)

**Decision Point**: If you're not working on compilers or observation specs, STOP READING HERE.

---

## Problem Statement

### Current Constraint

**Observations are hardcoded**:
```python
# Current: Hardcoded observation dimensions in environment
obs_dim = 29  # for L1 full observability
obs_dim = 54  # for L2 POMDP

# BAC compiler (TASK-003) will need to hardcode these shapes
# UAC compiler (TASK-004) has no target schema for variables
```

**Variables are ad-hoc**:
```python
# Meters stored as raw arrays with magic indices
meters[0] = energy  # Index 0 is energy... because we said so
meters[3] = money   # Index 3 is money... because we said so

# Actions can't declare "I read energy and write position"
# World model can't condition on "which world physics am I in?"
```

**No typed contracts**:
- BAC compiler has no spec for "what observations exist?"
- UAC compiler has no schema for "what does a variable definition look like?"
- ACS can't declare which variables an action reads/writes
- Engine has no way to expose `world_config_hash` for BLOCKER 2 solution

### Why This Is Technical Debt, Not Design

**Test**: "If removed, would the system be more expressive or more fragile?"

**Answer**: More expressive

- ✅ Enables: BAC compiler to generate observation heads dynamically
- ✅ Enables: UAC compiler to validate variable definitions
- ✅ Enables: ACS to declare action dependencies explicitly
- ✅ Enables: World model to condition on world configuration (BLOCKER 2 solution)
- ✅ Enables: Teleport actions, custom affordance state, arbitrary stash/pop behavior
- ❌ Does NOT: Break existing code (purely additive)

**Conclusion**: Technical debt that blocks two critical compilers.

### Impact of Current Constraint

**Cannot Create**:

- BAC compiler (TASK-003) without hardcoding observation shapes
- UAC compiler (TASK-004) without ad-hoc variable handling
- Teleport actions (no place to store `home_pos` variable)
- Stack/queue operations (no `energy_delta_stack` variable)
- World-model curriculum adaptation (no `world_config_hash` observable)

**Pedagogical Cost**:

- Students can't see "what variables exist in this universe?"
- Can't explain "energy is index 0" declaratively
- Can't demonstrate "agent's private memory" vs "public state"

**Research Cost**:

- Can't experiment with custom variables without engine changes
- Can't A/B test "does agent need to see home_pos?"
- Can't implement Theory of Mind (needs belief over hidden variables)

**From Analysis**: This is a foundational cross-cutting layer that three major tasks depend on.

---

## Solution Overview

### Design Principle

**Core Philosophy**: Make stateful variables first-class citizens in UAC with typed contracts that BAC, ACS, and Engine all bind to.

**Key Insight**: Variables are not implementation details—they're part of the universe specification. Declaring them in YAML makes the universe inspectable, reproducible, and composable.

### Architecture Changes

**1. VFS Module (NEW)**: Defines schemas, registry, observation builder
   - `schema.py`: VariableDef, ObservationField, WriteSpec
   - `registry.py`: Runtime variable storage (get/set with access control)
   - `observation_builder.py`: Generates observation specs for BAC compiler

**2. ACS Extension**: Action schema gains `reads` and `writes` fields
   - Actions declare which variables they read (for masks)
   - Actions declare which variables they write (for effects)

**3. BAC Interface**: Observation specs replace hardcoded dimensions
   - BAC compiler (TASK-003) consumes `ObservationField` list
   - Generates input head dimensions dynamically

**4. UAC Interface**: Variable definitions become compilation target
   - UAC compiler (TASK-004) generates `List[VariableDef]`
   - Engine instantiates VFS registry from these definitions

### Compatibility Strategy

**Backward Compatibility**:

- Existing configs work unchanged (VFS is opt-in initially)
- Meters continue working as arrays in Phase 1
- Migration to VFS-backed meters in Phase 2

**Migration Path**:

- Phase 1: VFS exists alongside legacy meters
- Phase 2 (post-TASK-004): Migrate meters to VFS storage
- Phase 3: Deprecate legacy meter arrays

**Versioning**:

- VFS schema version in `variables.yaml`
- UAC compiler (TASK-004) validates version compatibility

---

## Detailed Design

### Phase 1: Minimal Foundation (16-20 hours)

**Objective**: Establish typed contracts for variables, observations, and action dependencies without complex runtime behavior.

**Changes**:

- **File**: `src/townlet/vfs/__init__.py` (NEW)
  - Module initialization
  - Export public API

- **File**: `src/townlet/vfs/schema.py` (NEW)
  - `VariableDef`: Variable declaration (id, scope, type, lifetime, access control)
  - `ObservationField`: Observation exposure (source variable, shape, normalization)
  - `WriteSpec`: Action write declaration (variable_id, expression)
  - `NormalizationSpec`: Min/max or standardization parameters

- **File**: `src/townlet/vfs/registry.py` (NEW)
  - `VariableRegistry`: Runtime storage
    - `__init__(variable_defs)`: Initialize with schema
    - `get(scope, var_id)`: Get variable value (with access check)
    - `set(scope, var_id, value)`: Set variable value (with access check)
    - `get_observation_spec()`: Return list of ObservationField for BAC

- **File**: `src/townlet/vfs/observation_builder.py` (NEW)
  - `ObservationBuilder`: Constructs observation specs
    - `build_observation_spec(variables, exposures)`: Generate ObservationSpec
    - Returns shape, dtype, normalization for each field

- **File**: `src/townlet/acs/action_schema.py` (MODIFY)
  - Extend `ActionSchema` dataclass:
    - Add `reads: List[str] = field(default_factory=list)`
    - Add `writes: List[WriteSpec] = field(default_factory=list)`

- **File**: `configs/templates/variables.yaml` (NEW)
  - Template showing variable definitions
  - Examples: scalar, vec2i, vec3i, bool types
  - Examples: global, agent, agent_private scopes

**Schema Examples**:

```python
@dataclass
class VariableDef:
    """Variable declaration in UAC."""
    id: str                      # e.g., "home_pos", "energy", "world_config_hash"
    scope: Literal["global", "agent", "agent_private"]
    type: Literal["scalar", "vec2i", "vec3i", "bool"]
    lifetime: Literal["tick", "episode"]  # tick = derived feature, episode = persistent
    readable_by: List[str]      # ["agent", "engine", "acs"]
    writable_by: List[str]      # ["actions", "engine"]
    default: Any                # Default value (must match type)
    description: str = ""

@dataclass
class ObservationField:
    """Observation exposure specification for BAC."""
    id: str                      # e.g., "obs_home_pos"
    source_variable: str         # e.g., "home_pos"
    exposed_to: List[str]        # ["agent"] for agent_private
    shape: List[int]             # [2] for vec2i, [] for scalar
    normalization: Optional[NormalizationSpec] = None

@dataclass
class NormalizationSpec:
    kind: Literal["minmax", "standardization"]
    min: Optional[List[float]] = None   # For minmax
    max: Optional[List[float]] = None   # For minmax
    mean: Optional[float] = None        # For standardization
    std: Optional[float] = None         # For standardization

@dataclass
class WriteSpec:
    """Action write declaration."""
    variable_id: str             # Which variable to write
    expression: str              # e.g., "agent.pos", "clamp(energy - 5, 0, 100)"
```

**Tests**:

- [ ] `tests/test_vfs/test_schema.py`: Schema validation
  - VariableDef validation (valid types, scopes, lifetimes)
  - ObservationField validation (shape matches variable type)
  - NormalizationSpec validation (min < max, etc.)

- [ ] `tests/test_vfs/test_registry.py`: Registry operations
  - Get/set variables by scope
  - Access control enforcement (readable_by, writable_by)
  - Default values applied correctly
  - get_observation_spec() returns correct fields

- [ ] `tests/test_vfs/test_observation_builder.py`: Observation builder
  - Build observation spec from variables + exposures
  - Shape inference from variable types (vec2i → [2], scalar → [])
  - Normalization parameters passed through

- [ ] `tests/test_acs/test_action_schema_extension.py`: Action schema
  - ActionSchema with reads/writes fields
  - Serialization/deserialization preserves reads/writes

**Success Criteria**:

- ✅ All VFS schemas defined and validated
- ✅ VariableRegistry implements get/set with access checks
- ✅ ObservationBuilder generates specs from variable definitions
- ✅ ActionSchema extended with reads/writes fields
- ✅ Unit tests pass (15+ tests total)
- ✅ Example `variables.yaml` in templates

**Time Estimate**: 16-20 hours
- Schema design and implementation: 4 hours
- Registry implementation: 6 hours
- Observation builder: 4 hours
- ACS extension: 2 hours
- Tests: 4-6 hours

---

### Phase 2: Advanced Features (Deferred to Post-TASK-004)

**Objective**: Add complex types, feature derivation, incremental masks after learning from UAC compiler.

**Out of Scope for Phase 1**:

- ❌ Feature derivation graphs (manual features in Engine for now)
- ❌ Incremental mask recomputation (ACS does full recompute)
- ❌ Complex types (stacks, queues, maps, structs)
- ❌ Entity/episode scopes (agent-level only initially)
- ❌ Persistent lifetime (episode-scoped only)
- ❌ Full access control policies (basic readable/writable only)

**Phase 2 Scope** (3-4 days, TASK-002D or part of TASK-004):

- ✅ `FeatureDef`: Derived variables with dependency graphs
- ✅ `vfs/compiler.py`: Compile feature expressions into execution graph
- ✅ `vfs/incremental_mask.py`: Dependency-based mask recomputation
- ✅ Complex types: stack, queue, map, struct
- ✅ Full scope support: entity, episode
- ✅ Persistent lifetime (saved across runs)

**Triggers for Phase 2**:

1. UAC compiler (TASK-004) complete and working
2. Real variable patterns observed in UAC YAMLs
3. Need for feature derivation (e.g., `can_teleport_home`)
4. Performance profiling shows mask recomputation bottleneck

---

## Testing Strategy

### Test Coverage Requirements

**Unit Tests**:

- `vfs/`: 90%+ coverage (new code, no legacy constraints)
- `acs/action_schema.py`: 85%+ coverage (modified file)

**Integration Tests**:

- [ ] **Test 1: Variable Definition to Observation**
  - Define variable in schema → Store in registry → Expose as observation → BAC can read spec
  - Validates end-to-end flow without UAC compiler

- [ ] **Test 2: Action Read/Write Annotations**
  - Define action with reads/writes → ACS validates dependencies → Engine can query
  - Validates ACS extension

- [ ] **Test 3: Access Control Enforcement**
  - Agent tries to write read-only variable → raises AccessError
  - Agent tries to read agent_private of another agent → raises AccessError

### Regression Testing

**Critical Paths**:

- [ ] Existing ACS tests still pass (action schema is backward compatible)
- [ ] Existing configs without variables.yaml still load
- [ ] Meters continue working as arrays (no migration yet)

**Performance Testing**:

- [ ] Registry get/set operations: <1µs per operation
- [ ] Observation spec generation: <10ms for 100 variables

---

## Migration Guide

### For Existing Configs

**Before** (Phase 1 - no changes required):

```yaml
# Existing configs work unchanged
# Meters still in environment, not VFS
```

**After** (Phase 2 - optional migration):

```yaml
# configs/L1_baseline/variables.yaml (NEW)
version: "1.0"
variables:
  - id: "energy"
    scope: "agent"
    type: "scalar"
    lifetime: "episode"
    readable_by: ["agent", "engine", "acs"]
    writable_by: ["actions", "engine"]
    default: 1.0
    description: "Agent energy (0.0-1.0)"

  - id: "home_pos"
    scope: "agent_private"
    type: "vec2i"
    lifetime: "episode"
    readable_by: ["agent", "engine"]
    writable_by: ["actions", "engine"]
    default: [0, 0]
    description: "Agent's remembered home position"

exposed_observations:
  - id: "obs_energy"
    source_variable: "energy"
    exposed_to: ["agent"]
    shape: []
    normalization:
      kind: "minmax"
      min: 0.0
      max: 1.0

  - id: "obs_home_pos"
    source_variable: "home_pos"
    exposed_to: ["agent"]
    shape: [2]
    normalization:
      kind: "minmax"
      min: [0, 0]
      max: [7, 7]  # Assuming 8×8 grid
```

**Migration Script**: Not needed (Phase 1 is additive)

### For Existing Checkpoints

**Compatibility**: Full (Phase 1 doesn't touch checkpoints)

---

## Examples

### Example 1: Simple Variable Definition

**Config** (`configs/templates/variables.yaml`):

```yaml
version: "1.0"
variables:
  - id: "world_config_hash"
    scope: "global"
    type: "scalar"
    lifetime: "episode"
    readable_by: ["agent", "engine", "acs"]
    writable_by: ["engine"]
    default: 0
    description: "Hash of current world configuration (for BLOCKER 2 solution)"

exposed_observations:
  - id: "obs_world_config_hash"
    source_variable: "world_config_hash"
    exposed_to: ["agent"]
    shape: []
    normalization: null  # No normalization for hash
```

**Usage** (Engine sets, BAC reads):

```python
# Engine: Set world config hash at environment init
registry.set("global", "world_config_hash", compute_world_hash(uac))

# BAC compiler: Query observation spec
obs_fields = registry.get_observation_spec()
# Returns: [ObservationField(id="obs_world_config_hash", shape=[], ...)]

# Agent: Observes hash in state
obs = env.reset()
world_hash = obs["world_config_hash"]  # Can condition world model on this
```

**Output**: World model can now detect when world physics changed (BLOCKER 2 solution)

### Example 2: Action Read/Write Annotations

**Config** (action schema):

```yaml
actions:
  - id: "set_home_here"
    params: []
    reads: []  # No reads needed
    writes:
      - variable_id: "home_pos"
        expression: "agent.pos"  # Copy current position to home_pos
    pre: []
    effect: []  # Effect handled by VFS write

  - id: "teleport_home"
    params: []
    reads: ["home_pos"]  # Reads home position
    writes:
      - variable_id: "agent.pos"
        expression: "home_pos"  # Set position to home_pos
      - variable_id: "energy"
        expression: "clamp(energy - 5, 0, 100)"  # Cost 5 energy
    pre:
      - expr: "distance(agent.pos, home_pos) > 0"  # Not already at home
    effect: []
```

**Usage**:

```python
# ACS: Query which variables an action depends on
action_schema = load_action_schema("teleport_home")
print(action_schema.reads)   # ["home_pos"]
print(action_schema.writes)  # [WriteSpec("agent.pos", ...), WriteSpec("energy", ...)]

# Engine: Execute action with variable binding
registry.set("agent_private", "home_pos", [3, 4])  # Agent set home earlier
action_result = execute_action("teleport_home", agent_id=0)
# Engine reads home_pos, writes agent.pos and energy
```

**Output**: Teleport action with explicit variable dependencies

---

## Acceptance Criteria

### Must Have (Blocking)

- [ ] `VariableDef` schema defined and validated
- [ ] `ObservationField` schema defined and validated
- [ ] `WriteSpec` schema defined and validated
- [ ] `VariableRegistry` implements get/set with access control
- [ ] `ObservationBuilder` generates observation specs
- [ ] `ActionSchema` extended with `reads` and `writes` fields
- [ ] All unit tests pass (15+ tests)
- [ ] Integration test: variable → registry → observation → BAC spec
- [ ] Example `variables.yaml` in config templates
- [ ] Documentation: README in `src/townlet/vfs/` explaining purpose

### Should Have (Important)

- [ ] Registry performance: <1µs per get/set
- [ ] Clear error messages for access violations
- [ ] Type validation (vec2i must be 2-element list)
- [ ] Scope validation (agent_private only accessible by owning agent)

### Could Have (Future - Phase 2)

- [ ] Feature derivation compiler (deferred)
- [ ] Incremental mask optimization (deferred)
- [ ] Complex types: stack, queue, map (deferred)
- [ ] Entity/episode scopes (deferred)

---

## Risk Assessment

### Technical Risks

**Risk 1: Schema may need refinement after UAC compiler (TASK-004)**

- **Severity**: Medium
- **Mitigation**: Keep Phase 1 scope minimal, expect iteration
- **Contingency**: Schema changes are cheap before many systems depend on them

**Risk 2: Performance of registry get/set might not scale**

- **Severity**: Low
- **Mitigation**: Profile in Phase 2, optimize if needed
- **Contingency**: Registry is simple dict lookups, very fast

**Risk 3: Access control model may be too simplistic**

- **Severity**: Low
- **Mitigation**: Start with basic readable_by/writable_by, extend in Phase 2
- **Contingency**: Schema supports list of readers/writers, easy to extend

### Blocking Dependencies

- ✅ **TASK-002A** (Stratum): Complete
- ✅ **TASK-002B** (ACS): Complete
- ⚠️ **TASK-003** (UAC Core DTOs): Partially blocked by this task (circular dependency - see note below)

**Circular Dependency Resolution**:

- TASK-003 defines UAC DTOs (bars, affordances, cascades)
- TASK-002C defines VFS DTOs (variables, observations)
- **Resolution**: These are parallel concerns. VFS variables are separate from UAC bars/affordances.
- **Sequencing**: Do TASK-002C first (simpler scope), then TASK-003 can reference VFS schemas.

### Impact Radius

**Files Modified**: ~8 files
- 4 new files in `src/townlet/vfs/`
- 1 modified file in `src/townlet/acs/`
- 1 new config template
- 2+ test files

**Tests Added**: 15+ tests
**Breaking Changes**: None (purely additive)

**Blast Radius**: Small

- VFS is new subsystem, no legacy code depends on it
- ACS extension is backward compatible (new fields optional)
- Existing configs work unchanged

---

## Effort Breakdown

### Detailed Estimates

**Schema Design & Implementation**: 4 hours

- Define VariableDef, ObservationField, WriteSpec dataclasses: 2 hours
- Define NormalizationSpec and type enums: 1 hour
- Validation logic (Pydantic or manual): 1 hour

**Registry Implementation**: 6 hours

- VariableRegistry class structure: 2 hours
- get/set with access control: 2 hours
- get_observation_spec() method: 1 hour
- Error handling and edge cases: 1 hour

**Observation Builder**: 4 hours

- ObservationBuilder class: 2 hours
- build_observation_spec() logic: 2 hours

**ACS Extension**: 2 hours

- Extend ActionSchema dataclass: 0.5 hours
- Update serialization/deserialization: 0.5 hours
- Update existing tests: 1 hour

**Testing**: 4-6 hours

- Unit tests for schemas: 1 hour
- Unit tests for registry: 2 hours
- Unit tests for observation builder: 1 hour
- Integration tests: 1-2 hours

**Documentation & Examples**: 2 hours

- VFS module README: 1 hour
- Example variables.yaml template: 1 hour

**Total**: 16-20 hours (2-3 days)

**Confidence**: High

### Assumptions

- No complex feature derivation needed in Phase 1
- Access control is simple readable_by/writable_by lists
- Registry uses in-memory dict (no persistence in Phase 1)
- UAC compiler (TASK-004) will handle YAML → VariableDef generation

---

## Future Work (Explicitly Out of Scope)

### Not Included in This Task (Phase 2 - Post-TASK-004)

1. **Feature Derivation Compiler**
   - **Why Deferred**: Need to learn actual feature patterns from UAC YAMLs first
   - **Follow-up Task**: TASK-002D or part of TASK-004B (UAC Capabilities)

2. **Incremental Mask Recomputation**
   - **Why Deferred**: Performance optimization premature before profiling
   - **Follow-up Task**: Part of ACS optimization pass

3. **Complex Types (stacks, queues, maps, structs)**
   - **Why Deferred**: Bar_push example is motivating use case, but not critical path
   - **Follow-up Task**: TASK-002D

4. **Entity/Episode Scopes**
   - **Why Deferred**: Agent-level variables sufficient for compilers
   - **Follow-up Task**: When multi-entity worlds become priority

5. **Persistent Lifetime**
   - **Why Deferred**: Episode scope sufficient for current use cases
   - **Follow-up Task**: When cross-run state becomes requirement

### Enables Future Tasks

- **TASK-003**: UAC Core DTOs can reference VFS schemas
- **TASK-004A**: UAC Compiler can generate List[VariableDef]
- **TASK-005**: Brain as Code can consume observation specs
- **Feature**: Teleport actions (depends on VFS variables)
- **Feature**: Stack/queue operations (depends on complex types in Phase 2)

---

## References

### Related Documentation

- **Design Discussion**: This task document (TASK-002C analysis above)
- **Architecture**: `docs/architecture/UNIVERSE_AS_CODE.md` (UAC layer)
- **Architecture**: `docs/architecture/BRAIN_AS_CODE.md` (BAC layer)
- **Blocker**: `docs/architecture/hld/review/review-07-critical-blockers.md` (BLOCKER 2 solution)

### Related Tasks

- **Prerequisites**: TASK-002A (Stratum), TASK-002B (ACS)
- **Parallel Work**: TASK-003 (UAC Core DTOs - can proceed in parallel)
- **Follow-up**: TASK-004A (UAC Compiler), TASK-005 (Brain as Code)
- **Future**: TASK-002D (VFS Phase 2 - advanced features)

### Code References

- `src/townlet/acs/action_schema.py` - Action schema to be extended
- `src/townlet/stratum/interface.py` - Substrate interface (inspiration for contracts)
- `configs/templates/` - Config template directory for variables.yaml

---

## Notes for Implementer

### Before Starting

- [ ] Read TASK-002A and TASK-002B to understand stratum and ACS foundations
- [ ] Review `docs/architecture/UNIVERSE_AS_CODE.md` for UAC context
- [ ] Read BLOCKER 2 in `review-07-critical-blockers.md` (world_config_hash motivation)
- [ ] Check TASK-003 and TASK-004 specs to understand downstream consumers

### During Implementation

- [ ] Start with schemas (`schema.py`) - get types right first
- [ ] Implement registry with simple dict storage (optimize later)
- [ ] Write tests as you go (TDD recommended for schemas)
- [ ] Keep Phase 2 features out of scope (feature derivation, complex types, etc.)
- [ ] Document why access control is checked (governance requirement)

### Before Marking Complete

- [ ] All 15+ unit tests pass
- [ ] Integration test demonstrates variable → observation flow
- [ ] Example `variables.yaml` demonstrates each variable type
- [ ] VFS module has README explaining purpose and usage
- [ ] ActionSchema extension is backward compatible (existing tests pass)
- [ ] Task file updated with completion date and lessons learned

---

**IMPLEMENTATION NOTE**: This is a **foundational cross-cutting layer**. Resist the urge to add complex features in Phase 1. The goal is to establish typed contracts that unblock compilers, not to build the perfect variable system. Learn from UAC compiler (TASK-004) usage before adding complexity.

---

**END OF TASK SPECIFICATION**
