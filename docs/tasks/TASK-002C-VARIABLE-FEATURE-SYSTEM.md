# TASK-002C: Variable & Feature System (VFS) - Phase 1 Foundation

**Status**: Planned
**Priority**: High
**Estimated Effort**: 28-36 hours (4-5 days)
**Dependencies**: TASK-002A (Substrate), TASK-002B (Composable Actions)
**Enables**: TASK-004A (UAC Compiler), TASK-005 (Brain as Code), BLOCKER 2 (World Config Hash)
**Parallel Work**: TASK-003 (UAC Core DTOs - independent, can proceed in parallel)
**Created**: 2025-11-06
**Updated**: 2025-11-07 (Deep dive review - effort revised, scope clarifications added)
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

## Scope Semantics (CRITICAL CLARIFICATION)

**Problem**: The meaning of `scope` values must be precisely defined to avoid implementation ambiguity.

### Scope Definitions

**`global`**: Single value shared by entire universe
- **Storage**: Single tensor element (not per-agent)
- **Access**: All agents observe same value
- **Use cases**: `world_config_hash`, `time_of_day`, `total_population`
- **Example**: World configuration hash for BLOCKER 2 solution

**`agent`**: Per-agent value, publicly readable
- **Storage**: Per-agent tensor `[num_agents]` or `[num_agents, dims]`
- **Access**: All agents can read any agent's value
- **Use cases**: `position`, `energy`, `health` (public meters)
- **Example**: Agent positions visible to all for spatial reasoning

**`agent_private`**: Per-agent value, private to owner
- **Storage**: Per-agent tensor `[num_agents]` or `[num_agents, dims]`
- **Access**: Only owning agent can read their own value
- **Use cases**: `home_pos` (remembered location), `private_goals`, `beliefs`
- **Example**: Agent's remembered home position for teleport action

### Observation Implications

**Global variables**: Broadcast to all agents (1 value → num_agents copies)
**Agent variables**: Each agent sees all agent values (full matrix)
**Agent_private variables**: Each agent sees only their own value (diagonal extraction)

---

## Variable Type System (EXPANDED)

The type system must support all substrate dimensionalities (1D-100D) and both discrete/continuous spaces.

### Supported Types (Phase 1)

**Scalar Types:**
- `scalar`: Single float value
  - Shape: `[]` (0-dimensional)
  - Example: `energy`, `health`, `world_config_hash`

**Fixed-Dimension Vector Types:**
- `vec2i`: 2D integer vector (Grid2D positions)
  - Shape: `[2]`
  - Example: `position` on 8×8 grid

- `vec3i`: 3D integer vector (Grid3D positions)
  - Shape: `[3]`
  - Example: `position` on 3D grid

**Variable-Dimension Vector Types (NEW):**
- `vecNi`: N-dimensional integer vector
  - Shape: `[dims]` where `dims` specified in schema
  - Example: `position` on 7D GridND
  - **Required field**: `dims: int` (e.g., `dims: 7` for GridND)

- `vecNf`: N-dimensional float vector
  - Shape: `[dims]` where `dims` specified in schema
  - Example: `position` in continuous 4D space
  - **Required field**: `dims: int`

**Boolean Type:**
- `bool`: Boolean flag
  - Shape: `[]` (stored as 0.0 or 1.0)
  - Example: `is_at_home`, `panic_mode_active`

### Type Schema Update

```python
@dataclass
class VariableDef:
    """Variable declaration in UAC."""
    id: str
    scope: Literal["global", "agent", "agent_private"]
    type: Literal["scalar", "vec2i", "vec3i", "vecNi", "vecNf", "bool"]
    dims: Optional[int] = None  # Required for vecNi/vecNf
    lifetime: Literal["tick", "episode"]
    readable_by: List[str]
    writable_by: List[str]
    default: Any
    description: str = ""

    @model_validator(mode="after")
    def validate_dims(self) -> "VariableDef":
        """vecNi and vecNf require dims field."""
        if self.type in ["vecNi", "vecNf"] and self.dims is None:
            raise ValueError(f"Type {self.type} requires dims field")
        if self.type not in ["vecNi", "vecNf"] and self.dims is not None:
            raise ValueError(f"Type {self.type} cannot have dims field")
        return self
```

### Phase 2 Types (Deferred)

- `categorical`: Discrete choice (needs `categories` list)
- `stack`: LIFO collection (needs `element_type`, `capacity`)
- `queue`: FIFO collection (needs `element_type`, `capacity`)
- `map`: Key-value store (needs `key_type`, `value_type`)
- `struct`: Composite type (needs field definitions)

---

## Detailed Design

### Phase 1: Minimal Foundation (28-36 hours)

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

- **File**: `src/townlet/environment/action_config.py` (MODIFY)
  - Extend `ActionConfig` Pydantic model:
    - Add `reads: List[str] = Field(default_factory=list)`
    - Add `writes: List[WriteSpec] = Field(default_factory=list)`

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
    type: Literal["scalar", "vec2i", "vec3i", "vecNi", "vecNf", "bool"]  # Expanded type system
    dims: Optional[int] = None   # Required for vecNi/vecNf (e.g., dims=7 for 7D GridND)
    lifetime: Literal["tick", "episode"]  # tick = derived feature, episode = persistent
    readable_by: List[str]      # ["agent", "engine", "acs"]
    writable_by: List[str]      # ["actions", "engine"]
    default: Any                # Default value (must match type)
    description: str = ""

    @model_validator(mode="after")
    def validate_dims(self) -> "VariableDef":
        """vecNi and vecNf require dims field."""
        if self.type in ["vecNi", "vecNf"] and self.dims is None:
            raise ValueError(f"Type {self.type} requires dims field")
        return self

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

**Tests** (25-30 tests total):

- [ ] `tests/test_vfs/test_schema.py`: Schema validation (8-10 tests)
  - VariableDef validation (valid types, scopes, lifetimes)
  - vecNi/vecNf require dims field (validation test)
  - ObservationField validation (shape matches variable type)
  - NormalizationSpec validation (min < max, etc.)
  - Type system: scalar, vec2i, vec3i, vecNi, vecNf, bool
  - Scope semantics: global, agent, agent_private

- [ ] `tests/test_vfs/test_registry.py`: Registry operations (8-10 tests)
  - Get/set variables by scope (global/agent/agent_private)
  - Access control enforcement (readable_by, writable_by)
  - Default values applied correctly
  - get_observation_spec() returns correct fields
  - Scope storage: global (1 value), agent (num_agents), agent_private (num_agents)
  - Error handling: invalid scope, missing variable, access violation

- [ ] `tests/test_vfs/test_observation_builder.py`: Observation builder (5-6 tests)
  - Build observation spec from variables + exposures
  - Shape inference from variable types (vec2i → [2], scalar → [], vecNi → [dims])
  - Normalization parameters passed through
  - Support all substrate types (Grid2D, Grid3D, GridND, Continuous, Aspatial)

- [ ] `tests/test_vfs/test_observation_dimension_regression.py`: **CRITICAL VALIDATION** (4-5 tests)
  - **VFS observation_dim must equal current hardcoded calculation**
  - Test all curriculum levels: L0_minimal, L0_5_dual, L1_full, L2_pomdp, L3_temporal
  - Full observability: VFS dim == substrate.get_observation_dim() + meters + affordances + 4
  - Partial observability: VFS dim == window_size^position_dim + position_dim + meters + affordances + 4
  - **If these tests fail, checkpoints will be incompatible!**

- [ ] `tests/test_environment/test_action_config_extension.py`: Action config extension (2-3 tests)
  - ActionConfig with reads/writes fields
  - Serialization/deserialization preserves reads/writes
  - Backward compatibility: ActionConfig without reads/writes still loads

**Success Criteria**:

- ✅ All VFS schemas defined and validated (including expanded type system)
- ✅ Scope semantics clearly documented (global/agent/agent_private)
- ✅ VariableRegistry implements get/set with access checks
- ✅ ObservationBuilder generates specs from variable definitions
- ✅ ActionConfig extended with reads/writes fields (not ActionSchema - corrected path)
- ✅ **All 25-30 unit tests pass** (increased from 15+)
- ✅ **Observation dimension regression tests pass** (CRITICAL - prevents checkpoint breakage)
- ✅ Example `variables.yaml` in templates

**Time Estimate**: 28-36 hours (4-5 days)
- Schema design and implementation: 4 hours (unchanged)
- Registry implementation: **8-10 hours** (increased - scope semantics, access control complexity)
- Observation builder: **6-8 hours** (increased - must support all substrate types)
- ACS extension: 2 hours (unchanged)
- Tests: **10-12 hours** (increased - 25-30 tests, regression validation critical)
- Documentation & Examples: 2 hours (unchanged)

**Effort Increase Rationale**:
- Scope semantics (global/agent/agent_private) require careful tensor management
- Type system expanded (vecNi, vecNf for N-dimensional substrates)
- Observation dimension validation critical (checkpoint compatibility)
- Test count realistic based on existing integration test complexity (200-500 lines each)

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

- [ ] **Observation dimension compatibility** (CRITICAL - MUST PASS)
  - VFS-generated `observation_dim` must equal current hardcoded calculation
  - Test ALL configs: L0_minimal, L0_5_dual, L1_full, L2_pomdp, L3_temporal
  - Formula verification: VFS sum(field.shape_size) == env.observation_dim
  - **Why critical**: Different dim → all checkpoints incompatible!

- [ ] Existing ActionConfig tests still pass (backward compatible)
- [ ] Existing configs without variables.yaml still load
- [ ] Meters continue working as arrays (no migration yet)

**Performance Testing**:

- [ ] Registry get/set operations: <1µs per operation
- [ ] Observation spec generation: <10ms for 100 variables

**Validation Test Example**:

```python
def test_vfs_observation_dim_backward_compatible():
    """CRITICAL: VFS must generate identical observation_dim to current code."""
    for config_name in ["L0_minimal", "L0_5_dual", "L1_full", "L2_pomdp", "L3_temporal"]:
        # Current environment (hardcoded calculation)
        env_current = VectorizedHamletEnv.from_config(config_name)

        # VFS-based calculation
        variables = load_variables_from_config(config_name)
        obs_spec = ObservationBuilder.build_observation_spec(variables)
        vfs_dim = sum(field.shape_size for field in obs_spec.fields)

        # MUST be identical
        assert vfs_dim == env_current.observation_dim, (
            f"Config {config_name}: VFS dim {vfs_dim} != current dim {env_current.observation_dim}. "
            f"Checkpoint incompatibility detected!"
        )
```

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

**Risk 1: Observation Dimension Compatibility** ⚠️ **CRITICAL - NEW**

- **Severity**: HIGH
- **Problem**: VFS must generate **identical** observation_dim to current hardcoded calculation
- **Impact**: Different dim → all existing checkpoints incompatible (breaking change)
- **Mitigation**:
  - Create regression test suite comparing VFS dim to current env.observation_dim
  - Test ALL configs (L0_minimal, L0_5_dual, L1_full, L2_pomdp, L3_temporal)
  - Fail fast if dimensions mismatch
- **Contingency**: If VFS generates different dims, must either:
  1. Fix VFS calculation to match current (preferred)
  2. Declare breaking change and invalidate all checkpoints (last resort)
- **Test Example**: See "Regression Testing" section above

**Risk 2: Schema may need refinement after UAC compiler (TASK-004)**

- **Severity**: Medium
- **Mitigation**: Keep Phase 1 scope minimal, expect iteration
- **Contingency**: Schema changes are cheap before many systems depend on them

**Risk 3: Scope Semantics Complexity** ⚠️ **MEDIUM - NEW**

- **Severity**: Medium
- **Problem**: global/agent/agent_private require careful tensor management
- **Challenge**: Global (1 value) vs agent (num_agents values) storage patterns
- **Mitigation**:
  - Clear documentation added (see "Scope Semantics" section)
  - Unit tests validate storage shapes for each scope
  - Registry enforces scope contracts at runtime
- **Contingency**: If scope semantics prove insufficient, extend in Phase 2

**Risk 4: Variable Type System Completeness** ⚠️ **LOW - NEW**

- **Severity**: Low
- **Problem**: vecNi/vecNf added for N-dimensional support, may need more types
- **Mitigation**: Phase 2 explicitly defers complex types (categorical, stack, queue)
- **Contingency**: Type system extensible, add types as needed

**Risk 5: Performance of registry get/set might not scale**

- **Severity**: Low
- **Mitigation**: Profile in Phase 2, optimize if needed
- **Contingency**: Registry is simple dict lookups, very fast (existing AgentRuntimeRegistry proves pattern)

**Risk 6: Access control model may be too simplistic**

- **Severity**: Low
- **Mitigation**: Start with basic readable_by/writable_by, extend in Phase 2
- **Contingency**: Schema supports list of readers/writers, easy to extend

### Blocking Dependencies

- ✅ **TASK-002A** (Substrate Abstraction): Complete (renamed from "Stratum" in final implementation)
- ✅ **TASK-002B** (Composable Action Space): Complete
- ⚠️ **TASK-003** (UAC Core DTOs): **Independent** - can proceed in parallel

**TASK-003 Relationship Clarification**:

- **Original spec claimed**: TASK-002C blocks TASK-003
- **Reality**: TASK-003 defines UAC DTOs (bars, affordances, cascades) which are **independent** of VFS
- **Evidence**: `cascade_config.py` already implemented in codebase (150 lines, working)
- **Resolution**: TASK-003 and TASK-002C are **parallel work**
  - VFS defines variable contracts (for TASK-004A compiler, TASK-005 BAC)
  - TASK-003 defines universe element contracts (bars, affordances, cascades)
  - No circular dependency - different concerns
- **Sequencing**: Either order works; TASK-002C slightly simpler scope, so recommend first

### Impact Radius

**Files Modified**: ~10 files
- 4 new files in `src/townlet/vfs/` (schema, registry, observation_builder, __init__)
- 1 modified file in `src/townlet/environment/action_config.py` (corrected from acs/)
- 1 new config template (`configs/templates/variables.yaml`)
- 4-5 test files (schema, registry, observation_builder, dimension regression, action_config extension)

**Tests Added**: 25-30 tests (increased from 15+ to reflect realistic complexity)
**Breaking Changes**: None (purely additive in Phase 1)

**Blast Radius**: Small

- VFS is new subsystem, no legacy code depends on it
- ActionConfig extension is backward compatible (new fields optional with defaults)
- Existing configs work unchanged
- **Critical constraint**: VFS observation_dim must match current calculation (regression tests enforce)

---

## Effort Breakdown

### Detailed Estimates (REVISED)

**Schema Design & Implementation**: 4 hours (unchanged)

- Define VariableDef, ObservationField, WriteSpec dataclasses: 2 hours
- Define NormalizationSpec and expanded type enums (vecNi, vecNf): 1 hour
- Validation logic (Pydantic validators for dims field): 1 hour

**Registry Implementation**: 8-10 hours (increased from 6h)

- VariableRegistry class structure: 2 hours
- get/set with scope-aware access control: 3-4 hours (global vs agent vs agent_private tensor shapes)
- get_observation_spec() method: 1 hour
- Error handling and edge cases (access violations, scope validation): 2-3 hours
- **Reason for increase**: Scope semantics (global=1 value, agent=num_agents, agent_private=num_agents) require careful tensor management

**Observation Builder**: 6-8 hours (increased from 4h)

- ObservationBuilder class: 2 hours
- build_observation_spec() logic: 2-3 hours
- Support all substrate types (Grid2D, Grid3D, GridND, Continuous, Aspatial): 2-3 hours
- **Reason for increase**: Must generate specs for all substrate dimensionalities, ensure dimension compatibility

**ActionConfig Extension**: 2 hours (unchanged)

- Extend ActionConfig Pydantic model: 0.5 hours
- Update serialization/deserialization: 0.5 hours
- Update existing tests: 1 hour

**Testing**: 10-12 hours (increased from 4-6h)

- Unit tests for schemas (8-10 tests): 2 hours
- Unit tests for registry (8-10 tests): 3 hours
- Unit tests for observation builder (5-6 tests): 2 hours
- **Observation dimension regression tests (4-5 tests - CRITICAL)**: 2-3 hours
- Integration tests (variable → observation flow): 2 hours
- ActionConfig extension tests (2-3 tests): 1 hour
- **Reason for increase**: 25-30 tests total, regression validation complex (must test all configs), existing integration tests are 200-500 lines each

**Documentation & Examples**: 2 hours (unchanged)

- VFS module README: 1 hour
- Example variables.yaml template: 1 hour

**Total**: 28-36 hours (4-5 days)

**Confidence**: High (90%)

**Comparison to Original Estimate**:
- Original: 16-20 hours (2-3 days)
- Revised: 28-36 hours (4-5 days)
- Increase: +12-16 hours (+75% effort)
- **Justification**: Original underestimated scope semantics complexity, test count, and observation dimension validation criticality

### Assumptions

- No complex feature derivation needed in Phase 1
- Access control is simple readable_by/writable_by lists (extensible in Phase 2)
- Registry uses in-memory dict (no persistence in Phase 1)
- UAC compiler (TASK-004) will handle YAML → VariableDef generation
- **New assumption**: VFS observation_dim must match current env calculation (regression tests enforce)

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

- `src/townlet/environment/action_config.py` - ActionConfig to be extended (corrected path)
- `src/townlet/substrate/base.py` - Substrate interface (inspiration for contracts, renamed from stratum)
- `src/townlet/environment/observation_builder.py` - Current observation construction (integration point)
- `src/townlet/population/runtime_registry.py` - AgentRuntimeRegistry (pattern for VariableRegistry)
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

## Revision History

### 2025-11-07: Deep Dive Review (Implementation Readiness Assessment)

**Changes Made**:

1. **Effort Estimate Updated**: 16-20h → **28-36h** (4-5 days)
   - Registry: 6h → 8-10h (scope semantics complexity)
   - ObservationBuilder: 4h → 6-8h (all substrate types)
   - Testing: 4-6h → 10-12h (25-30 tests, regression critical)

2. **Scope Semantics Clarified**: Added dedicated section defining global/agent/agent_private
   - Global: Single value (tensor shape [1])
   - Agent: Per-agent values (tensor shape [num_agents])
   - Agent_private: Per-agent private values (tensor shape [num_agents], access-controlled)

3. **Variable Type System Expanded**: Added vecNi, vecNf for N-dimensional substrates
   - vecNi: N-dimensional integer vector (requires dims field)
   - vecNf: N-dimensional float vector (requires dims field)
   - Supports GridND (4D-100D) and ContinuousND substrates

4. **Observation Dimension Validation**: **CRITICAL** new requirement
   - VFS-generated observation_dim MUST equal current hardcoded calculation
   - Regression tests for ALL configs (L0, L0.5, L1, L2, L3)
   - Prevents checkpoint incompatibility (breaking change)

5. **Test Count Increased**: 15+ → **25-30 tests**
   - Added observation dimension regression tests (4-5 tests - critical)
   - Increased per-module test counts based on existing test complexity
   - Example validation test provided

6. **New Risks Identified**:
   - Risk 1: Observation dimension compatibility (HIGH severity - checkpoint breakage)
   - Risk 3: Scope semantics complexity (MEDIUM severity - tensor management)
   - Risk 4: Variable type system completeness (LOW severity - extensible)

7. **File Path Corrections**:
   - Fixed: `acs/action_schema.py` → `environment/action_config.py`
   - Fixed: `stratum/interface.py` → `substrate/base.py` (renamed in TASK-002A)
   - Added: Additional code references for integration points

8. **TASK-003 Relationship Clarified**:
   - Updated from "blocks TASK-003" to "parallel work - independent"
   - Evidence: cascade_config.py already implemented (150 lines)
   - No circular dependency - different concerns

9. **Dependencies Updated**:
   - "TASK-002A (Stratum)" → "TASK-002A (Substrate)" (final naming)
   - Added "BLOCKER 2 (World Config Hash)" to enabled tasks

**Implementation Readiness Score**: 8.5/10 (was unscored)

**Recommendation**: **CONDITIONAL GO** - Proceed after reviewing updated effort estimate and critical validation requirements.

**Key Takeaways**:
- Original estimate underestimated scope semantics and testing complexity by ~75%
- Observation dimension compatibility is CRITICAL - must not break checkpoints
- All prerequisites met, integration points clean, patterns established
- Risk mitigation strategies clear, no showstoppers identified

---

**END OF TASK SPECIFICATION**
