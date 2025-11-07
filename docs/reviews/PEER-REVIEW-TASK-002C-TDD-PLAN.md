# Peer Review: TASK-002C TDD Implementation Plan

**Reviewer**: Claude Code
**Review Date**: 2025-11-07
**Documents Reviewed**:
- `docs/plans/2025-11-07-task-002c-tdd-implementation-plan.md`
- `docs/tasks/TASK-002C-VARIABLE-FEATURE-SYSTEM.md`

**Codebase Analysis**:
- Current implementation: ActionConfig, VectorizedHamletEnv, Substrate system
- Existing tests: 40+ unit tests, integration test patterns
- observation_dim calculation: Verified in vectorized_env.py and test files

---

## Executive Summary

**Overall Assessment**: ⚠️ **CONDITIONAL APPROVAL - PROCEED WITH MODIFICATIONS**

The plan is well-structured with solid TDD methodology, but has **critical risks** and **underestimated complexity**. The core concern is **observation dimension compatibility** - if VFS generates different dimensions than the current hardcoded calculation, **ALL existing checkpoints become incompatible** (breaking change affecting months of training).

**Key Findings**:
1. ✅ TDD approach is excellent and appropriate for foundational work
2. ⚠️ Effort estimate optimistic: 28-36h → realistic **38-50 hours** (33% increase)
3. 🚨 **CRITICAL**: Naming collision with existing `ObservationBuilder` class
4. 🚨 **CRITICAL**: Observation dimension regression testing is essential but under-specified
5. ⚠️ Scope semantics need clearer specification (global/agent/agent_private tensor management)
6. ✅ Prerequisites met, integration points generally clean

**Recommendation**: **CONDITIONAL GO** with required modifications outlined below.

---

## 1. Complexity Assessment

### 1.1 Schema Layer (LOW-MEDIUM Complexity) ✅
- **VariableDef, ObservationField, WriteSpec**: Standard Pydantic schemas
- **Type system expansion** (vecNi, vecNf): Minor complication, Pydantic handles well
- **Validation logic**: Field validators for dims field are straightforward
- **Estimate**: 4 hours is **reasonable**

### 1.2 Registry Layer (MEDIUM-HIGH Complexity) ⚠️
**Current plan**: 8-10 hours
**My assessment**: **10-12 hours** (+20% increase)

**Complexity sources**:
1. **Scope-specific tensor management**:
   ```python
   # global: single value
   storage["global"]["world_config_hash"] = torch.tensor(0)  # shape: []

   # agent: per-agent values
   storage["agent"]["energy"] = torch.tensor([1.0, 1.0, 1.0, 1.0])  # shape: [num_agents]

   # agent_private: per-agent with access control
   storage["agent_private"]["home_pos"] = torch.tensor([[0,0], [0,0], [0,0], [0,0]])  # shape: [num_agents, 2]
   ```
   Each scope has different initialization, indexing, and access patterns.

2. **Access control enforcement**:
   - `readable_by` / `writable_by` validation
   - Agent_private requires checking caller identity
   - Error handling for violations

3. **Type-specific default initialization**:
   - scalar → fill value
   - vec2i → broadcast [0, 0]
   - vecNi → broadcast [0] * dims
   - Must handle both global (1 value) and agent (num_agents values) shapes

**Why 10-12h is more realistic**:
- Tensor shape management is error-prone (indexing, broadcasting)
- Access control has many edge cases (agent reading other agent's private variable)
- Type system has 6 types × 3 scopes = 18 initialization patterns
- Debugging tensor shape mismatches will consume time

### 1.3 Observation Builder (MEDIUM Complexity) ⚠️
**Current plan**: 6-8 hours
**My assessment**: **6-8 hours OK, but needs renaming**

🚨 **CRITICAL ISSUE**: Naming collision detected!
```python
# EXISTING:
src/townlet/environment/observation_builder.py - ObservationBuilder (75 lines)

# PLAN WANTS TO CREATE:
src/townlet/vfs/observation_builder.py - ObservationBuilder (new class)
```

**Impact**: Importing confusion, test ambiguity, documentation conflicts

**Required fix**: Rename VFS class to avoid collision
- Suggestion: `VFSObservationSpecBuilder` or `VariableObservationSpecBuilder`
- Update all references in plan
- Add integration note explaining relationship to existing ObservationBuilder

**Complexity is reasonable** given:
- Shape inference is straightforward mapping (vec2i → [2], scalar → [])
- Must support all substrate types (Grid2D, Grid3D, GridND, Continuous, Aspatial)
- Integration with existing system needs clarification

### 1.4 Critical Regression Testing (HIGH Complexity) 🚨
**Current plan**: 3-4 hours (part of overall testing)
**My assessment**: **UNDERESTIMATED - needs dedicated analysis**

**The problem**:
```python
# Current observation_dim calculation (from vectorized_env.py):
# Full observability:
self.observation_dim = (
    self.substrate.get_observation_dim() +  # Grid2D: 2, Grid3D: 3, Aspatial: 0
    meter_count +                            # Always 8
    (self.num_affordance_types + 1) +       # Varies by config
    4                                        # Temporal features (always added)
)

# Partial observability (POMDP):
window_size = 2 * vision_range + 1
self.observation_dim = (
    window_size ** position_dim +           # 5^2 = 25 for Grid2D vision_range=2
    substrate.position_dim +                # 2 for Grid2D
    meter_count +                           # 8
    (self.num_affordance_types + 1) +      # Varies
    4                                       # Temporal
)

# VFS must produce IDENTICAL dimension or ALL checkpoints break!
```

**Current dimensions by config** (from existing tests):
- L0_minimal: 29 dims
- L0_5_dual_resource: 29 dims
- L1_full_observability: 29 dims
- L2_partial_observability: 54 dims (POMDP)
- L3_temporal_mechanics: 29 dims

**Challenges**:
1. **Chicken-and-egg**: Need variable definitions for each config, but configs don't have variables.yaml yet
2. **Reverse engineering**: Must map current hardcoded structure to VFS variable definitions
3. **Formula complexity**: Different formulas for full vs partial observability
4. **Multiple substrate types**: Grid2D, Grid3D, GridND, Continuous, Aspatial each have different encoding

**Why this is CRITICAL**:
- Single dimension mismatch → ALL checkpoints incompatible
- No gradual migration possible
- Affects months of training work
- Breaking change requires re-training from scratch

---

## 2. Risk Assessment

### 2.1 HIGH RISKS 🚨

#### Risk 1: Observation Dimension Incompatibility (SEVERITY: CRITICAL)
**Problem**: VFS-generated observation_dim must **exactly match** current hardcoded calculation.

**Impact**:
- Different dimension → ALL existing checkpoints incompatible
- Network architecture mismatch → cannot load weights
- Months of training invalidated
- Breaking change with no migration path

**Evidence from codebase**:
```python
# tests/test_townlet/integration/test_substrate_observations.py:86
assert obs.shape[1] == env.observation_dim, \
    f"observation_dim={env.observation_dim} doesn't match actual obs.shape[1]={obs.shape[1]}"
```

**Current mitigation in plan**: TDD Cycle 5 (observation dimension regression tests)

**My assessment**: ⚠️ **Mitigation is necessary but under-specified**

**Problems with current mitigation**:
1. Cycle 5 is too late - dimension validation should happen **before** implementation
2. Test assumes `load_variables_from_config()` exists - it doesn't
3. No clear process for creating reference variable definitions
4. Formula differences (full vs POMDP) not explicitly addressed

**Required improvements**:
- Add **"Cycle 0: Observation Structure Reverse Engineering"** (4-6 hours)
  - Document exact observation_dim formulas for each config
  - Create reference variable definitions matching current structure
  - Build dimension validation script
  - Validate references produce correct dimensions BEFORE implementing VFS
- Update Cycle 5 to use these reference definitions
- Add dimension to checkpoint metadata for future validation

**Contingency if dimensions mismatch**:
1. **Preferred**: Fix VFS calculation to match current (adjust variable definitions)
2. **Last resort**: Declare breaking change, invalidate all checkpoints, re-train

#### Risk 2: Naming Collision with Existing ObservationBuilder (SEVERITY: HIGH)
**Problem**: Plan creates `vfs/observation_builder.py::ObservationBuilder` but `environment/observation_builder.py::ObservationBuilder` already exists.

**Impact**:
- Import confusion (`from townlet.vfs.observation_builder import ObservationBuilder` vs existing)
- Test ambiguity (which ObservationBuilder are we testing?)
- Documentation conflicts
- Code reviewer confusion

**Current mitigation in plan**: None - **plan does not mention existing ObservationBuilder**

**Required fix**:
1. Rename VFS class to `VFSObservationSpecBuilder` or `VariableObservationSpecBuilder`
2. Update all references in plan (search for "ObservationBuilder" and clarify)
3. Add note explaining relationship to existing `environment.observation_builder.ObservationBuilder`
4. Document integration path

### 2.2 MEDIUM RISKS ⚠️

#### Risk 3: Scope Semantics Implementation Complexity (SEVERITY: MEDIUM)
**Problem**: global/agent/agent_private scopes have different tensor management patterns that are under-specified.

**Ambiguities in current plan**:
```python
# What exactly does this return?
registry.get("agent", "energy", agent_id=None, reader="engine")
# → All agents' values? Single agent? Error?

# How is access control enforced?
registry.get("agent_private", "home_pos", agent_id=1, reader="agent")
# → Which agent is reading? How do we know it's agent 1?

# What about cross-agent reads?
registry.get("agent", "position", agent_id=5, reader="agent")
# → Can agent 0 read agent 5's position? (should be yes for "agent" scope)
```

**Required clarification**:
1. Document exact semantics of `get(scope, var_id, agent_id, reader)`
2. Specify tensor indexing for each scope:
   - global + agent_id=None → return scalar
   - agent + agent_id=X → return storage[X]
   - agent + agent_id=None → return storage (all agents)
   - agent_private + agent_id=X → check reader == agent X, then return
3. Add edge case tests:
   - Agent reading another agent's agent_private (should error)
   - Agent reading another agent's public "agent" variable (should succeed)
   - Engine reading agent_private (depends on readable_by list)

#### Risk 4: Integration Path Unclear (SEVERITY: MEDIUM)
**Problem**: Plan doesn't clearly specify how VFS integrates with existing VectorizedHamletEnv observation system.

**Questions**:
1. Where in `VectorizedHamletEnv.__init__()` does VFS get initialized?
2. Does existing `environment.observation_builder.ObservationBuilder` get replaced?
3. How do observations transition from current system to VFS system in Phase 1?
4. What's the migration story for existing configs?

**Current plan says**: "Backward compatible - VFS is opt-in initially"

**But unclear**:
- What triggers VFS path vs legacy path?
- How do both systems coexist?
- When does migration happen?

**Required specification**:
- Add integration diagram showing VFS in VectorizedHamletEnv
- Document decision logic (if variables.yaml exists, use VFS; else use legacy)
- Add tests showing both systems coexist
- Clarify Phase 2 migration timeline

### 2.3 LOW RISKS ✅

#### Risk 5: Type System Completeness (SEVERITY: LOW)
**Problem**: vecNi/vecNf added for N-dimensional support, may need more types later.

**Mitigation in plan**: Phase 2 explicitly defers complex types (categorical, stack, queue) ✅

**My assessment**: Low risk - type system is extensible, Pydantic makes adding types easy

#### Risk 6: Performance (SEVERITY: LOW)
**Problem**: Registry get/set might not scale.

**My assessment**: Low risk - dict lookups are O(1), existing `AgentRuntimeRegistry` proves pattern works

---

## 3. Executability Assessment

### 3.1 Prerequisites ✅
- ✅ **TASK-002A** (Substrate): Complete - substrate system exists and working
- ✅ **TASK-002B** (Composable Actions): Complete - ActionConfig uses Pydantic
- ✅ **Test infrastructure**: pytest, fixtures, 40+ existing tests
- ✅ **Pydantic**: Already in use (ActionConfig, substrate configs)
- ✅ **Patterns established**: AgentRuntimeRegistry shows registry pattern works

### 3.2 Implementation Path ✅ (with modifications)
- ✅ TDD approach is well-defined (RED-GREEN-REFACTOR)
- ✅ Each cycle has clear deliverables
- ✅ Test examples are concrete and realistic
- ⚠️ Needs Cycle 0 addition (observation reverse engineering)
- ⚠️ Needs ObservationBuilder renaming throughout

### 3.3 Effort Estimate ⚠️ (OPTIMISTIC)

**Plan's revised estimate**: 28-36 hours (4-5 days)

**My detailed breakdown**:

| Component | Plan | My Estimate | Rationale |
|-----------|------|-------------|-----------|
| **Cycle 0: Reverse Engineering** | 0h | **4-6h** | NEW - document current observation structure, create reference variables |
| **Cycle 1: Schema** | 4h | 4h | Reasonable - Pydantic schemas straightforward |
| **Cycle 2: Registry** | 8-10h | **10-12h** | +20% - scope semantics complexity, access control edge cases |
| **Cycle 3: Observation Builder** | 6-8h | 6-8h | OK - but must rename class |
| **Cycle 4: ActionConfig Extension** | 2h | 2h | Reasonable - simple Pydantic fields |
| **Cycle 5: Dimension Regression** | 3-4h | 3-4h | OK after Cycle 0 - validation with references |
| **Cycle 6: Integration Tests** | 2-3h | **3-4h** | +1h - more complex flow with both observation builders |
| **Cycle 7: Config Templates** | 1h | 1h | Reasonable |
| **Documentation** | 2h | 2h | Reasonable |
| **Integration Specification** | 0h | **2h** | NEW - document VFS integration with existing system |
| **TOTAL** | **28-36h** | **38-50h** | **+33% increase** |

**Confidence**: Medium (60%)

**Why more conservative**:
1. **Tensor shape debugging** consumes time (experience shows 2-3h for shape issues)
2. **Access control edge cases** often reveal unexpected behaviors (1-2h)
3. **Observation dimension validation** is non-negotiable and complex (Cycle 0 needed)
4. **Integration with existing system** requires careful thought (2h for spec)
5. **Buffer for unknowns** (10-15% of estimate)

### 3.4 Critical Validation Path 🚨

**The observation dimension regression testing is ESSENTIAL but has a chicken-and-egg problem:**

Plan's Cycle 5 test:
```python
def test_vfs_observation_dim_backward_compatible():
    for config_name in ["L0_minimal", "L0_5_dual", "L1_full", "L2_pomdp", "L3_temporal"]:
        # Current environment (hardcoded calculation)
        env_current = VectorizedHamletEnv.from_config(config_name)

        # VFS-based calculation
        variables = load_variables_from_config(config_name)  # ← DOESN'T EXIST YET!
        obs_spec = ObservationBuilder.build_observation_spec(variables)
        vfs_dim = sum(field.shape_size for field in obs_spec.fields)

        assert vfs_dim == env_current.observation_dim
```

**Problems**:
1. `load_variables_from_config()` doesn't exist
2. Configs don't have `variables.yaml` yet (they're created in Cycle 7)
3. How do we create variable definitions that match current structure?

**Required solution: Add Cycle 0 before implementation**

**Cycle 0: Observation Structure Reverse Engineering (4-6 hours)**

1. **Document current observation formulas** (1h):
   ```python
   # Full observability:
   obs_dim = substrate.get_observation_dim() + 8 + (num_affordances + 1) + 4

   # Partial observability:
   obs_dim = (vision_window_size ** position_dim) + position_dim + 8 + (num_affordances + 1) + 4
   ```

2. **Create reference variable definitions for each config** (2-3h):
   ```yaml
   # configs/L1_full_observability/variables_reference.yaml
   variables:
     - id: "position"
       scope: "agent"
       type: "vec2i"
       # ...
     - id: "energy"
       scope: "agent"
       type: "scalar"
       # ... (repeat for all 8 meters)

   exposed_observations:
     - id: "obs_position"
       source_variable: "position"
       shape: [2]  # substrate.get_observation_dim() = 2 for Grid2D
     # ... map to current observation structure
   ```

3. **Build validation script** (1h):
   ```python
   def validate_reference_dimensions():
       for config_name, expected_dim in [
           ("L0_minimal", 29),
           ("L1_full", 29),
           ("L2_pomdp", 54),
           # ...
       ]:
           env = load_env(config_name)
           variables = load_reference_variables(config_name)

           # Compute expected dimension from variables
           computed_dim = compute_observation_dim(variables)

           assert computed_dim == expected_dim, \
               f"{config_name}: Expected {expected_dim}, got {computed_dim}"
   ```

4. **Validate before implementing VFS** (30min):
   - Run validation script
   - Ensure all configs produce correct dimensions
   - Document any discrepancies
   - Fix reference definitions until validation passes

**Benefits of Cycle 0**:
- De-risks Cycle 5 (regression tests can use reference definitions)
- Provides concrete examples for implementation
- Catches dimension issues before building VFS
- Creates documentation of current observation structure
- Enables early validation without full VFS implementation

---

## 4. Specific Recommendations

### 4.1 CRITICAL - Must Fix Before Proceeding 🚨

#### 1. Add Cycle 0: Observation Structure Reverse Engineering
**Time**: 4-6 hours
**Deliverables**:
- Documented observation_dim formulas for all configs
- Reference variables.yaml for each config (L0, L0.5, L1, L2, L3)
- Validation script confirming references produce correct dimensions
- Documentation of dimension mapping (current → VFS)

**Rationale**: De-risks critical observation dimension compatibility

#### 2. Rename VFS ObservationBuilder to Avoid Collision
**Time**: 30 minutes
**Changes**:
- `ObservationBuilder` → `VFSObservationSpecBuilder` (or similar)
- Update all references in plan
- Add note about existing `environment.observation_builder.ObservationBuilder`
- Document relationship between two classes

**Rationale**: Prevents import confusion, test ambiguity, documentation conflicts

#### 3. Create Integration Specification
**Time**: 2 hours
**Deliverables**:
- Document how VFS integrates with VectorizedHamletEnv
- Specify decision logic (VFS vs legacy path)
- Add tests showing both systems coexist
- Clarify Phase 2 migration timeline

**Rationale**: Removes ambiguity about how VFS fits into existing system

### 4.2 IMPORTANT - Should Fix for Robustness ⚠️

#### 4. Clarify Scope Semantics
**Time**: 1-2 hours
**Changes**:
- Document exact semantics of `registry.get(scope, var_id, agent_id, reader)`
- Specify tensor indexing patterns for each scope
- Add edge case tests (agent reading another agent's private variable, etc.)
- Add examples to docstrings

**Example specification**:
```python
def get(scope: str, var_id: str, agent_id: Optional[int] = None, reader: str = "agent"):
    """Get variable value with access control.

    Semantics:
    - scope="global", agent_id=None → return scalar tensor []
    - scope="global", agent_id=X → Error (global has no agent_id)
    - scope="agent", agent_id=None → return all agents [num_agents, ...]
    - scope="agent", agent_id=X → return agent X's value [...]
    - scope="agent_private", agent_id=None → Error (must specify agent)
    - scope="agent_private", agent_id=X, reader="agent" → return if reader owns agent X
    """
```

#### 5. Extend Registry Testing Estimate
**Time**: +2-3 hours
**Reason**: Access control edge cases, scope semantics validation
**New estimate**: 10-12 hours (vs 8-10h planned)

#### 6. Add Integration Tests for Dual Observation Systems
**Time**: +1 hour
**Tests**:
- Legacy ObservationBuilder still works
- VFS ObservationSpecBuilder produces compatible specs
- Both can coexist in same environment
- Decision logic works correctly

### 4.3 NICE TO HAVE - Quality Improvements ✅

#### 7. Add Dimension Formula Documentation
**Time**: 1 hour
**Location**: `src/townlet/vfs/README.md`
**Content**:
```markdown
## Observation Dimension Formulas

### Full Observability
obs_dim = substrate.get_observation_dim() + meter_count + (num_affordances + 1) + 4

- substrate.get_observation_dim(): Grid2D=2, Grid3D=3, Aspatial=0
- meter_count: Always 8 (energy, health, satiation, money, mood, social, fitness, hygiene)
- num_affordances + 1: One-hot encoding (affordance types + "none")
- 4: Temporal features (time_sin, time_cos, interaction_progress, lifetime_progress)

### Partial Observability (POMDP)
obs_dim = window_size^position_dim + position_dim + meter_count + (num_affordances + 1) + 4

- window_size: (2 * vision_range + 1) - e.g., vision_range=2 → 5×5 window
- position_dim: 2 for Grid2D, 3 for Grid3D
```

#### 8. Add Property-Based Tests for Access Control
**Time**: 2-3 hours
**Library**: hypothesis
**Tests**:
- Any reader not in readable_by list should be rejected
- Any writer not in writable_by list should be rejected
- Agent_private variables only accessible by owner

---

## 5. Revised Implementation Plan

### Phase 0: Pre-Implementation (NEW)

**Cycle 0: Observation Structure Reverse Engineering (4-6 hours)**
1. Document current observation_dim formulas (1h)
2. Create reference variables.yaml for each config (2-3h)
3. Build dimension validation script (1h)
4. Validate references produce correct dimensions (30min)

**Cycle 0.5: Plan Updates (1 hour)**
1. Rename ObservationBuilder → VFSObservationSpecBuilder (30min)
2. Create integration specification (30min)

### Phase 1: Implementation (TDD Cycles)

**Cycles 1-7**: As planned with modifications:
- Cycle 2: Registry (10-12h instead of 8-10h)
- Cycle 3: Use renamed VFSObservationSpecBuilder
- Cycle 5: Use reference variables from Cycle 0
- Cycle 6: Add dual observation system tests (+1h)

### Revised Timeline

| Phase | Planned | Revised | Delta |
|-------|---------|---------|-------|
| **Pre-Implementation (Cycle 0-0.5)** | 0h | **5-7h** | +5-7h |
| **Schema (Cycle 1)** | 4h | 4h | 0h |
| **Registry (Cycle 2)** | 8-10h | **10-12h** | +2h |
| **Observation Builder (Cycle 3)** | 6-8h | 6-8h | 0h |
| **ActionConfig Extension (Cycle 4)** | 2h | 2h | 0h |
| **Dimension Regression (Cycle 5)** | 3-4h | 3-4h | 0h |
| **Integration Tests (Cycle 6)** | 2-3h | **3-4h** | +1h |
| **Config Templates (Cycle 7)** | 1h | 1h | 0h |
| **Documentation** | 2h | 2h | 0h |
| **Integration Spec** | 0h | **2h** | +2h |
| **TOTAL** | **28-36h** | **38-50h** | **+10-14h (+33%)** |

---

## 6. Decision Matrix

### Proceed If:
- ✅ Team accepts revised 38-50 hour estimate
- ✅ Cycle 0 (reverse engineering) is added to plan
- ✅ VFS observation builder is renamed to avoid collision
- ✅ Scope semantics are more fully specified
- ✅ Integration specification is documented
- ✅ Team acknowledges observation dimension compatibility as CRITICAL risk

### Do Not Proceed If:
- ❌ Observation dimension compatibility cannot be validated upfront
- ❌ Team needs delivery in <35 hours (estimate too optimistic)
- ❌ Scope semantics remain ambiguous
- ❌ Integration with existing ObservationBuilder is unclear
- ❌ Team unwilling to risk checkpoint compatibility

---

## 7. Summary & Final Recommendation

### Strengths of the Plan ✅
1. **TDD approach**: Excellent methodology for foundational work
2. **Comprehensive test coverage**: 25-30 tests appropriate for scope
3. **Clear deliverables**: Each cycle has concrete outputs
4. **Risk awareness**: Plan identifies critical risks (dimension compatibility)
5. **Backward compatibility**: ActionConfig extension handled well
6. **Prerequisites met**: All dependencies complete, patterns established

### Weaknesses of the Plan ⚠️
1. **Optimistic effort estimate**: 28-36h → realistic 38-50h (+33%)
2. **Missing Cycle 0**: Observation reverse engineering should come first
3. **Naming collision**: Unaddressed conflict with existing ObservationBuilder
4. **Scope semantics**: Under-specified tensor management for scopes
5. **Integration path**: How VFS fits into existing system unclear
6. **Chicken-and-egg**: Cycle 5 regression tests depend on variables that don't exist yet

### Critical Risks 🚨
1. **Observation dimension incompatibility** (HIGH) - checkpoint breakage
2. **Naming collision** (HIGH) - import/test confusion
3. **Scope semantics bugs** (MEDIUM) - tensor shape errors
4. **Integration ambiguity** (MEDIUM) - coexistence with legacy system

### Final Recommendation

**CONDITIONAL APPROVAL - PROCEED WITH MODIFICATIONS**

This is a **well-designed plan** with solid TDD methodology, but it has **critical gaps** that must be addressed:

**REQUIRED before starting implementation**:
1. ✅ Add Cycle 0 (observation reverse engineering, 4-6h)
2. ✅ Rename VFS observation builder (30min)
3. ✅ Create integration specification (2h)
4. ✅ Accept revised 38-50h estimate

**RECOMMENDED for robustness**:
5. ⚠️ Clarify scope semantics (1-2h)
6. ⚠️ Extend registry testing estimate (+2-3h)
7. ⚠️ Add dual observation system tests (+1h)

**With these modifications**:
- **Complexity**: MEDIUM-HIGH (manageable with TDD)
- **Risk**: MEDIUM (critical risks mitigated)
- **Executability**: HIGH (clear path, realistic estimate)
- **Delivery**: 38-50 hours (5-6 days) ← realistic

**Without modifications**:
- **Risk**: HIGH (observation compatibility, naming collision)
- **Likelihood of schedule overrun**: 75%
- **Likelihood of checkpoint breakage**: 30%

**Bottom Line**: This task is **executable and valuable**, but needs **10-14 additional hours** of planning and implementation. The core TDD approach is sound. The critical path is observation dimension compatibility - **validate first, implement second**.

---

## 8. Appendix: Code References

### Current Observation Dimension Calculation
**File**: `src/townlet/environment/vectorized_env.py:249-264`
```python
# Full observability:
self.observation_dim = (
    self.substrate.get_observation_dim() + meter_count + (self.num_affordance_types + 1)
)

# Partial observability:
self.observation_dim = (
    self.substrate.get_observation_dim() + meter_count + (self.num_affordance_types + 1)
)

# Always add temporal features:
self.observation_dim += 4
```

### Existing ObservationBuilder
**File**: `src/townlet/environment/observation_builder.py:11-149`
- Class: `ObservationBuilder`
- Methods: `build_observations()`, `_build_full_observations()`, `_build_partial_observations()`
- **Conflict**: Plan wants to create `vfs/observation_builder.py::ObservationBuilder`

### Validation Tests
**File**: `tests/test_townlet/integration/test_substrate_observations.py:67-86`
```python
def test_observation_dim_matches_actual_observation(test_config_pack_path):
    env = VectorizedHamletEnv(...)
    obs = env.reset()
    assert obs.shape[1] == env.observation_dim
```

### ActionConfig Schema
**File**: `src/townlet/environment/action_config.py:8-73`
- Uses Pydantic BaseModel
- Has field validators
- Good pattern for VFS schemas

---

**Review Completed**: 2025-11-07
**Next Step**: Address critical modifications, then proceed with Cycle 0
