# QUICK-05: Structured Observation Masking & Encoders

**Status**: ✅ COMPLETED
**Priority**: High
**Estimated Effort**: 4 hours
**Actual Effort**: ~6 hours (8 tasks implemented via TDD)
**Dependencies**: TASK-004A (compiler pipeline), QUICK-004 (test remediation)
**Created**: 2025-02-14
**Completed**: 2025-02-15

**Keywords**: observation-mask, compiler, VFS, networks, RND
**Subsystems**: compiler, environment, agent networks, exploration
**Files**: `src/townlet/vfs/schema.py`, `src/townlet/universe/adapters/vfs_adapter.py`, `src/townlet/agent/networks.py`

---

## AI-Friendly Summary (Skim This First!)

**What**: Add compiler-emitted observation activity masks plus structured encoders so runtime networks only learn from curriculum-active dimensions.
**Why**: Current flat obs vectors waste ~70% of inputs on permanent zeros, slowing learning and breaking intrinsic novelty.
**Scope**: Extend VFS schemas/configs, compiler DTOs, runtime wiring, feedforward + RND models to use per-group masks; no curriculum content changes.

**Quick Assessment**:

- Current State: ✅ Compiler emits `ObservationActivity` with active_mask and group_slices; ✅ RND applies masking; ✅ StructuredQNetwork implemented.
- Goal: ✅ Compiler outputs `ObservationActivity` metadata + masks; ✅ Structured network/RND use masked groups.
- Impact: Faster convergence for early curricula, less RND drift, cleaner portability path.
- **Status**: All 8 implementation tasks completed and committed (8 commits total).

---

## Problem Statement

### Context

Portability requires a fixed ~50-dim observation vector, but L0.0 only uses ~14 dims. Flattened encoders feed zeros for unused meters/affordances/temporal channels, so gradients dilute and novelty signals collapse. We need compiler-authored metadata that says which fields matter per curriculum level and runtime models that respect it.

### Current Limitations

**What Doesn't Work**:

- Flat `ObservationSpec` lacks activity or semantic grouping, so consumers cannot distinguish padding from live data.
- `SimpleQNetwork`/`RNDNetwork` ingest all dims equally, wasting capacity and corrupting novelty estimates.

**What We're Missing**:

- A persisted observation mask keyed to compiler UUIDs for checkpoint safety.
- Structured encoders that zero inactive bars/affordances while keeping portability.

### Use Cases

**Primary Use Case**: L0.0 agents learn using only health, position, 3×3 grid, and one affordance without modifying configs or networks between curricula.

**Secondary Use Cases**:

- RND novelty tracks only active dims, so intrinsic motivation remains meaningful.
- Curriculum upgrades reuse checkpoints by expanding masks and copying weights by field UUID.

---

## Solution Design

### Overview

Enhance VFS observation fields with explicit `semantic_type` + `curriculum_active` metadata. Compiler converts this into an `ObservationActivity` DTO that travels with `CompiledUniverse`. Runtime environments expose the mask to populations. Structured encoders slice groups (bars, spatial, affordance, temporal), apply mask buffers, and feed combined features to Q-networks and RND.

### Technical Approach

**Implementation Steps**:

1. Extend VFS schema/template + configs with semantic + curriculum flags.
2. Build `ObservationActivity` in compiler adapters, include in `CompiledUniverse`/`RuntimeUniverse`.
3. Wire mask into `VectorizedHamletEnv`/`VectorizedPopulation`, expose grouped slices.
4. Add `StructuredQNetwork` + masked RND pipeline that consume the new metadata.

**Key Design Decisions**:

- **Compiler-owned mask**: Ensures mask/UUID alignment is versioned and validated like `observation_field_uuids`.
- **Group encoders over per-dim masks**: Maintains portability (fixed bar count) while minimizing zero-noise via learned encoders plus element-wise mask buffers.

### Edge Cases

**Must Handle**:

- Partial observability: spatial slice swaps between full-grid vs local window but stays in same group width.
- Aspatial substrates: skip spatial/position encoders entirely when the compiler reports zero dims.

---

## Implementation Plan

### Phase 1: VFS Schema & Config Metadata (45 min)

**Files**: `src/townlet/vfs/schema.py`, `configs/**/variables_reference.yaml`, `docs/tasks/templates/QUICK-TEMPLATE.md`

**Changes**:

```python
class ObservationField(BaseModel):
    semantic_type: Literal["bars","spatial","affordance","temporal","custom"] = "custom"
    curriculum_active: bool = True
```

Update templates/sample configs to set semantic + active flags for each field (e.g., only `health` true in L0.0). Document in READMEs.

**Testing**:

- [x] ✅ VFS schema accepts semantic_type and curriculum_active fields
- [x] ✅ Config loader accepts new fields (all configs updated)

### Phase 2: Compiler ObservationActivity DTO (60 min)

**Files**: `src/townlet/universe/dto/observation_spec.py`, `src/townlet/universe/adapters/vfs_adapter.py`, `src/townlet/universe/compiled.py`, `src/townlet/universe/runtime.py`

**Changes**:

```python
@dataclass(frozen=True)
class ObservationActivity:
    active_mask: tuple[bool, ...]
    group_slices: dict[str, slice]
    active_field_uuids: tuple[str, ...]
```

Generate mask by walking VFS fields, flatten shapes, store group slices per `semantic_type`. Add to compiled/runtime DTOs and MessagePack cache serialization.

**Testing**:

- [x] ✅ New unit test verifying mask matches spec dims (`test_compiled_universe_activity.py`)
- [x] ✅ Cache round-trip test with backward compatibility for old caches

### Phase 3: Runtime Wiring (45 min)

**Files**: `src/townlet/environment/vectorized_env.py`, `src/townlet/population/vectorized.py`, `src/townlet/exploration/adaptive_intrinsic.py`

**Changes**:

- Store `observation_activity` on env/runtime.
- Pass mask to population, register as `torch.bool` buffer on networks/RND.
- Update RND observation handling to apply mask before feeding embeddings.

**Testing**:

- [x] ✅ Unit tests verify env exposes observation_activity (`test_env_observation_activity.py`)
- [x] ✅ RND masking integration tests with real observation activity

### Phase 4: Structured Encoders & Masked RND (90 min)

**Files**: `src/townlet/agent/networks.py`, `src/townlet/exploration/rnd.py`, `tests/test_townlet/_fixtures/networks.py`

**Changes**:

- Add `StructuredQNetwork` with bar/spatial/affordance/temporal encoders using group masks.
- Update population to select structured network by config flag (default for simple obs).
- Apply same mask in RND (active dims only) and store `active_field_uuids` in checkpoints for compatibility checks.

**Testing**:

- [x] ✅ Unit tests for StructuredQNetwork forward pass + group encoders (`test_structured_qnetwork.py`)
- [x] ✅ RND unit tests verifying masked inputs (`test_rnd_masking.py` - 7 tests)

### Phase 5: Validation & Docs (30 min)

**Verification Steps**:

1. [x] ✅ All unit tests pass (20 tests across VFS, universe, agent, exploration)
2. [x] ✅ Network selection tests verify config validation and instantiation
3. [x] ✅ Documentation updated (this file marked complete)
4. [x] ✅ All 13 training.yaml configs updated with mask_unused_obs field

---

## Testing Strategy

**Test Requirements**:

- **Unit Tests**: VFS schema defaults, ObservationActivity builder, StructuredQNetwork mask application, RND masked forward.
- **Integration Tests**: Existing data-flow test updated to assert masked dims, plus environment reset/step path verifying mask broadcast.
- **Property Tests**: None.

**Coverage Target**: ≥80% for new modules (`observation_activity`, structured network).

**Test-Driven Development**:

- [x] ✅ TDD methodology followed for all 8 tasks (RED → GREEN → REFACTOR)
- [x] ✅ Tests written first, watched them fail, then implemented minimal code to pass
- [x] ✅ All commits included test results showing GREEN phase before commit

---

## Acceptance Criteria

**Must Have**:

- [x] ✅ VFS schema + configs support `semantic_type` and `curriculum_active`
- [x] ✅ Compiler emits `ObservationActivity` with validated mask and UUID list
- [x] ✅ Runtime env/population expose mask tensors to networks/RND
- [x] ✅ Structured network + RND apply masks and pass updated tests
- [x] ✅ All test suites pass (20 new tests, all existing tests still pass)
- [x] ✅ ObservationActivity includes active_field_uuids for checkpoint compatibility

**Success Metrics**:

- Masked dim count matches manually computed active dims per config.
- Simple smoke training on L0.0 converges ≥25% faster (fewer episodes) due to higher SNR.

---

## Risk Assessment

**Technical Risks**:

- ✅ **LOW**: Template/config churn (easy to update).
- ⚠️ **MEDIUM**: Mask/UUID mismatches could break checkpoint compatibility.
- ❌ **HIGH**: None.

**Mitigation**:

- Add compiler validation to ensure mask length equals `ObservationSpec.total_dims`.
- Store `active_field_uuids` in checkpoints and compare before loading.

---

## Future Enhancements (Out of Scope)

**Not Included**:

- Dynamic runtime masking (changing active dims mid-training) → future TASK for curriculum gating.
- Attention-based masking instead of static encoders → potential research follow-up.

**Rationale**: Keep QUICK scope to compiler metadata + structured encoders; adaptive masks require larger architecture changes.

---

## Completion Summary

**Implementation Timeline**:
- Task 1: VFS schema fields (semantic_type, curriculum_active) ✅
- Task 2: ObservationActivity DTO creation ✅
- Task 3: VFS adapter builds ObservationActivity ✅
- Task 4: Wire ObservationActivity into Compiled/Runtime Universe ✅
- Task 5: Expose ObservationActivity to Environment ✅
- Task 6: Wire active_mask to Population and RND ✅
- Task 7: Create StructuredQNetwork with group encoders ✅
- Task 8: Add mask_unused_obs config and network selection ✅

**Commits**: 8 total (1 per task, following TDD methodology)

**Files Modified**:
- `src/townlet/vfs/schema.py` - Added semantic_type and curriculum_active fields
- `src/townlet/universe/dto/observation_activity.py` - New DTO for observation masking
- `src/townlet/universe/adapters/vfs_adapter.py` - Builds ObservationActivity from VFS fields
- `src/townlet/universe/compiled.py` - Added observation_activity field with serialization
- `src/townlet/universe/runtime.py` - Added observation_activity field
- `src/townlet/universe/compiler.py` - Builds observation_activity during compilation
- `src/townlet/environment/vectorized_env.py` - Exposes observation_activity
- `src/townlet/exploration/rnd.py` - Applies active_mask to zero padding dimensions
- `src/townlet/exploration/adaptive_intrinsic.py` - Passes active_mask to RND
- `src/townlet/agent/networks.py` - Added StructuredQNetwork class
- `src/townlet/config/population.py` - Added mask_unused_obs and "structured" network type
- `src/townlet/population/vectorized.py` - Instantiates StructuredQNetwork based on config
- `src/townlet/demo/runner.py` - Conditionally passes active_mask based on config
- `src/townlet/demo/live_inference.py` - Conditionally passes active_mask based on config
- All 13 `configs/*/training.yaml` - Added mask_unused_obs field

**Tests Added**:
- `tests/test_townlet/unit/vfs/test_vfs_observation_activity.py` (3 tests)
- `tests/test_townlet/unit/universe/test_compiled_universe_activity.py` (6 tests)
- `tests/test_townlet/unit/environment/test_env_observation_activity.py` (4 tests)
- `tests/test_townlet/unit/exploration/test_rnd_masking.py` (7 tests)
- `tests/test_townlet/unit/agent/test_structured_qnetwork.py` (7 tests)
- `tests/test_townlet/unit/agent/test_network_selection.py` (6 tests)

**Total Test Coverage**: 33 new tests, all passing

**Key Features Delivered**:
1. Compiler-emitted observation activity metadata with active_mask and group_slices
2. Semantic grouping of observations (spatial, bars, affordances, temporal, custom)
3. RND networks mask padding dimensions for improved sample efficiency
4. StructuredQNetwork with group encoders for better inductive bias
5. Configuration support for mask_unused_obs flag (backward compatible)
6. Full TDD methodology with RED → GREEN → REFACTOR cycle

---

## References

**Related Tasks**:

- TASK-004A: Universe compiler implementation (base DTO plumbing)
- TASK-002C: Variable & Feature System (original VFS integration)

**Code Files**:

- `src/townlet/universe/adapters/vfs_adapter.py` — builds observation spec from VFS fields
- `src/townlet/agent/networks.py` — Q-network definitions (Simple + Recurrent + Structured)
- `src/townlet/exploration/rnd.py` — intrinsic exploration with mask awareness
- `src/townlet/universe/dto/observation_activity.py` — observation masking DTO
