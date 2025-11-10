# QUICK-05: Structured Observation Masking & Encoders

**Status**: Planned
**Priority**: High
**Estimated Effort**: 4 hours
**Dependencies**: TASK-004A (compiler pipeline), QUICK-004 (test remediation)
**Created**: 2025-02-14
**Completed**: _TBD_

**Keywords**: observation-mask, compiler, VFS, networks, RND
**Subsystems**: compiler, environment, agent networks, exploration
**Files**: `src/townlet/vfs/schema.py`, `src/townlet/universe/adapters/vfs_adapter.py`, `src/townlet/agent/networks.py`

---

## AI-Friendly Summary (Skim This First!)

**What**: Add compiler-emitted observation activity masks plus structured encoders so runtime networks only learn from curriculum-active dimensions.
**Why**: Current flat obs vectors waste ~70% of inputs on permanent zeros, slowing learning and breaking intrinsic novelty.
**Scope**: Extend VFS schemas/configs, compiler DTOs, runtime wiring, feedforward + RND models to use per-group masks; no curriculum content changes.

**Quick Assessment**:

- Current State: ❌ Compiler lacks notion of active vs padded dims; ❌ SimpleQNetwork/RND consume all dims blindly.
- Goal: ✅ Compiler outputs `ObservationActivity` metadata + masks; ✅ Structured network/RND use masked groups.
- Impact: Faster convergence for early curricula, less RND drift, cleaner portability path.

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

- [ ] `tests/test_townlet/unit/vfs/test_observation_builder.py::test_field_defaults`
- [ ] Config lint (ensure loader accepts new keys).

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

- [ ] New unit test verifying mask matches spec dims.
- [ ] Cache round-trip test in `tests/test_townlet/unit/universe/test_compiled_universe.py`.

### Phase 3: Runtime Wiring (45 min)

**Files**: `src/townlet/environment/vectorized_env.py`, `src/townlet/population/vectorized.py`, `src/townlet/exploration/adaptive_intrinsic.py`

**Changes**:

- Store `observation_activity` on env/runtime.
- Pass mask to population, register as `torch.bool` buffer on networks/RND.
- Update RND observation handling to apply mask before feeding embeddings.

**Testing**:

- [ ] Integration test `tests/test_townlet/integration/test_data_flows.py` ensures env + population agree on masked dims.

### Phase 4: Structured Encoders & Masked RND (90 min)

**Files**: `src/townlet/agent/networks.py`, `src/townlet/exploration/rnd.py`, `tests/test_townlet/_fixtures/networks.py`

**Changes**:

- Add `StructuredQNetwork` with bar/spatial/affordance/temporal encoders using group masks.
- Update population to select structured network by config flag (default for simple obs).
- Apply same mask in RND (active dims only) and store `active_field_uuids` in checkpoints for compatibility checks.

**Testing**:

- [ ] Unit tests for new network forward pass + mask behavior.
- [ ] RND unit verifying masked inputs reduce to active dims.

### Phase 5: Validation & Docs (30 min)

**Verification Steps**:

1. [ ] `uv run pytest tests/test_townlet/unit/vfs tests/test_townlet/unit/universe tests/test_townlet/unit/agent`.
2. [ ] `uv run pytest tests/test_townlet/integration/test_data_flows.py`.
3. [ ] Update `docs/architecture/hld/review/review-05-observation-space-specification.md` with mask overview.
4. [ ] Smoke run `scripts/validate_substrate_runtime.py` on L0_0 + L1 configs.

---

## Testing Strategy

**Test Requirements**:

- **Unit Tests**: VFS schema defaults, ObservationActivity builder, StructuredQNetwork mask application, RND masked forward.
- **Integration Tests**: Existing data-flow test updated to assert masked dims, plus environment reset/step path verifying mask broadcast.
- **Property Tests**: None.

**Coverage Target**: ≥80% for new modules (`observation_activity`, structured network).

**Test-Driven Development**:

- [ ] Add failing unit tests for ObservationActivity + structured encoder slices before implementation.
- [ ] Ensure integration test fails without mask wiring.
- [ ] Implement minimal code to flip tests to green, then refactor.

---

## Acceptance Criteria

**Must Have**:

- [ ] VFS schema + configs support `semantic_type` and `curriculum_active`.
- [ ] Compiler emits `ObservationActivity` with validated mask and UUID list.
- [ ] Runtime env/population expose mask tensors to networks/RND.
- [ ] Structured network + RND apply masks and pass updated tests.
- [ ] All test suites pass (`uv run pytest`).
- [ ] Checkpoint compatibility checks include `active_field_uuids`.

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

## References

**Related Tasks**:

- TASK-004A: Universe compiler implementation (base DTO plumbing).
- TASK-002C: Variable & Feature System (original VFS integration).

**Code Files**:

- `src/townlet/universe/adapters/vfs_adapter.py` — builds observation spec from VFS fields.
- `src/townlet/agent/networks.py` — Q-network definitions (Simple + Recurrent).
- `src/townlet/exploration/rnd.py` — intrinsic exploration (needs mask awareness).
- `tests/test_townlet/integration/test_data_flows.py` — validates obs dimension flows end-to-end.
