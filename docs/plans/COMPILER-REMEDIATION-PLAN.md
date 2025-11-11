# Compiler Implementation Remediation Plan

**Status**: ACTIVE
**Created**: 2025-11-11
**Branch**: 004a-compiler-implementation
**Based On**: Code review by superpowers:code-reviewer agent

---

## Executive Summary

The Universe Compiler implementation is **architecturally excellent** but blocked by config migration issues. This plan addresses:

- **2 CRITICAL issues** (blocking merge) - Estimated 4-6 hours
- **3 IMPORTANT issues** (fix before TASK-005) - Estimated 6-8 hours
- **4 MINOR issues** (nice to have) - Estimated 2-3 hours

**Total Estimated Effort**: 12-17 hours (1.5-2 days)

---

## Phase 1: Critical Issues (BLOCKING MERGE)

### CRITICAL-1: Config Migration Incomplete

**Problem**: 5+ config packs use legacy affordance schema that compiler rejects

**Affected Configs**:
```bash
configs/L0_0_minimal/affordances.yaml          ❌ Has required_ticks, no effect_pipeline
configs/L1_3D_house/affordances.yaml           ❌ Legacy format
configs/L1_continuous_1D/affordances.yaml      ❌ Legacy format
configs/L1_continuous_2D/affordances.yaml      ❌ Legacy format
configs/L1_continuous_3D/affordances.yaml      ❌ Legacy format
```

**Already Migrated**:
```bash
configs/L0_5_dual_resource/affordances.yaml    ✅ Modern format
configs/L1_full_observability/affordances.yaml ✅ Modern format
configs/L2_partial_observability/affordances.yaml ✅ Modern format
configs/L3_temporal_mechanics/affordances.yaml ✅ Modern format
```

**Solution**: Execute migration script on remaining configs

**Tasks**:
1. Test migration script on L0_0_minimal (validate output)
2. Backup all legacy configs
3. Run migration script on all 5 legacy configs:
   ```bash
   python scripts/migrate_affordances.py configs/L0_0_minimal/affordances.yaml
   python scripts/migrate_affordances.py configs/L1_3D_house/affordances.yaml
   python scripts/migrate_affordances.py configs/L1_continuous_1D/affordances.yaml
   python scripts/migrate_affordances.py configs/L1_continuous_2D/affordances.yaml
   python scripts/migrate_affordances.py configs/L1_continuous_3D/affordances.yaml
   ```
4. Handle templates separately (may need manual review):
   ```bash
   python scripts/migrate_affordances.py configs/templates/affordances.yaml
   ```
5. Check test configs (may be intentionally legacy for testing):
   ```bash
   # Review these manually - may be test fixtures
   configs/aspatial_test/affordances.yaml
   configs/test/affordances.yaml
   ```

**Validation**:
```bash
# Verify all configs compile
for dir in configs/L0_0_minimal configs/L0_5_dual_resource configs/L1_* configs/L2_* configs/L3_*; do
    echo "Testing $dir..."
    python -m townlet.compiler compile "$dir" || echo "FAILED: $dir"
done
```

**Estimated Effort**: 2-3 hours

---

### CRITICAL-2: Test Suite Failures

**Current Status**:
- Universe unit tests: **37 failed, 34 passed, 50 errors**
- Integration tests: **16 failed, 4 passed, 11 errors**

**Root Cause**: Invalid configs (CRITICAL-1)

**Solution Approach**:
1. Fix CRITICAL-1 first (config migration)
2. Re-run test suite to identify remaining issues
3. Fix any real bugs uncovered

**Tasks**:
1. After CRITICAL-1 complete, run universe unit tests:
   ```bash
   uv run pytest tests/test_townlet/unit/universe/ -v --tb=short
   ```
2. Document any remaining failures
3. Fix real bugs (vs config issues)
4. Run integration tests:
   ```bash
   uv run pytest tests/test_townlet/integration/test_checkpointing.py -v
   ```
5. Update test fixtures if needed

**Expected Outcome**: 90%+ pass rate after config migration

**Estimated Effort**: 2-3 hours

---

## Phase 2: Important Issues (Fix Before TASK-005)

### IMPORTANT-1: VectorizedHamletEnv Integration Cleanup

**Problem**: Environment manually reconstructs affordance DTOs, violating "compile once, execute many"

**Location**: `src/townlet/environment/vectorized_env.py:33-69`

**Current Code**:
```python
def _build_affordance_collection(raw_affordances: tuple[Any, ...]) -> AffordanceConfigCollection:
    # Manual reconstruction from raw DTOs
    # This duplicates logic that should be in compiler
```

**Solution**: Move affordance collection preparation into compiler

**Tasks**:
1. Add `prepared_affordance_collection: AffordanceConfigCollection` to `CompiledUniverse`
2. Build it in compiler Stage 6 or 7 (optimization/emit)
3. Update `RuntimeDTO` to include pre-built collection
4. Remove `_build_affordance_collection()` from VectorizedHamletEnv
5. Update environment to use `runtime.affordance_collection` directly

**Benefits**:
- Eliminates DTO reconstruction at runtime
- Faster environment initialization
- True "compile once, execute many"

**Estimated Effort**: 2-3 hours

---

### IMPORTANT-2: Documentation Gaps

**Missing Documentation**:
1. Migration guide for pre-compiler → compiler transition
2. Troubleshooting guide expansion
3. ObservationSpec contract documentation
4. Cache invalidation behavior clarification

**Tasks**:

#### Task 2.1: Migration Guide
Create `docs/MIGRATION-PRE-COMPILER-TO-COMPILER.md`:
- Schema changes (legacy → modern affordance format)
- How to migrate existing training runs
- Checkpoint compatibility considerations
- Breaking changes summary
- FAQ section

**Estimated Effort**: 1-2 hours

#### Task 2.2: Troubleshooting Guide
Expand `docs/UNIVERSE-COMPILER.md` §6 (Troubleshooting):
- Common compilation errors with solutions
- How to debug cross-validation failures
- Cache invalidation patterns
- Performance optimization tips

**Estimated Effort**: 1 hour

#### Task 2.3: ObservationSpec Documentation
Add to `docs/architecture/COMPILER_ARCHITECTURE.md`:
- ObservationSpec contract details
- How VFS fields map to observation dimensions
- How to extend observation space
- Multi-agent observation handling

**Estimated Effort**: 1 hour

**Total Documentation Effort**: 3-4 hours

---

### IMPORTANT-3: Complete CuesCompiler

**Problem**: CuesCompiler missing domain coverage validation and multi-agent support

**Location**: `src/townlet/universe/cues_compiler.py:143`

**Missing Features**:
1. **Domain coverage validation** - Visual cues should cover full [0.0, 1.0] domain
2. **Multi-agent cue observation spec** - For L5+ multi-agent scenarios

**Tasks**:

#### Task 3.1: Domain Coverage Validation
```python
def _validate_domain_coverage(self, cues: Sequence[VisualCue], errors, formatter):
    """Validate that visual cues cover the full [0.0, 1.0] domain."""
    # Sort by threshold
    # Check for gaps between cues
    # Warn if first threshold > 0.0 or last < 1.0
```

**Estimated Effort**: 1 hour

#### Task 3.2: Multi-Agent Observation Spec
```python
def build_cue_observation_spec(self, multi_agent: bool = False) -> list[ObservationField]:
    """Build observation fields for cues (self + others if multi-agent)."""
    # Self cues (always included)
    # Other agent cues (if multi_agent=True)
```

**Estimated Effort**: 2 hours

**Total CuesCompiler Effort**: 3 hours

---

## Phase 3: Minor Issues (Nice to Have)

### MINOR-1: Provenance ID Optimization

**Problem**: Provenance includes torch/pydantic versions, causing unnecessary cache invalidation

**Location**: `src/townlet/universe/compiler.py:1260-1267`

**Solution**: Split into compiler_provenance (code only) vs full_provenance (includes deps)

**Tasks**:
1. Add `compiler_provenance` field (for cache key)
2. Keep `full_provenance` field (for debugging)
3. Update cache key to use `compiler_provenance`
4. Document in UNIVERSE-COMPILER.md

**Estimated Effort**: 30 minutes

---

### MINOR-2: Document Circular Import Strategy

**Problem**: Lazy import in `CompiledUniverse.create_environment()` suggests circular dependency concern

**Location**: `src/townlet/universe/compiled.py:98-99`

**Solution**: Add comment explaining import strategy

**Tasks**:
1. Add docstring explaining why import is lazy
2. Document the dependency graph in COMPILER_ARCHITECTURE.md
3. Consider refactoring if circular dependency is real

**Estimated Effort**: 30 minutes

---

### MINOR-3: Vectorize Stage 6 Optimization

**Problem**: Nested Python loop for action mask pre-computation

**Location**: `src/townlet/universe/compiler.py:1430-1440`

**Solution**: Use vectorized operations

**Tasks**:
1. Replace nested loop with torch operations
2. Benchmark before/after (should be <1ms difference)
3. Document performance improvement

**Estimated Effort**: 1 hour

---

### MINOR-4: Improve Error Message Actionability

**Problem**: Some error messages could include fix examples

**Location**: Various validation errors throughout compiler

**Solution**: Add fix hints to error messages

**Tasks**:
1. Audit all error messages in compiler
2. Add "Valid options: [...]" or "Example fix: ..." to messages
3. Update error message tests

**Estimated Effort**: 1 hour

---

## Implementation Order

### Day 1 (6-8 hours) - CRITICAL PATH

**Morning (4 hours)**:
1. ✅ Test migration script on L0_0_minimal
2. ✅ Execute migration on all 5 legacy configs
3. ✅ Validate all configs compile successfully
4. ✅ Commit migrated configs

**Afternoon (2-4 hours)**:
5. ✅ Run universe unit tests
6. ✅ Fix any real bugs (not config issues)
7. ✅ Run integration tests
8. ✅ Document results

**Exit Criteria**: Test suite passing, all configs compiling

---

### Day 2 (6-9 hours) - IMPORTANT ISSUES

**Morning (3-4 hours)**:
1. ✅ Clean up VectorizedHamletEnv integration
2. ✅ Move affordance collection to compiler
3. ✅ Test environment initialization

**Afternoon (3-5 hours)**:
4. ✅ Write migration documentation
5. ✅ Expand troubleshooting guide
6. ✅ Complete CuesCompiler domain coverage
7. ✅ Complete CuesCompiler multi-agent support

**Exit Criteria**: Integration clean, docs complete, CuesCompiler feature-complete

---

### Day 3 (Optional - 3 hours) - MINOR ISSUES

**As time permits**:
1. Optimize provenance ID computation
2. Document circular import strategy
3. Vectorize Stage 6 optimization
4. Improve error message actionability

**Exit Criteria**: Code quality improvements complete

---

## Validation Checklist

### Before Merge
- [ ] All 5 legacy configs migrated
- [ ] All config packs compile without errors
- [ ] Universe unit tests: >90% pass rate
- [ ] Integration tests: >90% pass rate
- [ ] CI validation passes
- [ ] No mypy errors in compiler modules
- [ ] No ruff warnings in compiler modules

### Before TASK-005
- [ ] VectorizedHamletEnv integration cleaned up
- [ ] Migration documentation complete
- [ ] Troubleshooting guide expanded
- [ ] CuesCompiler domain coverage implemented
- [ ] CuesCompiler multi-agent support implemented
- [ ] ObservationSpec contract documented

### Optional (Nice to Have)
- [ ] Provenance ID optimization
- [ ] Circular import strategy documented
- [ ] Stage 6 vectorization complete
- [ ] Error messages improved

---

## Risk Assessment

### Low Risk
- Config migration (script exists, well-tested)
- Documentation (no code changes)
- Minor optimizations (low impact)

### Medium Risk
- VectorizedHamletEnv integration cleanup (touches core execution path)
- CuesCompiler completion (new functionality)

### Mitigation
- Test thoroughly after each change
- Run full integration test suite
- Keep changes incremental
- Commit after each successful task

---

## Success Criteria

**CRITICAL (Merge Blocker)**:
- ✅ All configs compile successfully
- ✅ Test suite passing (>90%)
- ✅ CI validation green

**IMPORTANT (Before TASK-005)**:
- ✅ VectorizedHamletEnv integration clean
- ✅ Documentation complete
- ✅ CuesCompiler feature-complete

**MINOR (Nice to Have)**:
- ✅ Code quality improvements
- ✅ Performance optimizations
- ✅ Enhanced error messages

---

## Notes

- **Parallel Work**: Documentation can be done in parallel with code changes
- **Test-Driven**: Run tests after each change to catch regressions early
- **Incremental Commits**: Commit after each completed task
- **Review Checkpoints**: Review progress after Day 1 and Day 2

---

## Appendix: Migration Script Details

**Location**: `scripts/migrate_affordances.py`

**What it does**:
1. Detects legacy format: `required_ticks`, `effects_per_tick`, `completion_bonus`
2. Creates `capabilities: [{type: multi_tick, required_ticks: N}]`
3. Creates `effect_pipeline: {per_tick: [...], on_completion: [...]}`
4. Preserves all other fields unchanged
5. Writes back with proper YAML formatting

**Safe to run**: Idempotent (already-migrated affordances unchanged)

**Example transformation**:
```yaml
# BEFORE (legacy)
- id: bed
  name: Bed
  required_ticks: 5
  effects_per_tick:
    - {meter: energy, amount: 0.2}
  completion_bonus:
    - {meter: energy, amount: 0.4}

# AFTER (modern)
- id: bed
  name: Bed
  capabilities:
    - {type: multi_tick, required_ticks: 5}
  effect_pipeline:
    per_tick:
      - {meter: energy, amount: 0.2}
    on_completion:
      - {meter: energy, amount: 0.4}
```
