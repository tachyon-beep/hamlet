# HAMLET Work Packages - Architecture Analysis Follow-Up

**Source**: Architecture Analysis (2025-11-13)
**Report**: `docs/arch-analysis-2025-11-13-1532/04-final-report.md`
**Status**: Pre-release (zero users, zero downloads)

This document provides triaged work packages addressing the 15 technical concerns identified in the comprehensive architecture analysis. Packages are prioritized by severity and include implementation options where multiple credible approaches exist.

---

## Critical Priority (Blocking Public Release)

### WP-C1: Complete Recording Criteria Integration

**Severity**: Critical
**Effort**: 2 hours
**Blocking Issue**: Advertised feature non-functional (misleading documentation)

**Problem**: RecordingCriteria evaluator fully implemented (4 criterion types: periodic, stage_transitions, performance, stage_boundaries) but RecordingWriter uses simple inline periodic check instead of calling evaluator. Operators can configure criteria but they don't work.

**Current State**: `recording/recorder.py` lines 254-262 inline check, `recording/criteria.py` unused
**Root Cause**: TASK-005A Phase 2 incomplete

**Implementation** (single credible approach):
1. Modify `RecordingWriter._should_record_episode()` to call `criteria.should_record(episode_metadata)`
2. Pass RecordingCriteria instance to RecordingWriter constructor
3. Update all RecordingWriter instantiation sites to construct RecordingCriteria from config
4. Add integration test verifying stage_transitions criterion triggers recording
5. Update `docs/guides/recording.md` with criteria configuration examples

**Validation**:
- Run `pytest tests/test_townlet/integration/test_recording_criteria.py`
- Test stage_transitions criterion captures episodes before curriculum advancement
- Test performance criterion captures top 10% and bottom 10% episodes
- Verify periodic criterion still works (backwards compatibility)

**Impact**: Unlocks advertised functionality for researchers to capture interesting episodes automatically

---

### WP-C2: Deprecate Legacy Brain As Code Paths

**Severity**: Critical
**Effort**: 8 hours
**Blocking Issue**: Dual code paths double testing burden, confuse future developers

**Problem**: Network/optimizer/loss instantiation has two paths (brain_config vs legacy parameters). Legacy path exists when brain_config=None but has no test coverage. Per pre-release freedom principle, zero backwards compatibility needed.

**Current State**: `population/vectorized.py` lines 143-192 dual network init
**Root Cause**: TASK-005 partially complete

**Implementation Options**:

#### Option A: Hard Break (Recommended - aligns with pre-release policy)
**Effort**: 8 hours
**Approach**:
1. Remove legacy network_type parameter from VectorizedPopulation
2. Delete hardcoded hyperparameters (hidden_dim=256, learning_rate=3e-4, gamma=0.99)
3. Add validation: `if brain_config is None: raise ValueError("brain.yaml required")`
4. Update all test fixtures to use brain.yaml (12 test files affected)
5. Delete obsolete TODO(BRAIN_AS_CODE) comments

**Pros**: Clean break, no maintenance burden, enforces single source of truth
**Cons**: Breaks any external scripts (but zero users per pre-release policy)

#### Option B: Deprecation Warning with Timeline (NOT recommended)
**Effort**: 12 hours
**Approach**:
1. Add deprecation warning when brain_config=None
2. Set sunset date (e.g., "legacy path removed in v0.5")
3. Maintain dual paths until sunset
4. Remove at v0.5 release

**Pros**: Gentler transition for hypothetical users
**Cons**: Violates pre-release freedom principle, doubles testing burden, adds 4 hours work

**Decision**: **Option A** - Hard break aligns with CLAUDE.md pre-release policy

---

### WP-C3: Consolidate Cascade Systems

**Severity**: Critical
**Effort**: 16 hours
**Blocking Issue**: Operators confused which cascade system applies, bugs when behavior differs

**Problem**: Both CascadeEngine (legacy config-driven) and MeterDynamics (modern tensor-driven) exist. Some code paths use one, others use the other. Confusing which is active.

**Current State**: `environment/cascade_engine.py` (331 lines), `environment/meter_dynamics.py` (187 lines)
**Root Cause**: MeterDynamics introduced to replace CascadeEngine but migration incomplete

**Implementation Options**:

#### Option A: Complete Migration to MeterDynamics (Recommended)
**Effort**: 16 hours
**Approach**:
1. Audit all CascadeEngine call sites (grep returns 8 files)
2. Verify MeterDynamics has feature parity (especially circular cascade detection)
3. Migrate remaining code paths to `MeterDynamics.apply_depletion_and_cascades()`
4. Delete `environment/cascade_engine.py` (331 lines removed)
5. Update tests: delete `test_cascade_engine.py`, verify `test_meter_dynamics.py` comprehensive
6. Update `docs/config-schemas/cascades.md` clarifying MeterDynamics is active

**Pros**: Modern GPU-native implementation, removes confusion, simplifies environment init
**Cons**: Requires careful validation that MeterDynamics handles all CascadeEngine edge cases

#### Option B: Keep CascadeEngine as Fallback (NOT recommended)
**Effort**: 8 hours
**Approach**:
1. Add config flag: `cascade_mode: "legacy" | "modern"`
2. Environment dispatcher selects CascadeEngine vs MeterDynamics
3. Default to MeterDynamics, allow legacy mode for "compatibility"
4. Document both systems

**Pros**: Safety net if MeterDynamics has bugs
**Cons**: Violates pre-release freedom, doubles testing burden, keeps confusion

**Decision**: **Option A** - Complete migration, delete legacy per pre-release policy

---

## Medium Priority (Maintenance Burden)

### WP-M1: Modularize Large Monolithic Files

**Severity**: Medium
**Effort**: 40 hours
**Impact**: Developer productivity (cognitive load, merge conflicts, IDE performance)

**Problem**: Three files exceed 1000 lines handling multiple responsibilities:
- `compiler.py` (2542 lines, all 7 stages)
- `vectorized_env.py` (1531 lines, init + observation + actions + rewards + tracking)
- `dac_engine.py` (917 lines, 9 extrinsic + 11 shaping + modifiers)

**Implementation Options**:

#### Option A: Extract by Responsibility (Recommended)
**Effort**: 40 hours (16 + 16 + 8)

**Phase 1 - Universe Compiler** (16 hours):
```
universe/
├── compiler.py (orchestrator, ~200 lines)
└── stages/
    ├── parse.py (Stage 1)
    ├── symbol_table.py (Stage 2)
    ├── resolve.py (Stage 3)
    ├── cross_validate.py (Stage 4)
    ├── metadata.py (Stage 5)
    ├── optimize.py (Stage 6)
    └── emit.py (Stage 7)
```

**Phase 2 - Vectorized Environment** (16 hours):
```
environment/
├── vectorized_env.py (orchestrator, ~400 lines)
├── env_observation_builder.py (ObservationBuilder class)
├── env_action_executor.py (ActionExecutor class)
└── env_state_manager.py (StateManager class)
```

**Phase 3 - DAC Engine** (8 hours):
```
environment/
├── dac_engine.py (orchestrator, ~200 lines)
├── dac_modifiers.py (ModifierCompiler)
├── dac_extrinsic.py (ExtrinsicCompiler, 9 strategies)
└── dac_shaping.py (ShapingCompiler, 11 bonuses)
```

**Pros**: Clear separation of concerns, easier navigation, smaller files for code review
**Cons**: More files to manage, need to preserve external API

#### Option B: Extract by Feature (Alternative)
**Effort**: 36 hours
**Approach**: Group by feature domains instead of pipeline stages
- `compiler_yaml_processing.py` (Stages 1-2)
- `compiler_validation.py` (Stages 3-4)
- `compiler_codegen.py` (Stages 5-7)

**Pros**: Fewer files, feature cohesion
**Cons**: Mixes concerns, less obvious boundaries

**Decision**: **Option A** - Responsibility-based extraction clearer for future maintainers

---

### WP-M2: Consolidate POMDP Validation

**Severity**: Medium
**Effort**: 8 hours
**Impact**: Inconsistent error messages, easy to miss validation for new substrate types

**Problem**: POMDP compatibility checks scattered across 4 locations:
- `vectorized_env.py` __init__ lines 252-303
- `substrate_action_validator.py` (separate file)
- `compiler.py` Stage 4 cross-validation
- `observation_builder.py`

**Implementation Options**:

#### Option A: Centralize in Compiler Stage 4 (Recommended)
**Effort**: 8 hours
**Approach**:
1. Create `substrate/pomdp_validator.py` with single `validate_pomdp_compatibility(substrate_config, vision_range, observation_encoding)`
2. Call from UAC Stage 4 only (fail-fast at compile time)
3. Remove validation from Environment, ObservationBuilder (trust compiled artifact)
4. Delete `substrate_action_validator.py` if no other logic remains
5. Consolidate error messages (all say "UAC-POMDP-001: GridND with N≥4 not supported for POMDP")

**Pros**: Fail-fast at compile time, single source of truth, consistent error messages
**Cons**: Environment can't independently validate (relies on compiler)

#### Option B: Centralize in Substrate Classes (Alternative)
**Effort**: 12 hours
**Approach**:
1. Add `validate_pomdp_compatibility()` method to Substrate ABC
2. Each substrate implements validation (Grid2D returns True, GridND checks N<4)
3. Compiler Stage 4 calls `substrate.validate_pomdp_compatibility(vision_range)`
4. Environment init also calls (defensive programming)

**Pros**: Substrate-specific logic lives with substrate, runtime safety check
**Cons**: Duplication (compiler + environment both check), 4 extra hours

**Decision**: **Option A** - Compiler-only validation per compilation pipeline pattern

---

### WP-M3: Implement VFS Phase 2 Expression Evaluation

**Severity**: Medium
**Effort**: 80 hours
**Impact**: Enables custom variables and derived features without code changes

**Problem**: WriteSpec.expression stored as string (no AST parsing/execution). Operators can't define:
- Derived features: `energy_deficit = 1.0 - energy`
- Environmental phenomena: `raining = random() < 0.3 if hour in [6,18] else False`
- Action effects: `energy += 0.1 * fitness` (expression-based increments)

**Current State**: `vfs/schema.py` WriteSpec has string expression field, unused
**Root Cause**: Phase 2 deferred during TASK-002C (Phase 1 prioritized)

**Implementation Options**:

#### Option A: Embedded DSL with AST Parser (Recommended for safety)
**Effort**: 80 hours
**Approach**:
1. Define VFS expression grammar (operators: +,-,*,/,<,>,==,and,or,not, functions: min,max,clamp,random)
2. Implement parser: `vfs/expression_parser.py` using pyparsing or lark
3. AST → bytecode compiler for sandboxed execution
4. Variable resolution: expressions can reference other VFS variables
5. Type checking: ensure energy (float) + agent_count (int) = type error caught at compile time
6. Security: no import, no exec, no file access (whitelist math, random modules only)
7. Integration: ObservationBuilder evaluates expressions at compile time (constant folding) or runtime (dynamic values)

**Pros**: Safe (no arbitrary code execution), type-checked, compile-time optimization
**Cons**: 80 hours effort, new DSL to learn for operators

#### Option B: Python eval() with Restricted Globals (Simpler but risky)
**Effort**: 24 hours
**Approach**:
1. Use Python's `eval(expression, {'__builtins__': {}}, safe_globals)`
2. Provide safe_globals: `{'energy': <variable_value>, 'min': min, 'max': max}`
3. Catch SyntaxError, NameError at compile time
4. Document: "expressions are Python syntax with restricted builtins"

**Pros**: Operators already know Python, 56 hours faster
**Cons**: Security risk (eval exploits), no type checking, hard to optimize

#### Option C: Defer Until Needed (Pragmatic)
**Effort**: 0 hours
**Approach**: L0-L3 curriculum levels don't need custom variables. Implement when L4+ require it.

**Pros**: Zero effort now, implement when requirements clear
**Cons**: Operators stuck with auto-generated variables

**Decision**: **Option C** (defer) then **Option A** when L4+ levels need custom variables. Not blocking for current curriculum.

---

### WP-M4: Refactor Intrinsic Reward Double-Counting Prevention

**Severity**: Medium
**Effort**: 16 hours
**Impact**: Fragile coordination prone to bugs, confusing for new developers

**Problem**: Complex implicit contract between DAC and Population to avoid double-counting intrinsic rewards. DAC includes intrinsic in `env.step()` rewards, so Population stores zeros for `intrinsic_rewards` in replay buffer. Extensive comments throughout codebase warn about this.

**Current State**: `population/vectorized.py` lines 657-673, 706-707
**Root Cause**: DAC retrofitted after Population designed

**Implementation Options**:

#### Option A: Explicit Component Return (Recommended)
**Effort**: 16 hours
**Approach**:
1. DAC.calculate_rewards() returns tuple: `(extrinsic, intrinsic_component, shaping)`
2. Environment.step() sums components: `total_reward = extrinsic + (intrinsic_component * intrinsic_weight) + shaping`
3. Population stores all three components separately in replay buffer
4. Population combines at training time with configurable `intrinsic_weight` (enables ablation studies)
5. Update tests to verify components stored/retrieved correctly

**Pros**: Explicit contract (type-checked), enables intrinsic weight experimentation, no double-counting risk
**Cons**: Changes Environment.step() return signature (breaks checkpoints - but pre-release okay)

#### Option B: DAC Flag for "Already Composed" (Minimal change)
**Effort**: 4 hours
**Approach**:
1. Add flag to DriveAsCodeConfig: `return_composed: bool = True`
2. If True (default), DAC returns composed reward (current behavior)
3. If False, DAC returns components separately
4. Population checks flag, stores components or zeros accordingly

**Pros**: Minimal code changes, backwards compatible
**Cons**: Doesn't solve fragility, just makes implicit contract explicit

**Decision**: **Option A** - Explicit components enable better intrinsic weight experimentation

---

## Low Priority (Documentation/Polish)

### WP-L1: Generalize Recording/Visualization for Custom Configs

**Severity**: Low
**Effort**: 8 hours
**Impact**: Video export breaks for non-standard configs (L0_0_minimal, aspatial, custom meters)

**Problem**: Hardcoded assumptions:
- `video_renderer.py` assumes 8 standard meters with names ["energy", "hygiene", ...]
- `demo/live_inference.py` hardcodes grid size width=8, height=8
- `video_renderer.py` assumes 2D positions (crashes on aspatial substrates)

**Implementation** (single credible approach):
1. Read `meter_name_to_index` from CompiledUniverse metadata (dynamic meter count)
2. Infer grid size from substrate config or `max(affordance_positions)` if variable
3. Conditionally render grid only for spatial substrates (Grid2D/3D/ND, Continuous)
4. Aspatial: render meters-only dashboard (like AspatialView.vue)
5. Add integration test: export video for L0_0_minimal (3×3), aspatial config

**Validation**:
- `pytest tests/test_recording/test_video_export_generalization.py`
- Export L0_0_minimal (3×3 grid, 1 meter) video - should succeed
- Export aspatial config video - should show meters-only dashboard

---

### WP-L2: Centralize Error Handling with Error Code Registry

**Severity**: Low
**Effort**: 16 hours
**Impact**: Usability (operators can't grep error codes, inconsistent UX)

**Problem**: Error codes scattered ("UAC-VAL-001") with no central registry. Validation error messages inconsistent (some warn, some fail for same severity). No `docs/error-codes.md` troubleshooting guide.

**Implementation Options**:

#### Option A: Enum-Based Error Registry (Recommended)
**Effort**: 16 hours
**Approach**:
1. Create `townlet/error_codes.py`:
```python
class ErrorCode(Enum):
    UAC_VAL_001 = ("Grid size exceeds maximum", "Reduce width/height in substrate.yaml")
    UAC_VAL_002 = ("Meter count exceeds limit", "Reduce meters in bars.yaml")
    VFS_VAL_001 = ("Unknown variable reference", "Check variables_reference.yaml")
```
2. Update all raise sites: `raise ValidationError(ErrorCode.UAC_VAL_001.format(width=200))`
3. Generate `docs/error-codes.md` from enum docstrings
4. Standardize: errors raise ValueError/ValidationError, warnings use logger.warning with code

**Pros**: Centralized, searchable, auto-generated docs
**Cons**: 16 hours to update ~60 error sites

#### Option B: Error Code Comments (Minimal)
**Effort**: 2 hours
**Approach**: Add comments at each error site: `# Error code: UAC-VAL-001`

**Pros**: Minimal effort
**Cons**: Doesn't solve inconsistency, no searchable registry

**Decision**: **Option A** if time permits, **Option B** as quick fix

---

### WP-L3: Improve Test Coverage for Untested Features

**Severity**: Low
**Effort**: 24 hours
**Impact**: Bugs in untested code paths go undetected

**Problem**:
- PrioritizedReplayBuffer uses CPU storage (GPU tests missing)
- StructuredQNetwork limited adoption (no brain.yaml support)
- Multi-agent TensorBoard logging untested (HAMLET currently single-agent)

**Implementation Options**:

#### Option A: Add Comprehensive Tests (Recommended if features needed)
**Effort**: 24 hours
**Approach**:
1. **PrioritizedReplayBuffer**: Add `test_prioritized_replay_buffer_gpu.py` verifying CPU↔GPU transfers, TD-error updates, beta annealing
2. **StructuredQNetwork**: Add StructuredConfig to brain.yaml ArchitectureConfig union, add integration test
3. **Multi-agent TensorBoard**: Add `test_tensorboard_multi_agent.py` with 2+ agents, verify agent_id prefixes in metrics

**Pros**: Confidence in features, catch bugs early
**Cons**: 24 hours effort for currently unused features

#### Option B: Delete Untested Features (Pragmatic for pre-release)
**Effort**: 4 hours
**Approach**:
1. Delete PrioritizedReplayBuffer (not used in any curriculum level)
2. Delete StructuredQNetwork (SimpleQNetwork and RecurrentSpatialQNetwork sufficient)
3. Remove multi-agent TensorBoard code (HAMLET is single-agent)
4. Add back when needed for L5+ multi-agent levels

**Pros**: Reduces codebase bloat, eliminates untested code
**Cons**: Features unavailable for future use

**Decision**: **Option B** (delete) per pre-release freedom principle, add back when L5+ requires

---

## Summary: Effort Estimates and Prioritization

### Critical (26 hours total - must complete before v1.0)
- **WP-C1**: Recording Criteria Integration - 2 hours
- **WP-C2**: Deprecate Legacy Brain As Code - 8 hours
- **WP-C3**: Consolidate Cascade Systems - 16 hours

### Medium (120 hours total - complete before public marketing)
- **WP-M1**: Modularize Large Files - 40 hours
- **WP-M2**: Consolidate POMDP Validation - 8 hours
- **WP-M3**: VFS Phase 2 Expression Evaluation - 80 hours (defer until L4+)
- **WP-M4**: Refactor Intrinsic Reward Coordination - 16 hours

### Low (48 hours total - nice-to-have polish)
- **WP-L1**: Generalize Recording/Visualization - 8 hours
- **WP-L2**: Centralize Error Handling - 16 hours
- **WP-L3**: Improve Test Coverage (or delete features) - 24 hours

### Recommended Pre-Release Roadmap

**Sprint 1** (26 hours) - Critical blocking issues:
1. WP-C1: Recording Criteria Integration (2h)
2. WP-C2: Deprecate Legacy Paths (8h)
3. WP-C3: Consolidate Cascades (16h)

**Sprint 2** (24 hours) - High-value medium items:
4. WP-M2: Consolidate POMDP Validation (8h)
5. WP-M4: Refactor Intrinsic Reward (16h)

**Sprint 3** (40 hours) - Developer productivity:
6. WP-M1: Modularize Large Files (40h)

**Deferred**:
7. WP-M3: VFS Phase 2 (wait for L4+ requirements)
8. WP-L1, WP-L2, WP-L3 (post-v1.0 polish)

**Total Critical Path**: 90 hours (Sprints 1-3)
