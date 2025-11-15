Title: Compiler injects hidden default vision_range=3 for POMDP local_window

Severity: high
Status: RESOLVED
Resolved Date: 2025-11-15

Subsystem: universe/compiler
Affected Version/Branch: main

Affected Files:
- `src/townlet/universe/compiler.py:726` (auto-generate local_window dims)

Description:
- When `environment.partial_observability` is true, the compiler computes the `local_window` variable with `vision_range = raw_configs.environment.vision_range or 3`.
- This silently defaults `vision_range` to 3 if the config omits it or sets it to 0/None, violating the no-defaults principle and creating potential mismatch with runtime (which enforces explicit settings).

Reproduction:
1) Omit `vision_range` with `partial_observability: true` in a pack.
2) Compile → observation spec includes a local_window sized for r=3.
3) Runtime env (`VectorizedHamletEnv`) requires explicit `vision_range` (see validation) and can error or be inconsistent.

Expected Behavior:
- If POMDP is enabled, `vision_range` must be explicitly provided; otherwise the compiler should fail with a clear error.

Actual Behavior:
- Compiler silently assumes `vision_range=3`.

Root Cause:
- Use of `or 3` when constructing dims for `local_window`.

Proposed Fix (Breaking OK):
- Require `vision_range` when `partial_observability` is true; raise a compilation error if missing or invalid.
- Remove the implicit fallback `or 3`.

Migration Impact:
- Packs must specify `vision_range` explicitly for POMDP.

Alternatives Considered:
- Keep default; conflicts with project policy (no-defaults) and drifts from runtime validation.

Tests:
- Add compile-time failure test for missing `vision_range` in POMDP.

Owner: compiler

---

## RESOLUTION

**Fixed By**: Systematic debugging process using superpowers:systematic-debugging skill
**Resolution Date**: 2025-11-15

### Implementation Summary

**Phase 1 (Root Cause Investigation)**: Identified that compiler line 390 used `vision_range = raw_configs.environment.vision_range or 3` which silently changed explicit 0 → 3, violating no-defaults principle. Config schema correctly REQUIRES vision_range (Pydantic enforces this), but compiler had implicit fallback that could override operator's explicit choices.

**Phase 2 (Pattern Analysis)**: Found that ALL 10 config packs explicitly specify vision_range (including full obs configs where it's not used). Compared to BUG-30 fix pattern (operating_hours required). Confirmed vision_range=0 is VALID (ge=0 constraint, means 1×1 window) and should be respected, not silently changed to 3.

**Phase 3 (Hypothesis)**: Confirmed that removing `or 3` fallback enforces no-defaults principle without breaking anything, since config schema already requires vision_range and all configs provide explicit values.

**Phase 4 (Implementation)**:

1. **Test Creation** (`tests/test_townlet/unit/universe/test_vision_range_no_defaults.py`):
   - Created failing test `test_compiler_respects_vision_range_zero` demonstrating bug
   - Test verified compiler changed 0 → 3 (1×1 window became 7×7 = 49 dims)
   - Added baseline test `test_compiler_respects_vision_range_two` for regression

2. **Compiler Fix** (`src/townlet/universe/compiler.py:390`):
   - Changed from: `vision_range = raw_configs.environment.vision_range or 3`
   - To: `vision_range = raw_configs.environment.vision_range`
   - Added comment: `# no hidden defaults - BUG-18 fix`

3. **Test Fixture Cleanup** (`tests/test_townlet/unit/universe/test_symbol_table.py:91`):
   - Added missing `operating_hours=[0, 24]` to test affordance (BUG-30 cleanup)

### Test Results
- Vision range tests: 2/2 PASSED ✓
- Universe compiler tests: 262/262 PASSED ✓
- L2 verification: vision_range=2 → 25 dims (5×5 window) ✓
- Total: 262/262 PASSED ✓

### Files Modified
1. `src/townlet/universe/compiler.py` - Removed `or 3` fallback (line 390)
2. `tests/test_townlet/unit/universe/test_vision_range_no_defaults.py` - New test file (2 tests)
3. `tests/test_townlet/unit/universe/test_symbol_table.py` - BUG-30 cleanup (1 fixture)

### Migration Notes
- All 10 config packs already have explicit vision_range values
- No migration required - all existing configs already compliant
- Breaking change appropriate for pre-release project (zero users)
- Aligns with CLAUDE.md no-defaults principle

### Impact
- ✅ Eliminates hidden default that violated no-defaults principle
- ✅ Respects operator's explicit vision_range=0 choice (1×1 window for extreme POMDP)
- ✅ Config schema already enforces required vision_range (Pydantic validation)
- ✅ All implicit default logic removed from compiler
- ✅ Consistent with BUG-30 fix pattern (operating_hours)
