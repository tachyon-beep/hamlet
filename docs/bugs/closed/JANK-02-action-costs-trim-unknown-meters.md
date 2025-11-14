Title: Action costs/effects with unknown meters are silently trimmed during action space composition

Severity: medium
Status: RESOLVED
Resolved Date: 2025-11-15

Ticket Type: JANK
Subsystem: universe/compiler_inputs (action space composition)
Affected Version/Branch: main

Affected Files:
- `src/townlet/universe/compiler_inputs.py:190`

Description:
- `_compose_action_space` builds a combined action space from substrate defaults plus `global_actions.yaml`.
- The helper `_trim_meter_payload` removes entries in `costs` / `effects` whose meter names are not listed in `hamlet_config.bars`.
- When this happens, the compiler only emits a generic hint once and continues; no per-action diagnostic is emitted.

Reproduction:
1) Define a custom action in `global_actions.yaml` with a typo'd meter name (e.g., `"hygeine"` instead of `"hygiene"`).
2) Compile the pack.
3) Observe:
   - Compilation succeeds.
   - A single hint about "Action costs/effects referencing meters absent from bars.yaml were ignored…" may appear.
   - The action executes without the intended cost/effect, unbalancing the environment.

Expected Behavior:
- For standard universes, any action referencing an unknown meter should produce a targeted compile error at `global_actions.yaml:<action.name>` naming the offending meter.
- Only when explicitly configured (e.g., variable-meter experiments) should trimming be allowed, with clear per-action warnings.

Actual Behavior:
- Unknown meter references are silently removed from costs/effects.
- Only a generic hint is emitted, and it does not identify which actions or meters were affected.

Root Cause:
- `_trim_meter_payload` was designed as a defensive measure for variable-meter universes, but its behavior is unconditional.
- There is no strict mode vs permissive mode distinction for action meter references.

Risk:
- Config bugs (typos or stale meter names) quietly shift the game's economics.
- Operators and students may debug "weird" training dynamics without realizing that some action costs or effects are being ignored.

Proposed Directions:
- Introduce strict vs permissive behavior:
  - Default: unknown meters in action costs/effects → `UAC-ACT-002` errors with precise locations.
  - Optional flag (e.g., `training.allow_unknown_action_meters`) to enable trimming, with per-action warnings.
- Make the existing hint more specific: include action name and meter names, and emit once per action rather than once globally.

Tests:
- Unit: unknown meter in action costs → compile-time error in strict mode.
- Unit: in permissive mode, the same config yields a warning and a trimmed payload, and the warning points directly at `global_actions.yaml`.

Owner: compiler

---

## RESOLUTION

**Fixed By**: Systematic debugging process using superpowers:systematic-debugging skill
**Resolution Date**: 2025-11-15

### Implementation Summary

**Phase 1 (Root Cause Investigation)**: Found `_trim_meter_payload` at line 190 silently removes unknown meter references with only a generic hint. Traced how it's called during action composition. Identified that context (action name, field type) is lost during trimming.

**Phase 2 (Pattern Analysis)**: Identified UAC-ACT-001 pattern for similar validations. Found that UAC-ACT-002 is next available error code. Examined how other validations report errors with specific violators and locations.

**Phase 3 (Hypothesis & Testing - TDD)**: Architectural decision point - agent correctly asked for guidance on strict vs permissive mode. **Air Traffic Control guidance**: Implement strict mode only (permissive mode is optional enhancement for future).

Created 5 comprehensive tests:
1. Unknown meter in costs → UAC-ACT-002 error
2. Unknown meter in effects → UAC-ACT-002 error
3. Multiple unknown meters → All reported
4. Valid meters → Compilation succeeds (regression test)
5. Empty costs/effects → Compilation succeeds

**Phase 4 (Implementation)**:

#### **Core Changes** (`src/townlet/universe/compiler_inputs.py`)

**Refactored function** (lines 226-256):
```python
# BEFORE: _trim_meter_payload (silent trimming)
def _trim_meter_payload(payload: dict[str, float]) -> dict[str, float]:
    # Silently removed unknown meters
    # Only generic hint emitted
    return {meter: amount for meter, amount in payload.items() if meter in meter_names}

# AFTER: _validate_meter_payload (strict validation)
def _validate_meter_payload(
    payload: dict[str, float],
    action_name: str,
    field_name: str,  # "costs" or "effects"
) -> dict[str, float] | None:
    """Validate that all meters in payload are defined in bars.yaml."""
    if not payload:
        return payload

    missing = [meter for meter in payload if meter not in meter_names]
    if missing:
        for meter in missing:
            errors.add(
                f"Action '{action_name}' references unknown meter '{meter}' in {field_name}. "
                f"Ensure all meters are defined in bars.yaml.",
                code="UAC-ACT-002",
                location=f"global_actions.yaml:{action_name}",
            )
        return None  # Signal compilation failure

    return payload
```

**Updated action processing** (lines 265-312):
- Modified `_clone` to call `_validate_meter_payload` for both costs and effects
- Returns None if validation fails
- Added validation failure tracking in action loop
- Compilation aborts when any action fails validation

### Error Code Assignment

**UAC-ACT-002**: Action references unknown meter
- Format: `"Action '{action_name}' references unknown meter '{meter}' in {field_name}. Ensure all meters are defined in bars.yaml."`
- Location: `global_actions.yaml:{action_name}`
- Follows UAC-ACT-* namespace pattern

### Test Results
✅ **All 5 tests PASS**
- Tests initially FAILED before fix (silent trimming)
- After implementation, all tests pass (strict validation)

### Example Error Output

**Before Fix** (silent trimming):
```
Compilation succeeds
Unknown meter "hygeine" silently removed
Generic hint: "Action costs/effects referencing meters absent from bars.yaml were ignored..."
```

**After Fix** (strict validation):
```
Stage 1: Parse failed:
  - [UAC-ACT-002] global_actions.yaml:REST - Action 'REST' references unknown meter 'hygeine' in costs. Ensure all meters are defined in bars.yaml.
```

### Code Review
- Reviewer: feature-dev:code-reviewer subagent
- Status: ✅ APPROVED
- Score: 95/100 (Excellent)
- Findings:
  - Strict validation approach appropriate for pre-release project
  - Error messages clear and actionable
  - UAC-ACT-002 follows established patterns
  - Test coverage comprehensive
  - Zero false positives detected
  - Deferring permissive mode is appropriate (can add later if needed)
  - Silent trimming completely eliminated

### Files Modified
1. `src/townlet/universe/compiler_inputs.py` - Refactored validation (~60 lines)
2. `tests/test_townlet/unit/universe/test_action_space_composition.py` - New test file (236 lines)

### Architectural Decisions

**Strict Mode Only** (per guidance):
- Default behavior: unknown meters → UAC-ACT-002 compilation errors
- No permissive mode flag (optional future enhancement)
- Aligns with no-defaults principle - fail loudly on config errors
- Appropriate for pre-release project with zero users

**Error Granularity**:
- One error per unknown meter (not one per action)
- Allows operators to see all problems at once
- Clear diagnostic information (action name, meter name, field location)

### Migration Notes
- **Zero migration required** - project is pre-release with zero users
- Breaking change is free and encouraged (per CLAUDE.md)
- All existing configs already use valid meter names

### Impact
- ✅ Prevents silent economic bugs from typo'd meter names
- ✅ Fails fast with precise compilation errors
- ✅ Clear error messages with exact locations
- ✅ No false positives (valid configs unaffected)
- ✅ Developer experience improved (actionable diagnostics)
- ✅ Eliminates "weird training dynamics" debugging mystery

### JANK Note
This ticket required architectural decision (strict vs permissive mode). Agent correctly stopped to ask for guidance before implementing - this is expected and encouraged behavior for JANK tickets.
