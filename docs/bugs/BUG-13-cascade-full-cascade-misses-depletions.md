Title: CascadeEngine.apply_full_cascade omits base depletions (doc mismatch)

Severity: medium
Status: open

Subsystem: environment/cascade
Affected Version/Branch: main

Affected Files:
- `src/townlet/environment/cascade_engine.py:300`

Description:
- Docstring claims `apply_full_cascade` applies base depletions, modulations, and cascades.
- Implementation iterates `execution_order` but never applies base depletions.

Reproduction:
- Invoke `apply_full_cascade` on meters and compare to sequence that includes `apply_base_depletions`.

Expected Behavior:
- Either: apply depletions inside `apply_full_cascade`, or the docstring reflects the true behavior.

Actual Behavior:
- Depletions omitted; downstream code must remember to apply them separately.

Root Cause:
- Refactor drift; base depletions left to external orchestration, but docs not updated.

Proposed Fix (Breaking OK):
- Apply `apply_base_depletions` first inside `apply_full_cascade` (breaking behavior for callers depending on current behavior), or
- Update docs to remove depletions from the described behavior and keep semantics as-is.

Migration Impact:
- If we change behavior, any code that already applies depletions separately must stop calling it twice.

Alternatives Considered:
- Keep current behavior and explicitly rename method to `apply_full_cascade_without_depletions` (too verbose) or document clearly.

Tests:
- Add test covering `apply_full_cascade` vs explicit sequence.

Owner: environment
