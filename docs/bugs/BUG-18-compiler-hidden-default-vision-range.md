Title: Compiler injects hidden default vision_range=3 for POMDP local_window

Severity: high
Status: open

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
