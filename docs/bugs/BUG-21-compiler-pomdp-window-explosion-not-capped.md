Title: Compiler does not cap POMDP local_window size for 3D substrates

Severity: medium
Status: open

Subsystem: universe/compiler
Affected Version/Branch: main

Affected Files:
- `src/townlet/universe/compiler.py:738` (3D local_window dims)

Description:
- For 3D POMDP, the compiler computes local_window dims as `(2r+1)^3` without enforcing a practical limit.
- Runtime (`VectorizedHamletEnv`) explicitly rejects 3D windows > 5×5×5 (=125 cells) to avoid memory explosion.
- This can lead to a compiled observation spec that the runtime refuses to use.

Reproduction:
1) Configure Grid3D with `partial_observability: true` and `vision_range: 4` (9×9×9 window = 729 cells).
2) Compile → spec includes huge window; runtime env raises error on reset.

Expected Behavior:
- Compiler should enforce the same cap and fail fast with a clear error.

Actual Behavior:
- Mismatch between compile-time and runtime validation.

Root Cause:
- Missing compiler-side constraint mirroring.

Proposed Fix (Breaking OK):
- Add a compile-time check: if 3D POMDP and `(2r+1)^3 > 125`, raise a compilation error with remediation guidance.

Migration Impact:
- Packs must reduce 3D vision_range or disable POMDP; aligns compile/run-time behavior.

Alternatives Considered:
- Auto-truncate; violates no-defaults and hides intent.

Tests:
- Add compile-time failure test for oversized 3D local windows.

Owner: compiler
