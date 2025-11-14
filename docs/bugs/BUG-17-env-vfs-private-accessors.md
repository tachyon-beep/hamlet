Title: Vectorized env uses VFS private attributes for presence checks

Severity: low
Status: open

Subsystem: environment/vectorized + VFS
Affected Version/Branch: main

Affected Files:
- `src/townlet/environment/vectorized_env.py:1093` (velocity vars)
- `src/townlet/environment/vectorized_env.py:698` (grid_encoding var)

Description:
- The environment probes `self.vfs_registry._definitions` (private attribute) to check if variables exist before write.
- This couples the env to internal VFS structures and bypasses any future invariants the registry might impose.

Reproduction:
- Read code; private member access is brittle by design.

Expected Behavior:
- Use a public `has(var_name)` or similar method exposed by the registry to check variable presence.

Actual Behavior:
- Access to `_definitions` directly.

Root Cause:
- Lack of an explicit public API in the registry for presence checks, or convenience shortcut.

Proposed Fix (Breaking OK):
- Add `contains(name: str) -> bool` to `VariableRegistry` and replace `_definitions` access in env.

Migration Impact:
- None for users; internal refactor only. If the registry API changes, update other locations accordingly.

Alternatives Considered:
- Try/except around `set`; heavier and less clear.

Tests:
- None needed beyond ensuring env writes do not error when variables absent.

Owner: env+vfs
