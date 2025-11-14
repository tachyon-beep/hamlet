Title: Add `contains()` and `get_optional()` to VariableRegistry

Severity: low
Status: open

Subsystem: VFS/registry
Affected Version/Branch: main

Description:
- Callers (env) currently peek into `_definitions` to check presence and avoid errors, which is a private attribute.

Proposed Enhancement:
- Add `contains(var_id: str) -> bool` to check presence via public API.
- Add `get_optional(var_id: str, reader: str) -> Tensor | None` to retrieve or return None if absent.

Migration Impact:
- Replace private attribute usage in env with public methods; improves encapsulation.

Tests:
- Simple presence checks and optional gets.

Owner: VFS
