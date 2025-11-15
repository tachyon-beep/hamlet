Title: Add bulk get/set and read-only view accessors to VariableRegistry

Severity: low
Status: open

Subsystem: VFS/registry
Affected Version/Branch: main

Description:
- Repeated `get()` clones and individual `set()` calls increase overhead in hot paths.

Proposed Enhancement:
- Add `get_many(var_ids: list[str], reader: str) -> dict[str, Tensor]` to fetch multiple variables efficiently.
- Add `set_many(updates: dict[str, Tensor], writer: str)` to apply multiple updates.
- Add `get_view(var_id: str, reader: str)` returning a read-only tensor view (document risks), restricted to privileged readers.

Migration Impact:
- Optional; hot-path users (env/engines) can opt in for perf.

Tests:
- Validate shapes/dtypes and access control across bulk ops.

Owner: VFS
