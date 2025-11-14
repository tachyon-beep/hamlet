Title: Allow custom variables to opt out of automatic agent exposure

Severity: low
Status: open

Subsystem: universe/compiler + VFS
Affected Version/Branch: main

Description:
- Compiler auto-exposes all variables (standard + custom) to agents.
- Some custom variables may be intended for engine-only use or logging.

Proposed Enhancement:
- Extend `VariableDef` with a flag (e.g., `expose_to_agent: bool = True`).
- Respect it when generating observation exposures; allow engine-only variables to remain hidden while still in the registry.

Migration Impact:
- Defaults keep current behavior; opt-out on a per-variable basis.

Tests:
- Variable with `expose_to_agent=false` should not appear in observation spec.

Owner: compiler
