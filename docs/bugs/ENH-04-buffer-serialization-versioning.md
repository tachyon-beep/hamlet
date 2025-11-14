Title: Version serialized buffer state to support schema evolution

Severity: low
Status: open

Subsystem: training/replay-buffer (all)
Affected Version/Branch: main

Description:
- Serialized payloads lack a `version` field, making schema changes brittle (e.g., reward schema for sequential buffer).

Proposed Enhancement:
- Add `version: int` to serialized dicts, starting at 1.
- `load_from_serialized` branches by version and performs migrations (e.g., infer split rewards from combined if needed).

Migration Impact:
- Backwards compatible if `version` omitted defaults to 0 with current behavior, or we document breaking change pre-release.

Tests:
- Round-trip tests for versioned payloads; migration tests for older shapes.

Owner: training
