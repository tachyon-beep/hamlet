Title: Missing observation_activity in cached universes silently yields empty activity/masks

Severity: low
Status: open

Ticket Type: JANK
Subsystem: universe/compiled + runtime (observation activity)
Affected Version/Branch: main

Affected Files:
- `src/townlet/universe/compiled.py:232`

Description:
- `CompiledUniverse.save_to_cache` persists `observation_activity` into the MessagePack payload.
- `CompiledUniverse.load_from_cache` reconstructs `ObservationActivity` when present; if it is missing, it fabricates a default:
  - `active_mask=()`
  - `group_slices={}`
  - `active_field_uuids=()`
- This silent fallback is intended for backward compatibility with older caches.

Reproduction:
1) Produce a compiled universe and manually remove `observation_activity` from the cache payload (or simulate an older format).
2) Load it via `CompiledUniverse.load_from_cache`.
3) Observe that:
   - `observation_activity` exists but has no active fields or group slices.
   - Structured consumers (group-aware networks, RND masks) see “no grouping” without any warning.

Expected Behavior:
- When `observation_activity` is missing but the current compiler would generate non-empty group metadata:
  - Either the cache should be considered too old and force recompilation, or
  - A clear warning should indicate that grouping information is unavailable and flat behavior is being used.

Actual Behavior:
- The fallback to an empty `ObservationActivity` is entirely silent.
- The environment and training loop still function with flat observations, so tests pass, but any grouping-dependent logic quietly degrades.

Root Cause:
- Backwards-compatibility convenience in `load_from_cache` opted for a non-breaking default rather than a compatibility check.
- There is no schema versioning attached to `observation_activity` to detect when the cached format lags behind the current compiler’s capabilities.

Risk:
- Subtle bugs in observation grouping, structured encoders, or RND active masks can be introduced by stale caches.
- Operators may believe they are using structured observation groups when they are actually using a flat layout.

Proposed Directions:
- Add a minimal compatibility guard:
  - When `observation_activity` is missing but `observation_spec` clearly implies grouped semantics, log a warning or raise a “cache too old; recompile” error.
- Optionally attach a schema or compiler version stamp to `observation_activity` in metadata to make future format changes explicit and testable.

Tests:
- Unit: simulate an older cache without `observation_activity` and assert that a warning or error is raised (depending on chosen policy).
- Regression: ensure current compiler always writes `observation_activity` and that round-trips preserve grouping info.

Owner: compiler/runtime
Links:
- `src/townlet/universe/compiled.py:200–260`
