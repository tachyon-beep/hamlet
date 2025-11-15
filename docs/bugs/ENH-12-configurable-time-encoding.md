Title: Make time-of-day encoding configurable (scalar vs sin/cos)

Severity: low
Status: open

Subsystem: environment/vectorized + VFS
Affected Version/Branch: main

Description:
- Current env exposes time using two floats: `time_sin`, `time_cos` (24h cycle).
- Some consumers expect a single scalar (e.g., normalized hour in [0, 1]).

Proposed Enhancement:
- Add a config flag in universe/env (e.g., `temporal_encoding: "sin_cos" | "scalar"`).
- When `scalar`, expose `time_norm = hour / 24.0` and remove `time_sin/time_cos` from observation; adjust VFS spec accordingly.
- Default remains `sin_cos` to avoid wrap-around discontinuity at midnight.

Migration Impact:
- Changing to `scalar` reduces temporal dims by 1 (from 2 to 1); update tests and networks expecting 4 temporal channels.

Tests:
- Adjust `calculate_expected_observation_dim` when in scalar mode.

Owner: environment
