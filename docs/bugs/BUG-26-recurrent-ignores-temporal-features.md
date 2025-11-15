Title: RecurrentSpatialQNetwork ignores temporal features in observation tail

Severity: low
Status: open

Subsystem: agent/networks
Affected Version/Branch: main

Affected Files:
- `src/townlet/agent/networks.py:190` (commented ignore of temporal tail)

Description:
- The recurrent network splits observation into grid/position/meters/affordance and ignores any remaining temporal features (e.g., time_sin, time_cos, progress).
- With temporal mechanics enabled, this discards potentially useful information for policies.

Reproduction:
- Enable temporal mechanics; observation includes time features; forward does not consume them.

Expected Behavior:
- Either include a temporal encoder and concatenate, or explicitly shape the observation without temporal features when using this network.

Actual Behavior:
- Temporal features are silently ignored.

Root Cause:
- Initial focus on spatial/meter signals; temporal left for future work.

Proposed Fix (Breaking OK):
- Add optional temporal encoder (e.g., 3→16 MLP) controlled by `enable_temporal_features` and include in LSTM input dim.

Migration Impact:
- Changes input schema; require config and env alignment.

Alternatives Considered:
- Rely on DAC shaping for time; still valuable to expose raw time to network.

Tests:
- Unit test that forward consumes temporal dims when enabled.

Owner: agent
