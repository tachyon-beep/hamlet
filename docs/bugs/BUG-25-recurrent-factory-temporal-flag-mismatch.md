Title: NetworkFactory.build_recurrent hardcodes enable_temporal_features=False

Severity: medium
Status: open

Subsystem: agent/network-factory
Affected Version/Branch: main

Affected Files:
- `src/townlet/agent/network_factory.py:73`

Description:
- When building the recurrent network from BrainConfig, the factory always passes `enable_temporal_features=False`.
- The environment may include temporal features in the observation (time_sin, time_cos, progress), and code elsewhere passes the env flag when constructing networks directly.

Reproduction:
- Use brain_config path; env with temporal features; network constructed via factory will ignore them.

Expected Behavior:
- Factory should accept `enable_temporal_features: bool` or infer from env metadata and pass through.

Actual Behavior:
- Flag is hardcoded to False.

Root Cause:
- Phase 2 shortcut left in code.

Proposed Fix (Breaking OK):
- Add `enable_temporal_features` parameter to `build_recurrent` and thread it from calling sites; default to explicit value (no default in configs per policy).

Migration Impact:
- Callers must supply the flag; improves parity with direct construction path.

Alternatives Considered:
- Keep ignoring temporal features; underutilizes available signals.

Tests:
- Construct with `enable_temporal_features=True` and confirm input split accounts for trailing temporal dims.

Owner: agent
