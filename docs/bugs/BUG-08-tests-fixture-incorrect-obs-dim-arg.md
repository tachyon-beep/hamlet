Title: Test fixture passes unsupported `obs_dim` to ReplayBuffer constructor

Severity: medium
Status: open

Subsystem: tests-fixtures
Affected Version/Branch: main

Affected Files:
- `tests/test_townlet/_fixtures/training.py:36`

Description:
- `ReplayBuffer` does not accept an `obs_dim` parameter, but the fixture passes it.
- This fixture would throw if invoked; likely dead code but confusing.

Reproduction:
- Import the fixture or call it directly; Python will error on unexpected keyword argument.

Expected Behavior:
- Fixture matches the buffer’s constructor, or the class supports early preallocation via `obs_dim`.

Actual Behavior:
- Mismatch between fixture and API.

Root Cause:
- API drift or leftover from an earlier design.

Proposed Fix (Breaking OK):
- Either: remove `obs_dim=...` from fixture, or
- Add optional `obs_dim` to `ReplayBuffer.__init__` to preallocate (and document), updating code accordingly.

Migration Impact:
- If modifying ReplayBuffer, update docs and tests to reflect preallocation option.

Alternatives Considered:
- Keep as-is; leaves confusing/unusable fixture.

Tests:
- Ensure fixture composes and the buffer can be used immediately in tests.

Owner: tests
Links:
- N/A
