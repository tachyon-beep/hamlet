Title: RecordingCriteria transition_episodes set grows unbounded

Severity: medium
Status: open

Ticket Type: JANK
Subsystem: recording/criteria
Affected Version/Branch: main

Affected Files:
- `src/townlet/recording/criteria.py:44`
- `docs/WORK-PACKAGES.md:324`
- `docs/arch-analysis-2025-11-13-1532/02-subsystem-catalog.md:564`

Description:
- The `RecordingCriteria` evaluator tracks curriculum stage transitions using a `transition_episodes: set[int]`, but this set is never pruned or bounded.
- For long-running training jobs (e.g., 1M+ episodes), this set can grow to contain one entry per stage transition, which will typically be every time the curriculum progresses or regresses between stages.
- Even though each element is “just an int”, the unbounded growth is unnecessary and surprising, especially in a subsystem meant to be lightweight and low-overhead.

Reproduction:
1) Enable `stage_transitions` recording in the recording configuration and wire `RecordingCriteria` into `RecordingWriter` (as per BUG-31), or instantiate `RecordingCriteria` directly in a test harness.
2) Simulate a long curriculum run that toggles stages (e.g., 1→2→1→2 with many episodes in between), passing `EpisodeMetadata` instances with increasing `episode_id` and changing `curriculum_stage`.
3) Inspect `criteria.transition_episodes`:
   - It continuously accumulates new episode IDs and never evicts old ones, even when they are outside any plausible `record_before`/`record_after` window.

Expected Behavior:
- `transition_episodes` should be treated as a sliding window or bounded cache:
  - Only the last N transitions (where N is derived from `record_before`/`record_after`) need to be retained to support the “before/after transition” windows.
  - Older transition markers should be safely discarded once all episodes that could fall into their windows are past.

Actual Behavior:
- `_mark_transition()` simply does `self.transition_episodes.add(episode_id)` and never removes entries.
- `_check_stage_transitions()` iterates over the entire set for every episode evaluation, which grows in cost and memory usage over time.

Root Cause:
- The initial implementation optimized for clarity of the transition-window logic and assumed short-to-medium-length training runs.
- No eviction policy or max-size bound was added to `transition_episodes`, and there is no awareness of the configured `record_before` / `record_after` windows when deciding which transitions are still relevant.

Risk:
- Memory footprint grows linearly with the number of stage transitions across a training run, which is unnecessary and potentially surprising for operators running long experiments.
- The per-episode evaluation cost for `stage_transitions` increases as the set grows, which could introduce small but unnecessary overhead in the recording subsystem.
- Once BUG-31 integrates `RecordingCriteria` into production, this jank becomes user-visible for long runs.

Proposed Directions:
- Short-term guardrail:
  - Convert `transition_episodes` from an unbounded `set[int]` to a `deque[int]` with `maxlen` derived from the maximum horizon of the before/after windows (e.g., a multiple of `record_before + record_after`).
  - Periodically drop transitions that are certainly out of window for the current episode ID.
- Long-term cleanup:
  - Consider making the transition-window logic purely local to the current stage and recent history (e.g., track only the last transition episode per stage).
  - Document the expected complexity and memory profile of `RecordingCriteria` so future additions (e.g., new criteria) follow the same pattern.

Tests:
- Extend `tests/test_townlet/unit/recording/test_criteria.py`:
  - Add a stress-style unit test that feeds in a large number of transitions and asserts that the size of `transition_episodes` remains bounded and that before/after windows still behave correctly.

Owner: recording/criteria
Links:
- `docs/arch-analysis-2025-11-13-1532/02-subsystem-catalog.md:564`
- `docs/WORK-PACKAGES.md:324` (WP-L1 mentions recording/visualization generalization; same area of concern)
