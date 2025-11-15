Title: Recording criteria config (beyond periodic) ignored in production

Severity: critical
Status: open

Subsystem: recording/criteria + recorder
Affected Version/Branch: main

Affected Files:
- `src/townlet/recording/recorder.py:222`
- `src/townlet/recording/criteria.py:13`
- `configs/L0_5_dual_resource/training.yaml:87`
- `docs/manual/RECORDING_SYSTEM_SUMMARY.md:120`
- `docs/manual/REPLAY_USAGE.md:41`
- `docs/WORK-PACKAGES.md:324`

Description:
- The recording configuration exposes multiple criteria (`stage_transitions`, `performance`, `stage_boundaries`) and the manual describes them as active, but `RecordingWriter._should_record_episode()` only implements a simple periodic check and ignores all non-periodic criteria and the richer `RecordingCriteria` evaluator.
- As a result, training configs that enable non-periodic criteria silently behave as if only periodic recording were configured, and `recording_reason` values stored in SQLite never reflect the more specific reason strings (`periodic_100`, `top_10pct`, `stage_2_pre_transition`, etc.) described in the docs.

Reproduction:
- Configure recording criteria in `configs/L0_5_dual_resource/training.yaml`:
  - Set `recording.enabled: true`.
  - Under `recording.criteria`, enable `stage_transitions`, `performance`, and `stage_boundaries` with non-default values (as currently shown in that file).
- Run a short training session via `scripts/run_demo.py` until several episodes complete with varying rewards and at least one curriculum stage transition.
- Inspect `episode_recordings` rows using `DemoDatabase.list_recordings()` or directly querying SQLite:
  - All `recording_reason` values will be the constant `"periodic"`.
  - Turning off the periodic criterion while leaving others enabled will result in **no** episodes recorded, even when stage transitions or performance criteria should match.

Expected Behavior:
- Recording criteria should match the documented semantics:
  - Periodic, stage transitions, performance percentile, and stage boundary criteria should each be able to trigger recording.
  - OR logic: any enabled criterion that matches should cause the episode to be recorded.
  - `recording_reason` should reflect the specific criterion that triggered recording (e.g., `periodic_50`, `top_10.0pct`, `before_transition_105`, `stage_2_first_3`), so operators can filter replays by reason as described in `REPLAY_USAGE.md`.

Actual Behavior:
- `RecordingWriter._should_record_episode()` only reads `config["criteria"]["periodic"]` and applies a simple modulo test. All other criteria in the recording config are effectively no-ops.
- `recording_reason` is hard-coded to `"periodic"` in `_write_episode()`, regardless of which criterion conceptually triggered the recording, and regardless of the richer reason strings produced by `RecordingCriteria`.
- Docs and example payloads in `RECORDING_SYSTEM_SUMMARY.md` and `REPLAY_USAGE.md` show `recording_reason: "periodic_100"` and other reason codes that never occur when using the current production recording path.

Root Cause:
- `RecordingCriteria` is fully implemented in `criteria.py` (periodic, stage_transitions, performance, stage_boundaries with OR logic and stateful tracking), but it is never instantiated or used by `RecordingWriter`.
- `RecordingWriter._should_record_episode()` implements an inline periodic-only check instead of delegating to `RecordingCriteria.should_record()`.
- The training config schema for recording criteria (`training.yaml`) was updated to use names like `lookback`, `lookahead`, `top_percentile`, `history_window`, `record_first_n`, `record_last_n`, but there is no DTO layer mapping these names onto the `RecordingCriteria` fields (`record_before`, `record_after`, `top_percent`, `window`, `first_n`, `last_n`), so even if the evaluator were wired up, the raw config would not match its expected structure.

Proposed Fix (Breaking OK):
- Introduce a small DTO or adapter layer for recording config that:
  - Normalizes training YAML keys (`lookback`, `lookahead`, `top_percentile`, `bottom_percentile`, `history_window`, `record_first_n`, `record_last_n`) into the names used by `RecordingCriteria` (`record_before`, `record_after`, `top_percent`, `bottom_percent`, `window`, `first_n`, `last_n`).
  - Optionally validates that at least one criterion is enabled when `recording.enabled` is true.
- In `EpisodeRecorder.__init__`, construct a `RecordingCriteria` instance and pass it into `RecordingWriter` (or have `RecordingWriter` create it from the normalized config).
- Replace `RecordingWriter._should_record_episode()` with a thin wrapper around the evaluator:
  - Call `self.criteria.should_record(metadata)` and use the returned `(should_record, reason)` tuple.
  - Propagate the `reason` string into `_write_episode()` so that `DemoDatabase.insert_recording()` receives the correct `recording_reason`.
- Update `tests/test_townlet/unit/recording/test_recorder.py` and `test_criteria.py` to cover the integrated path, ensuring that enabling each criterion in config results in recordings with the expected `recording_reason` and that disabling all criteria results in no recordings.

Migration Impact:
- Existing training configs:
  - `configs/L0_5_dual_resource/training.yaml` already uses the new field names (lookback/top_percentile/etc.), which will become active once the adapter layer is added.
  - Other configs that only configure `periodic` will continue to behave the same, except that `recording_reason` will be more specific (e.g., `periodic_50` instead of `periodic`).
- Operators who relied on constant `"periodic"` in downstream tooling will need to update their filters to handle the more precise codes (e.g., treat `recording_reason LIKE "periodic_%"` as the periodic bucket).
- This change makes previously dormant config fields live; documentation should explicitly note that pre-v1.0 recordings may have only `"periodic"` reasons and incomplete coverage.

Alternatives Considered:
- Keep periodic-only behavior and treat RecordingCriteria as an experimental, unused component:
  - Rejected because it contradicts both the manual and `WORK-PACKAGES.md` (WP-C1: Recording Criteria Integration).
- Implement only a subset of criteria (e.g., periodic + performance) and drop stage-based ones:
  - Rejected; stage-based criteria are important for pedagogical story-telling and are already documented.

Tests:
- Extend unit tests:
  - `tests/test_townlet/unit/recording/test_criteria.py` to remain the source of truth for evaluator behavior.
  - `tests/test_townlet/unit/recording/test_recorder.py` to exercise the integrated path with a small in-memory `RecordingCriteria` and a mock database, asserting both that recordings are written and that `recording_reason` matches the evaluator’s reason string.
- Add an integration-style test:
  - Under `tests/test_townlet/integration/test_recording_recorder.py`, configure recording criteria via a small YAML fragment (or dict), run a few synthetic episodes, and assert that the database contains the expected mix of reasons (periodic, performance, stage-boundary) when conditions are met.

Owner: recording subsystem
Links:
- `docs/WORK-PACKAGES.md:324` (WP-C1: Recording Criteria Integration)
- `docs/manual/RECORDING_SYSTEM_SUMMARY.md`
- `docs/manual/REPLAY_USAGE.md`
- `docs/arch-analysis-2025-11-13-1532/04-final-report.md`
