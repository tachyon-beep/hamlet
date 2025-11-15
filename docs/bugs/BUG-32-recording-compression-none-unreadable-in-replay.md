Title: Recording files written with compression=none cannot be replayed

Severity: medium
Status: open

Subsystem: recording/replay
Affected Version/Branch: main

Affected Files:
- `src/townlet/recording/recorder.py:265`
- `src/townlet/recording/replay.py:49`
- `tests/test_townlet/unit/recording/test_recorder.py:306`

Description:
- The recording pipeline supports a `compression` setting in `EpisodeRecorder`/`RecordingWriter` that can be set to `"none"` (no compression), but `ReplayManager` always assumes LZ4 compression when loading episodes.
- When `compression` is set to `"none"`, `RecordingWriter` writes raw msgpack bytes to disk using a `.msgpack.lz4` filename, and `ReplayManager.load_episode()` attempts to decompress those bytes with `lz4.frame.decompress`, which fails with a `LZ4FrameError`.
- This makes recordings produced with `compression: "none"` unreadable by both the replay manager and downstream tools (e.g., live inference replay and video export).

Reproduction:
- Configure recording with no compression (e.g., in `training.yaml` under `recording`):
  - `compression: "none"`.
- Run a short training session with recording enabled until at least one episode is recorded.
- Use `DemoDatabase.list_recordings()` to find the corresponding recording file path, or inspect the recordings directory directly.
- Attempt to load the episode via `ReplayManager`:
  - Instantiate `DemoDatabase` and `ReplayManager(recordings_base_dir=checkpoint_root)`.
  - Call `load_episode(episode_id)` for the recorded episode.
- Observed behavior:
  - `lz4.frame.decompress` raises an exception because the file is plain msgpack data without an LZ4 frame header.
  - `ReplayManager.load_episode()` logs an error and returns `False`, making the recording unusable.

Expected Behavior:
- Episodes recorded with `compression: "none"` should be loadable via `ReplayManager`, or the system should explicitly prohibit/validate unsupported compression modes before writing files.
- The on-disk representation should be self-consistent:
  - Either the file extension and metadata clearly indicate compression type, or the loader should auto-detect whether LZ4 compression is present and fall back to raw msgpack when appropriate.

Actual Behavior:
- `RecordingWriter._write_episode()`:
  - Writes raw `msgpack.packb` bytes when `compression != "lz4"` but still uses a `.msgpack.lz4` filename and records `compressed_size` accordingly.
- `ReplayManager.load_episode()`:
  - Unconditionally calls `lz4.frame.decompress(compressed_data)` and then msgpack-unpacks the result.
  - For raw msgpack files, `lz4.frame.decompress` fails and no fallback path exists, so the episode cannot be loaded at all.

Root Cause:
- The compression configuration was implemented only on the write side (`RecordingWriter._write_episode`) without updating the reader (`ReplayManager`) to support non-LZ4 formats or to detect compression type.
- The filename extension `.msgpack.lz4` is used regardless of the `compression` setting, so the loader has no signal that the file actually contains uncompressed msgpack data.

Proposed Fix (Breaking OK):
- Introduce explicit compression metadata and align file naming/reading with it:
  - Option A (preferred): Add a small header in `episode_data` (e.g., `"compression": "lz4" | "none"`) and keep the filename as `.msgpack` or `.msgpack.lz4` accordingly.
    - In `RecordingWriter._write_episode()`, set the header and choose the extension based on `compression`.
    - In `ReplayManager.load_episode()`, detect compression from either the header or filename:
      - If `compression == "lz4"` (or extension ends with `.lz4`), call `lz4.frame.decompress`.
      - If `compression == "none"` (or extension is `.msgpack`), skip decompression and feed raw bytes into `msgpack.unpackb`.
  - Option B (minimal): Keep the existing filename but have `ReplayManager.load_episode()` attempt `lz4.frame.decompress` first and, on `LZ4FrameError`, retry by passing the raw bytes to `msgpack.unpackb`.
- Add validation:
  - When constructing `EpisodeRecorder`, validate that `compression` is one of `"lz4"` or `"none"`, and warn or error on unknown values.

Migration Impact:
- Existing recordings:
  - All current production configs use the default `"lz4"` compression, so the on-disk format for existing runs remains valid.
  - Any experimental runs that used `"none"` are currently unreadable; after the fix, they may become readable if the loader gains a fallback path.
- File naming:
  - If extensions are changed (`.msgpack` vs `.msgpack.lz4`), tooling that statically assumes a specific suffix will need minor updates, but this is acceptable pre-v1.0.

Alternatives Considered:
- Remove support for `compression: "none"`:
  - Simpler implementation but loses a potentially useful debugging option and contradicts existing tests that exercise the no-compression branch.
- Keep current behavior and simply document that `"none"` is unsupported:
  - Rejected; silent failure at replay time is confusing and violates the principle of least surprise.

Tests:
- Extend `tests/test_townlet/integration/test_recording_replay_manager.py`:
  - Add a case that writes an episode using `compression: "none"` and asserts that `ReplayManager.load_episode()` successfully returns `True` and yields the correct metadata and steps.
- Extend `tests/test_townlet/unit/recording/test_recorder.py`:
  - Verify that `_write_episode()` sets any new compression header correctly and chooses the expected filename extension.

Owner: recording subsystem
Links:
- `docs/WORK-PACKAGES.md:324` (WP-L1 mentions recording/visualization robustness)
- `docs/arch-analysis-2025-11-13-1532/02-subsystem-catalog.md`
