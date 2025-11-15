Title: SequentialReplayBuffer serialize/load assumes split rewards; accepts combined rewards in store

Severity: high
Status: open

Subsystem: training/replay-buffer (sequential)
Affected Version/Branch: main

Affected Files:
- `src/townlet/training/sequential_replay_buffer.py:75`
- `src/townlet/training/sequential_replay_buffer.py:213`
- `src/townlet/training/sequential_replay_buffer.py:244`

Description:
- `store_episode` accepts either `rewards` (combined) or split `rewards_extrinsic`/`rewards_intrinsic`.
- `serialize` and `load_from_serialized` unconditionally read/write split rewards only.
- Episodes stored with combined `rewards` cannot be serialized/restored → KeyError or silent contract mismatch.

Reproduction:
1) Create buffer; call `store_episode` with only `rewards` key.
2) Call `serialize()` then `load_from_serialized()` on a fresh buffer.
3) Observe KeyError when accessing `rewards_extrinsic`/`rewards_intrinsic` during serialize/load.

Expected Behavior:
- Either: unified schema (always split OR always combined) flows through store/serialize/load.
- Or: both representations are supported consistently by serialize/load.

Actual Behavior:
- Mixed acceptance in `store_episode`, but serialization assumes split rewards only.

Root Cause:
- Divergent schema support between `store_episode` (lines 71–76) and (de)serialization (lines 213–219, 242–246).

Proposed Fix (Breaking OK):
- Standardize on split rewards for sequential episodes:
  - Reject `rewards` in `store_episode` with a clear error asking for split fields.
  - Keep serialize/load as-is (split).
- Alternative (non-preferred): Accept `rewards` in serialize/load by materializing `rewards_extrinsic=0` and `rewards_intrinsic=rewards`.

Migration Impact:
- Any callsites/tests passing `rewards` must switch to split fields.
- Update fixtures/builders to emit `rewards_extrinsic` and `rewards_intrinsic`.

Alternatives Considered:
- Retain dual support and branch in serialize/load; increases complexity and ongoing risk.

Tests:
- Add unit test: store episode with combined rewards → expect ValueError.
- Ensure existing tests that use split rewards still pass.

Owner: training
Links:
- N/A
