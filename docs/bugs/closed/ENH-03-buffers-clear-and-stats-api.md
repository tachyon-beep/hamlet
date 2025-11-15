Title: Add `clear()` and `stats()` API to buffers for manageability

Severity: low
Status: RESOLVED
Resolved Date: 2025-11-15

Subsystem: training/replay-buffer (standard, sequential, PER)
Affected Version/Branch: main

Description:
- No simple way to reset buffers mid-run or inspect occupancy/memory usage without internal poking.

Proposed Enhancement:
- Add `clear()` method to reset size/position and drop stored data.
- Add `stats()` returning dict: `{size, capacity, occupancy_ratio, memory_bytes (approx), device}`.
- Sequential: include `num_episodes` and `num_transitions`.

Migration Impact:
- Backwards compatible; new API only.

Tests:
- Unit tests for `clear()` idempotence and `stats()` content.

Owner: training

---

## RESOLUTION

**Implemented By**: Systematic debugging process using superpowers:systematic-debugging skill
**Resolution Date**: 2025-11-15

### Implementation Summary

**Phase 1 (Investigation)**: Analyzed current buffer state management across all three buffer types, identified state variables, storage mechanisms, and memory allocation patterns.

**Phase 2 (Pattern Analysis)**: No existing `clear()` or `stats()` patterns in codebase. Designed consistent API across all buffer types using PyTorch tensor introspection for memory calculation.

**Phase 3 (Design & Testing - TDD)**: Created comprehensive test suites first (29 new tests total):
- ReplayBuffer: 10 tests (4 clear + 6 stats)
- SequentialReplayBuffer: 7 tests (3 clear + 4 stats)
- PrioritizedReplayBuffer: 11 tests (5 clear + 6 stats)

**Phase 4 (Implementation)**:

#### **ReplayBuffer** (`src/townlet/training/replay_buffer.py`)

**`clear()` method** (lines 136-150):
- Resets size/position counters to 0
- Sets all storage tensors to None (enables garbage collection)
- Idempotent - safe to call multiple times
- Buffer remains fully functional after clearing

**`stats()` method** (lines 152-191):
- Returns dict with size, capacity, occupancy_ratio, memory_bytes, device
- Memory calculation: `.element_size() * .numel()` for all tensors
- Handles empty buffer case (returns 0 memory when observations is None)
- Occupancy ratio guards against capacity=0

#### **SequentialReplayBuffer** (`src/townlet/training/sequential_replay_buffer.py`)

**`clear()` method** (lines 50-57):
- Clears episode list, resets transition counter
- Enables garbage collection of episode tensors

**`stats()` method** (lines 59-89):
- Base stats: size, capacity, occupancy_ratio, memory_bytes, device
- Additional: num_episodes, num_transitions (episode-specific metrics)
- Memory iterates through all episodes and tensors

#### **PrioritizedReplayBuffer** (`src/townlet/training/prioritized_replay_buffer.py`)

**`clear()` method** (lines 198-213):
- Resets size/position counters
- Sets storage tensors to None
- Resets priorities array to zeros (PER-specific state)
- Resets max_priority to 1.0 (critical for correct prioritization)

**`stats()` method** (lines 215-253):
- Includes NumPy priorities array in memory calculation (`.nbytes`)
- Handles empty buffer case
- Consistent API with other buffer types

### Test Results
✅ **All tests pass: 120/120**
- 91 existing tests (no regressions)
- 29 new tests for `clear()` and `stats()` API

### Code Review
- Reviewer: feature-dev:code-reviewer subagent
- Status: ✅ APPROVED
- Findings:
  - Implementation correct for all three buffer types
  - Memory calculations accurate
  - API consistent across buffer types with appropriate extensions
  - Test coverage comprehensive
  - Fully backwards compatible
  - Minor: One redundant import statement (non-blocking style issue)

### Files Modified
1. `src/townlet/training/replay_buffer.py` - Added `clear()` and `stats()`
2. `src/townlet/training/sequential_replay_buffer.py` - Added `clear()` and `stats()`
3. `src/townlet/training/prioritized_replay_buffer.py` - Added `clear()` and `stats()`
4. `tests/test_townlet/unit/training/test_replay_buffers.py` - Added 10 tests
5. `tests/test_townlet/unit/training/test_sequential_replay_buffer.py` - Added 7 tests
6. `tests/test_townlet/unit/training/test_prioritized_replay_buffer.py` - Added 11 tests

### Migration Notes
- **Zero migration required** - New methods only, no changes to existing APIs
- Fully backwards compatible
- Type-safe with proper annotations

### API Documentation

**`clear()` Method**:
- Purpose: Reset buffer to empty state and deallocate storage
- Signature: `def clear(self) -> None`
- Idempotent (safe to call multiple times)
- Buffer remains fully functional after clearing

**`stats()` Method**:
- Purpose: Return buffer statistics for introspection and monitoring
- Signature: `def stats(self) -> dict[str, Any]`
- Returns: size, capacity, occupancy_ratio, memory_bytes, device
- SequentialReplayBuffer also returns: num_episodes, num_transitions

### Impact
- ✅ Simple reset mechanism for mid-training buffer clearing
- ✅ Runtime introspection of buffer state and memory usage
- ✅ Support for debugging, monitoring, and resource management
- ✅ Consistent API across all three buffer types
- ✅ Enhanced buffer manageability with zero breaking changes
