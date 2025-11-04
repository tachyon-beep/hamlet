# TASK-007: Live Training Visualization Streaming

**Status**: Planned
**Priority**: HIGH (enables real-time pedagogy)
**Estimated Effort**: 9 hours
**Dependencies**: None
**Enables**: TASK-008 (can use live viz for Q-value display)
**Created**: 2025-11-05
**Completed**: TBD

**Keywords**: live-viz, episode-streaming, telemetry, websocket, real-time, training-visualization, callback-queue, pedagogy
**Subsystems**: recording, demo (UnifiedServer, DemoRunner, LiveInferenceServer)
**Architecture Impact**: Minor (adds streaming path, existing checkpoint path unchanged)
**Breaking Changes**: No

---

## AI-Friendly Summary (Skim This First!)

**What**: Stream training episodes directly from DemoRunner to frontend with <1 second latency (vs current 100+ episode delay from checkpoint polling)

**Why**: Students need to see "what the agent is learning RIGHT NOW" for effective real-time pedagogy and engagement

**Scope**: Episode streaming via callback + queue; does NOT include model abstraction (see TASK-008)

**Quick Assessment**:

- **Current Limitation**: Visualization lags training by 100+ episodes (checkpoint save frequency), shows reconstructed episodes (not actual training)
- **After Implementation**: <1s latency, shows actual training episodes as they complete
- **Unblocks**: Real-time observation of exploration→exploitation, curriculum progression, reward hacking
- **Impact Radius**: 3 files modified (EpisodeRecorder, UnifiedServer, LiveInferenceServer)

**Decision Point**: If you're implementing model abstraction (TASK-008), you can do these in parallel or sequentially. If not working on inference refactor, STOP READING HERE.

---

## Problem Statement

### Current Constraint

**Visualization lags training by 100+ episodes:**

```
Training Episode 5000 completes
    ↓ (wait 99 more episodes)
Training Episode 5099 completes
    ↓
Save checkpoint_ep05100.pt (~500ms disk I/O)
    ↓
LiveInferenceServer polls for new checkpoint
    ↓
Load checkpoint (~500ms)
    ↓
Run NEW episode (not training episode!)
    ↓
Frontend displays

TOTAL LATENCY: 100 episodes + 1 second
```

**Problems:**
1. **Delay**: Students watch training from 100 episodes ago
2. **Mismatch**: Visualization episodes ≠ actual training episodes
   - Different random seeds
   - Different exploration (inference uses epsilon from checkpoint, training uses adaptive exploration)
3. **Inefficiency**: Expensive checkpoint save/load for every episode view
4. **Waste**: Most checkpoints never viewed (saved every 100, viewed every ~1000)

**Example code showing limitation** (src/townlet/demo/live_inference.py:290):

```python
async def _check_and_load_checkpoint(self) -> bool:
    """Check for new checkpoints and load if available."""
    checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_ep*.pt"))
    latest_checkpoint = checkpoints[-1]

    if latest_checkpoint == self.current_checkpoint_path:
        return False  # No new checkpoint = can't show new episode

    # Load 50MB checkpoint just to get model weights
    checkpoint = torch.load(latest_checkpoint, weights_only=False)
    self.population.q_network.load_state_dict(checkpoint["population_state"]["q_network"])
```

### Why This Is Technical Debt, Not Design

**Test**: "If removed, would the system be more expressive or more fragile?"

**Answer**: More expressive

- ✅ Enables: Real-time observation of learning dynamics
- ✅ Enables: Immediate feedback on hyperparameter changes
- ✅ Enables: Live debugging (see agent behavior NOW, not 100 episodes ago)
- ❌ Does NOT: Break checkpoint-based replay (existing recording system unchanged)

**Conclusion**: Technical debt from when training and visualization were separate processes

### Impact of Current Constraint

**Cannot Observe**:

- Real-time exploration→exploitation transition
- Exact moment when curriculum stage advances
- Actual training episodes (only reconstructed ones)
- Immediate effects of hyperparameter changes

**Pedagogical Cost**:

- Students watch "training from the past," not "training happening now"
- Delays discovery of interesting behaviors (reward hacking, etc.)
- Reduces engagement (no immediate feedback)

**Research Cost**:

- Can't debug training issues in real-time
- Hyperparameter tuning requires waiting for checkpoints
- Harder to catch transient behaviors

**From Analysis**: High-leverage change (9 hours → 100x latency improvement + shows actual training)

---

## Solution Overview

### Design Principle

**Core Philosophy**: "Stream training episodes as they complete, don't wait for checkpoints"

**Key Insight**: Training and visualization run in the same process (UnifiedServer), so they can share data directly via in-memory queue. No disk I/O needed!

### Architecture Changes

**1. EpisodeRecorder**: Add callback hook for live streaming

- Accept optional `live_stream_callback` parameter
- Call callback on `finish_episode()` with episode data
- Non-blocking (uses queue, never blocks training)

**2. UnifiedServer**: Create thread-safe queue for episode passing

- `queue.Queue(maxsize=10)` for bounded memory
- Training thread writes, inference thread reads
- Drops episodes if queue full (acceptable for live viz)

**3. LiveInferenceServer**: Consume episodes from queue

- Background task: `_stream_training_episodes()`
- Async consume from queue
- Broadcast to WebSocket clients

**4. No Changes**: Keep existing recording system (disk + database) unchanged

### Compatibility Strategy

**Backward Compatibility**:

- Callback is optional parameter (default None)
- Existing recording system unchanged
- LiveInferenceServer keeps checkpoint mode for standalone usage

**Migration Path**:

- No migration needed (additive change only)
- Users can enable live streaming by passing queue to UnifiedServer

**Versioning**:

- No version changes needed (API addition only)

---

## Detailed Design

### Phase 1: Add Callback to EpisodeRecorder (2 hours)

**Objective**: EpisodeRecorder accepts callback and calls it on episode completion

**Changes**:

- File: `src/townlet/recording/recorder.py`
  - Add `live_stream_callback` parameter to `__init__`
  - In `finish_episode()`: call callback with episode data dict
  - Handle callback exceptions gracefully (log, don't crash)
  - Episode data format:
    ```python
    # Episode data structure
    {
        "episode_id": int,
        "metadata": EpisodeMetadata (as dict),
        "steps": List[RecordedStep],  # All steps in episode
        "timestamp": float,
    }

    # RecordedStep structure (from src/townlet/recording/data_structures.py)
    RecordedStep = {
        "step": int,              # Step number within episode
        "position": (int, int),   # Agent (x, y) position
        "meters": tuple[float],   # 8 meter values (energy, hygiene, ...)
        "action": int,            # Action taken (0-5)
        "reward": float,          # Extrinsic reward
        "intrinsic_reward": float,  # RND novelty reward
        "done": bool,             # Terminal state flag
        "q_values": Optional[List[float]],  # Q-values for all actions (if recorded)
        "epsilon": Optional[float],         # Exploration rate
        "action_masks": Optional[List[bool]],  # Valid actions
        "time_of_day": Optional[int],       # Temporal mechanics (if enabled)
        "interaction_progress": Optional[float],  # Multi-tick interaction (if enabled)
    }
    ```

**Tests**:

- [ ] Unit test: Callback called with correct data
- [ ] Unit test: Callback failure doesn't crash training
- [ ] Unit test: Callback not called if None
- [ ] Existing recording tests still pass

**Success Criteria**: EpisodeRecorder can stream episodes via callback without affecting disk recording

---

### Phase 2: Wire Queue in UnifiedServer (1 hour)

**Objective**: UnifiedServer creates queue and connects training to visualization

**Changes**:

- File: `src/townlet/demo/unified_server.py`
  - Add import: `import queue` (stdlib)
  - Create `self.live_episode_queue = queue.Queue(maxsize=10)` in `__init__`
  - Define `on_episode_complete(episode_data)` callback:
    ```python
    def on_episode_complete(episode_data: dict):
        """Callback for EpisodeRecorder when episode completes.

        Runs in training thread, must be non-blocking.
        """
        try:
            self.live_episode_queue.put_nowait(episode_data)
        except queue.Full:
            logger.debug(f"Live queue full, dropping episode {episode_data['episode_id']}")
    ```
  - Pass callback to DemoRunner on creation:
    ```python
    self.runner = DemoRunner(
        config_dir=str(self.config_dir),
        training_config_path=str(self.training_config_path),
        db_path=str(db_path),
        checkpoint_dir=str(self.checkpoint_dir),
        max_episodes=self.total_episodes,
        live_episode_callback=on_episode_complete,  # NEW: Pass callback
    )
    ```
  - Pass queue to LiveInferenceServer on creation:
    ```python
    self.inference_server = LiveInferenceServer(
        checkpoint_dir=self.checkpoint_dir,
        port=self.inference_port,
        # ... other params ...
        live_episode_queue=self.live_episode_queue,  # NEW: Pass queue
    )
    ```

**Tests**:

- [ ] Integration test: Queue passes data from training to inference
- [ ] Test: Queue full handling (drops episodes, doesn't block)

**Success Criteria**: Episodes flow from training thread to inference thread via queue

---

### Phase 3: Consumer in LiveInferenceServer (2 hours)

**Objective**: LiveInferenceServer consumes episodes and broadcasts to WebSocket clients

**Changes**:

- File: `src/townlet/demo/live_inference.py`
  - Add `live_episode_queue` parameter to `__init__`
  - Set `self.mode = "live" if live_episode_queue else "checkpoint"`
  - Add async background task:
    ```python
    async def _stream_training_episodes(self):
        """Consume episodes from queue and broadcast."""
        while True:
            try:
                episode_data = await asyncio.get_running_loop().run_in_executor(
                    None,
                    lambda: self.live_episode_queue.get(timeout=0.1)
                )
                await self._broadcast_training_episode(episode_data)
            except queue.Empty:
                await asyncio.sleep(0.05)
    ```
  - Add `_broadcast_training_episode()` method:
    - Send "episode_start" message
    - Replay steps with configurable delay
    - Send "episode_end" message

**Tests**:

- [ ] Integration test: Episodes broadcast to WebSocket clients
- [ ] Test: Step-by-step replay at correct speed
- [ ] Test: Multiple clients receive same data

**Success Criteria**: Frontend receives training episodes within <1s of completion

---

### Phase 4: Update DemoRunner (1 hour)

**Objective**: DemoRunner accepts and passes callback to EpisodeRecorder

**Changes**:

- File: `src/townlet/demo/runner.py`
  - Add `live_episode_callback` parameter to `__init__`
  - Store as `self.live_episode_callback`
  - Pass to EpisodeRecorder when creating:
    ```python
    self.recorder = EpisodeRecorder(
        config=recording_cfg,
        output_dir=recording_output_dir,
        database=self.db,
        curriculum=self.curriculum,
        live_stream_callback=self.live_episode_callback,  # NEW
    )
    ```

**Migration**:

- [ ] Old code works (callback=None by default)
- [ ] New code enables streaming (callback provided)

**Success Criteria**: DemoRunner wires callback from UnifiedServer to EpisodeRecorder

---

### Phase 5: Integration & Polish (2 hours)

**Objective**: End-to-end testing and edge case handling

**Testing**:

- [ ] Full integration test: Training → Queue → Viz → Frontend
- [ ] Test: Queue full scenario (drops episodes gracefully)
- [ ] Test: No episodes lost under normal load
- [ ] Performance test: Training throughput unaffected

**Edge Cases**:

- [ ] Queue full: Drop episode, log warning
- [ ] Callback exception: Log error, continue training
- [ ] No clients connected: Episodes dropped (no backpressure)
- [ ] Client connects mid-episode: Waits for next episode start

**Documentation**:

- [ ] Update `CLAUDE.md` with live streaming architecture
- [ ] Update `manual/UNIFIED_SERVER_USAGE.md` with live mode
- [ ] Comment in code explaining queue flow

**Success Criteria**: System handles all edge cases gracefully, full end-to-end test passes

---

## Testing Strategy

### Test Coverage Requirements

**Unit Tests**:

- EpisodeRecorder: Callback mechanism (100% coverage)
- UnifiedServer: Queue handling (100% coverage)
- LiveInferenceServer: Episode consumption (100% coverage)

**Integration Tests**:

- [ ] Test: Training episode → Queue → Broadcast → Frontend
- [ ] Test: Multiple episodes stream correctly
- [ ] Test: Queue full handling (no blocking)
- [ ] Test: Callback failure handling (training continues)

**Performance Testing**:

- [ ] Training throughput unchanged (<5% regression)
- [ ] Memory usage bounded (queue never exceeds ~1MB)
- [ ] Latency <1s from episode end to frontend display

### Regression Testing

**Critical Paths**:

- [ ] Existing recording system works (disk + database)
- [ ] Checkpoint-based inference still works (for standalone usage)
- [ ] All existing tests pass

**Performance Testing**:

- [ ] Benchmark: Training episodes/minute remains within 5%
- [ ] Benchmark: Memory usage stays bounded

---

## Migration Guide

### For Existing Configs

**No config changes needed** (additive feature only)

### For Existing Code

**Before** (checkpoint polling):

```python
# UnifiedServer starts training and inference independently
self.training_thread = threading.Thread(target=self._run_training)
self.inference_thread = threading.Thread(target=self._run_inference)
```

**After** (with live streaming):

```python
# Create queue for episode streaming
self.live_episode_queue = queue.Queue(maxsize=10)

# Define callback
def on_episode_complete(episode_data):
    try:
        self.live_episode_queue.put_nowait(episode_data)
    except queue.Full:
        logger.debug(f"Dropping episode {episode_data['episode_id']}")

# Pass callback to training
self.runner = DemoRunner(..., live_episode_callback=on_episode_complete)

# Pass queue to inference
self.inference_server = LiveInferenceServer(..., live_episode_queue=self.live_episode_queue)
```

**Compatibility**: Old code still works (callback=None, queue=None → checkpoint mode)

---

## Examples

### Example 1: Watch Training Live

**Usage**:

```bash
# Terminal 1: Start training with unified server
python scripts/run_demo.py --config configs/L2_partial_observability --episodes 10000

# Terminal 2: Start frontend
cd frontend && npm run dev

# Browser: http://localhost:5173
# See training episodes as they complete (<1s latency)
```

**Output**: Frontend shows actual training episodes in real-time

### Example 2: Standalone Inference (Checkpoint Mode)

**Usage**:

```python
# Run inference server without training (checkpoint mode)
from townlet.demo.live_inference import run_server

run_server(
    checkpoint_dir="runs/L2_partial_observability/2025-11-05_123456/checkpoints",
    port=8766,
    # No live_episode_queue → checkpoint polling mode
)
```

**Output**: Falls back to checkpoint polling (existing behavior)

---

## Acceptance Criteria

### Must Have (Blocking)

- [ ] Episodes stream from training to frontend with <1s latency
- [ ] Shows actual training episodes (not reconstructed)
- [ ] Memory bounded (queue ~1MB max)
- [ ] Training throughput unchanged (<5% regression)
- [ ] All tests pass (unit + integration)
- [ ] No breaking changes (backward compatible)
- [ ] Documentation updated

### Should Have (Important)

- [ ] Graceful handling of queue full (drops episodes, logs warning)
- [ ] Callback exceptions don't crash training
- [ ] Multiple WebSocket clients supported

### Could Have (Future)

- [ ] Episode filtering (show only failures, only stage transitions)
- [ ] Playback speed controls from frontend
- [ ] Episode compression (LZ4) for smaller queue memory

---

## Risk Assessment

### Technical Risks

**Risk 1: Training thread blocking**

- **Severity**: HIGH (would slow down training)
- **Mitigation**: Use `queue.put_nowait()` (never blocks)
- **Contingency**: Drop episodes if queue full (acceptable for live viz)

**Risk 2: Memory usage from large episodes**

- **Severity**: MEDIUM
- **Mitigation**: Bounded queue (maxsize=10), ~1MB total
- **Contingency**: Can reduce queue size or compress episodes

**Risk 3: Queue synchronization issues**

- **Severity**: LOW
- **Mitigation**: Use stdlib `queue.Queue` (thread-safe by design)
- **Contingency**: Extensive testing of concurrent access

### Blocking Dependencies

- ✅ **NONE**: All prerequisites exist (EpisodeRecorder, UnifiedServer, LiveInferenceServer)

### Impact Radius

**Files Modified**: 4
- `src/townlet/recording/recorder.py`
- `src/townlet/demo/unified_server.py`
- `src/townlet/demo/live_inference.py`
- `src/townlet/demo/runner.py`

**Tests Added**: ~8 tests (unit + integration)

**Breaking Changes**: None

**Blast Radius**: Small (additive change only, existing paths unchanged)

---

## Effort Breakdown

### Detailed Estimates

**Phase 1**: 2 hours
- Add callback parameter: 0.5h
- Call callback on finish_episode: 0.5h
- Exception handling: 0.5h
- Tests: 0.5h

**Phase 2**: 1 hour
- Create queue in UnifiedServer: 0.25h
- Define callback: 0.25h
- Wire to DemoRunner and LiveInferenceServer: 0.5h

**Phase 3**: 2 hours
- Add _stream_training_episodes: 1h
- Add _broadcast_training_episode: 0.5h
- Tests: 0.5h

**Phase 4**: 1 hour
- Add parameter to DemoRunner: 0.25h
- Pass to EpisodeRecorder: 0.25h
- Tests: 0.5h

**Phase 5**: 3 hours
- Integration testing: 1.5h
- Edge case handling: 1h
- Documentation: 0.5h

**Total**: 9 hours

**Confidence**: HIGH (well-scoped, standard patterns)

### Assumptions

- EpisodeRecorder already captures all step data
- UnifiedServer runs training and inference in same process
- queue.Queue provides sufficient thread safety
- ~100KB per episode × 10 queue size = ~1MB memory

---

## Future Work (Explicitly Out of Scope)

### Not Included in This Task

1. **Episode Filtering**
   - **Why Deferred**: Core streaming is sufficient for MVP
   - **Follow-up Task**: Future enhancement (3-4h)

2. **Playback Controls from Frontend**
   - **Why Deferred**: Step delay is configurable server-side
   - **Follow-up Task**: Future enhancement (2-3h)

3. **Episode Compression**
   - **Why Deferred**: Memory usage acceptable without compression
   - **Follow-up Task**: If queue memory becomes issue (2-3h)

### Enables Future Tasks

- **TASK-008**: Can use live viz for real-time Q-value display
- **Future**: Multi-agent streaming (show all agents simultaneously)
- **Future**: Episode filtering and selection

---

## References

### Related Documentation

- **Research**: `docs/research/RESEARCH-LIVE-TRAINING-TELEMETRY.md`
- **Synthesis**: `docs/research/RESEARCH-INFERENCE-ARCHITECTURE-SYNTHESIS.md`
- **Architecture**: `docs/architecture/TOWNLET_HLD.md`
- **Manual**: `docs/manual/UNIFIED_SERVER_USAGE.md`

### Related Tasks

- **Parallel Work**: TASK-008 (Model Abstraction) - can be done in same PR
- **Enables**: TASK-008 (can use HamletModel for Q-value computation)

### Code References

- `src/townlet/recording/recorder.py:22` - EpisodeRecorder class
- `src/townlet/demo/unified_server.py:28` - UnifiedServer class
- `src/townlet/demo/live_inference.py:48` - LiveInferenceServer class
- `src/townlet/demo/runner.py:50` - DemoRunner class

---

## Notes for Implementer

### Before Starting

- [ ] Read `docs/research/RESEARCH-LIVE-TRAINING-TELEMETRY.md` for design rationale
- [ ] Understand thread safety requirements (training vs inference threads)
- [ ] Review EpisodeRecorder to understand current recording flow
- [ ] Can be implemented in same PR as TASK-008 (both touch LiveInferenceServer)

### During Implementation

- [ ] Use `queue.put_nowait()` (never blocks training thread)
- [ ] Log dropped episodes at DEBUG level (expected behavior)
- [ ] Test with multiple episodes to verify no memory leaks
- [ ] Keep callback simple and fast (<1ms)

### Before Marking Complete

- [ ] All acceptance criteria met
- [ ] Full integration test passes (training → viz → frontend)
- [ ] Memory profiling shows bounded usage
- [ ] Training throughput unchanged
- [ ] Documentation updated (CLAUDE.md, manual)

---

**END OF TASK SPECIFICATION**
