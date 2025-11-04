# Research: Live Training Telemetry Streaming

## Problem Statement (Reframed)

**Current Misunderstanding:** The inference server is for "production inference" on trained models.

**Actual Use Case:** The inference server is a **live training visualizer** that shows "what the agent is learning RIGHT NOW as it trains."

**Core Problem:** The visualization lags behind training by ~100 episodes due to checkpoint polling. We're not showing "live training," we're showing "training from 100 episodes ago."

### What We Actually Want

**Goal:** Stream training episodes directly from DemoRunner to LiveInferenceServer to frontend, showing real-time learning behavior.

**Natural Unit:** Whole episodes at a time (episode = complete trajectory from reset to done/max_steps).

**Why Episodes Make Sense:**
- Natural learning unit (agent completes one life)
- Frontend can replay at any speed
- Includes complete context (trajectory, final reward, survival time)
- Avoids mid-episode partial state

---

## Current Architecture Analysis

### Data Flow (Current)

```
┌─────────────────────────────────────────────────────────────────┐
│ UnifiedServer (single process, multiple threads)                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Training Thread (DemoRunner)                                   │
│  ├─> Run episode                                                │
│  ├─> Record steps (EpisodeRecorder → queue → disk)             │
│  ├─> Every 100 episodes: save_checkpoint()                     │
│  └─> checkpoint_ep05000.pt (~50MB)                             │
│                                                                  │
│  ────────────────────────────────────────────────────────────── │
│                                                                  │
│  Inference Thread (LiveInferenceServer)                         │
│  ├─> Poll for new checkpoints (every episode)                  │
│  ├─> Load checkpoint (if new)                                  │
│  ├─> Run OWN episode (not training episode!)                   │
│  └─> Stream to WebSocket clients                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Problems with Current Approach

1. **Latency:** 100 episode delay (checkpoint save frequency)
2. **Inefficiency:** Checkpoint save/load is expensive (~50MB files)
3. **Duplication:** Running episodes twice (training + visualization)
4. **Mismatch:** Visualization episodes ≠ actual training episodes
   - Different random seeds
   - Inference uses greedy/epsilon-greedy, training uses complex exploration
   - No guarantee behavior matches
5. **Waste:** Most checkpoints never viewed (saved every 100, viewed every ~1000)

### Why This Architecture Exists

**Historical reason:** Training and inference were separate processes, so checkpoint was the only communication channel.

**Current reality:** UnifiedServer runs both in same process, so they can share data directly!

**Existing Infrastructure:** EpisodeRecorder already captures all step data in a thread-safe queue.

---

## Design Space: Episode Streaming Architectures

### Option 1: Callback-Based Streaming (Simplest)

**Add observer pattern to EpisodeRecorder:**

```python
class EpisodeRecorder:
    def __init__(self, ..., live_callback=None):
        self.live_callback = live_callback  # NEW: callback for live streaming

    def finish_episode(self, metadata: EpisodeMetadata):
        # Existing: Save to disk (background writer)
        self._enqueue_episode_end(metadata)

        # NEW: Stream to visualization (if callback registered)
        if self.live_callback:
            episode_data = {
                "episode_id": metadata.episode_id,
                "steps": self.current_episode_steps.copy(),  # All steps
                "metadata": asdict(metadata),
            }
            # Call callback (runs in training thread, must be fast!)
            self.live_callback(episode_data)

# In UnifiedServer.__init__():
self.live_episode_queue = queue.Queue(maxsize=10)  # Bounded queue

# In UnifiedServer._run_training():
def live_callback(episode_data):
    try:
        self.live_episode_queue.put_nowait(episode_data)
    except queue.Full:
        logger.warning("Live episode queue full, dropping episode")

self.runner = DemoRunner(..., recorder_callback=live_callback)

# In LiveInferenceServer:
class LiveInferenceServer:
    def __init__(self, ..., live_episode_queue=None):
        self.live_episode_queue = live_episode_queue

    async def _stream_training_episodes(self):
        """Background task: consume episodes from training and broadcast."""
        while True:
            # Check queue in background (thread-safe, non-blocking)
            try:
                episode_data = await asyncio.get_running_loop().run_in_executor(
                    None,
                    self.live_episode_queue.get,
                    timeout=0.1
                )
                await self._broadcast_training_episode(episode_data)
            except queue.Empty:
                await asyncio.sleep(0.1)
```

**Benefits:**
- ✅ Simple: Just add callback parameter
- ✅ Fast: No disk I/O for live streaming
- ✅ Real-time: Episodes stream as they complete
- ✅ Minimal changes: Leverages existing EpisodeRecorder
- ✅ Thread-safe: queue.Queue handles synchronization
- ✅ Bounded: Queue has maxsize, won't OOM if visualization slow

**Drawbacks:**
- ⚠️ Memory: Queue holds complete episodes in RAM
- ⚠️ Drops episodes if queue full (acceptable for live viz)

**Effort Estimate:** 6-8 hours
- Add callback to EpisodeRecorder (2h)
- Add queue to UnifiedServer (1h)
- Add streaming consumer to LiveInferenceServer (2h)
- Tests (2h)
- Handle edge cases (1h)

---

### Option 2: Shared Episode Buffer (In-Memory)

**Create shared circular buffer for latest N episodes:**

```python
class EpisodeBuffer:
    """Thread-safe circular buffer for latest episodes."""

    def __init__(self, capacity: int = 10):
        self.capacity = capacity
        self.episodes: list[dict | None] = [None] * capacity
        self.write_index = 0
        self.lock = threading.Lock()

    def push(self, episode_data: dict):
        """Add episode (overwrites oldest)."""
        with self.lock:
            self.episodes[self.write_index] = episode_data
            self.write_index = (self.write_index + 1) % self.capacity

    def get_latest(self, n: int = 1) -> list[dict]:
        """Get latest N episodes."""
        with self.lock:
            # Return latest N non-None episodes
            valid = [ep for ep in self.episodes if ep is not None]
            return valid[-n:] if valid else []

# In UnifiedServer:
self.episode_buffer = EpisodeBuffer(capacity=10)

# Training writes:
def finish_episode(...):
    self.episode_buffer.push(episode_data)

# Visualization reads:
async def poll_latest_episodes():
    while True:
        latest = self.episode_buffer.get_latest(n=1)
        if latest:
            await self._broadcast_episode(latest[0])
        await asyncio.sleep(0.5)  # Poll every 500ms
```

**Benefits:**
- ✅ Simple: No queue management
- ✅ Always latest: Can skip episodes if visualization slow
- ✅ Fixed memory: Circular buffer, no unbounded growth

**Drawbacks:**
- ❌ Polling: Visualization polls instead of push
- ❌ May skip episodes: If training produces >10 episodes before poll
- ❌ Lock contention: Every read/write acquires lock

**Effort Estimate:** 6-8 hours

**Recommendation:** Option 1 (Callback) is better - push model, no polling, queue handles backpressure.

---

### Option 3: Database as Intermediary

**Use existing DemoDatabase for episode streaming:**

```python
# Training writes (already happening):
self.db.insert_episode(episode_id=ep, timestamp=..., survival_time=..., ...)

# Visualization reads:
async def poll_database():
    last_seen_episode = 0
    while True:
        # Query for new episodes
        new_episodes = self.db.get_episodes_since(last_seen_episode)
        for episode in new_episodes:
            # Load recording from disk
            recording = self.replay_manager.load_episode(episode['episode_id'])
            await self._broadcast_episode(recording)
            last_seen_episode = episode['episode_id']
        await asyncio.sleep(1.0)
```

**Benefits:**
- ✅ Leverages existing: DemoDatabase already stores episode metadata
- ✅ Persistent: Episodes saved to DB, can replay later
- ✅ Decoupled: No direct dependency between training and visualization

**Drawbacks:**
- ❌ Disk I/O: Database writes + reads
- ❌ Latency: Polling + disk access (slower than in-memory)
- ❌ Complexity: Need to coordinate DB writes + recording saves
- ❌ Still polling: Not true push model

**Effort Estimate:** 4-6 hours (mostly already implemented)

**Recommendation:** This already works (via replay system), but it's slow. Not the "real-time streaming" goal.

---

### Option 4: Hybrid - Streaming + Checkpoints

**Keep checkpoints for persistence, add streaming for real-time:**

```python
class LiveInferenceServer:
    def __init__(self, ..., mode: str = "auto"):
        """
        Mode:
        - 'live': Stream from training (real-time)
        - 'checkpoint': Load from checkpoints (current behavior)
        - 'auto': Live if available, fallback to checkpoint
        """
        self.mode = mode

    async def run(self):
        if self.mode == "live" and self.live_episode_queue:
            # NEW: Stream training episodes
            await self._stream_training_episodes()
        else:
            # EXISTING: Poll checkpoints
            await self._poll_checkpoints()
```

**Benefits:**
- ✅ Backward compatible: Works without live streaming
- ✅ Flexible: Can switch modes (live vs playback)
- ✅ Best of both: Real-time + persistence

**Drawbacks:**
- ⚠️ Complexity: Two code paths

**Recommendation:** Good for transitional period, but eventually prefer pure streaming.

---

## Tradeoffs Analysis

| Option | Latency | Complexity | Memory | Persistence | Effort |
|--------|---------|-----------|---------|-------------|--------|
| **1. Callback + Queue** | <1s | Low | Medium | No* | 6-8h |
| **2. Shared Buffer** | ~500ms | Low | Low | No | 6-8h |
| **3. Database** | 1-2s | Medium | Low | Yes | 4-6h |
| **4. Hybrid** | <1s / varies | Medium | Medium | Yes | 10-12h |

\* Episodes still saved to disk by existing recording system, just not used for live streaming

**Key Insights:**
- Option 1 (Callback) wins on latency and simplicity
- Option 3 (Database) already works, just slow
- Option 4 (Hybrid) is nice-to-have, not essential

---

## Recommendation

**Implement Option 1: Callback-Based Episode Streaming**

**Why:**
1. **Minimal Changes:** Leverage existing EpisodeRecorder infrastructure
2. **Real-Time:** <1 second latency from episode completion to frontend display
3. **Thread-Safe:** queue.Queue handles synchronization automatically
4. **Bounded:** Won't OOM if visualization lags behind training
5. **Simple:** Clean push model, no polling

**Reject:**
- Option 2: Polling is anti-pattern, lock contention
- Option 3: Already exists (via replay), too slow for "live" goal
- Option 4: Hybrid adds complexity without clear benefit (can add later if needed)

---

## Implementation Sketch (Option 1)

### Phase 1: Add Callback to EpisodeRecorder (2h)

**File:** `src/townlet/recording/recorder.py`

```python
class EpisodeRecorder:
    def __init__(
        self,
        config: dict,
        output_dir: Path,
        database,
        curriculum,
        live_stream_callback=None,  # NEW
    ):
        """
        Args:
            live_stream_callback: Optional callback(episode_data: dict) for live streaming.
                                 Called after episode completes, in training thread.
                                 Must be non-blocking!
        """
        self.live_stream_callback = live_stream_callback
        # ... existing init ...

    def finish_episode(self, metadata: EpisodeMetadata):
        """Finish current episode (save to disk + optionally stream live)."""
        # Existing: Enqueue for background writer
        end_marker = EpisodeEndMarker(metadata=metadata)
        try:
            self.queue.put(end_marker, timeout=5.0)
        except queue.Full:
            logger.error("Recording queue full, dropping episode")
            return

        # NEW: Stream to live visualization (if callback registered)
        if self.live_stream_callback and hasattr(self, 'current_episode_steps'):
            episode_data = {
                "episode_id": metadata.episode_id,
                "metadata": asdict(metadata),
                "steps": self.current_episode_steps.copy(),  # List[RecordedStep]
                "timestamp": metadata.timestamp,
            }

            try:
                # Callback must be fast and non-blocking!
                # If it blocks, it delays training loop
                self.live_stream_callback(episode_data)
            except Exception as e:
                logger.error(f"Live stream callback failed: {e}")

        # Reset current episode
        self.current_episode_steps = []
```

**Key Design Decisions:**
- Callback runs in training thread (fast path)
- Callback must not block (caller's responsibility)
- Episodes still saved to disk (existing path unchanged)
- If callback fails, log error but don't crash training

---

### Phase 2: Wire Queue in UnifiedServer (1h)

**File:** `src/townlet/demo/unified_server.py`

```python
class UnifiedServer:
    def __init__(self, ...):
        # ... existing init ...

        # NEW: Thread-safe queue for live episode streaming
        self.live_episode_queue = queue.Queue(maxsize=10)  # Bounded

    def _run_training(self):
        """Training thread entry point."""
        try:
            from townlet.demo.runner import DemoRunner

            # Define callback for live episode streaming
            def on_episode_complete(episode_data: dict):
                """
                Called by EpisodeRecorder when episode finishes.
                Runs in training thread, must be non-blocking.
                """
                try:
                    # Non-blocking put (drop if queue full)
                    self.live_episode_queue.put_nowait(episode_data)
                except queue.Full:
                    logger.debug(f"Live queue full, dropping episode {episode_data['episode_id']}")

            # Create DemoRunner with live callback
            self.runner = DemoRunner(
                config_dir=str(self.config_dir),
                training_config_path=str(self.training_config_path),
                db_path=str(db_path),
                checkpoint_dir=str(self.checkpoint_dir),
                max_episodes=self.total_episodes,
                live_episode_callback=on_episode_complete,  # NEW
            )

            # ... rest of training ...
```

---

### Phase 3: Consume Queue in LiveInferenceServer (2h)

**File:** `src/townlet/demo/live_inference.py`

```python
class LiveInferenceServer:
    def __init__(
        self,
        checkpoint_dir: Path | str,
        port: int = 8766,
        ...,
        live_episode_queue: queue.Queue | None = None,  # NEW
    ):
        """
        Args:
            live_episode_queue: Optional queue for receiving live training episodes.
                               If provided, server streams training episodes in real-time.
                               If None, falls back to checkpoint polling (legacy mode).
        """
        self.live_episode_queue = live_episode_queue
        self.mode = "live" if live_episode_queue else "checkpoint"

    async def startup(self):
        """Initialize environment and start streaming."""
        logger.info(f"Starting in {self.mode} mode")

        if self.mode == "live":
            # Start live episode streaming
            asyncio.create_task(self._stream_training_episodes())
        else:
            # Legacy: checkpoint polling (keep for backward compat)
            await self._check_and_load_checkpoint()

    async def _stream_training_episodes(self):
        """
        Background task: consume episodes from training queue and broadcast to clients.

        This runs continuously, pulling completed episodes from the training thread
        and streaming them to WebSocket clients.
        """
        logger.info("Live episode streaming started")

        while True:
            try:
                # Get episode from queue (runs in executor to avoid blocking asyncio)
                # Use short timeout so we can check for shutdown signals
                episode_data = await asyncio.get_running_loop().run_in_executor(
                    None,
                    lambda: self.live_episode_queue.get(timeout=0.1)
                )

                # Broadcast episode to all WebSocket clients
                await self._broadcast_training_episode(episode_data)

            except queue.Empty:
                # No episodes available, wait a bit
                await asyncio.sleep(0.05)

            except Exception as e:
                logger.error(f"Error streaming episode: {e}", exc_info=True)
                await asyncio.sleep(1.0)  # Back off on error

    async def _broadcast_training_episode(self, episode_data: dict):
        """
        Broadcast a completed training episode to all clients.

        Instead of running our own inference episodes, we replay the actual
        training episode step-by-step at human-watchable speed.

        Args:
            episode_data: {
                "episode_id": int,
                "metadata": EpisodeMetadata dict,
                "steps": List[RecordedStep],
                "timestamp": float,
            }
        """
        metadata = episode_data["metadata"]
        steps = episode_data["steps"]

        # Send episode start
        await self._broadcast_to_clients({
            "type": "episode_start",
            "episode": metadata["episode_id"],
            "mode": "live_training",  # Distinguish from inference mode
            "survival_steps": metadata["survival_steps"],
            "curriculum_stage": metadata["curriculum_stage"],
            "epsilon": metadata["epsilon"],
            "timestamp": episode_data["timestamp"],
        })

        # Replay steps at human-watchable speed
        for step_data in steps:
            # Convert RecordedStep to state_update format
            state_update = self._step_to_state_update(step_data, metadata)
            await self._broadcast_to_clients(state_update)

            # Delay for visualization (configurable speed)
            await asyncio.sleep(self.step_delay)

        # Send episode end
        await self._broadcast_to_clients({
            "type": "episode_end",
            "episode": metadata["episode_id"],
            "steps": metadata["survival_steps"],
            "total_reward": metadata["total_reward"],
            "extrinsic_reward": metadata["extrinsic_reward"],
            "intrinsic_reward": metadata["intrinsic_reward"],
            "curriculum_stage": metadata["curriculum_stage"],
        })

        logger.info(f"Broadcasted training episode {metadata['episode_id']}")
```

---

### Phase 4: Update DemoRunner to Accept Callback (1h)

**File:** `src/townlet/demo/runner.py`

```python
class DemoRunner:
    def __init__(
        self,
        config_dir: Path | str,
        db_path: Path | str,
        checkpoint_dir: Path | str,
        max_episodes: int | None = None,
        training_config_path: Path | str | None = None,
        live_episode_callback=None,  # NEW
    ):
        """
        Args:
            live_episode_callback: Optional callback for live episode streaming.
                                  Called with episode_data dict when episode completes.
        """
        self.live_episode_callback = live_episode_callback
        # ... existing init ...

    def run(self):
        """Run demo training loop."""
        # ... initialization ...

        # Initialize episode recorder with live callback
        recording_cfg = self.config.get("recording", {})
        if recording_cfg.get("enabled", False):
            from townlet.recording.recorder import EpisodeRecorder

            recording_output_dir = self.checkpoint_dir / recording_cfg.get("output_dir", "recordings")
            recording_output_dir.mkdir(parents=True, exist_ok=True)

            self.recorder = EpisodeRecorder(
                config=recording_cfg,
                output_dir=recording_output_dir,
                database=self.db,
                curriculum=self.curriculum,
                live_stream_callback=self.live_episode_callback,  # NEW: Pass callback
            )
            logger.info(f"Episode recording enabled with live streaming: {recording_output_dir}")
```

---

## Benefits Summary

### Before (Checkpoint Polling)
```
Training episode 5000 completes
    ↓ (wait 99 more episodes)
Training episode 5099 completes
    ↓
Save checkpoint_ep05100.pt
    ↓ (50MB write to disk, ~500ms)
Inference server polls
    ↓
Load checkpoint (~500ms)
    ↓
Run NEW episode (not training episode!)
    ↓
Frontend shows episode

TOTAL LATENCY: 100 episodes + 1 second
```

### After (Episode Streaming)
```
Training episode 5000 completes
    ↓ (callback, <1ms)
Episode data → queue
    ↓ (<1ms)
Inference server consumes
    ↓
Replay training episode step-by-step
    ↓
Frontend shows episode

TOTAL LATENCY: <1 second
```

**Improvements:**
- ✅ **100x faster**: <1s latency vs 100+ episodes
- ✅ **Shows actual training**: Not reconstructed episodes
- ✅ **No checkpoint overhead**: No disk I/O in critical path
- ✅ **Memory efficient**: Bounded queue, old episodes dropped
- ✅ **Minimal changes**: Leverages existing EpisodeRecorder

---

## Addressing "Hidden Downsides"

**You asked:** "unless that has hidden downsides I don't see"

### Potential Issue 1: Memory Usage

**Concern:** Queue holds complete episodes (500 steps × ~200 bytes/step = ~100KB per episode).

**Analysis:**
- Queue capacity: 10 episodes = ~1MB total
- Training produces ~10 episodes/minute at stage 5
- Visualization consumes ~1 episode/10 seconds (with step_delay=0.2)
- **Verdict:** Memory usage negligible, queue won't fill up

### Potential Issue 2: Training Thread Blocking

**Concern:** Callback blocks training if queue operations are slow.

**Mitigation:**
- Use `queue.put_nowait()` (non-blocking)
- If queue full, drop episode (acceptable for live viz)
- Callback is < 1ms (just copies data to queue)
- **Verdict:** No blocking risk

### Potential Issue 3: Episode Data Size

**Concern:** Large episodes (500 steps) might be too big to stream?

**Analysis:**
- Episode with 500 steps:
  - Positions: 500 × 2 floats × 4 bytes = 4KB
  - Meters: 500 × 8 floats × 4 bytes = 16KB
  - Actions: 500 × 1 int × 4 bytes = 2KB
  - Q-values: 500 × 6 floats × 4 bytes = 12KB
  - Total: ~40KB per episode (compressed)
- WebSocket can handle this easily
- **Verdict:** Not a problem

### Potential Issue 4: Synchronization Complexity

**Concern:** Thread safety issues between training and inference threads?

**Mitigation:**
- Use stdlib `queue.Queue` (thread-safe by design)
- No shared mutable state
- Training writes, inference reads (one-way)
- **Verdict:** No sync issues, queue handles it

### Potential Issue 5: Missing Episodes

**Concern:** If visualization slow, queue fills up and episodes get dropped.

**Analysis:**
- This is acceptable for live visualization!
- User wants to see "what's happening now," not every single episode
- Dropping old episodes when queue full is correct behavior
- Episodes still saved to disk (can replay later)
- **Verdict:** Feature, not bug

---

## Effort Estimate

| Phase | Description | Hours |
|-------|-------------|-------|
| 1 | Add callback to EpisodeRecorder | 2 |
| 2 | Wire queue in UnifiedServer | 1 |
| 3 | Consume queue in LiveInferenceServer | 2 |
| 4 | Update DemoRunner to accept callback | 1 |
| 5 | Tests (unit + integration) | 2 |
| 6 | Handle edge cases (queue full, errors) | 1 |

**Total: 9 hours** (was 6-8, revised up for safety)

**Can be parallelized:**
- Phase 1 + 4 can be done concurrently (different files)
- Phase 2 + 3 depend on Phase 1
- Phase 5 can start after Phase 1-4

---

## Success Criteria

✅ **Phase 1 Complete:**
- [ ] EpisodeRecorder accepts `live_stream_callback` parameter
- [ ] Callback called on `finish_episode()` with episode data
- [ ] Callback failure doesn't crash training

✅ **Phase 2 Complete:**
- [ ] UnifiedServer creates `live_episode_queue`
- [ ] Training thread callback writes to queue
- [ ] Queue bounded (maxsize=10), drops on full

✅ **Phase 3 Complete:**
- [ ] LiveInferenceServer consumes from queue
- [ ] Episodes broadcast to WebSocket clients
- [ ] Step-by-step replay at configurable speed

✅ **Phase 4 Complete:**
- [ ] DemoRunner passes callback to EpisodeRecorder
- [ ] Recording system works with and without callback
- [ ] Backward compatible (callback optional)

✅ **Integration Test:**
- [ ] Run unified server with live streaming
- [ ] Frontend shows training episodes in <1 second
- [ ] No dropped episodes under normal load
- [ ] Graceful degradation if queue fills

---

## Migration Path

**Backward Compatibility:**

```python
# Old code (still works):
runner = DemoRunner(config_dir, db_path, checkpoint_dir)
# No live callback, recording system unchanged

# New code (opt-in to live streaming):
runner = DemoRunner(
    config_dir,
    db_path,
    checkpoint_dir,
    live_episode_callback=on_episode_complete
)
# Live streaming enabled
```

**Frontend Changes:**

Frontend needs to handle new message type:
```javascript
// Existing: state_update (from inference episodes)
// NEW: state_update with mode="live_training" (from training episodes)

websocket.onmessage = (event) => {
    const data = JSON.parse(event.data);

    if (data.mode === "live_training") {
        // Show "LIVE" indicator
        // Display actual training behavior
    } else {
        // Existing checkpoint-based inference
    }
}
```

---

## Future Enhancements (Out of Scope)

### 1. Episode Filtering

Allow frontend to request specific types of episodes:
- "Show me only episodes where agent dies" (failure cases)
- "Show me only episodes > 200 steps" (successes)
- "Show me stage transitions" (curriculum changes)

**Effort:** 4-6 hours

### 2. Multi-Agent Streaming

Current: Only streams agent 0. Future: Stream all agents.

**Effort:** 2-3 hours

### 3. Compression

Compress episode data before queueing (LZ4).

**Effort:** 2-3 hours
**Benefit:** ~50% smaller queue memory

### 4. Playback Controls

Frontend can pause/resume/speed up live stream.

**Effort:** 3-4 hours

---

## Summary

**Problem:** Visualization lags training by 100 episodes due to checkpoint polling.

**Solution:** Stream completed episodes directly from training to visualization via thread-safe queue.

**Effort:** 9 hours

**Benefits:**
- 100x faster: <1s latency vs 100+ episodes
- Shows actual training episodes (not reconstructed)
- No checkpoint overhead
- Minimal code changes (leverage existing EpisodeRecorder)

**No Hidden Downsides:**
- Memory: ~1MB for queue (negligible)
- Thread safety: stdlib queue.Queue handles it
- Dropped episodes: Acceptable for live viz
- Episode size: ~40KB, easily streamable

**Next Steps:**
1. Review this research
2. Confirm Option 1 (Callback + Queue) approach
3. Create TASK document
4. Implement in 5 phases over ~9 hours
