# Episode Replay Recording System - Design Document

**Date**: 2025-11-03
**Status**: Design Complete, Ready for Implementation
**Author**: Claude (via brainstorming session)

---

## Executive Summary

This document describes the design of an **Episode Replay Recording System** for the Hamlet/Townlet training environment. The system captures agent episodes during training, stores them efficiently on disk, and provides playback capabilities through both the web UI and headless video export.

### Key Features

- **Selective Recording**: Captures episodes based on configurable criteria (periodic, stage transitions, performance extremes, stage boundaries)
- **Async Architecture**: Non-blocking queue-based recording with <5% training overhead
- **Multi-Format Playback**: Web UI replay and MP4 video export for YouTube streaming
- **Efficient Storage**: ~20 MB per 1,000 episodes using MessagePack + LZ4 compression
- **TDD-Ready**: Designed for test-driven development with clear component boundaries

### Motivation

**Problem**: Training runs 200× faster than visualization (1,000 steps/sec vs 5 steps/sec). While watching one episode, the model trains for 200 episodes and evolves significantly. This makes live streaming from training impractical - viewers would be watching obsolete model behavior.

**Solution**: Record actual training episodes to disk, then replay them at human-watchable speeds. This enables:

- Watching exact episodes that shaped the agent's learning
- Creating YouTube videos from recorded episodes
- Debugging training issues by replaying problematic episodes
- Analyzing curriculum progression systematically

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Data Structures](#data-structures)
3. [Recording System](#recording-system)
4. [Storage Format](#storage-format)
5. [Recording Criteria](#recording-criteria)
6. [Curriculum API Extension](#curriculum-api-extension)
7. [Playback System](#playback-system)
8. [Video Export](#video-export)
9. [Configuration](#configuration)
10. [Implementation Plan](#implementation-plan)
11. [Testing Strategy](#testing-strategy)
12. [Open Questions](#open-questions)

---

## Architecture Overview

### System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                        TRAINING PROCESS                          │
│                                                                   │
│  ┌────────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │   Runner   │───▶│   Recorder   │───▶│  Queue (bounded) │   │
│  │ (main loop)│    │  (capture)   │    │   (lock-free)    │   │
│  └────────────┘    └──────────────┘    └──────────────────┘   │
│                                                 │                │
└─────────────────────────────────────────────────┼────────────────┘
                                                  │
                                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                     BACKGROUND WRITER THREAD                     │
│                                                                   │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐ │
│  │  Batch Pull  │───▶│   Criteria   │───▶│     Storage      │ │
│  │ (from queue) │    │  Evaluator   │    │ (disk + SQLite)  │ │
│  └──────────────┘    └──────────────┘    └──────────────────┘ │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │   PLAYBACK & ANALYSIS        │
                    │                              │
                    │  • Inference Server (web)    │
                    │  • Video Renderer (headless) │
                    │  • Python API (analysis)     │
                    └──────────────────────────────┘
```

### Data Flow

#### Hot Path (Training Loop - microseconds overhead)

```python
# After each environment step:
agent_state = self.population.step_population(self.env)

# Non-blocking capture:
self.recorder.record_step(
    step=current_step,
    positions=self.env.positions[0].clone(),      # [2] - x, y
    meters=self.env.meters[0].clone(),            # [8] - all meters
    action=agent_state.actions[0].item(),         # int
    reward=agent_state.rewards[0].item(),         # float
    intrinsic_reward=agent_state.intrinsic_rewards[0].item(),
    done=agent_state.dones[0].item(),             # bool
    q_values=q_values if available else None,     # [6] - optional
)
```

#### Background Writer (milliseconds overhead)

```python
while running:
    # Pull batch from queue (non-blocking)
    batch = queue.pull_batch(max_size=100, timeout=0.1)

    # Buffer until episode boundary
    for item in batch:
        if item.is_episode_end:
            # Evaluate recording criteria
            should_record = evaluator.should_record(episode_metadata)

            if should_record:
                # Serialize → compress → write
                data = msgpack.packb(episode_buffer)
                compressed = lz4.frame.compress(data)
                write_file(f"episode_{episode_id:06d}.msgpack.lz4", compressed)
                db.insert_recording(episode_id, metadata, path)

            episode_buffer.clear()
```

### Design Rationale

**Why async queue approach?**

- Training loop does cheap clones (no I/O blocking)
- Background thread handles expensive work (compression, disk writes)
- Bounded queue prevents memory explosion if writer can't keep up
- Graceful degradation: drops oldest frames under extreme load

**Why not synchronous writes?**

- Disk I/O blocks training (5-10% overhead → ~10-20% in practice)
- NFS or slow disks would cause significant slowdown
- Harder to add features like cloud upload or streaming later

**Why not in-memory circular buffer?**

- Requires ~200 MB RAM for 1,000 episodes
- Data loss on crash (episodes not yet flushed)
- More complex lifecycle management
- User accepts 5-10% overhead, so simplicity wins

---

## Data Structures

### In-Memory Representations

```python
@dataclass(frozen=True, slots=True)
class RecordedStep:
    """Single step of episode recording.

    Optimized for:
    - Fast clone from GPU tensors
    - Minimal memory footprint (~100 bytes/step)
    - Lock-free queue compatibility
    """
    step: int                          # Step number within episode
    position: tuple[int, int]          # Agent (x, y)
    meters: tuple[float, ...]          # 8 meters, normalized [0,1]
    action: int                        # Action taken (0-5)
    reward: float                      # Extrinsic reward
    intrinsic_reward: float            # RND novelty reward
    done: bool                         # Terminal state
    q_values: tuple[float, ...] | None # Optional Q-values for all actions

    # Optional temporal mechanics (Level 2.5+)
    time_of_day: int | None = None
    interaction_progress: float | None = None


@dataclass(frozen=True, slots=True)
class EpisodeMetadata:
    """Episode-level metadata for recording decisions."""
    episode_id: int
    survival_steps: int
    total_reward: float
    extrinsic_reward: float
    intrinsic_reward: float
    curriculum_stage: int
    epsilon: float
    intrinsic_weight: float
    timestamp: float

    # Affordance context
    affordance_layout: dict[str, tuple[int, int]]  # name → (x, y)
    affordance_visits: dict[str, int]              # name → count


@dataclass
class EpisodeEndMarker:
    """Sentinel value marking episode boundary in queue."""
    metadata: EpisodeMetadata
```

### Memory Footprint

Per-step data (worst case):

- Position: 16 bytes (2× int64)
- Meters: 64 bytes (8× float64)
- Action: 8 bytes (int64)
- Rewards: 16 bytes (2× float64)
- Done: 1 byte (bool)
- Q-values: 48 bytes (6× float64) - optional
- Temporal: 16 bytes (int + float) - optional

**Total**: ~100-150 bytes/step (without compression)

500-step episode = ~50-75 KB uncompressed

---

## Recording System

### Core Components

#### 1. EpisodeRecorder (main interface)

```python
class EpisodeRecorder:
    """Main interface for episode recording.

    Thread-safe, non-blocking capture of episode data.
    """

    def __init__(
        self,
        config: dict,
        output_dir: Path,
        database: DemoDatabase,
        curriculum: AdversarialCurriculum,
    ):
        self.config = config
        self.output_dir = output_dir
        self.database = database
        self.curriculum = curriculum

        # Queue for passing data to writer thread
        max_queue_size = config.get("max_queue_size", 1000)
        self.queue: queue.Queue = queue.Queue(maxsize=max_queue_size)

        # Writer thread
        self.writer = RecordingWriter(
            queue=self.queue,
            config=config,
            output_dir=output_dir,
            database=database,
            curriculum=curriculum,
        )
        self.writer_thread = threading.Thread(
            target=self.writer.writer_loop,
            daemon=True,
        )
        self.writer_thread.start()

    def record_step(
        self,
        step: int,
        positions: torch.Tensor,
        meters: torch.Tensor,
        action: int,
        reward: float,
        intrinsic_reward: float,
        done: bool,
        q_values: torch.Tensor | None = None,
        time_of_day: int | None = None,
        interaction_progress: float | None = None,
    ):
        """Record a single step (non-blocking).

        Clones tensors to prevent training loop from blocking.
        """
        recorded_step = RecordedStep(
            step=step,
            position=(positions[0].item(), positions[1].item()),
            meters=tuple(meters.tolist()),
            action=action,
            reward=reward,
            intrinsic_reward=intrinsic_reward,
            done=done,
            q_values=tuple(q_values.tolist()) if q_values is not None else None,
            time_of_day=time_of_day,
            interaction_progress=interaction_progress,
        )

        try:
            self.queue.put_nowait(recorded_step)
        except queue.Full:
            # Drop frame if queue is full (graceful degradation)
            logger.warning("Recording queue full, dropping frame")

    def finish_episode(self, metadata: EpisodeMetadata):
        """Mark episode boundary (non-blocking)."""
        marker = EpisodeEndMarker(metadata=metadata)
        try:
            self.queue.put_nowait(marker)
        except queue.Full:
            logger.error("Recording queue full, episode metadata lost")

    def shutdown(self):
        """Graceful shutdown: drain queue and stop writer thread."""
        self.writer.stop()
        self.writer_thread.join(timeout=10.0)
```

#### 2. RecordingWriter (background thread)

```python
class RecordingWriter:
    """Background thread for writing episode recordings."""

    def __init__(
        self,
        queue: queue.Queue,
        config: dict,
        output_dir: Path,
        database: DemoDatabase,
        curriculum: AdversarialCurriculum,
    ):
        self.queue = queue
        self.config = config
        self.output_dir = output_dir
        self.database = database
        self.criteria = RecordingCriteria(config, curriculum)

        self.episode_buffer: list[RecordedStep] = []
        self.running = True

    def writer_loop(self):
        """Main writer thread loop."""
        while self.running:
            try:
                item = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if isinstance(item, RecordedStep):
                self.episode_buffer.append(item)

            elif isinstance(item, EpisodeEndMarker):
                # Evaluate recording criteria
                should_record, reason = self.criteria.should_record(item.metadata)

                if should_record:
                    self._write_episode(item.metadata, reason)
                    logger.info(
                        f"Recorded episode {item.metadata.episode_id} "
                        f"(reason: {reason}, {len(self.episode_buffer)} steps)"
                    )

                # Clear buffer regardless
                self.episode_buffer.clear()

    def _write_episode(self, metadata: EpisodeMetadata, reason: str):
        """Serialize, compress, and write episode to disk."""
        # Build episode data structure
        episode_data = {
            "version": 1,
            "metadata": asdict(metadata),
            "steps": [asdict(step) for step in self.episode_buffer],
            "affordances": metadata.affordance_layout,
        }

        # Serialize with msgpack
        serialized = msgpack.packb(episode_data, use_bin_type=True)

        # Compress with LZ4
        compressed = lz4.frame.compress(serialized, compression_level=0)

        # Write to file
        episode_id = metadata.episode_id
        file_path = self.output_dir / f"episode_{episode_id:06d}.msgpack.lz4"
        file_path.write_bytes(compressed)

        # Index in database
        self.database.insert_recording(
            episode_id=episode_id,
            file_path=str(file_path.relative_to(self.output_dir.parent)),
            metadata=metadata,
            reason=reason,
            file_size=len(serialized),
            compressed_size=len(compressed),
        )

    def stop(self):
        """Signal writer thread to stop."""
        self.running = False
```

---

## Storage Format

### On-Disk Format (MessagePack + LZ4)

```python
# episode_012345.msgpack.lz4
{
    "version": 1,
    "metadata": {
        "episode_id": 12345,
        "survival_steps": 487,
        "total_reward": 423.7,
        "extrinsic_reward": 410.2,
        "intrinsic_reward": 13.5,
        "curriculum_stage": 3,
        "epsilon": 0.15,
        "intrinsic_weight": 0.3,
        "timestamp": 1699123456.78,
        "affordance_layout": {
            "Bed": [2, 3],
            "Hospital": [5, 1],
            # ... more affordances
        },
        "affordance_visits": {
            "Bed": 15,
            "Hospital": 2,
            # ... more visits
        }
    },
    "steps": [
        {
            "step": 0,
            "position": [3, 5],
            "meters": [1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0],
            "action": 4,  # INTERACT
            "reward": 1.0,
            "intrinsic_reward": 0.15,
            "done": false,
            "q_values": [0.8, 0.7, 0.9, 0.6, 1.2, 0.5],
            # Optional temporal fields:
            "time_of_day": 12,
            "interaction_progress": 0.33,
        },
        # ... more steps
    ],
    "affordances": {  # Redundant with metadata, but convenient
        "Bed": [2, 3],
        "Hospital": [5, 1],
        # ...
    }
}
```

### Compression Characteristics

- **Uncompressed**: ~100-200 bytes/step → 50-100 KB/episode (500 steps)
- **LZ4 compression**: ~60-70% reduction → **15-30 KB/episode**
- **MessagePack overhead**: ~5% vs raw binary

### Database Schema Extension

```sql
CREATE TABLE episode_recordings (
    episode_id INTEGER PRIMARY KEY,
    file_path TEXT NOT NULL,              -- Relative path to .msgpack.lz4
    timestamp REAL NOT NULL,

    -- Episode metadata (for queries without loading file)
    survival_steps INTEGER NOT NULL,
    total_reward REAL NOT NULL,
    extrinsic_reward REAL NOT NULL,
    intrinsic_reward REAL NOT NULL,
    curriculum_stage INTEGER NOT NULL,
    epsilon REAL NOT NULL,
    intrinsic_weight REAL NOT NULL,

    -- Recording metadata
    recording_reason TEXT NOT NULL,       -- "periodic", "stage_transition", etc.
    file_size_bytes INTEGER,              -- Uncompressed size
    compressed_size_bytes INTEGER,        -- Compressed size

    FOREIGN KEY (episode_id) REFERENCES episodes(id)
);

CREATE INDEX idx_recordings_stage ON episode_recordings(curriculum_stage);
CREATE INDEX idx_recordings_reward ON episode_recordings(total_reward DESC);
CREATE INDEX idx_recordings_reason ON episode_recordings(recording_reason);
CREATE INDEX idx_recordings_timestamp ON episode_recordings(timestamp);
```

### Storage Estimates

**10,000 episodes, 10% recorded (1,000 episodes):**

- Average episode: 450 steps
- Average compressed size: 20 KB
- **Total episode data**: 1,000 × 20 KB = **20 MB**
- **SQLite metadata**: ~2 MB
- **Total new storage**: **~22 MB** for 10K episodes

---

## Recording Criteria

### Configuration (YAML)

```yaml
recording:
  enabled: true
  output_dir: "recordings"  # Relative to run output dir

  # Recording criteria (all are OR'd together)
  criteria:
    periodic:
      enabled: true
      interval: 100  # Record every 100th episode

    stage_transitions:
      enabled: true
      record_before: 5   # Record last 5 episodes before transition
      record_after: 10   # Record first 10 episodes after transition

    performance:
      enabled: true
      top_percent: 1.0      # Top 1% by total reward
      bottom_percent: 1.0   # Bottom 1% (failures)
      window: 100           # Evaluate over last 100 episodes

    stage_boundaries:
      enabled: true
      first_n: 10    # First 10 episodes at each stage
      last_n: 10     # Last 10 episodes at each stage (uses curriculum API)

  # Storage settings
  compression: "lz4"  # "lz4", "none", or "zstd"
  include_q_values: true  # Store Q-values (larger files, useful for analysis)
  max_queue_size: 1000    # Bounded queue size (graceful degradation)
```

### Criteria Evaluator

```python
class RecordingCriteria:
    """Evaluates whether an episode should be recorded.

    Supports multiple criteria evaluated in OR fashion:
    - Periodic: Every Nth episode
    - Stage transitions: Before/after curriculum changes
    - Performance: Top/bottom percentiles
    - Stage boundaries: First/last N episodes per stage
    """

    def __init__(self, config: dict, curriculum: AdversarialCurriculum):
        self.config = config
        self.curriculum = curriculum
        self.episode_history: deque[EpisodeMetadata] = deque(maxlen=1000)
        self.stage_episode_counts: dict[int, int] = defaultdict(int)
        self.last_stage: int | None = None
        self.transition_episodes: set[int] = set()

    def should_record(self, metadata: EpisodeMetadata) -> tuple[bool, str]:
        """Determine if episode should be recorded.

        Returns:
            (should_record, reason) - reason for logging/debugging
        """
        episode_id = metadata.episode_id
        stage = metadata.curriculum_stage

        # Track stage transitions
        if self.last_stage is not None and stage != self.last_stage:
            self._mark_transition(episode_id)
        self.last_stage = stage

        # Update history
        self.episode_history.append(metadata)
        self.stage_episode_counts[stage] += 1

        # Evaluate criteria (short-circuit on first match)
        criteria = self.config.get("criteria", {})

        # 1. Periodic recording
        if criteria.get("periodic", {}).get("enabled", False):
            interval = criteria["periodic"]["interval"]
            if episode_id % interval == 0:
                return True, f"periodic_{interval}"

        # 2. Stage transition episodes
        if criteria.get("stage_transitions", {}).get("enabled", False):
            before = criteria["stage_transitions"]["record_before"]
            after = criteria["stage_transitions"]["record_after"]

            for trans_ep in self.transition_episodes:
                if trans_ep - before <= episode_id < trans_ep:
                    return True, f"before_transition_{trans_ep}"
                if trans_ep <= episode_id < trans_ep + after:
                    return True, f"after_transition_{trans_ep}"

        # 3. Performance-based (top/bottom percentiles)
        if criteria.get("performance", {}).get("enabled", False):
            window = criteria["performance"]["window"]
            top_pct = criteria["performance"]["top_percent"]
            bottom_pct = criteria["performance"]["bottom_percent"]

            if len(self.episode_history) >= window:
                rewards = [ep.total_reward for ep in self.episode_history]
                current_reward = metadata.total_reward

                top_threshold = np.percentile(rewards, 100 - top_pct)
                if current_reward >= top_threshold:
                    return True, f"top_{top_pct}pct"

                bottom_threshold = np.percentile(rewards, bottom_pct)
                if current_reward <= bottom_threshold:
                    return True, f"bottom_{bottom_pct}pct"

        # 4. Stage boundaries (first/last N episodes per stage)
        if criteria.get("stage_boundaries", {}).get("enabled", False):
            first_n = criteria["stage_boundaries"]["first_n"]
            last_n = criteria["stage_boundaries"]["last_n"]

            stage_count = self.stage_episode_counts[stage]

            # First N episodes at this stage
            if stage_count <= first_n:
                return True, f"stage_{stage}_first_{stage_count}"

            # Last N episodes (using curriculum API)
            stage_info = self.curriculum.get_stage_info(agent_idx=0)
            if stage_info["likely_transition_soon"]:
                return True, f"stage_{stage}_pre_transition"

        return False, "no_match"

    def _mark_transition(self, episode_id: int):
        """Mark an episode as a stage transition point."""
        self.transition_episodes.add(episode_id)
        logger.info(f"Stage transition detected at episode {episode_id}")
```

---

## Curriculum API Extension

### Problem

To record "last N episodes before stage transition," we need to predict when a transition is imminent. The curriculum makes this decision internally but doesn't expose it.

### Solution: Stage Info API

**Add to `AdversarialCurriculum` class:**

```python
class AdversarialCurriculum:
    """Adversarial curriculum with stage boundary prediction."""

    def get_stage_info(self, agent_idx: int = 0) -> dict:
        """Get detailed stage information for an agent.

        Returns:
            {
                "current_stage": int,
                "episodes_at_stage": int,
                "min_episodes_required": int,
                "can_transition": bool,
                "survival_rate": float,
                "survival_threshold_advance": float,
                "survival_threshold_retreat": float,
                "entropy": float,
                "entropy_gate": float,
                "likely_transition_soon": bool,  # Heuristic prediction
            }
        """
        stage = self.agent_stages[agent_idx].item()
        tracker = self.performance_trackers[agent_idx]

        # Calculate survival rate
        recent_survivals = tracker.survival_history[-self.survival_window:]
        survival_rate = sum(recent_survivals) / len(recent_survivals) if recent_survivals else 0.0

        # Calculate action entropy
        recent_entropy = tracker.entropy_history[-self.survival_window:]
        avg_entropy = sum(recent_entropy) / len(recent_entropy) if recent_entropy else 0.0

        # Check minimum episodes requirement
        episodes_at_stage = tracker.episodes_at_current_stage
        can_transition = episodes_at_stage >= self.min_steps_at_stage

        # Heuristic: likely to transition soon if within 5% of threshold
        likely_advance = (
            can_transition
            and survival_rate >= self.survival_advance_threshold * 0.95
            and avg_entropy >= self.entropy_gate
        )
        likely_retreat = (
            can_transition
            and survival_rate <= self.survival_retreat_threshold * 1.05
        )

        return {
            "current_stage": int(stage),
            "episodes_at_stage": episodes_at_stage,
            "min_episodes_required": self.min_steps_at_stage,
            "can_transition": can_transition,
            "survival_rate": survival_rate,
            "survival_threshold_advance": self.survival_advance_threshold,
            "survival_threshold_retreat": self.survival_retreat_threshold,
            "entropy": avg_entropy,
            "entropy_gate": self.entropy_gate,
            "likely_transition_soon": likely_advance or likely_retreat,
        }
```

### PerformanceTracker Update

**Add episode counting:**

```python
@dataclass
class PerformanceTracker:
    """Track performance for curriculum decisions."""
    survival_history: list[bool]
    entropy_history: list[float]
    episodes_at_current_stage: int = 0  # NEW

    def reset_stage(self):
        """Reset tracking when stage changes."""
        self.survival_history.clear()
        self.entropy_history.clear()
        self.episodes_at_current_stage = 0  # NEW

    def record_episode(self, survived: bool, entropy: float):
        """Record episode outcome."""
        self.survival_history.append(survived)
        self.entropy_history.append(entropy)
        self.episodes_at_current_stage += 1  # NEW
```

---

## Playback System

### Inference Server Extension

**Add replay mode to `LiveInferenceServer`:**

```python
class LiveInferenceServer:
    """WebSocket server for live training and replay playback."""

    def __init__(self, ...):
        # ... existing init ...
        self.replay_mode: bool = False
        self.replay_data: dict | None = None
        self.replay_step_index: int = 0

    async def load_replay(self, episode_id: int) -> bool:
        """Load episode recording for playback."""
        # Query database
        recording = self.database.get_recording(episode_id)
        if not recording:
            return False

        # Load and decompress
        file_path = Path(recording["file_path"])
        compressed = file_path.read_bytes()
        decompressed = lz4.frame.decompress(compressed)
        episode_data = msgpack.unpackb(decompressed, raw=False)

        # Store replay data
        self.replay_mode = True
        self.replay_data = episode_data
        self.replay_step_index = 0

        logger.info(f"Loaded replay for episode {episode_id}")
        return True

    async def step_replay(self) -> dict | None:
        """Get next step in replay (same format as live inference)."""
        if not self.replay_mode or self.replay_data is None:
            return None

        steps = self.replay_data["steps"]
        if self.replay_step_index >= len(steps):
            return None  # Replay finished

        step_data = steps[self.replay_step_index]
        metadata = self.replay_data["metadata"]

        # Build state update (SAME FORMAT as live inference)
        update = {
            "type": "state_update",
            "mode": "replay",  # NEW: distinguish replay from live
            "episode_id": metadata["episode_id"],
            "step": step_data["step"],
            "cumulative_reward": sum(s["reward"] for s in steps[:self.replay_step_index + 1]),
            "grid": {
                "width": self.env.grid_size,
                "height": self.env.grid_size,
                "agents": [{
                    "id": "agent_0",
                    "x": step_data["position"][0],
                    "y": step_data["position"][1],
                    "color": "blue",
                    "last_action": step_data["action"],
                }],
                "affordances": [
                    {"type": name, "x": pos[0], "y": pos[1]}
                    for name, pos in self.replay_data["affordances"].items()
                ],
            },
            "agent_meters": {
                "agent_0": {"meters": {...}}  # 8 meters
            },
            "q_values": step_data.get("q_values"),
            "replay_metadata": {
                "total_steps": len(steps),
                "survival_steps": metadata["survival_steps"],
                "total_reward": metadata["total_reward"],
                "curriculum_stage": metadata["curriculum_stage"],
            },
        }

        # Add temporal mechanics if present
        if "time_of_day" in step_data:
            update["temporal"] = {
                "time_of_day": step_data["time_of_day"],
                "interaction_progress": step_data.get("interaction_progress", 0.0),
            }

        self.replay_step_index += 1
        return update
```

### WebSocket Protocol Extension

```python
# Client → Server
{
    "type": "load_replay",
    "episode_id": 12345
}

{
    "type": "list_recordings",
    "filters": {
        "stage": 3,              # Optional
        "reason": "periodic_100", # Optional
        "min_reward": 400.0,     # Optional
    }
}

{
    "type": "replay_control",
    "action": "play" | "pause" | "step" | "seek",
    "seek_step": 123  # For seek action
}

# Server → Client
{
    "type": "recordings_list",
    "recordings": [
        {
            "episode_id": 100,
            "survival_steps": 487,
            "total_reward": 456.3,
            "curriculum_stage": 3,
            "recording_reason": "periodic_100",
            "timestamp": 1699123456.78,
        },
        # ... more recordings
    ]
}

{
    "type": "replay_loaded",
    "episode_id": 12345,
    "metadata": {...},
    "total_steps": 487
}

{
    "type": "replay_finished",
    "episode_id": 12345
}
```

### Database Query Methods

```python
class DemoDatabase:
    """SQLite database for training metrics and recordings."""

    def insert_recording(
        self,
        episode_id: int,
        file_path: str,
        metadata: EpisodeMetadata,
        reason: str,
        file_size: int,
        compressed_size: int,
    ):
        """Insert recording metadata."""
        self.conn.execute(
            """
            INSERT INTO episode_recordings (
                episode_id, file_path, timestamp,
                survival_steps, total_reward, extrinsic_reward, intrinsic_reward,
                curriculum_stage, epsilon, intrinsic_weight,
                recording_reason, file_size_bytes, compressed_size_bytes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                episode_id, file_path, metadata.timestamp,
                metadata.survival_steps, metadata.total_reward,
                metadata.extrinsic_reward, metadata.intrinsic_reward,
                metadata.curriculum_stage, metadata.epsilon, metadata.intrinsic_weight,
                reason, file_size, compressed_size,
            )
        )
        self.conn.commit()

    def get_recording(self, episode_id: int) -> dict | None:
        """Get recording metadata by episode ID."""
        row = self.conn.execute(
            "SELECT * FROM episode_recordings WHERE episode_id = ?",
            (episode_id,)
        ).fetchone()
        return dict(row) if row else None

    def list_recordings(
        self,
        stage: int | None = None,
        reason: str | None = None,
        min_reward: float | None = None,
        max_reward: float | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Query recordings with optional filters."""
        query = "SELECT * FROM episode_recordings WHERE 1=1"
        params = []

        if stage is not None:
            query += " AND curriculum_stage = ?"
            params.append(stage)

        if reason is not None:
            query += " AND recording_reason = ?"
            params.append(reason)

        if min_reward is not None:
            query += " AND total_reward >= ?"
            params.append(min_reward)

        if max_reward is not None:
            query += " AND total_reward <= ?"
            params.append(max_reward)

        query += " ORDER BY episode_id DESC LIMIT ?"
        params.append(limit)

        rows = self.conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]
```

### Frontend Changes

**Minimal changes needed** - existing visualization already handles state updates!

**Add:**

1. Replay controls (play/pause/seek buttons)
2. Recordings browser (dropdown or list)
3. Mode indicator ("REPLAY" badge)

**Reuse:**

- Grid visualization ✅
- Meter displays ✅
- Q-value display ✅
- Temporal indicators ✅

---

## Video Export

### Frame Renderer

**Core rendering module (`src/townlet/recording/video_renderer.py`):**

```python
class EpisodeVideoRenderer:
    """Renders episode recordings to video frames.

    Matches web UI visualization:
    - Grid with agents and affordances
    - Meter bars for all 8 meters
    - Episode metadata (step, reward, stage)
    - Temporal indicators (if applicable)
    """

    def __init__(
        self,
        grid_size: int,
        dpi: int = 100,          # 100 = 1600×900, 150 = 2400×1350
        figsize: tuple[float, float] = (16, 9),
        style: str = "dark",
    ):
        self.grid_size = grid_size
        self.dpi = dpi
        self.figsize = figsize
        self.style = style

        # Color schemes (match frontend)
        if style == "dark":
            self.bg_color = "#1a1a1a"
            self.grid_color = "#333333"
            self.text_color = "#ffffff"
            self.agent_color = "#4dabf7"
        else:
            self.bg_color = "#ffffff"
            self.grid_color = "#cccccc"
            self.text_color = "#000000"
            self.agent_color = "#1971c2"

        self.affordance_colors = {
            "Bed": "#8b5cf6", "Bathroom": "#3b82f6", "Kitchen": "#10b981",
            "Job": "#f59e0b", "Hospital": "#ef4444", "Gym": "#ec4899",
            "Bar": "#14b8a6", "Park": "#22c55e",
        }

    def render_frame(self, step_data: dict, metadata: dict) -> np.ndarray:
        """Render a single frame.

        Returns:
            RGB image array (H, W, 3) with dtype=uint8
        """
        fig = Figure(figsize=self.figsize, dpi=self.dpi, facecolor=self.bg_color)
        canvas = FigureCanvasAgg(fig)

        # Layout: [grid | meters | metadata]
        gs = fig.add_gridspec(1, 3, width_ratios=[2, 1, 1])

        self._render_grid(fig.add_subplot(gs[0, 0]), step_data)
        self._render_meters(fig.add_subplot(gs[0, 1]), step_data["meters"])
        self._render_metadata(fig.add_subplot(gs[0, 2]), step_data, metadata)

        # Convert to RGB array
        canvas.draw()
        width, height = canvas.get_width_height()
        image = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        image = image.reshape((height, width, 4))[:, :, :3]  # Drop alpha

        plt.close(fig)
        return image

    # _render_grid(), _render_meters(), _render_metadata() implementations...
```

### Video Export CLI Tool

```python
#!/usr/bin/env python3
"""Export episode recording to video."""

def export_episode_to_video(
    episode_id: int,
    database_path: Path,
    output_path: Path,
    fps: int = 30,
    speed: float = 1.0,
    dpi: int = 100,
):
    """Export episode recording to MP4 video.

    Args:
        episode_id: Episode ID to export
        database_path: Path to training database
        output_path: Output video file path
        fps: Frames per second (30 for smooth, 10 for slower)
        speed: Playback speed multiplier (1.0 = real-time)
        dpi: Video resolution (100 = 1600×900, 150 = 2400×1350)
    """
    # Load recording
    db = DemoDatabase(database_path)
    recording = db.get_recording(episode_id)
    if not recording:
        raise ValueError(f"No recording found for episode {episode_id}")

    # Load and decompress
    file_path = Path(recording["file_path"])
    compressed = file_path.read_bytes()
    decompressed = lz4.frame.decompress(compressed)
    episode_data = msgpack.unpackb(decompressed, raw=False)

    # Initialize renderer
    renderer = EpisodeVideoRenderer(grid_size=8, dpi=dpi, style="dark")

    # Render frames to temporary directory
    frames_dir = output_path.parent / f"frames_{episode_id}"
    frames_dir.mkdir(exist_ok=True)

    print(f"Rendering {len(episode_data['steps'])} frames...")
    for i, step_data in enumerate(tqdm(episode_data["steps"])):
        frame = renderer.render_frame(step_data, episode_data["metadata"])
        Image.fromarray(frame).save(frames_dir / f"frame_{i:05d}.png")

    # Encode with ffmpeg
    print(f"Encoding video at {fps} fps...")
    effective_fps = fps * speed
    (
        ffmpeg
        .input(str(frames_dir / "frame_%05d.png"), framerate=effective_fps)
        .output(str(output_path), vcodec="libx264", pix_fmt="yuv420p", crf=18)
        .overwrite_output()
        .run(quiet=True)
    )

    # Cleanup
    for frame_file in frames_dir.glob("*.png"):
        frame_file.unlink()
    frames_dir.rmdir()

    print(f"✅ Video exported to {output_path}")
```

### Usage Examples

```bash
# Export episode 1234 at default settings (30fps, 1080p)
python -m townlet.recording.export_video 1234 \
    --database demo_level2.db \
    --output episode_1234.mp4

# Export at 60fps for YouTube (smoother)
python -m townlet.recording.export_video 1234 \
    --database demo_level2.db \
    --output episode_1234_60fps.mp4 \
    --fps 60

# Export in slow motion (0.5x speed)
python -m townlet.recording.export_video 1234 \
    --database demo_level2.db \
    --output episode_1234_slowmo.mp4 \
    --speed 0.5

# Export at 1620p for high-quality presentation
python -m townlet.recording.export_video 1234 \
    --database demo_level2.db \
    --output episode_1234_hq.mp4 \
    --dpi 150
```

### Batch Export Script

```bash
#!/bin/bash
# Export all top-performing episodes

DATABASE="demo_level2.db"
OUTPUT_DIR="videos"
mkdir -p "$OUTPUT_DIR"

# Export top 10 episodes by reward
sqlite3 "$DATABASE" "SELECT episode_id FROM episode_recordings ORDER BY total_reward DESC LIMIT 10" | \
while read episode_id; do
    echo "Exporting episode $episode_id..."
    python -m townlet.recording.export_video "$episode_id" \
        --database "$DATABASE" \
        --output "$OUTPUT_DIR/episode_${episode_id}.mp4" \
        --fps 30
done

echo "✅ All videos exported to $OUTPUT_DIR"
```

---

## Configuration

### Training Config Extension

**Add to `configs/*/training.yaml`:**

```yaml
recording:
  enabled: true
  output_dir: "recordings"

  criteria:
    periodic:
      enabled: true
      interval: 100

    stage_transitions:
      enabled: true
      record_before: 5
      record_after: 10

    performance:
      enabled: true
      top_percent: 1.0
      bottom_percent: 1.0
      window: 100

    stage_boundaries:
      enabled: true
      first_n: 10
      last_n: 10

  compression: "lz4"
  include_q_values: true
  max_queue_size: 1000
```

### Dependencies

**Add to `pyproject.toml`:**

```toml
[project.optional-dependencies]
recording = [
    "msgpack>=1.0.0",       # Serialization
    "lz4>=4.0.0",           # Compression
    "ffmpeg-python>=0.2.0", # Video export
    # matplotlib and Pillow already in deps
]
```

**System dependency:**

- `ffmpeg` binary (install via `apt install ffmpeg` or `brew install ffmpeg`)

---

## Implementation Plan

### Phase 1: Core Recording Infrastructure

**Files to create:**

- `src/townlet/recording/__init__.py`
- `src/townlet/recording/recorder.py` - `EpisodeRecorder`, `RecordingWriter`
- `src/townlet/recording/data_structures.py` - `RecordedStep`, `EpisodeMetadata`, `EpisodeEndMarker`

**Files to modify:**

- `src/townlet/demo/runner.py` - Integrate recorder
- `src/townlet/demo/database.py` - Add recording schema and methods

**Tests to write:**

- `tests/test_townlet/test_recording/test_recorder.py` - Core recording logic
- `tests/test_townlet/test_recording/test_data_structures.py` - Serialization roundtrips

### Phase 2: Recording Criteria

**Files to create:**

- `src/townlet/recording/criteria.py` - `RecordingCriteria`

**Files to modify:**

- `src/townlet/curriculum/adversarial.py` - Add `get_stage_info()` method
- `src/townlet/curriculum/adversarial.py` - Update `PerformanceTracker`

**Tests to write:**

- `tests/test_townlet/test_recording/test_criteria.py` - Each criterion independently
- `tests/test_townlet/test_curriculum/test_stage_info.py` - Curriculum API

### Phase 3: Playback System

**Files to modify:**

- `src/townlet/demo/live_inference.py` - Add replay mode
- `src/townlet/demo/database.py` - Add `get_recording()`, `list_recordings()`

**Tests to write:**

- `tests/test_townlet/test_recording/test_playback.py` - Replay loading and stepping

### Phase 4: Video Export

**Files to create:**

- `src/townlet/recording/video_renderer.py` - `EpisodeVideoRenderer`
- `src/townlet/recording/export_video.py` - CLI tool

**Scripts to create:**

- `scripts/export_recordings.sh` - Batch export script

**Tests to write:**

- `tests/test_townlet/test_recording/test_video_renderer.py` - Frame rendering
- Manual tests: Export a video and verify it matches web UI

### Phase 5: Integration & Documentation

**Files to modify:**

- `configs/L0_minimal/training.yaml` - Add recording config
- `CLAUDE.md` - Document recording system
- `README.md` - Add usage examples

**Tests to write:**

- `tests/test_townlet/test_recording/test_integration.py` - End-to-end test

---

## Testing Strategy

### Unit Tests

**Core recording:**

- Test queue overflow behavior (graceful degradation)
- Test serialization/deserialization roundtrips
- Test compression/decompression
- Mock database writes

**Criteria:**

- Test each criterion independently with edge cases
- Test OR logic (multiple criteria)
- Test stage transition detection

**Curriculum API:**

- Test `get_stage_info()` with various survival rates
- Test "likely_transition_soon" heuristic
- Test episode counting

**Video renderer:**

- Test frame generation without ffmpeg
- Test color schemes (dark/light)
- Verify frame dimensions match figsize × dpi

### Integration Tests

**End-to-end recording:**

1. Run mini training loop (10 episodes)
2. Configure periodic recording (every 5 episodes)
3. Verify 2 recordings saved
4. Load and replay one recording
5. Verify all steps present

**Video export:**

1. Create synthetic episode data
2. Export to video
3. Verify file exists and has correct duration
4. (Manual) Watch video and verify visually

### Performance Tests

**Recording overhead:**

1. Benchmark training loop with/without recording
2. Verify <5% slowdown
3. Test queue overflow scenario (slow disk writes)

---

## Open Questions

### 1. Frontend Replay UI Design

**Decision needed**: Should replay controls be:

- **Option A**: Separate tab/page in the frontend
- **Option B**: Integrated into main view with mode toggle
- **Option C**: Dedicated replay viewer (separate port)

**Recommendation**: Option B (integrated) - less code duplication, easier to compare live vs replay.

### 2. Video Renderer Style Matching

**Decision needed**: How closely should video frames match the web UI?

- **Option A**: Pixel-perfect match (requires extracting Vue component styles)
- **Option B**: Close approximation (colors/layout similar, details differ)
- **Option C**: Separate "publication" style (optimized for papers/presentations)

**Recommendation**: Option B initially, Option C as stretch goal.

### 3. Q-Values Storage

**Trade-off**: Including Q-values increases file size by ~30-50%.

**Options**:

- Always include (max detail, larger files)
- Configurable (default on)
- Never include (minimal storage)

**Recommendation**: Configurable (default on) - useful for debugging, disable if storage is tight.

### 4. Cloud Storage Integration

**Future consideration**: Should recordings support cloud upload (S3, GCS)?

**Not in scope for initial implementation**, but architecture supports it:

- Writer thread could push to cloud after disk write
- Database stores both local and cloud paths

---

## Dependencies

### Python Packages (New)

- `msgpack>=1.0.0` - Serialization
- `lz4>=4.0.0` - Compression
- `ffmpeg-python>=0.2.0` - Video export wrapper

### Python Packages (Existing)

- `matplotlib` - Frame rendering
- `Pillow` - Image manipulation
- `numpy` - Array operations
- `torch` - Already in deps

### System Dependencies

- `ffmpeg` binary - Video encoding (install via apt/brew)

---

## Success Criteria

### Functional Requirements

- ✅ Records episodes during training with <5% overhead
- ✅ Saves to disk with compression (target: 20 KB/episode)
- ✅ Supports all recording criteria (periodic, transitions, performance, boundaries)
- ✅ Replays episodes through web UI (reuses existing visualization)
- ✅ Exports episodes to MP4 video (1080p @ 30fps)
- ✅ Handles queue overflow gracefully (no crashes)

### Non-Functional Requirements

- ✅ Training overhead: <5% (measured with 1,000 episodes)
- ✅ Storage efficiency: <25 KB/episode compressed
- ✅ Video quality: YouTube-ready (1080p H.264)
- ✅ Code quality: 80%+ test coverage
- ✅ Documentation: README examples + CLAUDE.md integration

---

## Future Enhancements

### Near-Term (Next 3-6 months)

1. **Multi-agent recording**: Extend to population of agents
2. **Python analysis API**: Load episodes for programmatic analysis
3. **Replay speed control**: Adjust playback speed in web UI
4. **Seek functionality**: Jump to specific step in replay

### Long-Term (6-12 months)

1. **Cloud storage**: Auto-upload to S3/GCS
2. **Episode comparison**: Side-by-side replay of two episodes
3. **Highlights generation**: Auto-detect "interesting moments"
4. **Live streaming**: Direct RTMP push to YouTube/Twitch

---

## Conclusion

This design provides a complete episode replay recording system that:

- **Minimizes training impact** (<5% overhead via async queue)
- **Stores efficiently** (~20 KB/episode with LZ4 compression)
- **Enables multiple use cases** (debugging, presentations, YouTube)
- **Reuses existing code** (web UI visualization, database schema)
- **Follows TDD principles** (clear component boundaries, testable)

The system is ready for implementation following the phased plan outlined above.

---

## Appendix: File Structure

```
src/townlet/recording/
├── __init__.py
├── recorder.py               # EpisodeRecorder, RecordingWriter
├── data_structures.py        # RecordedStep, EpisodeMetadata, EpisodeEndMarker
├── criteria.py               # RecordingCriteria
├── video_renderer.py         # EpisodeVideoRenderer
└── export_video.py           # CLI tool for video export

tests/test_townlet/test_recording/
├── test_recorder.py
├── test_data_structures.py
├── test_criteria.py
├── test_playback.py
├── test_video_renderer.py
└── test_integration.py

scripts/
└── export_recordings.sh      # Batch export script

configs/*/training.yaml        # Add recording section

docs/plans/
└── 2025-11-03-episode-replay-recording-design.md  # This document
```
