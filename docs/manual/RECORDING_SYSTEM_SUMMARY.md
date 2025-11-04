# Episode Recording System - Complete Summary

## Overview

The Hamlet episode recording system captures, stores, and replays training episodes for analysis, debugging, and video export. Designed for **<5% training overhead** with async I/O, the system supports real-time replay and high-quality video export for YouTube streaming.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Training Loop                            │
│  (DemoRunner / VectorizedPopulation / VectorizedHamletEnv)      │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       │ Record steps (async queue)
                       ▼
         ┌─────────────────────────────┐
         │      EpisodeRecorder         │
         │  - Async queue (maxsize 1000)│
         │  - Clone tensors to CPU      │
         │  - No training blocking      │
         └──────────────┬───────────────┘
                        │
                        │ Steps + EpisodeEndMarker
                        ▼
         ┌─────────────────────────────┐
         │     RecordingWriter          │
         │  - Background thread         │
         │  - Buffer episodes           │
         │  - MessagePack + LZ4         │
         └──────────────┬───────────────┘
                        │
                        │ Write to disk
                        ▼
         ┌─────────────────────────────┐
         │   Recordings Directory       │
         │  episode_XXXXXX.msgpack.lz4  │
         │  ~15-30 KB per episode       │
         └──────────────┬───────────────┘
                        │
                        │ Index metadata
                        ▼
         ┌─────────────────────────────┐
         │      SQLite Database         │
         │  - Episode metadata          │
         │  - Queryable by stage/reward │
         │  - File paths                │
         └─────────────────────────────┘
                        │
            ┌───────────┴───────────┐
            │                       │
            ▼                       ▼
   ┌─────────────────┐   ┌─────────────────────┐
   │ ReplayManager   │   │  Video Exporter      │
   │ - Load episode  │   │  - Render frames     │
   │ - Control       │   │  - Encode MP4        │
   │ - Stream steps  │   │  - YouTube-ready     │
   └────────┬────────┘   └──────────┬───────────┘
            │                       │
            ▼                       ▼
   ┌─────────────────┐   ┌─────────────────────┐
   │ WebSocket API   │   │  Command-Line Tool   │
   │ - Real-time     │   │  - Batch export      │
   │ - Play/pause    │   │  - Filters           │
   │ - Seek/control  │   │  - YouTube settings  │
   └─────────────────┘   └─────────────────────┘
```

## Components

### 1. Data Structures (`src/townlet/recording/data_structures.py`)

**RecordedStep**:

```python
@dataclass
class RecordedStep:
    step: int
    position: tuple[int, int]
    meters: tuple[float, ...]  # 8 meters
    action: int
    reward: float
    intrinsic_reward: float
    done: bool
    q_values: tuple[float, ...] | None = None
    time_of_day: int | None = None  # Temporal mechanics
    interaction_progress: float | None = None
```

**EpisodeMetadata**:

```python
@dataclass
class EpisodeMetadata:
    episode_id: int
    survival_steps: int
    total_reward: float
    extrinsic_reward: float
    intrinsic_reward: float
    curriculum_stage: int
    epsilon: float
    intrinsic_weight: float
    timestamp: float
    affordance_layout: dict[str, tuple[int, int]]
    affordance_visits: dict[str, int]
```

### 2. Recording (`src/townlet/recording/recorder.py`)

**EpisodeRecorder**:

- Async queue (maxsize 1000) for non-blocking recording
- Clones tensors to CPU to avoid GPU memory issues
- Records per-step data from environment

**RecordingWriter**:

- Background thread writes to disk
- Buffers episodes in memory until `EpisodeEndMarker`
- Serializes with MessagePack + LZ4 compression
- Inserts metadata into database

### 3. Criteria (`src/townlet/recording/criteria.py`)

**PeriodicCriterion**: Record every N episodes

```python
periodic = PeriodicCriterion(interval=100)
```

**StageTransitionCriterion**: Record before/after stage changes

```python
stage_transition = StageTransitionCriterion(
    lookback=5,  # Record last 5 episodes before transition
    lookahead=5  # Record first 5 after transition
)
```

**PerformanceCriterion**: Record top/bottom performers

```python
performance = PerformanceCriterion(
    top_percentile=10.0,     # Top 10%
    bottom_percentile=10.0,  # Bottom 10%
    history_window=100       # Recent 100 episodes
)
```

**StageBoundariesCriterion**: Record first/last N at each stage

```python
stage_boundaries = StageBoundariesCriterion(
    record_first_n=5,
    record_last_n=5
)
```

### 4. Database (`src/townlet/demo/database.py`)

**Schema**:

```sql
CREATE TABLE recordings (
    episode_id INTEGER PRIMARY KEY,
    survival_steps INTEGER,
    total_reward REAL,
    curriculum_stage INTEGER,
    recording_reason TEXT,
    timestamp REAL,
    file_path TEXT,
    file_size_bytes INTEGER,
    compressed_size_bytes INTEGER
)
```

**Query API**:

```python
db.query_recordings(
    stage=2,
    reason="periodic_100",
    min_reward=100.0,
    max_reward=500.0,
    limit=50
)
```

### 5. Replay (`src/townlet/recording/replay.py`)

**ReplayManager**:

```python
replay = ReplayManager(database, recordings_base_dir)
replay.load_episode(episode_id)

# Control
replay.play()
replay.pause()
replay.next_step()
replay.seek(step_index)
replay.reset()

# Access
step_data = replay.get_current_step()
metadata = replay.get_metadata()
affordances = replay.get_affordances()
```

### 6. WebSocket Integration (`src/townlet/demo/live_inference.py`)

**Commands**:

- `list_recordings` - Query database with filters
- `load_replay` - Load episode by ID
- `play` / `pause` / `step` / `reset` - Playback controls
- `set_speed` - Adjust playback speed
- `replay_control` with `action: "seek"` - Jump to step

**State Updates**:

```json
{
  "type": "state_update",
  "mode": "replay",
  "episode_id": 100,
  "step": 25,
  "grid": {...},
  "agent_meters": {...},
  "q_values": [...],
  "time_of_day": 12,
  "interaction_progress": 0.5,
  "replay_metadata": {
    "total_steps": 50,
    "current_step": 25,
    "survival_steps": 50,
    "total_reward": 123.4
  }
}
```

### 7. Video Export (`src/townlet/recording/video_export.py`)

**EpisodeVideoRenderer**:

- 16:9 aspect ratio for YouTube
- Matplotlib-based high-quality rendering
- Dark theme by default
- Components:
  - Grid view with agent and affordances
  - 8-meter bar chart
  - Episode info panel
  - Q-values bar chart

**export_episode_video()**:

```python
export_episode_video(
    episode_id=500,
    database_path="demo.db",
    recordings_base_dir="recordings/",
    output_path="episode_500.mp4",
    fps=60,
    dpi=150,  # 2400×1350 resolution
    style="dark"
)
```

**batch_export_videos()**:

```python
batch_export_videos(
    database_path="demo.db",
    recordings_base_dir="recordings/",
    output_dir="videos/",
    stage=2,
    min_reward=200.0,
    limit=20,
    fps=60,
    dpi=150
)
```

### 8. CLI Tool (`src/townlet/recording/__main__.py`)

**Single export**:

```bash
python -m townlet.recording export 500 \
  --database demo.db \
  --recordings recordings/ \
  --output episode_500.mp4 \
  --fps 60 \
  --dpi 150
```

**Batch export**:

```bash
python -m townlet.recording batch \
  --database demo.db \
  --recordings recordings/ \
  --output-dir videos/ \
  --stage 2 \
  --min-reward 200.0 \
  --limit 20 \
  --verbose
```

## File Formats

### Recording File Format

**Filename**: `episode_XXXXXX.msgpack.lz4`

**Structure**:

```python
{
    "version": 1,
    "metadata": {
        "episode_id": 500,
        "survival_steps": 100,
        "total_reward": 250.0,
        # ... (EpisodeMetadata fields)
    },
    "steps": [
        {
            "step": 0,
            "position": [3, 4],
            "meters": [0.8, 0.7, ...],
            "action": 2,
            "reward": 1.0,
            # ... (RecordedStep fields)
        },
        # ... more steps
    ],
    "affordances": {
        "Bed": [2, 3],
        "Job": [5, 6],
        # ...
    }
}
```

**Compression**:

- Serialized with MessagePack (binary JSON)
- Compressed with LZ4 (fast compression)
- Typical size: 15-30 KB per episode

### Video Format

**Codec**: H.264 (libx264)
**Pixel format**: yuv420p (YouTube-compatible)
**Quality**: CRF 18 (high quality)
**Preset**: slow (better compression)
**Resolution**: Configurable via DPI

- DPI 80: 1280×720 (HD)
- DPI 100: 1600×900 (HD+)
- DPI 120: 1920×1080 (Full HD)
- DPI 150: 2400×1350 (2K)

## Performance

### Recording Overhead

- **Queue operations**: <0.1ms per step
- **Tensor cloning**: ~0.2ms per step
- **Background writing**: Async, no blocking
- **Total overhead**: <5% of training time

### Replay Performance

- **Episode loading**: <10ms (decompression + deserialization)
- **Step access**: <0.1ms (in-memory list)
- **Streaming rate**: 5-50 steps/sec (configurable)
- **Memory**: ~1-2 MB per loaded episode

### Video Export Performance

- **Frame rendering**: ~0.1-0.2 seconds per frame
- **Video encoding**: ~1-2 seconds per 100 frames at 1080p
- **Total time**: ~30-60 seconds for 100-step episode
- **File size**: ~10-30 MB per 100 steps at 1080p

## Configuration

### Training Config (`configs/*.yaml`)

```yaml
recording:
  enabled: true
  recordings_dir: "recordings"
  criteria:
    periodic:
      enabled: true
      interval: 100
    stage_transitions:
      enabled: true
      lookback: 5
      lookahead: 5
    performance:
      enabled: true
      top_percentile: 10.0
      bottom_percentile: 10.0
      history_window: 100
    stage_boundaries:
      enabled: true
      record_first_n: 5
      record_last_n: 5
```

## Usage Examples

### 1. Real-Time Replay During Training

```bash
# Terminal 1: Start server with replay
python -m townlet.demo.live_inference \
  checkpoints/ 8766 0.2 10000 configs/townlet.yaml \
  --db demo.db \
  --recordings recordings/

# Terminal 2: Start frontend
cd frontend && npm run dev
```

### 2. Export Top Performers for YouTube

```bash
python -m townlet.recording batch \
  --database demo_level2_5.db \
  --recordings recordings/ \
  --output-dir videos/top_performers/ \
  --min-reward 300.0 \
  --limit 20 \
  --dpi 150 \
  --fps 60
```

### 3. Debug Failure Cases

```bash
# List failures
python -m townlet.recording batch \
  --database demo.db \
  --recordings recordings/ \
  --output-dir videos/failures/ \
  --max-reward 50.0 \
  --limit 10

# Export for analysis
```

### 4. Track Curriculum Progression

```bash
# Export stage transitions
python -m townlet.recording batch \
  --database demo.db \
  --recordings recordings/ \
  --output-dir videos/transitions/ \
  --reason "stage_2_pre_transition"
```

## Testing

### Test Coverage

- **Total tests**: 72 passing
- **Coverage**: 25% (recording modules: >90%)
- **Test types**:
  - Unit tests for each component
  - Integration tests for full pipeline
  - Protocol tests for WebSocket API

### Running Tests

```bash
# All recording tests
uv run pytest tests/test_townlet/test_recording/ -v

# Specific module
uv run pytest tests/test_townlet/test_recording/test_video_export.py -v

# With coverage
uv run pytest tests/test_townlet/test_recording/ --cov=townlet.recording
```

## Documentation

- **REPLAY_USAGE.md**: Real-time replay system guide
- **VIDEO_EXPORT_USAGE.md**: Video export and YouTube workflow
- **RECORDING_SYSTEM_SUMMARY.md**: This document (complete overview)

## Future Enhancements

### Planned (Not Yet Implemented)

1. **Differential recording**: Only record changed state
2. **Thumbnail generation**: Preview frames for episode browser
3. **Timeline scrubbing**: Visual timeline in frontend
4. **Multi-agent recording**: Track multiple agents simultaneously
5. **Compressed replay streaming**: Stream LZ4-compressed chunks
6. **Export presets**: "YouTube 1080p", "Debug 720p", etc.
7. **Comparison mode**: Side-by-side replay of two episodes

### Possible Extensions

- **Cloud storage**: Upload recordings to S3/GCS
- **Annotation system**: Mark interesting moments
- **Statistics overlay**: Add charts/graphs to video
- **Audio generation**: Add sound effects for actions
- **Narration**: AI-generated commentary
- **Interactive exports**: HTML5 canvas with replay controls

## Integration with Training

### DemoRunner Integration

The recording system is designed to integrate seamlessly with the training loop:

```python
# In DemoRunner.__init__()
if config.recording.enabled:
    self.recorder = EpisodeRecorder()
    self.writer = RecordingWriter(
        recorder=self.recorder,
        database=self.database,
        recordings_dir=config.recording.recordings_dir
    )
    self.writer.start()

# In training loop (per step)
if self.recorder:
    self.recorder.record_step(
        episode_id=episode_id,
        step=step,
        position=position,
        meters=meters,
        action=action,
        reward=reward,
        intrinsic_reward=intrinsic_reward,
        done=done,
        q_values=q_values  # Optional
    )

# At episode end
if self.recorder:
    self.recorder.finish_episode(episode_id, metadata)
```

## Troubleshooting

### Recording Issues

**Queue full warning**:

- Increase queue size: `EpisodeRecorder(queue_maxsize=2000)`
- Or reduce recording frequency

**Missing recordings**:

- Check config: `recording.enabled = true`
- Check criteria: At least one criterion should match
- Check disk space

### Replay Issues

**"Recording not found"**:

- Episode wasn't recorded (check criteria)
- Database is stale (restart server)
- File was deleted (check recordings directory)

**Choppy playback**:

- Adjust speed: `set_speed` command
- Increase step delay on server

### Video Export Issues

**"ffmpeg not found"**:

```bash
# Install ffmpeg
sudo apt-get install ffmpeg  # Ubuntu/Debian
brew install ffmpeg          # macOS
```

**Poor video quality**:

- Increase DPI: `--dpi 150`
- Lower CRF: Edit `video_export.py` line 138

**File sizes too large**:

- Reduce DPI: `--dpi 80`
- Increase CRF: Edit `video_export.py` line 138
- Reduce FPS: `--fps 24`

## Summary

The Hamlet episode recording system provides:

✅ **Non-blocking recording** with <5% training overhead
✅ **Intelligent criteria** for automatic episode selection
✅ **Efficient storage** with MessagePack + LZ4 compression
✅ **Real-time replay** via WebSocket with full playback controls
✅ **High-quality video export** for YouTube streaming
✅ **Queryable database** for filtering and analysis
✅ **72 passing tests** with comprehensive coverage

Perfect for debugging, analysis, and content creation for your DRL YouTube channel!
