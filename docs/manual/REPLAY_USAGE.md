# Episode Replay System - Usage Guide

## Quick Start

### 1. Start Inference Server with Replay Support

```bash
cd /home/john/hamlet

# Start server with replay enabled
python -m townlet.demo.live_inference \
  checkpoints_level2_5 \
  8766 \
  0.2 \
  10000 \
  configs/townlet_level_2_5_temporal.yaml \
  --db demo_level2_5.db \
  --recordings recordings/
```

**Arguments**:
1. `checkpoints_level2_5` - Checkpoint directory (for live inference mode)
2. `8766` - WebSocket port
3. `0.2` - Step delay (5 steps/sec)
4. `10000` - Total episodes
5. `configs/townlet_level_2_5_temporal.yaml` - Training config
6. `--db demo_level2_5.db` - **Database path (enables replay)**
7. `--recordings recordings/` - **Recordings directory**

### 2. Connect Frontend

```bash
cd /home/john/hamlet/frontend
npm run dev
# Open http://localhost:5173
```

## WebSocket Protocol

### List Available Recordings

```javascript
// Request
{
  "type": "list_recordings",
  "filters": {
    "stage": 2,              // Optional: filter by curriculum stage
    "reason": "periodic_100", // Optional: filter by recording reason
    "min_reward": 100.0,      // Optional: minimum reward
    "max_reward": 500.0,      // Optional: maximum reward
    "limit": 50               // Optional: max results (default 100)
  }
}

// Response
{
  "type": "recordings_list",
  "recordings": [
    {
      "episode_id": 100,
      "survival_steps": 50,
      "total_reward": 123.4,
      "curriculum_stage": 2,
      "recording_reason": "periodic_100",
      "timestamp": 1699123456.78,
      "file_path": "recordings/episode_000100.msgpack.lz4",
      "file_size_bytes": 8192,
      "compressed_size_bytes": 5432
    },
    // ... more recordings
  ]
}
```

### Load Replay

```javascript
// Request
{
  "type": "load_replay",
  "episode_id": 100
}

// Response
{
  "type": "replay_loaded",
  "episode_id": 100,
  "metadata": {
    "survival_steps": 50,
    "total_reward": 123.4,
    "curriculum_stage": 2,
    "timestamp": 1699123456.78
  },
  "total_steps": 50
}
```

### Playback Controls

```javascript
// Play
{
  "command": "play"
}

// Pause
{
  "command": "pause"
}

// Single step
{
  "command": "step"
}

// Reset to beginning
{
  "command": "reset"
}

// Seek to specific step
{
  "type": "replay_control",
  "action": "seek",
  "seek_step": 25
}

// Adjust playback speed
{
  "command": "set_speed",
  "speed": 2.0  // 2x speed (10 steps/sec)
}
```

### State Updates

During replay, you'll receive `state_update` messages identical to live inference, with additional `replay_metadata`:

```javascript
{
  "type": "state_update",
  "mode": "replay",  // Distinguishes from live inference
  "episode_id": 100,
  "step": 25,
  "cumulative_reward": 25.0,
  "grid": {
    "width": 8,
    "height": 8,
    "agents": [
      {
        "id": "agent_0",
        "x": 3,
        "y": 4,
        "color": "blue",
        "last_action": 2
      }
    ],
    "affordances": [
      {"type": "Bed", "x": 2, "y": 3},
      {"type": "Job", "x": 5, "y": 6}
    ]
  },
  "agent_meters": {
    "agent_0": {
      "meters": {
        "energy": 0.8,
        "hygiene": 0.7,
        // ... etc
      }
    }
  },
  "q_values": [0.1, 0.2, 0.3, 0.4, 0.5],  // Optional
  "time_of_day": 12,  // Optional (temporal mechanics)
  "interaction_progress": 0.5,  // Optional (temporal mechanics)
  "replay_metadata": {
    "total_steps": 50,
    "current_step": 25,
    "survival_steps": 50,
    "total_reward": 123.4,
    "curriculum_stage": 2
  }
}
```

### Replay Finished

```javascript
{
  "type": "replay_finished",
  "episode_id": 100
}
```

## Common Use Cases

### Investigate Specific Episode

```javascript
// 1. List recordings with filters
ws.send(JSON.stringify({
  type: "list_recordings",
  filters: {
    stage: 2,
    min_reward: 200.0
  }
}));

// 2. Load specific episode
ws.send(JSON.stringify({
  type: "load_replay",
  episode_id: 500
}));

// 3. Play through slowly
ws.send(JSON.stringify({
  command: "set_speed",
  speed: 0.5  // Half speed
}));

ws.send(JSON.stringify({
  command: "play"
}));
```

### Compare Episodes

```javascript
// Load episode 100
ws.send(JSON.stringify({
  type: "load_replay",
  episode_id: 100
}));

// Step through manually
ws.send(JSON.stringify({
  command: "step"
}));

// Later, load episode 200 for comparison
ws.send(JSON.stringify({
  type: "load_replay",
  episode_id: 200
}));
```

### Find Problem Episodes

```javascript
// Find low-reward episodes (failures)
ws.send(JSON.stringify({
  type: "list_recordings",
  filters: {
    max_reward: 50.0,  // Failed early
    limit: 20
  }
}));
```

## Implementation Details

### Storage Format

- **File format**: MessagePack + LZ4 compression
- **File naming**: `episode_{id:06d}.msgpack.lz4`
- **Size**: ~15-30 KB per episode (compressed)
- **Location**: Recordings directory specified at server startup

### Database Schema

Recordings are indexed in SQLite with:
- episode_id (primary key)
- survival_steps
- total_reward
- curriculum_stage
- recording_reason
- timestamp
- file_path
- file_size_bytes
- compressed_size_bytes

### Modes

The inference server supports two modes:

1. **Live Inference Mode** (default): Runs agent on latest checkpoint
2. **Replay Mode**: Streams recorded episodes

Switch by loading a replay. Return to live inference by loading a checkpoint.

## Troubleshooting

### "Replay not available (no database)"

Server wasn't started with `--db` and `--recordings` arguments. Restart with replay parameters.

### "Recording not found"

Episode wasn't recorded or database is stale. List recordings to see available episodes:

```javascript
ws.send(JSON.stringify({
  type: "list_recordings",
  filters: {}
}));
```

### "Failed to load episode"

Recording file is missing or corrupted. Check `recordings/` directory:

```bash
ls -lh recordings/episode_*.msgpack.lz4
```

### Playback is choppy

Adjust step delay or speed:

```javascript
ws.send(JSON.stringify({
  command: "set_speed",
  speed: 0.5  // Slower playback
}));
```

## Advanced Features

### Seek to Specific Points

```javascript
// Jump to midpoint
ws.send(JSON.stringify({
  type: "replay_control",
  action: "seek",
  seek_step: 25  // Half of 50 total steps
}));
```

### Query by Recording Reason

Recording reasons include:
- `periodic_100` - Every 100th episode
- `stage_2_pre_transition` - Before stage transition
- `top_10.0pct` - Top 10% reward
- `bottom_10.0pct` - Bottom 10% reward (failures)
- `stage_2_first_5` - First 5 episodes at stage 2

```javascript
// Find all stage transition recordings
ws.send(JSON.stringify({
  type: "list_recordings",
  filters: {
    reason: "stage_2_pre_transition"
  }
}));
```

## Frontend Integration

The existing Vue frontend automatically handles replay mode:
- Grid visualization shows agent position and affordances
- Meters display agent state
- Q-values shown if recorded
- Temporal mechanics (day/night cycle, interaction progress) displayed
- **No code changes needed** - replay uses same state_update format!

## Performance

- **Server overhead**: Minimal (replay is async I/O)
- **Decompression**: <10ms per episode load
- **Streaming**: 5-50 steps/sec (configurable)
- **Memory**: ~1-2 MB per loaded episode

---

## Video Export

Want to export episodes to MP4 for YouTube or offline viewing? See **VIDEO_EXPORT_USAGE.md** for:
- Command-line video export tool
- Batch export multiple episodes
- YouTube-optimized rendering (1080p/1440p)
- Filtering by stage, reward, reason

**Quick example**:
```bash
# Export single episode
python -m townlet.recording export 500 \
  --database demo.db \
  --recordings recordings/ \
  --output episode_500.mp4 \
  --fps 60 \
  --dpi 150
```

---
