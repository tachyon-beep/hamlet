# Video Export - Usage Guide

Export recorded episodes to high-quality MP4 videos suitable for YouTube streaming.

## Quick Start

### Export Single Episode

```bash
python -m townlet.recording export 500 \
  --database demo_level2_5.db \
  --recordings recordings/ \
  --output episode_500.mp4 \
  --fps 60 \
  --dpi 150
```

### Batch Export Multiple Episodes

```bash
python -m townlet.recording batch \
  --database demo_level2_5.db \
  --recordings recordings/ \
  --output-dir videos/ \
  --stage 2 \
  --min-reward 200.0 \
  --limit 20 \
  --fps 60 \
  --dpi 150 \
  --verbose
```

## Command Reference

### Single Episode Export

```bash
python -m townlet.recording export <episode_id> \
  --database <db_path> \
  --recordings <recordings_dir> \
  --output <output.mp4> \
  [--fps 30] \
  [--speed 1.0] \
  [--dpi 100] \
  [--style dark]
```

**Arguments**:

- `episode_id`: Episode ID to export (required)
- `--database`: Path to demo database (required)
- `--recordings`: Base directory for recordings (required)
- `--output`: Output MP4 file path (required)
- `--fps`: Frames per second (default: 30)
- `--speed`: Playback speed multiplier (default: 1.0, 2.0 = 2x speed)
- `--dpi`: Rendering DPI (100 = 1600×900, 150 = 2400×1350, default: 100)
- `--style`: Visual style - "dark" or "light" (default: dark)

### Batch Export

```bash
python -m townlet.recording batch \
  --database <db_path> \
  --recordings <recordings_dir> \
  --output-dir <output_dir> \
  [--stage <stage>] \
  [--reason <reason>] \
  [--min-reward <min>] \
  [--max-reward <max>] \
  [--limit 100] \
  [--fps 30] \
  [--speed 1.0] \
  [--dpi 100] \
  [--style dark] \
  [--verbose]
```

**Filter Arguments**:

- `--stage`: Filter by curriculum stage (e.g., 2)
- `--reason`: Filter by recording reason (e.g., "periodic_100")
- `--min-reward`: Minimum reward threshold
- `--max-reward`: Maximum reward threshold
- `--limit`: Maximum number of videos to export (default: 100)

**Other Arguments**: Same as single export, plus:

- `--verbose` / `-v`: Enable verbose logging

## YouTube Optimization

### Recommended Settings

**For 1080p uploads (HD)**:

```bash
python -m townlet.recording export 500 \
  --database demo.db \
  --recordings recordings/ \
  --output episode_500.mp4 \
  --dpi 120 \
  --fps 60
```

- Resolution: 1920×1080
- File size: ~10-30 MB per 100 steps
- Quality: Excellent for YouTube

**For 1440p uploads (2K)**:

```bash
python -m townlet.recording export 500 \
  --database demo.db \
  --recordings recordings/ \
  --output episode_500.mp4 \
  --dpi 150 \
  --fps 60
```

- Resolution: 2400×1350
- File size: ~20-50 MB per 100 steps
- Quality: Premium YouTube quality

**For fast prototyping**:

```bash
python -m townlet.recording export 500 \
  --database demo.db \
  --recordings recordings/ \
  --output episode_500.mp4 \
  --dpi 80 \
  --fps 30
```

- Resolution: 1280×720
- File size: ~5-15 MB per 100 steps
- Quality: Good for quick review

### Speed Control

**Slow motion (0.5x)**:

```bash
--speed 0.5 --fps 30
```

- Effective playback: 15 FPS
- Good for detailed analysis

**Normal speed (1.0x)**:

```bash
--speed 1.0 --fps 30
```

- Standard playback

**Fast forward (2.0x)**:

```bash
--speed 2.0 --fps 30
```

- Effective playback: 60 FPS
- Good for long episodes

## Use Cases

### 1. Export All Stage 2 Episodes

```bash
python -m townlet.recording batch \
  --database demo_level2.db \
  --recordings recordings/ \
  --output-dir videos/stage_2/ \
  --stage 2 \
  --limit 50
```

### 2. Export Top Performers

```bash
python -m townlet.recording batch \
  --database demo_level2.db \
  --recordings recordings/ \
  --output-dir videos/top_performers/ \
  --min-reward 300.0 \
  --limit 20
```

### 3. Export Failure Cases

```bash
python -m townlet.recording batch \
  --database demo_level2.db \
  --recordings recordings/ \
  --output-dir videos/failures/ \
  --max-reward 50.0 \
  --limit 20
```

### 4. Export Pre-Transition Episodes

```bash
python -m townlet.recording batch \
  --database demo_level2.db \
  --recordings recordings/ \
  --output-dir videos/transitions/ \
  --reason "stage_2_pre_transition"
```

### 5. Export Periodic Checkpoints

```bash
python -m townlet.recording batch \
  --database demo_level2.db \
  --recordings recordings/ \
  --output-dir videos/checkpoints/ \
  --reason "periodic_100" \
  --limit 10
```

## Video Features

The exported videos include:

### Grid View (Left Panel)

- 8×8 environment grid
- Color-coded affordances (Bed, Job, Gym, etc.)
- Agent position with action arrows
- Movement visualization

### Meters Panel (Top Right)

- 8 agent meters with color-coded bars:
  - Energy (yellow)
  - Hygiene (blue)
  - Satiation (green)
  - Money (green)
  - Health (red)
  - Fitness (orange)
  - Mood (purple)
  - Social (teal)
- Percentage values

### Info Panel (Top Right)

- Episode ID
- Curriculum stage
- Current step / survival steps
- Action taken
- Reward received
- Total accumulated reward
- Temporal mechanics (if enabled):
  - Time of day
  - Interaction progress

### Q-Values Panel (Bottom Right)

- Bar chart of action Q-values
- Chosen action highlighted in green
- Value labels on bars

## Technical Details

### Video Format

- **Codec**: H.264 (libx264)
- **Pixel format**: yuv420p (YouTube-compatible)
- **CRF**: 18 (high quality, 0=lossless, 23=default, 51=worst)
- **Preset**: slow (better compression)
- **Container**: MP4

### Frame Generation

- **Backend**: matplotlib with Agg (non-interactive)
- **Aspect ratio**: 16:9 (YouTube standard)
- **Color scheme**: Dark theme by default
- **Font**: System default with bold for emphasis

### Performance

- **Rendering**: ~0.1-0.2 seconds per frame
- **Encoding**: ~1-2 seconds per 100 frames (depends on settings)
- **Total time**: ~30-60 seconds for 100-step episode at 1080p

### Requirements

- **ffmpeg**: Must be installed on system

  ```bash
  # Ubuntu/Debian
  sudo apt-get install ffmpeg

  # macOS
  brew install ffmpeg
  ```

- **Python packages**: Automatically installed with townlet
  - matplotlib
  - numpy
  - pillow

## Troubleshooting

### "ffmpeg not found"

Install ffmpeg on your system:

```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Verify installation
ffmpeg -version
```

### "Recording not found"

Episode wasn't recorded. List available recordings:

```bash
# Start inference server with replay support
python -m townlet.demo.live_inference \
  checkpoints/ 8766 0.2 10000 configs/townlet.yaml \
  --db demo.db \
  --recordings recordings/
```

Then check available episodes in frontend or query database directly.

### Video quality is poor

Increase DPI:

```bash
--dpi 150  # 2400×1350 resolution
```

Or increase CRF quality (edit video_export.py line 138):

```python
"-crf", "15",  # Higher quality (lower CRF = better quality)
```

### Export is too slow

Reduce DPI for faster rendering:

```bash
--dpi 80  # 1280×720 resolution
```

Or use faster ffmpeg preset (edit video_export.py line 142):

```python
"-preset", "faster",  # Faster encoding
```

### File sizes too large

Increase CRF (lower quality, smaller files):

```python
"-crf", "23",  # Default quality (smaller files)
```

Or reduce FPS:

```bash
--fps 24  # Cinematic frame rate (smaller files)
```

## Workflow for YouTube Streaming

### 1. Record Episodes During Training

Make sure recording is enabled in your config:

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
    performance:
      enabled: true
      top_percentile: 10.0
```

### 2. Batch Export After Training

```bash
python -m townlet.recording batch \
  --database demo_level2_5.db \
  --recordings recordings/ \
  --output-dir videos/stream_content/ \
  --min-reward 200.0 \
  --limit 50 \
  --dpi 150 \
  --fps 60 \
  --verbose
```

### 3. Upload to YouTube

- Use videos from `videos/stream_content/`
- Set title: "Hamlet DRL Training - Episode {episode_id} - Stage {stage} - Reward {reward}"
- Add description with metrics
- Use tags: "deep reinforcement learning", "AI training", "Hamlet", etc.

### 4. Create Compilation Videos

Combine multiple episodes using ffmpeg:

```bash
# Create file list
for f in videos/stream_content/*.mp4; do
  echo "file '$f'" >> filelist.txt
done

# Concatenate
ffmpeg -f concat -safe 0 -i filelist.txt -c copy compilation.mp4
```

---

**Next Steps**: See `REPLAY_USAGE.md` for real-time replay during training!
