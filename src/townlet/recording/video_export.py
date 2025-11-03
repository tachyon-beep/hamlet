"""
Video export for episode replay.

Exports recorded episodes to MP4 video files suitable for YouTube.
Uses EpisodeVideoRenderer for frame generation and ffmpeg for encoding.
"""

import logging
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from townlet.demo.database import DemoDatabase
from townlet.recording.replay import ReplayManager
from townlet.recording.video_renderer import EpisodeVideoRenderer

logger = logging.getLogger(__name__)


def export_episode_video(
    episode_id: int,
    database_path: Path | str,
    recordings_base_dir: Path | str,
    output_path: Path | str,
    fps: int = 30,
    speed: float = 1.0,
    dpi: int = 100,
    style: str = "dark",
    grid_size: int | None = None,
) -> bool:
    """Export episode to MP4 video.

    Args:
        episode_id: Episode ID to export
        database_path: Path to demo database
        recordings_base_dir: Base directory for recordings
        output_path: Output MP4 file path
        fps: Frames per second (default 30)
        speed: Playback speed multiplier (default 1.0)
        dpi: Dots per inch for rendering (default 100 = 1600×900)
        style: Visual style ("dark" or "light")
        grid_size: Grid size (auto-detect from position if None)

    Returns:
        True if export succeeded, False otherwise
    """
    database_path = Path(database_path)
    recordings_base_dir = Path(recordings_base_dir)
    output_path = Path(output_path)

    # Load episode
    logger.info(f"Loading episode {episode_id}...")
    db = DemoDatabase(database_path)
    replay = ReplayManager(db, recordings_base_dir)

    if not replay.load_episode(episode_id):
        logger.error(f"Failed to load episode {episode_id}")
        return False

    # Initialize renderer
    metadata = replay.get_metadata()
    affordances = replay.get_affordances()
    total_steps = replay.get_total_steps()

    # Auto-detect grid size from affordance positions if not provided
    if grid_size is None:
        affordance_positions = affordances.get("positions", affordances)
        max_coord = 0
        for pos in affordance_positions.values():
            max_coord = max(max_coord, pos[0], pos[1])
        grid_size = max_coord + 1  # Grid is 0-indexed
        logger.info(f"Auto-detected grid size: {grid_size}×{grid_size}")

    logger.info(
        f"Rendering {total_steps} frames at {dpi} DPI (stage {metadata['curriculum_stage']})..."
    )

    renderer = EpisodeVideoRenderer(grid_size=grid_size, dpi=dpi, style=style)

    # Render frames to temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        frame_paths = []

        for step_index in range(total_steps):
            replay.seek(step_index)
            step_data = replay.get_current_step()

            # Render frame
            frame = renderer.render_frame(step_data, metadata, affordances)

            # Save frame as PNG
            frame_path = tmpdir_path / f"frame_{step_index:06d}.png"
            frame_paths.append(frame_path)

            # Write frame (using PIL for simplicity)
            from PIL import Image

            img = Image.fromarray(frame)
            img.save(frame_path)

        logger.info(f"Rendered {len(frame_paths)} frames")

        # Encode with ffmpeg
        logger.info(f"Encoding to MP4 at {fps} FPS...")
        success = _encode_video_ffmpeg(
            tmpdir_path, output_path, fps=fps, speed=speed
        )

        if success:
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info(f"Video exported to {output_path} ({file_size_mb:.1f} MB)")
            return True
        else:
            logger.error("Video encoding failed")
            return False


def _encode_video_ffmpeg(
    frames_dir: Path, output_path: Path, fps: int = 30, speed: float = 1.0
) -> bool:
    """Encode frames to MP4 using ffmpeg.

    Args:
        frames_dir: Directory containing frame_*.png files
        output_path: Output MP4 path
        fps: Frames per second
        speed: Playback speed multiplier

    Returns:
        True if encoding succeeded
    """
    # Check ffmpeg availability
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("ffmpeg not found. Please install ffmpeg.")
        return False

    # Adjust FPS for speed
    effective_fps = int(fps * speed)

    # Build ffmpeg command
    # Use h264 codec with high quality settings suitable for YouTube
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output file
        "-framerate",
        str(effective_fps),
        "-pattern_type",
        "glob",
        "-i",
        str(frames_dir / "frame_*.png"),
        "-c:v",
        "libx264",  # H.264 codec
        "-pix_fmt",
        "yuv420p",  # YouTube-compatible pixel format
        "-crf",
        "18",  # High quality (0=lossless, 23=default, 51=worst)
        "-preset",
        "slow",  # Better compression
        str(output_path),
    ]

    try:
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True
        )
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"ffmpeg encoding failed: {e.stderr}")
        return False


def batch_export_videos(
    database_path: Path | str,
    recordings_base_dir: Path | str,
    output_dir: Path | str,
    stage: int | None = None,
    reason: str | None = None,
    min_reward: float | None = None,
    max_reward: float | None = None,
    limit: int = 100,
    fps: int = 30,
    speed: float = 1.0,
    dpi: int = 100,
    style: str = "dark",
) -> int:
    """Batch export multiple episodes to MP4.

    Args:
        database_path: Path to demo database
        recordings_base_dir: Base directory for recordings
        output_dir: Output directory for MP4 files
        stage: Filter by curriculum stage (optional)
        reason: Filter by recording reason (optional)
        min_reward: Minimum reward threshold (optional)
        max_reward: Maximum reward threshold (optional)
        limit: Maximum number of videos to export
        fps: Frames per second
        speed: Playback speed multiplier
        dpi: Dots per inch for rendering
        style: Visual style

    Returns:
        Number of successfully exported videos
    """
    database_path = Path(database_path)
    recordings_base_dir = Path(recordings_base_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Query database for recordings
    db = DemoDatabase(database_path)
    recordings = db.query_recordings(
        stage=stage,
        reason=reason,
        min_reward=min_reward,
        max_reward=max_reward,
        limit=limit,
    )

    logger.info(f"Found {len(recordings)} episodes to export")

    success_count = 0
    for recording in recordings:
        episode_id = recording["episode_id"]
        output_path = output_dir / f"episode_{episode_id:06d}.mp4"

        logger.info(f"Exporting episode {episode_id}...")

        success = export_episode_video(
            episode_id=episode_id,
            database_path=database_path,
            recordings_base_dir=recordings_base_dir,
            output_path=output_path,
            fps=fps,
            speed=speed,
            dpi=dpi,
            style=style,
        )

        if success:
            success_count += 1

    logger.info(f"Exported {success_count}/{len(recordings)} videos")
    return success_count
