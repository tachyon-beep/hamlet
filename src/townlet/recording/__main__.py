"""
Command-line interface for episode recording tools.

Usage:
    python -m townlet.recording.export_video <episode_id> --database demo.db --recordings recordings/ --output output.mp4
    python -m townlet.recording.batch_export --database demo.db --recordings recordings/ --output-dir videos/
"""

import argparse
import logging
import sys
from pathlib import Path

from townlet.recording.video_export import batch_export_videos, export_episode_video

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",  # Simple format for CLI output
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Export episode recordings to MP4 video"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Single episode export
    export_parser = subparsers.add_parser(
        "export", help="Export single episode to MP4"
    )
    export_parser.add_argument("episode_id", type=int, help="Episode ID to export")
    export_parser.add_argument(
        "--database", required=True, type=Path, help="Path to demo database"
    )
    export_parser.add_argument(
        "--recordings",
        required=True,
        type=Path,
        help="Base directory for recordings",
    )
    export_parser.add_argument(
        "--output", required=True, type=Path, help="Output MP4 file path"
    )
    export_parser.add_argument(
        "--fps", type=int, default=30, help="Frames per second (default: 30)"
    )
    export_parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier (default: 1.0)",
    )
    export_parser.add_argument(
        "--dpi",
        type=int,
        default=100,
        help="DPI for rendering (100=1600×900, 150=2400×1350, default: 100)",
    )
    export_parser.add_argument(
        "--style",
        choices=["dark", "light"],
        default="dark",
        help="Visual style (default: dark)",
    )

    # Batch export
    batch_parser = subparsers.add_parser("batch", help="Batch export multiple episodes")
    batch_parser.add_argument(
        "--database", required=True, type=Path, help="Path to demo database"
    )
    batch_parser.add_argument(
        "--recordings",
        required=True,
        type=Path,
        help="Base directory for recordings",
    )
    batch_parser.add_argument(
        "--output-dir", required=True, type=Path, help="Output directory for MP4 files"
    )
    batch_parser.add_argument(
        "--stage", type=int, help="Filter by curriculum stage"
    )
    batch_parser.add_argument(
        "--reason", type=str, help="Filter by recording reason"
    )
    batch_parser.add_argument(
        "--min-reward", type=float, help="Minimum reward threshold"
    )
    batch_parser.add_argument(
        "--max-reward", type=float, help="Maximum reward threshold"
    )
    batch_parser.add_argument(
        "--limit", type=int, default=100, help="Maximum number of videos (default: 100)"
    )
    batch_parser.add_argument(
        "--fps", type=int, default=30, help="Frames per second (default: 30)"
    )
    batch_parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier (default: 1.0)",
    )
    batch_parser.add_argument(
        "--dpi",
        type=int,
        default=100,
        help="DPI for rendering (100=1600×900, 150=2400×1350, default: 100)",
    )
    batch_parser.add_argument(
        "--style",
        choices=["dark", "light"],
        default="dark",
        help="Visual style (default: dark)",
    )
    batch_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.INFO if getattr(args, "verbose", False) else logging.WARNING
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    if args.command == "export":
        # Single episode export
        success = export_episode_video(
            episode_id=args.episode_id,
            database_path=args.database,
            recordings_base_dir=args.recordings,
            output_path=args.output,
            fps=args.fps,
            speed=args.speed,
            dpi=args.dpi,
            style=args.style,
        )

        if success:
            logger.info(f"✓ Exported episode {args.episode_id} to {args.output}")
            return 0
        else:
            logger.error(f"✗ Failed to export episode {args.episode_id}")
            return 1

    elif args.command == "batch":
        # Batch export
        count = batch_export_videos(
            database_path=args.database,
            recordings_base_dir=args.recordings,
            output_dir=args.output_dir,
            stage=args.stage,
            reason=args.reason,
            min_reward=args.min_reward,
            max_reward=args.max_reward,
            limit=args.limit,
            fps=args.fps,
            speed=args.speed,
            dpi=args.dpi,
            style=args.style,
        )

        logger.info(f"✓ Exported {count} videos to {args.output_dir}")
        return 0

    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
