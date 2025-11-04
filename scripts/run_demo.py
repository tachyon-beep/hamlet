#!/usr/bin/env python3
"""
Unified Demo Server Entry Point

Single-command interface for running Hamlet training, inference, and frontend together.

Usage:
    python run_demo.py --config configs/L1_full_observability --episodes 10000

Author: Hamlet Project
Date: November 2, 2025
"""

import argparse
import logging
import signal
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from townlet.demo.unified_server import UnifiedServer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Unified Hamlet Demo Server - Training, Inference, and Visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start training + inference (then run frontend separately)
  python run_demo.py --config configs/L1_full_observability --episodes 10000
  # In another terminal: cd frontend && npm run dev

  # Resume from checkpoint
  python run_demo.py --config configs/L1_full_observability \\
      --checkpoint-dir runs/L1_full_observability/2025-11-02_123456/checkpoints

  # Custom inference port
  python run_demo.py --config configs/L1_full_observability \\
      --episodes 5000 --inference-port 8800

Note:
  Frontend runs separately for better stability and Vue HMR support.
  Start frontend with: cd frontend && npm run dev
""",
    )

    # Required arguments
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration pack directory or training YAML (e.g., configs/L1_full_observability)",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Total number of training episodes to run (default: read from config YAML)",
    )

    # Optional arguments
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory for checkpoints. If not provided, auto-generated in runs/ structure",
    )

    parser.add_argument(
        "--inference-port",
        type=int,
        default=8766,
        help="Port for inference WebSocket server (default: 8766)",
    )

    parser.add_argument(
        "--debug", action="store_true", help="Enable debug-level logging for troubleshooting"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")

    # Determine config pack directory and training config file
    config_input = Path(args.config)
    if config_input.is_dir():
        config_dir = config_input
        config_file = config_dir / "training.yaml"
        if not config_file.exists():
            logger.error(f"Config folder '{config_dir}' does not contain training.yaml")
            sys.exit(1)
    else:
        config_file = config_input
        if not config_file.exists():
            logger.error(f"Config file not found: {config_file}")
            logger.error(
                "Provide a valid config path (directory with training.yaml or legacy YAML file)."
            )
            sys.exit(1)
        config_dir = config_file.parent

    # Read total_episodes from config if not provided via CLI
    total_episodes = args.episodes
    if total_episodes is None:
        try:
            import yaml
            with open(config_file) as f:
                config = yaml.safe_load(f)
                total_episodes = config.get('training', {}).get('max_episodes', None)
                if total_episodes is None:
                    logger.error("No total_episodes provided and training.max_episodes not found in config")
                    sys.exit(1)
                logger.debug(f"Read max_episodes={total_episodes} from config")
        except Exception as e:
            logger.error(f"Failed to read max_episodes from config: {e}")
            sys.exit(1)

    # Print startup banner
    logger.info("=" * 60)
    logger.info("Unified Demo Server Starting")
    logger.info("=" * 60)
    logger.info(f"Config Pack: {config_dir}")
    logger.info(f"Training Config: {config_file}")
    if args.episodes is not None:
        logger.info(f"Episodes: {total_episodes} (from --episodes flag)")
    else:
        logger.info(f"Episodes: {total_episodes} (from training config)")
    if args.checkpoint_dir:
        logger.info(f"Checkpoint Dir: {args.checkpoint_dir}")
    else:
        logger.info("Checkpoint Dir: Auto-generated in runs/ structure")
    logger.info(f"Inference Port: {args.inference_port}")
    logger.info("Note: Frontend runs separately (cd frontend && npm run dev)")
    logger.info("=" * 60)

    # Create unified server
    try:
        server = UnifiedServer(
            config_dir=str(config_dir),
            training_config_path=str(config_file),
            total_episodes=total_episodes,
            checkpoint_dir=args.checkpoint_dir,
            inference_port=args.inference_port,
        )
    except Exception as e:
        logger.error(f"Failed to create UnifiedServer: {e}")
        if args.debug:
            logger.exception("Full traceback:")
        sys.exit(1)

    # Register signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        """Handle SIGINT (Ctrl+C) and SIGTERM."""
        signal_name = signal.Signals(signum).name
        logger.info(f"\nShutdown signal received ({signal_name}). Gracefully stopping...")
        server.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start the server (blocks until shutdown)
    try:
        server.start()
    except KeyboardInterrupt:
        # Already handled by signal handler
        pass
    except Exception as e:
        logger.error(f"Unexpected error during server operation: {e}")
        if args.debug:
            logger.exception("Full traceback:")
        server.stop()
        sys.exit(1)

    # Clean exit
    logger.info("=" * 60)
    logger.info("All systems stopped. Goodbye!")
    logger.info("=" * 60)
    sys.exit(0)


if __name__ == "__main__":
    main()
