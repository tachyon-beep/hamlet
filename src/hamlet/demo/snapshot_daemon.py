"""Snapshot daemon for capturing demo visualizations."""

import asyncio
import logging
import time
from pathlib import Path
from datetime import datetime
import subprocess

from hamlet.demo.database import DemoDatabase

logger = logging.getLogger(__name__)


class SnapshotDaemon:
    """Periodically captures screenshots, GIFs, and CSV exports."""

    def __init__(
        self,
        db_path: Path | str,
        output_dir: Path | str,
        browser_url: str = "http://localhost:5173",
    ):
        """Initialize snapshot daemon.

        Args:
            db_path: Path to demo database
            output_dir: Output directory for snapshots
            browser_url: URL to screenshot
        """
        self.db_path = Path(db_path)
        self.output_dir = Path(output_dir)
        self.browser_url = browser_url

        # Create output directories
        (self.output_dir / "screenshots").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "gifs").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "exports").mkdir(parents=True, exist_ok=True)

        self.db = DemoDatabase(db_path)

        self.last_screenshot_time = 0
        self.last_gif_episode = 0
        self.last_csv_export_time = 0

    async def run(self):
        """Main daemon loop."""
        logger.info(f"Starting snapshot daemon")
        logger.info(f"Database: {self.db_path}")
        logger.info(f"Output: {self.output_dir}")
        logger.info(f"Browser: {self.browser_url}")

        while True:
            try:
                current_time = time.time()

                # Screenshot every 5 minutes
                if current_time - self.last_screenshot_time > 300:
                    await self.capture_screenshot()
                    self.last_screenshot_time = current_time

                # Check for GIF generation (every 50 episodes)
                if self.should_generate_gif():
                    await self.generate_heatmap_gif()

                # CSV export every hour
                if current_time - self.last_csv_export_time > 3600:
                    await self.export_metrics_csv()
                    self.last_csv_export_time = current_time

                # Check every minute
                await asyncio.sleep(60)

            except Exception as e:
                logger.error(f"Error in snapshot daemon: {e}")
                await asyncio.sleep(60)

    async def capture_screenshot(self):
        """Capture browser screenshot using headless Chrome."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / "screenshots" / f"screenshot_{timestamp}.png"

        try:
            # Use playwright for headless screenshots (requires: uv add playwright)
            # For now, use simple screencapture on macOS or import via selenium
            logger.info(f"Capturing screenshot: {output_path}")

            # TODO: Implement with playwright or selenium
            # For now, just log
            logger.warning("Screenshot capture not implemented yet (needs playwright/selenium)")

        except Exception as e:
            logger.error(f"Failed to capture screenshot: {e}")

    def should_generate_gif(self) -> bool:
        """Check if enough episodes have passed for GIF generation."""
        episodes = self.db.get_latest_episodes(limit=1)
        if not episodes:
            return False

        current_episode = episodes[0]['episode_id']

        # Generate GIF every 50 episodes
        if current_episode >= self.last_gif_episode + 50:
            return True

        return False

    async def generate_heatmap_gif(self):
        """Generate novelty heatmap GIF for last 50 episodes."""
        episodes = self.db.get_latest_episodes(limit=1)
        if not episodes:
            return

        current_episode = episodes[0]['episode_id']
        start_episode = self.last_gif_episode
        end_episode = current_episode

        output_path = self.output_dir / "gifs" / f"novelty_ep{start_episode:05d}-{end_episode:05d}.gif"

        try:
            logger.info(f"Generating novelty GIF: {output_path}")

            # TODO: Implement GIF generation from position_heatmap table
            # Query position_heatmap for episodes [start, end]
            # Generate frames
            # Compile to GIF using Pillow
            logger.warning("GIF generation not implemented yet")

            self.last_gif_episode = end_episode

        except Exception as e:
            logger.error(f"Failed to generate GIF: {e}")

    async def export_metrics_csv(self):
        """Export episodes table to CSV."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / "exports" / f"metrics_{timestamp}.csv"

        try:
            logger.info(f"Exporting metrics to CSV: {output_path}")

            # Use sqlite3 command line tool for export
            cmd = f'sqlite3 {self.db_path} ".mode csv" ".output {output_path}" "SELECT * FROM episodes"'
            subprocess.run(cmd, shell=True, check=True)

            logger.info(f"CSV export complete: {output_path}")

        except Exception as e:
            logger.error(f"Failed to export CSV: {e}")


def run_daemon(
    db_path: str = "demo_state.db",
    output_dir: str = "snapshots",
    browser_url: str = "http://localhost:5173",
):
    """Run snapshot daemon."""
    logging.basicConfig(level=logging.INFO)

    daemon = SnapshotDaemon(db_path, output_dir, browser_url)
    asyncio.run(daemon.run())


if __name__ == '__main__':
    import sys
    run_daemon(
        db_path=sys.argv[1] if len(sys.argv) > 1 else "demo_state.db",
        output_dir=sys.argv[2] if len(sys.argv) > 2 else "snapshots",
        browser_url=sys.argv[3] if len(sys.argv) > 3 else "http://localhost:5173",
    )
