"""
Replay manager for episode playback.

Handles loading, decompressing, and streaming recorded episodes.
"""

import logging
from pathlib import Path
from typing import Any
import msgpack
import lz4.frame

from townlet.recording.data_structures import deserialize_step, deserialize_metadata

logger = logging.getLogger(__name__)


class ReplayManager:
    """Manages replay of recorded episodes.

    Loads episode data from disk and provides step-by-step playback control.
    """

    def __init__(self, database, recordings_base_dir: Path):
        """Initialize replay manager.

        Args:
            database: Database instance for querying recordings
            recordings_base_dir: Base directory for recording files
        """
        self.database = database
        self.recordings_base_dir = Path(recordings_base_dir)

        # Current replay state
        self.episode_id: int | None = None
        self.metadata: dict | None = None
        self.steps: list[dict] = []
        self.affordances: dict = {}
        self.current_step_index: int = 0
        self.playing: bool = False

    def load_episode(self, episode_id: int) -> bool:
        """Load episode from database and decompress.

        Args:
            episode_id: Episode ID to load

        Returns:
            True if episode loaded successfully, False otherwise
        """
        # Query database for recording metadata
        recording = self.database.get_recording(episode_id)
        if recording is None:
            logger.error(f"Recording not found: episode {episode_id}")
            return False

        # Construct full file path
        file_path = self.recordings_base_dir / recording["file_path"]
        if not file_path.exists():
            logger.error(f"Recording file not found: {file_path}")
            return False

        try:
            # Read and decompress
            compressed_data = file_path.read_bytes()
            decompressed = lz4.frame.decompress(compressed_data)
            episode_data = msgpack.unpackb(decompressed, raw=False)

            # Store episode data
            self.episode_id = episode_id
            self.metadata = episode_data["metadata"]
            self.steps = episode_data["steps"]
            self.affordances = episode_data.get("affordances", {})
            self.current_step_index = 0
            self.playing = False

            logger.info(
                f"Loaded episode {episode_id}: {len(self.steps)} steps, "
                f"survival={self.metadata['survival_steps']}, "
                f"reward={self.metadata['total_reward']:.1f}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to load episode {episode_id}: {e}")
            return False

    def get_current_step(self) -> dict | None:
        """Get current step data.

        Returns:
            Step data dict or None if no episode loaded or at end
        """
        if not self.is_loaded():
            return None

        if self.current_step_index >= len(self.steps):
            return None

        return self.steps[self.current_step_index]

    def get_metadata(self) -> dict | None:
        """Get episode metadata.

        Returns:
            Metadata dict or None if no episode loaded
        """
        return self.metadata

    def get_affordances(self) -> dict:
        """Get affordance layout.

        Returns:
            Affordance layout dict {name: (x, y)}
        """
        return self.affordances

    def get_total_steps(self) -> int:
        """Get total number of steps in episode.

        Returns:
            Total steps, or 0 if no episode loaded
        """
        return len(self.steps) if self.is_loaded() else 0

    def get_current_step_index(self) -> int:
        """Get current step index.

        Returns:
            Current step index
        """
        return self.current_step_index

    def is_loaded(self) -> bool:
        """Check if an episode is loaded.

        Returns:
            True if episode is loaded
        """
        return self.episode_id is not None and len(self.steps) > 0

    def is_at_end(self) -> bool:
        """Check if replay is at the end.

        Returns:
            True if at end of episode
        """
        return self.current_step_index >= len(self.steps)

    def next_step(self) -> dict | None:
        """Advance to next step and return it.

        Returns:
            Next step data or None if at end
        """
        if not self.is_loaded() or self.is_at_end():
            return None

        self.current_step_index += 1
        return self.get_current_step()

    def seek(self, step_index: int) -> bool:
        """Seek to specific step index.

        Args:
            step_index: Target step index

        Returns:
            True if seek successful, False if out of bounds
        """
        if not self.is_loaded():
            return False

        if 0 <= step_index < len(self.steps):
            self.current_step_index = step_index
            return True

        return False

    def reset(self):
        """Reset replay to beginning."""
        self.current_step_index = 0
        self.playing = False

    def unload(self):
        """Unload current episode and clear state."""
        self.episode_id = None
        self.metadata = None
        self.steps = []
        self.affordances = {}
        self.current_step_index = 0
        self.playing = False

    def list_recordings(
        self,
        stage: int | None = None,
        reason: str | None = None,
        min_reward: float | None = None,
        max_reward: float | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query available recordings from database.

        Args:
            stage: Filter by curriculum stage
            reason: Filter by recording reason
            min_reward: Filter by minimum total reward
            max_reward: Filter by maximum total reward
            limit: Maximum number of recordings to return

        Returns:
            List of recording metadata dicts
        """
        return self.database.list_recordings(
            stage=stage,
            reason=reason,
            min_reward=min_reward,
            max_reward=max_reward,
            limit=limit,
        )
