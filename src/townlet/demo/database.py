"""SQLite database for multi-day demo state management."""

import sqlite3
from pathlib import Path
from typing import Any
from typing import Any


class DemoDatabase:
    """Manages SQLite database for demo metrics and state.

    Thread Safety:
        Uses check_same_thread=False and WAL mode for concurrent access.
        Safe for multiple readers and single writer (training process).
        Not safe for multiple concurrent writers without external synchronization.
    """

    def __init__(self, db_path: Path | str):
        """Initialize database, creating schema if needed.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Enable WAL mode for better concurrent access
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.row_factory = sqlite3.Row

        self._create_schema()

    def _create_schema(self):
        """Create database schema if it doesn't exist."""
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS episodes (
                episode_id INTEGER PRIMARY KEY,
                timestamp REAL NOT NULL,
                survival_time INTEGER NOT NULL,
                total_reward REAL NOT NULL,
                extrinsic_reward REAL NOT NULL,
                intrinsic_reward REAL NOT NULL,
                intrinsic_weight REAL NOT NULL,
                curriculum_stage INTEGER NOT NULL,
                epsilon REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_episodes_timestamp ON episodes(timestamp);

            CREATE TABLE IF NOT EXISTS affordance_visits (
                episode_id INTEGER NOT NULL,
                from_affordance TEXT NOT NULL,
                to_affordance TEXT NOT NULL,
                visit_count INTEGER NOT NULL,
                FOREIGN KEY (episode_id) REFERENCES episodes(episode_id)
            );
            CREATE INDEX IF NOT EXISTS idx_visits_episode ON affordance_visits(episode_id);

            CREATE TABLE IF NOT EXISTS position_heatmap (
                episode_id INTEGER NOT NULL,
                x INTEGER NOT NULL,
                y INTEGER NOT NULL,
                visit_count INTEGER NOT NULL,
                novelty_value REAL,
                FOREIGN KEY (episode_id) REFERENCES episodes(episode_id)
            );
            CREATE INDEX IF NOT EXISTS idx_heatmap_episode ON position_heatmap(episode_id);

            CREATE TABLE IF NOT EXISTS system_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
        """
        )
        self.conn.commit()

    def insert_episode(
        self,
        episode_id: int,
        timestamp: float,
        survival_time: int,
        total_reward: float,
        extrinsic_reward: float,
        intrinsic_reward: float,
        intrinsic_weight: float,
        curriculum_stage: int,
        epsilon: float,
    ):
        """Insert episode metrics into database.

        Args:
            episode_id: Episode number
            timestamp: Unix timestamp
            survival_time: Steps survived
            total_reward: Combined reward
            extrinsic_reward: Environment reward
            intrinsic_reward: RND novelty reward
            intrinsic_weight: Current intrinsic weight
            curriculum_stage: Current curriculum stage (1-5)
            epsilon: Current exploration epsilon
        """
        self.conn.execute(
            """INSERT OR REPLACE INTO episodes
               (episode_id, timestamp, survival_time, total_reward, extrinsic_reward,
                intrinsic_reward, intrinsic_weight, curriculum_stage, epsilon)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                episode_id,
                timestamp,
                survival_time,
                total_reward,
                extrinsic_reward,
                intrinsic_reward,
                intrinsic_weight,
                curriculum_stage,
                epsilon,
            ),
        )
        self.conn.commit()

    def get_latest_episodes(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get most recent episodes.

        Args:
            limit: Maximum number of episodes to return

        Returns:
            List of episode dictionaries
        """
        cursor = self.conn.execute("SELECT * FROM episodes ORDER BY episode_id DESC LIMIT ?", (limit,))
        return [dict(row) for row in cursor.fetchall()]

    def set_system_state(self, key: str, value: str):
        """Set system state key-value pair.

        Args:
            key: State key
            value: State value (will be converted to string)
        """
        self.conn.execute("INSERT OR REPLACE INTO system_state (key, value) VALUES (?, ?)", (key, str(value)))
        self.conn.commit()

    def get_system_state(self, key: str) -> str | None:
        """Get system state value.

        Args:
            key: State key

        Returns:
            State value or None if not found
        """
        cursor = self.conn.execute("SELECT value FROM system_state WHERE key = ?", (key,))
        row = cursor.fetchone()
        return row["value"] if row else None

    def insert_affordance_visits(self, episode_id: int, transitions: dict[str, dict[str, int]]):
        """Insert affordance transition counts for an episode.

        Args:
            episode_id: Episode number
            transitions: Dict mapping from_affordance -> {to_affordance: count}

        TODO: Implement in Task 2 when tracking affordance visits
        """
        pass

    def insert_position_heatmap(
        self,
        episode_id: int,
        positions: dict[tuple[int, int], int],
        novelty: dict[tuple[int, int], float] | None = None,
    ):
        """Insert position visit counts and novelty values for an episode.

        Args:
            episode_id: Episode number
            positions: Dict mapping (x, y) -> visit_count
            novelty: Optional dict mapping (x, y) -> novelty_value

        TODO: Implement in Task 5 for visualization
        """
        pass

    def get_position_heatmap(self, episode_id: int) -> list[dict[str, Any]]:
        """Get position heatmap data for an episode.

        Args:
            episode_id: Episode number

        Returns:
            List of position heatmap rows

        TODO: Implement in Task 5 for visualization
        """
        raise NotImplementedError("Position heatmap retrieval will be implemented in Task 5.")

    def close(self):
        """Close database connection."""
        self.conn.close()
