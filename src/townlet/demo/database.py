"""SQLite database for multi-day demo state management."""

import sqlite3
from pathlib import Path
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

            CREATE TABLE IF NOT EXISTS episode_recordings (
                episode_id INTEGER PRIMARY KEY,
                file_path TEXT NOT NULL,
                timestamp REAL NOT NULL,
                survival_steps INTEGER NOT NULL,
                total_reward REAL NOT NULL,
                extrinsic_reward REAL NOT NULL,
                intrinsic_reward REAL NOT NULL,
                curriculum_stage INTEGER NOT NULL,
                epsilon REAL NOT NULL,
                intrinsic_weight REAL NOT NULL,
                recording_reason TEXT NOT NULL,
                file_size_bytes INTEGER,
                compressed_size_bytes INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_recordings_stage ON episode_recordings(curriculum_stage);
            CREATE INDEX IF NOT EXISTS idx_recordings_reason ON episode_recordings(recording_reason);
            CREATE INDEX IF NOT EXISTS idx_recordings_reward ON episode_recordings(total_reward);
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

        Example:
            transitions = {
                "Bed": {"Hospital": 3, "Job": 1},
                "Hospital": {"Bed": 2}
            }
            # Inserts 3 rows:
            #   (episode_id, "Bed", "Hospital", 3)
            #   (episode_id, "Bed", "Job", 1)
            #   (episode_id, "Hospital", "Bed", 2)
        """
        if not transitions:
            return  # No transitions to insert (empty episode)

        rows = []
        for from_aff, to_affs in transitions.items():
            for to_aff, count in to_affs.items():
                rows.append((episode_id, from_aff, to_aff, count))

        self.conn.executemany(
            "INSERT INTO affordance_visits (episode_id, from_affordance, to_affordance, visit_count) VALUES (?, ?, ?, ?)", rows
        )
        self.conn.commit()

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

    def insert_recording(
        self,
        episode_id: int,
        file_path: str,
        metadata,  # EpisodeMetadata type hint would require import
        reason: str,
        file_size: int,
        compressed_size: int,
    ):
        """Insert recording metadata into database.

        Args:
            episode_id: Episode number
            file_path: Path to recording file
            metadata: EpisodeMetadata instance
            reason: Recording reason (e.g., 'periodic', 'stage_transition')
            file_size: Uncompressed file size in bytes
            compressed_size: Compressed file size in bytes
        """
        self.conn.execute(
            """INSERT OR REPLACE INTO episode_recordings
               (episode_id, file_path, timestamp, survival_steps, total_reward,
                extrinsic_reward, intrinsic_reward, curriculum_stage, epsilon,
                intrinsic_weight, recording_reason, file_size_bytes, compressed_size_bytes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                episode_id,
                file_path,
                metadata.timestamp,
                metadata.survival_steps,
                metadata.total_reward,
                metadata.extrinsic_reward,
                metadata.intrinsic_reward,
                metadata.curriculum_stage,
                metadata.epsilon,
                metadata.intrinsic_weight,
                reason,
                file_size,
                compressed_size,
            ),
        )
        self.conn.commit()

    def get_recording(self, episode_id: int) -> dict[str, Any] | None:
        """Get recording metadata by episode_id.

        Args:
            episode_id: Episode number

        Returns:
            Recording metadata dict or None if not found
        """
        cursor = self.conn.execute(
            "SELECT * FROM episode_recordings WHERE episode_id = ?",
            (episode_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def list_recordings(
        self,
        stage: int | None = None,
        reason: str | None = None,
        min_reward: float | None = None,
        max_reward: float | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List recordings with optional filters.

        Args:
            stage: Filter by curriculum stage
            reason: Filter by recording reason
            min_reward: Filter by minimum total reward
            max_reward: Filter by maximum total reward
            limit: Maximum number of recordings to return

        Returns:
            List of recording metadata dicts, ordered by episode_id DESC
        """
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

        cursor = self.conn.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    def close(self):
        """Close database connection."""
        self.conn.close()
