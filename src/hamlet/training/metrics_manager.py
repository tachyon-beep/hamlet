"""
Metrics management for training.

Handles TensorBoard, SQLite database, episode replays, and live broadcasting.
"""

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

from hamlet.training.config import MetricsConfig


class MetricsManager:
    """
    Comprehensive metrics tracking for training.

    Supports multiple output formats:
    - TensorBoard for real-time visualization
    - SQLite database for structured queries
    - Episode replay storage for analysis
    - Live broadcasting (WebSocket integration)
    """

    def __init__(self, config: MetricsConfig, experiment_name: str = "default"):
        """
        Initialize metrics manager.

        Args:
            config: Metrics configuration
            experiment_name: Name for organizing metrics
        """
        self.config = config
        self.experiment_name = experiment_name

        # TensorBoard writer
        self.tb_writer: Optional[SummaryWriter] = None
        if config.tensorboard and TENSORBOARD_AVAILABLE:
            tb_path = Path(config.tensorboard_dir) / experiment_name
            tb_path.mkdir(parents=True, exist_ok=True)
            self.tb_writer = SummaryWriter(str(tb_path))

        # SQLite database
        self.db_conn: Optional[sqlite3.Connection] = None
        if config.database:
            db_path = Path(config.database_path)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self.db_conn = sqlite3.connect(str(db_path))
            self._init_database()

        # Replay storage
        self.replay_dir: Optional[Path] = None
        if config.replay_storage:
            self.replay_dir = Path(config.replay_dir) / experiment_name
            self.replay_dir.mkdir(parents=True, exist_ok=True)

        # Episode counter for replay sampling
        self.episode_count = 0

        # Live broadcast subscribers (WebSocket connections)
        self.subscribers: List[Any] = []

    def log_episode(
        self,
        episode: int,
        agent_id: str,
        metrics: Dict[str, float],
    ):
        """
        Log episode-level metrics.

        Args:
            episode: Episode number
            agent_id: Agent identifier
            metrics: Dictionary of metric name -> value
        """
        # TensorBoard logging
        if self.tb_writer:
            for metric_name, value in metrics.items():
                tag = f"{agent_id}/{metric_name}"
                self.tb_writer.add_scalar(tag, value, episode)

        # Database logging
        if self.db_conn:
            self._log_to_database(episode, agent_id, metrics)

        # Live broadcast
        if self.config.live_broadcast and self.subscribers:
            self._broadcast_metrics(episode, agent_id, metrics)

    def log_step(
        self,
        step: int,
        agent_id: str,
        metrics: Dict[str, float],
    ):
        """
        Log step-level metrics (within episode).

        Args:
            step: Global step number
            agent_id: Agent identifier
            metrics: Dictionary of metric name -> value
        """
        # TensorBoard logging
        if self.tb_writer:
            for metric_name, value in metrics.items():
                tag = f"{agent_id}/step_{metric_name}"
                self.tb_writer.add_scalar(tag, value, step)

    def save_episode_replay(
        self,
        episode: int,
        agent_id: str,
        trajectory: List[Dict[str, Any]],
    ):
        """
        Save complete episode trajectory for replay analysis.

        Args:
            episode: Episode number
            agent_id: Agent identifier
            trajectory: List of step dictionaries containing states, actions, rewards
        """
        if not self.replay_dir:
            return

        # Sample based on replay_sample_rate
        if self.episode_count % int(1.0 / self.config.replay_sample_rate) != 0:
            return

        # Save trajectory as JSON
        replay_file = self.replay_dir / f"ep{episode}_{agent_id}.json"
        replay_data = {
            "episode": episode,
            "agent_id": agent_id,
            "timestamp": datetime.now().isoformat(),
            "trajectory": trajectory,
        }

        with open(replay_file, "w") as f:
            json.dump(replay_data, f, indent=2)

    def log_hyperparameters(self, hparams: Dict[str, Any], metrics: Dict[str, float]):
        """
        Log hyperparameters with final metrics for comparison.

        Args:
            hparams: Hyperparameter dictionary
            metrics: Final metric values
        """
        if self.tb_writer:
            self.tb_writer.add_hparams(hparams, metrics)

    def add_subscriber(self, subscriber: Any):
        """
        Add live metrics subscriber (WebSocket connection).

        Args:
            subscriber: Subscriber object with send method
        """
        if self.config.live_broadcast:
            self.subscribers.append(subscriber)

    def remove_subscriber(self, subscriber: Any):
        """
        Remove live metrics subscriber.

        Args:
            subscriber: Subscriber to remove
        """
        if subscriber in self.subscribers:
            self.subscribers.remove(subscriber)

    def increment_episode(self):
        """Increment episode counter for replay sampling."""
        self.episode_count += 1

    def query_metrics(
        self,
        agent_id: Optional[str] = None,
        metric_name: Optional[str] = None,
        min_episode: Optional[int] = None,
        max_episode: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query metrics from database.

        Args:
            agent_id: Filter by agent ID
            metric_name: Filter by metric name
            min_episode: Minimum episode number
            max_episode: Maximum episode number

        Returns:
            List of metric records
        """
        if not self.db_conn:
            return []

        query = "SELECT * FROM metrics WHERE 1=1"
        params = []

        if agent_id:
            query += " AND agent_id = ?"
            params.append(agent_id)

        if metric_name:
            query += " AND metric_name = ?"
            params.append(metric_name)

        if min_episode is not None:
            query += " AND episode >= ?"
            params.append(min_episode)

        if max_episode is not None:
            query += " AND episode <= ?"
            params.append(max_episode)

        cursor = self.db_conn.execute(query, params)
        columns = [desc[0] for desc in cursor.description]

        results = []
        for row in cursor:
            results.append(dict(zip(columns, row)))

        return results

    def close(self):
        """Clean up resources."""
        if self.tb_writer:
            self.tb_writer.close()

        if self.db_conn:
            self.db_conn.close()

    def _init_database(self):
        """Initialize SQLite database schema."""
        self.db_conn.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                episode INTEGER NOT NULL,
                agent_id TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                value REAL NOT NULL
            )
        """)

        # Create indexes for common queries
        self.db_conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_episode
            ON metrics(episode)
        """)

        self.db_conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_agent_metric
            ON metrics(agent_id, metric_name)
        """)

        self.db_conn.commit()

    def _log_to_database(
        self,
        episode: int,
        agent_id: str,
        metrics: Dict[str, float],
    ):
        """Log metrics to SQLite database."""
        timestamp = datetime.now().isoformat()

        for metric_name, value in metrics.items():
            self.db_conn.execute(
                """
                INSERT INTO metrics (timestamp, episode, agent_id, metric_name, value)
                VALUES (?, ?, ?, ?, ?)
                """,
                (timestamp, episode, agent_id, metric_name, value),
            )

        self.db_conn.commit()

    def _broadcast_metrics(
        self,
        episode: int,
        agent_id: str,
        metrics: Dict[str, float],
    ):
        """Broadcast metrics to live subscribers."""
        message = {
            "type": "metrics",
            "episode": episode,
            "agent_id": agent_id,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        }

        # Send to all subscribers (assumes WebSocket interface)
        for subscriber in self.subscribers[:]:  # Copy to avoid modification during iteration
            try:
                if hasattr(subscriber, "send"):
                    subscriber.send(json.dumps(message))
            except Exception:
                # Remove failed subscribers
                self.subscribers.remove(subscriber)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
