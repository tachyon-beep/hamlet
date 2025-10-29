"""
Tests for MetricsManager.
"""

import pytest
import tempfile
import json
import sqlite3
from pathlib import Path
from hamlet.training.metrics_manager import MetricsManager, TENSORBOARD_AVAILABLE
from hamlet.training.config import MetricsConfig


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def metrics_config(temp_dir):
    """Create metrics configuration for testing."""
    return MetricsConfig(
        tensorboard=True,
        tensorboard_dir=str(temp_dir / "tensorboard"),
        database=True,
        database_path=str(temp_dir / "metrics.db"),
        replay_storage=True,
        replay_dir=str(temp_dir / "replays"),
        replay_sample_rate=1.0,  # Save all episodes for testing
        live_broadcast=True,
    )


def test_metrics_manager_initialization(metrics_config, temp_dir):
    """Test that metrics manager initializes correctly."""
    manager = MetricsManager(metrics_config, experiment_name="test_exp")

    assert manager.experiment_name == "test_exp"
    assert manager.config == metrics_config

    # Check TensorBoard writer (if available)
    if TENSORBOARD_AVAILABLE:
        assert manager.tb_writer is not None
    else:
        assert manager.tb_writer is None

    # Check database connection
    assert manager.db_conn is not None

    # Check replay directory
    assert manager.replay_dir is not None
    assert manager.replay_dir.exists()

    manager.close()


def test_database_initialization(metrics_config):
    """Test that SQLite database is initialized with correct schema."""
    manager = MetricsManager(metrics_config, experiment_name="test_exp")

    # Check that tables exist
    cursor = manager.db_conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='metrics'"
    )
    tables = cursor.fetchall()
    assert len(tables) == 1

    # Check that indexes exist
    cursor = manager.db_conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index'"
    )
    indexes = cursor.fetchall()
    assert len(indexes) >= 2  # At least idx_episode and idx_agent_metric

    manager.close()


def test_log_episode_to_database(metrics_config):
    """Test logging episode metrics to database."""
    manager = MetricsManager(metrics_config, experiment_name="test_exp")

    metrics = {
        "total_reward": 100.5,
        "episode_length": 250,
        "epsilon": 0.5,
    }

    manager.log_episode(episode=1, agent_id="agent_0", metrics=metrics)

    # Query database
    cursor = manager.db_conn.execute(
        "SELECT * FROM metrics WHERE episode = 1 AND agent_id = 'agent_0'"
    )
    rows = cursor.fetchall()

    assert len(rows) == 3  # One row per metric

    # Check values
    metric_values = {row[4]: row[5] for row in rows}  # metric_name: value
    assert metric_values["total_reward"] == 100.5
    assert metric_values["episode_length"] == 250
    assert metric_values["epsilon"] == 0.5

    manager.close()


def test_log_multiple_episodes(metrics_config):
    """Test logging multiple episodes."""
    manager = MetricsManager(metrics_config, experiment_name="test_exp")

    for episode in range(5):
        metrics = {
            "total_reward": float(episode * 10),
            "episode_length": 100 + episode,
        }
        manager.log_episode(episode=episode, agent_id="agent_0", metrics=metrics)

    # Query all episodes
    cursor = manager.db_conn.execute("SELECT COUNT(*) FROM metrics")
    count = cursor.fetchone()[0]
    assert count == 10  # 5 episodes * 2 metrics each

    manager.close()


def test_query_metrics_all(metrics_config):
    """Test querying all metrics."""
    manager = MetricsManager(metrics_config, experiment_name="test_exp")

    # Log some data
    for episode in range(3):
        metrics = {"reward": float(episode * 10)}
        manager.log_episode(episode=episode, agent_id="agent_0", metrics=metrics)

    # Query all
    results = manager.query_metrics()
    assert len(results) == 3

    manager.close()


def test_query_metrics_by_agent(metrics_config):
    """Test querying metrics by agent ID."""
    manager = MetricsManager(metrics_config, experiment_name="test_exp")

    # Log data for multiple agents
    for agent_idx in range(2):
        for episode in range(3):
            metrics = {"reward": float(episode * 10)}
            manager.log_episode(
                episode=episode,
                agent_id=f"agent_{agent_idx}",
                metrics=metrics,
            )

    # Query specific agent
    results = manager.query_metrics(agent_id="agent_0")
    assert len(results) == 3
    assert all(r["agent_id"] == "agent_0" for r in results)

    manager.close()


def test_query_metrics_by_name(metrics_config):
    """Test querying metrics by name."""
    manager = MetricsManager(metrics_config, experiment_name="test_exp")

    # Log multiple metrics
    metrics = {
        "reward": 100.0,
        "length": 250,
        "epsilon": 0.5,
    }
    manager.log_episode(episode=1, agent_id="agent_0", metrics=metrics)

    # Query specific metric
    results = manager.query_metrics(metric_name="reward")
    assert len(results) == 1
    assert results[0]["metric_name"] == "reward"
    assert results[0]["value"] == 100.0

    manager.close()


def test_query_metrics_by_episode_range(metrics_config):
    """Test querying metrics by episode range."""
    manager = MetricsManager(metrics_config, experiment_name="test_exp")

    # Log episodes 0-9
    for episode in range(10):
        metrics = {"reward": float(episode)}
        manager.log_episode(episode=episode, agent_id="agent_0", metrics=metrics)

    # Query episodes 3-7
    results = manager.query_metrics(min_episode=3, max_episode=7)
    assert len(results) == 5  # Episodes 3, 4, 5, 6, 7
    episodes = [r["episode"] for r in results]
    assert min(episodes) == 3
    assert max(episodes) == 7

    manager.close()


def test_save_episode_replay(metrics_config, temp_dir):
    """Test saving episode replay."""
    manager = MetricsManager(metrics_config, experiment_name="test_exp")

    trajectory = [
        {"state": [1.0, 2.0], "action": 0, "reward": 1.0, "done": False},
        {"state": [1.1, 2.1], "action": 1, "reward": 2.0, "done": False},
        {"state": [1.2, 2.2], "action": 0, "reward": 3.0, "done": True},
    ]

    manager.save_episode_replay(
        episode=1,
        agent_id="agent_0",
        trajectory=trajectory,
    )

    # Check replay file exists
    replay_file = manager.replay_dir / "ep1_agent_0.json"
    assert replay_file.exists()

    # Load and verify content
    with open(replay_file) as f:
        data = json.load(f)

    assert data["episode"] == 1
    assert data["agent_id"] == "agent_0"
    assert data["trajectory"] == trajectory
    assert "timestamp" in data

    manager.close()


def test_replay_sample_rate(metrics_config, temp_dir):
    """Test that replay sampling rate works correctly."""
    # Set sample rate to 50%
    metrics_config.replay_sample_rate = 0.5
    manager = MetricsManager(metrics_config, experiment_name="test_exp")

    trajectory = [{"state": [1.0], "action": 0, "reward": 1.0, "done": True}]

    # Save 10 episodes
    for episode in range(10):
        manager.increment_episode()
        manager.save_episode_replay(
            episode=episode,
            agent_id="agent_0",
            trajectory=trajectory,
        )

    # Should have ~5 replay files (every 2nd episode)
    replay_files = list(manager.replay_dir.glob("*.json"))
    assert len(replay_files) == 5

    manager.close()


def test_log_step_metrics(metrics_config):
    """Test logging step-level metrics."""
    manager = MetricsManager(metrics_config, experiment_name="test_exp")

    # Log step metrics
    for step in range(5):
        metrics = {"loss": float(step * 0.1)}
        manager.log_step(step=step, agent_id="agent_0", metrics=metrics)

    # TensorBoard writer should receive step metrics
    # (No direct way to verify without reading TensorBoard files)

    manager.close()


def test_log_hyperparameters(metrics_config):
    """Test logging hyperparameters."""
    manager = MetricsManager(metrics_config, experiment_name="test_exp")

    hparams = {
        "learning_rate": 0.001,
        "gamma": 0.99,
        "epsilon": 1.0,
    }

    final_metrics = {
        "final_reward": 100.0,
        "episodes": 1000,
    }

    # Should not raise error
    manager.log_hyperparameters(hparams, final_metrics)

    manager.close()


def test_add_remove_subscriber(metrics_config):
    """Test adding and removing live broadcast subscribers."""
    manager = MetricsManager(metrics_config, experiment_name="test_exp")

    # Mock subscriber
    class MockSubscriber:
        def __init__(self):
            self.messages = []

        def send(self, message):
            self.messages.append(message)

    subscriber = MockSubscriber()

    # Add subscriber
    manager.add_subscriber(subscriber)
    assert len(manager.subscribers) == 1

    # Remove subscriber
    manager.remove_subscriber(subscriber)
    assert len(manager.subscribers) == 0

    manager.close()


def test_live_broadcast(metrics_config):
    """Test live broadcasting of metrics."""
    manager = MetricsManager(metrics_config, experiment_name="test_exp")

    # Mock subscriber
    class MockSubscriber:
        def __init__(self):
            self.messages = []

        def send(self, message):
            self.messages.append(message)

    subscriber = MockSubscriber()
    manager.add_subscriber(subscriber)

    # Log episode with broadcasting enabled
    metrics = {"reward": 100.0}
    manager.log_episode(episode=1, agent_id="agent_0", metrics=metrics)

    # Check subscriber received message
    assert len(subscriber.messages) == 1

    message = json.loads(subscriber.messages[0])
    assert message["type"] == "metrics"
    assert message["episode"] == 1
    assert message["agent_id"] == "agent_0"
    assert message["metrics"] == metrics

    manager.close()


def test_broadcast_with_failed_subscriber(metrics_config):
    """Test that failed subscribers are removed."""
    manager = MetricsManager(metrics_config, experiment_name="test_exp")

    # Mock subscriber that fails
    class FailingSubscriber:
        def send(self, message):
            raise Exception("Connection failed")

    subscriber = FailingSubscriber()
    manager.add_subscriber(subscriber)
    assert len(manager.subscribers) == 1

    # Log episode (should remove failed subscriber)
    metrics = {"reward": 100.0}
    manager.log_episode(episode=1, agent_id="agent_0", metrics=metrics)

    # Failed subscriber should be removed
    assert len(manager.subscribers) == 0

    manager.close()


def test_context_manager(metrics_config):
    """Test context manager functionality."""
    with MetricsManager(metrics_config, experiment_name="test_exp") as manager:
        metrics = {"reward": 100.0}
        manager.log_episode(episode=1, agent_id="agent_0", metrics=metrics)

    # Manager should be closed after context
    # Cannot easily verify, but should not raise error


def test_metrics_without_tensorboard(metrics_config):
    """Test metrics manager works without TensorBoard."""
    # Disable TensorBoard
    metrics_config.tensorboard = False

    manager = MetricsManager(metrics_config, experiment_name="test_exp")

    # Should still work
    metrics = {"reward": 100.0}
    manager.log_episode(episode=1, agent_id="agent_0", metrics=metrics)

    # Database should still have data
    results = manager.query_metrics()
    assert len(results) == 1

    manager.close()


def test_metrics_without_database(metrics_config):
    """Test metrics manager works without database."""
    # Disable database
    metrics_config.database = False

    manager = MetricsManager(metrics_config, experiment_name="test_exp")

    # Should still work
    metrics = {"reward": 100.0}
    manager.log_episode(episode=1, agent_id="agent_0", metrics=metrics)

    # Query should return empty list
    results = manager.query_metrics()
    assert len(results) == 0

    manager.close()


def test_metrics_without_replay_storage(metrics_config):
    """Test metrics manager works without replay storage."""
    # Disable replay storage
    metrics_config.replay_storage = False

    manager = MetricsManager(metrics_config, experiment_name="test_exp")

    trajectory = [{"state": [1.0], "action": 0, "reward": 1.0, "done": True}]

    # Should not raise error
    manager.save_episode_replay(
        episode=1,
        agent_id="agent_0",
        trajectory=trajectory,
    )

    manager.close()


def test_increment_episode_counter(metrics_config):
    """Test episode counter incrementation."""
    manager = MetricsManager(metrics_config, experiment_name="test_exp")

    assert manager.episode_count == 0

    manager.increment_episode()
    assert manager.episode_count == 1

    manager.increment_episode()
    assert manager.episode_count == 2

    manager.close()


def test_failure_event_queries_and_summary(metrics_config):
    """Ensure failure events can be queried and summarised."""
    manager = MetricsManager(metrics_config, experiment_name="test_exp")

    manager.log_failure_reason(episode=1, agent_id="agent_0", reason="energy_depleted")
    manager.log_failure_reason(episode=2, agent_id="agent_0", reason="money_depleted")
    manager.log_failure_reason(episode=3, agent_id="agent_1", reason="energy_depleted")

    all_events = manager.query_failure_events()
    assert len(all_events) == 3

    agent_events = manager.query_failure_events(agent_id="agent_0")
    assert len(agent_events) == 2

    limited_events = manager.query_failure_events(limit=1)
    assert len(limited_events) == 1

    summary = manager.get_failure_summary()
    summary_map = {f"{row['agent_id']}:{row['reason']}": row for row in summary}
    assert summary_map["agent_0:energy_depleted"]["count"] == 1
    assert summary_map["agent_0:money_depleted"]["count"] == 1
    assert summary_map["agent_1:energy_depleted"]["count"] == 1

    top_summary = manager.get_failure_summary(top_n=1)
    assert len(top_summary) == 1

    manager.close()

    # Re-open manager to ensure data is still accessible
    reopened = MetricsManager(metrics_config, experiment_name="test_exp")
    persisted = reopened.get_failure_summary(agent_id="agent_0")
    assert sum(row["count"] for row in persisted) == 2
    reopened.close()
