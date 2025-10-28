"""
Tests for ExperimentManager and CheckpointManager.
"""

import pytest
import tempfile
from pathlib import Path
from hamlet.training.experiment_manager import ExperimentManager, MLFLOW_AVAILABLE
from hamlet.training.checkpoint_manager import CheckpointManager
from hamlet.training.config import ExperimentConfig, AgentConfig
from hamlet.training.agent_manager import AgentManager


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def experiment_config(temp_dir):
    """Create experiment configuration for testing."""
    return ExperimentConfig(
        name="test_experiment",
        description="Test experiment",
        tracking_uri=str(temp_dir / "mlruns"),
    )


# ============================================================================
# ExperimentManager Tests
# ============================================================================


def test_experiment_manager_initialization(experiment_config):
    """Test that experiment manager initializes correctly."""
    manager = ExperimentManager(experiment_config)

    assert manager.config == experiment_config
    assert manager.active_run is None

    if MLFLOW_AVAILABLE:
        assert manager.experiment_id is not None


def test_start_and_end_run(experiment_config):
    """Test starting and ending an MLflow run."""
    if not MLFLOW_AVAILABLE:
        pytest.skip("MLflow not available")

    manager = ExperimentManager(experiment_config)

    # Start run
    manager.start_run(run_name="test_run")

    assert manager.active_run is not None
    assert manager.get_run_id() is not None

    # End run
    manager.end_run()

    assert manager.active_run is None


def test_log_params(experiment_config):
    """Test logging parameters."""
    if not MLFLOW_AVAILABLE:
        pytest.skip("MLflow not available")

    manager = ExperimentManager(experiment_config)
    manager.start_run()

    params = {
        "learning_rate": 0.001,
        "gamma": 0.99,
        "epsilon": 1.0,
    }

    # Should not raise error
    manager.log_params(params)

    manager.end_run()


def test_log_single_metric(experiment_config):
    """Test logging a single metric."""
    if not MLFLOW_AVAILABLE:
        pytest.skip("MLflow not available")

    manager = ExperimentManager(experiment_config)
    manager.start_run()

    # Should not raise error
    manager.log_metric("reward", 100.5, step=1)

    manager.end_run()


def test_log_multiple_metrics(experiment_config):
    """Test logging multiple metrics."""
    if not MLFLOW_AVAILABLE:
        pytest.skip("MLflow not available")

    manager = ExperimentManager(experiment_config)
    manager.start_run()

    metrics = {
        "reward": 100.5,
        "length": 250,
        "loss": 0.05,
    }

    # Should not raise error
    manager.log_metrics(metrics, step=1)

    manager.end_run()


def test_set_tags(experiment_config):
    """Test setting tags."""
    if not MLFLOW_AVAILABLE:
        pytest.skip("MLflow not available")

    manager = ExperimentManager(experiment_config)
    manager.start_run()

    # Single tag
    manager.set_tag("algorithm", "DQN")

    # Multiple tags
    manager.set_tags({"env": "HamletEnv", "version": "0.1.0"})

    manager.end_run()


def test_context_manager(experiment_config):
    """Test context manager functionality."""
    if not MLFLOW_AVAILABLE:
        pytest.skip("MLflow not available")

    with ExperimentManager(experiment_config) as manager:
        manager.start_run()
        manager.log_metric("test", 1.0)

    # Run should be ended after context


def test_experiment_manager_without_mlflow(experiment_config):
    """Test that experiment manager works gracefully without MLflow."""
    # Test that methods don't raise errors even if MLflow unavailable
    manager = ExperimentManager(experiment_config)

    manager.start_run()
    manager.log_params({"test": 1})
    manager.log_metric("test", 1.0)
    manager.log_metrics({"test": 1.0})
    manager.set_tag("test", "value")
    manager.set_tags({"test": "value"})
    manager.end_run()

    # Should not raise any errors


# ============================================================================
# CheckpointManager Tests
# ============================================================================


@pytest.fixture
def checkpoint_manager(temp_dir):
    """Create checkpoint manager for testing."""
    return CheckpointManager(
        checkpoint_dir=str(temp_dir / "checkpoints"),
        max_checkpoints=3,
        keep_best=True,
        metric_name="total_reward",
        metric_mode="max",
    )


@pytest.fixture
def test_agents():
    """Create test agents."""
    agent_manager = AgentManager()

    # Add two agents
    for i in range(2):
        config = AgentConfig(agent_id=f"agent_{i}", algorithm="dqn", state_dim=4, action_dim=2)
        agent_manager.add_agent(config)

    return {agent_id: agent_manager.get_agent(agent_id) for agent_id in agent_manager.get_agent_ids()}


def test_checkpoint_manager_initialization(checkpoint_manager, temp_dir):
    """Test that checkpoint manager initializes correctly."""
    assert checkpoint_manager.checkpoint_dir.exists()
    assert checkpoint_manager.max_checkpoints == 3
    assert checkpoint_manager.keep_best is True
    assert checkpoint_manager.metric_name == "total_reward"
    assert checkpoint_manager.metric_mode == "max"


def test_save_checkpoint(checkpoint_manager, test_agents):
    """Test saving a checkpoint."""
    episode = 10
    metrics = {"total_reward": 100.5, "episode_length": 250}

    checkpoint_path = checkpoint_manager.save_checkpoint(
        episode=episode,
        agents=test_agents,
        metrics=metrics,
    )

    assert checkpoint_path.exists()
    assert (checkpoint_path / "metadata.pt").exists()

    # Check agent files exist
    for agent_id in test_agents.keys():
        assert (checkpoint_path / f"{agent_id}.pt").exists()


def test_load_checkpoint(checkpoint_manager, test_agents):
    """Test loading a checkpoint."""
    # Save checkpoint
    episode = 10
    metrics = {"total_reward": 100.5}

    checkpoint_path = checkpoint_manager.save_checkpoint(
        episode=episode,
        agents=test_agents,
        metrics=metrics,
    )

    # Load checkpoint
    metadata = checkpoint_manager.load_checkpoint(checkpoint_path, test_agents)

    assert metadata["episode"] == episode
    assert metadata["metrics"] == metrics
    assert set(metadata["agent_ids"]) == set(test_agents.keys())


def test_load_checkpoint_not_found(checkpoint_manager, test_agents, temp_dir):
    """Test loading non-existent checkpoint raises error."""
    with pytest.raises(ValueError, match="Checkpoint not found"):
        checkpoint_manager.load_checkpoint(temp_dir / "nonexistent", test_agents)


def test_save_multiple_checkpoints(checkpoint_manager, test_agents):
    """Test saving multiple checkpoints."""
    for episode in range(5):
        metrics = {"total_reward": float(episode * 10)}
        checkpoint_manager.save_checkpoint(
            episode=episode,
            agents=test_agents,
            metrics=metrics,
        )

    # Should have max_checkpoints (3) saved
    checkpoints = checkpoint_manager.list_checkpoints()
    assert len(checkpoints) == 3


def test_keep_best_checkpoints(checkpoint_manager, test_agents):
    """Test that best checkpoints are kept."""
    # Save checkpoints with different rewards
    rewards = [10, 50, 20, 80, 30]  # Best: 80, 50, 30

    for episode, reward in enumerate(rewards):
        metrics = {"total_reward": reward}
        checkpoint_manager.save_checkpoint(
            episode=episode,
            agents=test_agents,
            metrics=metrics,
        )

    # Should keep top 3 by reward
    checkpoints = checkpoint_manager.list_checkpoints()
    assert len(checkpoints) == 3

    rewards_kept = [cp["metrics"]["total_reward"] for cp in checkpoints]
    assert set(rewards_kept) == {80, 50, 30}


def test_keep_recent_checkpoints(checkpoint_manager, test_agents):
    """Test keeping most recent checkpoints."""
    # Change to keep recent instead of best
    checkpoint_manager.keep_best = False

    # Save 5 checkpoints
    for episode in [0, 1, 2, 3, 4]:
        metrics = {"total_reward": 100.0}
        checkpoint_manager.save_checkpoint(
            episode=episode,
            agents=test_agents,
            metrics=metrics,
        )

    # Should keep episodes 2, 3, 4 (most recent 3)
    checkpoints = checkpoint_manager.list_checkpoints()
    assert len(checkpoints) == 3

    episodes_kept = [cp["episode"] for cp in checkpoints]
    assert set(episodes_kept) == {2, 3, 4}


def test_list_checkpoints(checkpoint_manager, test_agents):
    """Test listing checkpoints."""
    # Initially empty
    checkpoints = checkpoint_manager.list_checkpoints()
    assert len(checkpoints) == 0

    # Save some checkpoints
    for episode in range(3):
        metrics = {"total_reward": float(episode * 10)}
        checkpoint_manager.save_checkpoint(
            episode=episode,
            agents=test_agents,
            metrics=metrics,
        )

    # List checkpoints
    checkpoints = checkpoint_manager.list_checkpoints()
    assert len(checkpoints) == 3

    # Check structure
    for cp in checkpoints:
        assert "path" in cp
        assert "episode" in cp
        assert "metrics" in cp


def test_load_latest_checkpoint(checkpoint_manager, test_agents):
    """Test loading the latest checkpoint."""
    # Save checkpoints
    for episode in [0, 5, 10]:
        metrics = {"total_reward": 100.0}
        checkpoint_manager.save_checkpoint(
            episode=episode,
            agents=test_agents,
            metrics=metrics,
        )

    # Load latest
    metadata = checkpoint_manager.load_latest_checkpoint(test_agents)

    assert metadata is not None
    assert metadata["episode"] == 10  # Latest episode


def test_load_best_checkpoint(checkpoint_manager, test_agents):
    """Test loading the best checkpoint."""
    # Save checkpoints with different rewards
    for episode, reward in [(0, 10), (1, 50), (2, 30)]:
        metrics = {"total_reward": reward}
        checkpoint_manager.save_checkpoint(
            episode=episode,
            agents=test_agents,
            metrics=metrics,
        )

    # Load best
    metadata = checkpoint_manager.load_best_checkpoint(test_agents)

    assert metadata is not None
    assert metadata["episode"] == 1  # Episode with reward 50


def test_load_latest_no_checkpoints(checkpoint_manager, test_agents):
    """Test loading latest with no checkpoints returns None."""
    metadata = checkpoint_manager.load_latest_checkpoint(test_agents)
    assert metadata is None


def test_load_best_no_checkpoints(checkpoint_manager, test_agents):
    """Test loading best with no checkpoints returns None."""
    metadata = checkpoint_manager.load_best_checkpoint(test_agents)
    assert metadata is None


def test_delete_checkpoint(checkpoint_manager, test_agents):
    """Test deleting a checkpoint."""
    # Save checkpoint
    checkpoint_path = checkpoint_manager.save_checkpoint(
        episode=10,
        agents=test_agents,
        metrics={"total_reward": 100.0},
    )

    assert checkpoint_path.exists()

    # Delete checkpoint
    checkpoint_manager.delete_checkpoint(checkpoint_path)

    assert not checkpoint_path.exists()
    assert len(checkpoint_manager.checkpoints) == 0


def test_get_checkpoint_info(checkpoint_manager, test_agents):
    """Test getting checkpoint info."""
    # Initially empty
    info = checkpoint_manager.get_checkpoint_info()
    assert info["num_checkpoints"] == 0
    assert info["latest_episode"] is None

    # Save checkpoints
    for episode, reward in [(0, 10), (1, 50), (2, 30)]:
        metrics = {"total_reward": reward}
        checkpoint_manager.save_checkpoint(
            episode=episode,
            agents=test_agents,
            metrics=metrics,
        )

    # Get info
    info = checkpoint_manager.get_checkpoint_info()
    assert info["num_checkpoints"] == 3
    assert info["latest_episode"] == 2
    assert info["best_metric_value"] == 50
    assert info["best_episode"] == 1


def test_checkpoint_with_metadata(checkpoint_manager, test_agents):
    """Test saving checkpoint with additional metadata."""
    metadata = {
        "training_time": 123.45,
        "git_hash": "abc123",
    }

    checkpoint_path = checkpoint_manager.save_checkpoint(
        episode=10,
        agents=test_agents,
        metrics={"total_reward": 100.0},
        metadata=metadata,
    )

    # Load and verify metadata
    loaded_metadata = checkpoint_manager.load_checkpoint(checkpoint_path, test_agents)

    assert loaded_metadata["training_time"] == 123.45
    assert loaded_metadata["git_hash"] == "abc123"


def test_min_metric_mode(checkpoint_manager, test_agents):
    """Test checkpoint manager with min metric mode."""
    # Change to minimize metric (e.g., loss)
    checkpoint_manager.metric_mode = "min"
    checkpoint_manager.metric_name = "loss"

    # Save checkpoints with different losses
    for episode, loss in [(0, 1.0), (1, 0.1), (2, 0.5)]:
        metrics = {"loss": loss}
        checkpoint_manager.save_checkpoint(
            episode=episode,
            agents=test_agents,
            metrics=metrics,
        )

    # Load best (lowest loss)
    metadata = checkpoint_manager.load_best_checkpoint(test_agents)

    assert metadata["episode"] == 1  # Episode with loss 0.1
