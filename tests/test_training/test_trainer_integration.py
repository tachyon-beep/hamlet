"""
Integration tests for Trainer.

Tests the complete training pipeline end-to-end.
"""

import pytest
import tempfile
from pathlib import Path
from hamlet.training.trainer import Trainer
from hamlet.training.config import (
    FullConfig,
    ExperimentConfig,
    EnvironmentConfig,
    AgentConfig,
    TrainingConfig,
    MetricsConfig,
)


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def minimal_config(temp_dir):
    """Create minimal configuration for fast testing."""
    return FullConfig(
        experiment=ExperimentConfig(
            name="test_trainer",
            description="Integration test",
            tracking_uri=str(temp_dir / "mlruns"),
        ),
        environment=EnvironmentConfig(
            grid_width=8,
            grid_height=8,
        ),
        agents=[
            AgentConfig(
                agent_id="agent_0",
                algorithm="dqn",
                state_dim=70,
                action_dim=5,
                learning_rate=1e-3,
                gamma=0.99,
            )
        ],
        training=TrainingConfig(
            num_episodes=5,  # Very short for testing
            max_steps_per_episode=50,
            batch_size=32,
            learning_starts=100,
            target_update_frequency=2,
            save_frequency=3,
            log_frequency=1,
            checkpoint_dir=str(temp_dir / "checkpoints"),
            replay_buffer_size=1000,
        ),
        metrics=MetricsConfig(
            tensorboard=False,  # Disable for faster tests
            database=True,
            database_path=str(temp_dir / "metrics.db"),
            replay_storage=False,  # Disable for faster tests
            live_broadcast=False,
        ),
    )


def test_trainer_initialization(minimal_config):
    """Test that trainer initializes correctly."""
    trainer = Trainer(minimal_config)

    assert trainer.config == minimal_config
    assert trainer.env is not None
    assert trainer.agent_manager is not None
    assert trainer.metrics_manager is not None
    assert trainer.experiment_manager is not None
    assert trainer.checkpoint_manager is not None
    assert trainer.global_step == 0


def test_trainer_agent_creation(minimal_config):
    """Test that trainer creates agents from config."""
    trainer = Trainer(minimal_config)

    agent_ids = trainer.agent_manager.get_agent_ids()
    assert len(agent_ids) == 1
    assert "agent_0" in agent_ids

    agent = trainer.agent_manager.get_agent("agent_0")
    assert agent is not None
    assert agent.agent_id == "agent_0"


def test_run_single_episode(minimal_config):
    """Test running a single episode."""
    # Configure for just 1 episode
    minimal_config.training.num_episodes = 1
    minimal_config.training.learning_starts = 0  # Learn immediately

    trainer = Trainer(minimal_config)

    # Run episode
    episode_metrics = trainer._run_episode(episode=0)

    # Check metrics structure
    assert "agent_0" in episode_metrics
    agent_metrics = episode_metrics["agent_0"]

    assert "total_reward" in agent_metrics
    assert "episode_length" in agent_metrics
    assert "epsilon" in agent_metrics
    assert "buffer_size" in agent_metrics

    # Check that episode was recorded
    assert len(trainer.episode_rewards["agent_0"]) == 1
    assert len(trainer.episode_lengths["agent_0"]) == 1


def test_complete_training_run(minimal_config):
    """Test complete training run with all components."""
    trainer = Trainer(minimal_config)

    # Run training (should not raise errors)
    trainer.train()

    # Verify training completed
    assert len(trainer.episode_rewards["agent_0"]) == minimal_config.training.num_episodes
    assert trainer.global_step > 0

    # Check that metrics were logged to database
    metrics = trainer.metrics_manager.query_metrics()
    assert len(metrics) > 0

    # Check that checkpoints were created
    checkpoints = trainer.checkpoint_manager.list_checkpoints()
    assert len(checkpoints) > 0  # At least final checkpoint


def test_checkpoint_saving(minimal_config):
    """Test that checkpoints are saved correctly."""
    # Configure to save every episode
    minimal_config.training.num_episodes = 3
    minimal_config.training.save_frequency = 1

    trainer = Trainer(minimal_config)
    trainer.train()

    # Check checkpoints
    checkpoints = trainer.checkpoint_manager.list_checkpoints()
    assert len(checkpoints) >= 2  # Episodes 1, 2, and final


def test_target_network_updates(minimal_config):
    """Test that target networks are updated."""
    minimal_config.training.num_episodes = 5
    minimal_config.training.target_update_frequency = 2

    trainer = Trainer(minimal_config)

    agent = trainer.agent_manager.get_agent("agent_0")
    initial_target_weights = agent.target_network.state_dict()

    # Run training
    trainer.train()

    # Target network should have been updated
    final_target_weights = agent.target_network.state_dict()

    # Weights should be different (at least one parameter)
    weights_changed = False
    for key in initial_target_weights.keys():
        if not (initial_target_weights[key] == final_target_weights[key]).all():
            weights_changed = True
            break

    assert weights_changed


def test_epsilon_decay(minimal_config):
    """Test that epsilon decays during training."""
    minimal_config.training.num_episodes = 10

    trainer = Trainer(minimal_config)

    agent = trainer.agent_manager.get_agent("agent_0")
    initial_epsilon = agent.epsilon

    # Run training
    trainer.train()

    final_epsilon = agent.epsilon

    # Epsilon should have decayed
    assert final_epsilon < initial_epsilon


def test_metrics_logging(minimal_config):
    """Test that metrics are logged correctly."""
    minimal_config.training.num_episodes = 3
    minimal_config.training.log_frequency = 1

    trainer = Trainer(minimal_config)
    trainer.train()

    # Query metrics from database
    metrics = trainer.metrics_manager.query_metrics(agent_id="agent_0")

    # Should have metrics for each episode
    assert len(metrics) >= 3  # Multiple metrics per episode

    # Check metric types
    metric_names = {m["metric_name"] for m in metrics}
    assert "total_reward" in metric_names
    assert "episode_length" in metric_names


def test_trainer_from_yaml(minimal_config, temp_dir):
    """Test creating trainer from YAML file."""
    # Save config to YAML
    yaml_path = temp_dir / "test_config.yaml"
    minimal_config.to_yaml(str(yaml_path))

    # Create trainer from YAML
    trainer = Trainer.from_yaml(str(yaml_path))

    assert trainer.config.experiment.name == minimal_config.experiment.name
    assert trainer.config.training.num_episodes == minimal_config.training.num_episodes


def test_get_hyperparameters(minimal_config):
    """Test getting hyperparameters for logging."""
    trainer = Trainer(minimal_config)

    hparams = trainer._get_hyperparameters()

    assert "num_episodes" in hparams
    assert "learning_rate" in hparams
    assert "gamma" in hparams
    assert "batch_size" in hparams
    assert "buffer_size" in hparams
    assert "num_agents" in hparams

    assert hparams["num_episodes"] == minimal_config.training.num_episodes
    assert hparams["num_agents"] == 1


def test_get_final_metrics(minimal_config):
    """Test getting final training metrics."""
    minimal_config.training.num_episodes = 5

    trainer = Trainer(minimal_config)
    trainer.train()

    final_metrics = trainer._get_final_metrics()

    assert "final_reward_mean" in final_metrics
    assert "final_reward_std" in final_metrics
    assert "final_length_mean" in final_metrics
    assert "total_episodes" in final_metrics
    assert "total_steps" in final_metrics

    assert final_metrics["total_episodes"] == 5
    assert final_metrics["total_steps"] > 0


def test_buffer_mode_single_agent(minimal_config):
    """Test that buffer mode is per_agent for single agent."""
    trainer = Trainer(minimal_config)

    assert trainer.agent_manager.buffer_mode == "per_agent"


def test_learning_starts(minimal_config):
    """Test that learning only starts after learning_starts steps."""
    minimal_config.training.learning_starts = 500
    minimal_config.training.num_episodes = 2

    trainer = Trainer(minimal_config)

    # Run first episode
    episode_metrics = trainer._run_episode(episode=0)

    # Learning shouldn't have started yet if episode is short
    assert trainer.global_step < minimal_config.training.learning_starts
