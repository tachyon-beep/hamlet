"""
TDD Tests for P1.1: Complete Checkpointing System

Tests verify that all critical state is saved and restored correctly.
This ensures safe resume after interruption during multi-day training.

Test Structure:
1. Checkpoint Versioning
2. Target Network State (recurrent mode)
3. Training Counters
4. Curriculum State
5. Replay Buffer State
6. Environment Layout (affordance positions)
"""

import pytest
import torch

from townlet.curriculum.adversarial import AdversarialCurriculum
from townlet.curriculum.static import StaticCurriculum
from townlet.environment.vectorized_env import VectorizedHamletEnv
from townlet.exploration.adaptive_intrinsic import AdaptiveIntrinsicExploration
from townlet.population.vectorized import VectorizedPopulation
from townlet.training.replay_buffer import ReplayBuffer
from townlet.training.sequential_replay_buffer import SequentialReplayBuffer


@pytest.fixture
def device():
    """Test device (CPU for tests)."""
    return torch.device("cpu")


@pytest.fixture
def simple_env(device):
    """Simple feedforward environment."""
    return VectorizedHamletEnv(
        num_agents=1,
        grid_size=5,
        device=device,
        partial_observability=False,
        enable_temporal_mechanics=False,
    )


@pytest.fixture
def recurrent_env(device):
    """Recurrent LSTM environment."""
    return VectorizedHamletEnv(
        num_agents=1,
        grid_size=5,
        device=device,
        partial_observability=True,
        vision_range=2,
        enable_temporal_mechanics=False,
    )


@pytest.fixture
def curriculum(device):
    """Static curriculum for testing."""
    return StaticCurriculum(
        difficulty_level=0.5,
        reward_mode="shaped",
        depletion_multiplier=0.5,
    )


@pytest.fixture
def adversarial_curriculum(device):
    """Adversarial curriculum with state."""
    curriculum = AdversarialCurriculum(
        max_steps_per_episode=500,
        device=device,
    )
    curriculum.initialize_population(num_agents=1)
    return curriculum


def create_exploration(obs_dim: int, device):
    """Helper to create exploration strategy with correct obs_dim."""
    return AdaptiveIntrinsicExploration(
        obs_dim=obs_dim,
        device=device,
        initial_intrinsic_weight=1.0,
        variance_threshold=100.0,
        survival_window=100,
    )


class TestCheckpointVersioning:
    """Test that checkpoints include version information."""

    def test_checkpoint_has_version_field(self, simple_env, curriculum, device):
        """Checkpoint should include version number for backwards compatibility."""
        exploration = create_exploration(simple_env.observation_dim, device)
        population = VectorizedPopulation(
            env=simple_env,
            curriculum=curriculum,
            exploration=exploration,
            agent_ids=[0],
            device=device,
            obs_dim=simple_env.observation_dim,
            action_dim=6,
            network_type="simple",
        )

        checkpoint = population.get_checkpoint_state()

        assert "version" in checkpoint
        assert isinstance(checkpoint["version"], int)
        assert checkpoint["version"] >= 2  # v2 includes all P1.1 fixes


class TestTargetNetworkCheckpointing:
    """Test target network state is saved and restored for all network types."""

    def test_target_network_saved_when_exists(self, recurrent_env, curriculum, device):
        """Target network state should be in checkpoint if network exists."""
        exploration = create_exploration(recurrent_env.observation_dim, device)
        population = VectorizedPopulation(
            env=recurrent_env,
            curriculum=curriculum,
            exploration=exploration,
            agent_ids=[0],
            device=device,
            obs_dim=recurrent_env.observation_dim,
            action_dim=6,
            network_type="recurrent",
        )

        checkpoint = population.get_checkpoint_state()

        assert "target_network" in checkpoint
        assert checkpoint["target_network"] is not None
        assert isinstance(checkpoint["target_network"], dict)

    def test_target_network_saved_for_simple_network(self, simple_env, curriculum, device):
        """Target network should also be saved for simple feed-forward networks."""
        exploration = create_exploration(simple_env.observation_dim, device)
        population = VectorizedPopulation(
            env=simple_env,
            curriculum=curriculum,
            exploration=exploration,
            agent_ids=[0],
            device=device,
            obs_dim=simple_env.observation_dim,
            action_dim=6,
            network_type="simple",
        )

        checkpoint = population.get_checkpoint_state()

        assert "target_network" in checkpoint
        assert checkpoint["target_network"] is not None
        assert isinstance(checkpoint["target_network"], dict)

    def test_target_network_restored_correctly(self, recurrent_env, curriculum, device):
        """Target network weights should match after save/load cycle."""
        exploration = create_exploration(recurrent_env.observation_dim, device)
        population = VectorizedPopulation(
            env=recurrent_env,
            curriculum=curriculum,
            exploration=exploration,
            agent_ids=[0],
            device=device,
            obs_dim=recurrent_env.observation_dim,
            action_dim=6,
            network_type="recurrent",
        )

        # Train a bit to change weights
        recurrent_env.reset()
        population.reset()
        for _ in range(10):
            population.step_population(recurrent_env)

        # Save checkpoint
        checkpoint = population.get_checkpoint_state()
        original_target_weights = checkpoint["target_network"]["lstm.weight_ih_l0"].clone()

        # Create new population and restore
        new_population = VectorizedPopulation(
            env=recurrent_env,
            curriculum=curriculum,
            exploration=exploration,
            agent_ids=[0],
            device=device,
            obs_dim=recurrent_env.observation_dim,
            action_dim=6,
            network_type="recurrent",
        )
        new_population.load_checkpoint_state(checkpoint)

        # Verify target network matches
        restored_target_weights = new_population.target_network.state_dict()["lstm.weight_ih_l0"]
        assert torch.allclose(original_target_weights, restored_target_weights)


class TestTrainingCountersCheckpointing:
    """Test training counters are saved and restored."""

    def test_total_steps_saved(self, simple_env, curriculum, device):
        """total_steps counter should be in checkpoint."""
        exploration = create_exploration(simple_env.observation_dim, device)
        population = VectorizedPopulation(
            env=simple_env,
            curriculum=curriculum,
            exploration=exploration,
            agent_ids=[0],
            device=device,
            obs_dim=simple_env.observation_dim,
            action_dim=6,
            network_type="simple",
        )

        # Simulate some training
        simple_env.reset()
        population.reset()
        for _ in range(50):
            population.step_population(simple_env)

        checkpoint = population.get_checkpoint_state()

        assert "total_steps" in checkpoint
        assert checkpoint["total_steps"] == population.total_steps
        assert checkpoint["total_steps"] > 0

    def test_training_step_counter_saved_recurrent(self, recurrent_env, curriculum, device):
        """training_step_counter should be saved for recurrent networks."""
        exploration = create_exploration(recurrent_env.observation_dim, device)
        population = VectorizedPopulation(
            env=recurrent_env,
            curriculum=curriculum,
            exploration=exploration,
            agent_ids=[0],
            device=device,
            obs_dim=recurrent_env.observation_dim,
            action_dim=6,
            network_type="recurrent",
        )

        # Simulate training
        recurrent_env.reset()
        population.reset()
        for _ in range(20):
            population.step_population(recurrent_env)

        checkpoint = population.get_checkpoint_state()

        assert "training_step_counter" in checkpoint
        assert checkpoint["training_step_counter"] == population.training_step_counter

    def test_counters_restored_correctly(self, simple_env, curriculum, device):
        """Counters should restore to exact values."""
        exploration = create_exploration(simple_env.observation_dim, device)
        population = VectorizedPopulation(
            env=simple_env,
            curriculum=curriculum,
            exploration=exploration,
            agent_ids=[0],
            device=device,
            obs_dim=simple_env.observation_dim,
            action_dim=6,
            network_type="simple",
        )

        # Train to specific step count
        simple_env.reset()
        population.reset()
        for _ in range(73):  # Odd number to test exact restoration
            population.step_population(simple_env)

        original_total_steps = population.total_steps
        checkpoint = population.get_checkpoint_state()

        # Create new population and restore
        new_population = VectorizedPopulation(
            env=simple_env,
            curriculum=curriculum,
            exploration=exploration,
            agent_ids=[0],
            device=device,
            obs_dim=simple_env.observation_dim,
            action_dim=6,
            network_type="simple",
        )
        new_population.load_checkpoint_state(checkpoint)

        assert new_population.total_steps == original_total_steps


class TestCurriculumCheckpointing:
    """Test curriculum state is saved and restored."""

    def test_adversarial_curriculum_checkpoint_method_exists(self, adversarial_curriculum):
        """AdversarialCurriculum should have checkpoint_state method."""
        assert hasattr(adversarial_curriculum, "checkpoint_state")
        assert callable(adversarial_curriculum.checkpoint_state)

    def test_adversarial_curriculum_load_method_exists(self, adversarial_curriculum):
        """AdversarialCurriculum should have load_checkpoint_state method."""
        assert hasattr(adversarial_curriculum, "load_checkpoint_state")
        assert callable(adversarial_curriculum.load_checkpoint_state)

    def test_adversarial_curriculum_stage_saved(self, adversarial_curriculum):
        """Curriculum stage should be in checkpoint."""
        # Advance to stage 2
        adversarial_curriculum.tracker.agent_stages[0] = 2

        checkpoint = adversarial_curriculum.checkpoint_state()

        assert "agent_stages" in checkpoint
        assert checkpoint["agent_stages"][0] == 2

    def test_adversarial_curriculum_tracker_state_saved(self, adversarial_curriculum):
        """All tracker state should be in checkpoint."""
        # Set some tracker state
        adversarial_curriculum.tracker.steps_at_stage[0] = 1500
        adversarial_curriculum.tracker.prev_avg_reward[0] = 42.5

        checkpoint = adversarial_curriculum.checkpoint_state()

        assert "steps_at_stage" in checkpoint
        assert "prev_avg_reward" in checkpoint
        assert checkpoint["steps_at_stage"][0] == 1500
        assert abs(checkpoint["prev_avg_reward"][0] - 42.5) < 1e-6

    def test_adversarial_curriculum_restored_correctly(self, device):
        """Curriculum should restore to exact state after save/load."""
        curriculum = AdversarialCurriculum(
            max_steps_per_episode=500,
            device=device,
        )
        curriculum.initialize_population(num_agents=1)

        # Set specific state
        curriculum.tracker.agent_stages[0] = 3
        curriculum.tracker.steps_at_stage[0] = 2500
        curriculum.tracker.prev_avg_reward[0] = 123.45

        checkpoint = curriculum.checkpoint_state()

        # Create new curriculum and restore
        new_curriculum = AdversarialCurriculum(
            max_steps_per_episode=500,
            device=device,
        )
        new_curriculum.initialize_population(num_agents=1)
        new_curriculum.load_checkpoint_state(checkpoint)

        assert new_curriculum.tracker.agent_stages[0] == 3
        assert new_curriculum.tracker.steps_at_stage[0] == 2500
        assert abs(new_curriculum.tracker.prev_avg_reward[0] - 123.45) < 1e-6


class TestReplayBufferCheckpointing:
    """Test replay buffer contents are saved and restored."""

    def test_replay_buffer_serialize_method_exists(self, device):
        """ReplayBuffer should have serialize method."""
        buffer = ReplayBuffer(capacity=1000, device=device)
        assert hasattr(buffer, "serialize")
        assert callable(buffer.serialize)

    def test_replay_buffer_deserialize_method_exists(self, device):
        """ReplayBuffer should have load_from_serialized method."""
        buffer = ReplayBuffer(capacity=1000, device=device)
        assert hasattr(buffer, "load_from_serialized")
        assert callable(buffer.load_from_serialized)

    def test_replay_buffer_serializes_contents(self, device):
        """Buffer serialization should include all transitions."""
        buffer = ReplayBuffer(capacity=100, device=device)

        # Add some transitions
        for i in range(10):
            obs = torch.randn(1, 40)
            action = torch.tensor([i % 5])
            reward_ext = torch.tensor([float(i)])
            reward_int = torch.tensor([float(i * 2)])
            next_obs = torch.randn(1, 40)
            done = torch.tensor([False])

            buffer.push(obs, action, reward_ext, reward_int, next_obs, done)

        serialized = buffer.serialize()

        assert "size" in serialized
        assert serialized["size"] == 10
        assert "position" in serialized
        assert "observations" in serialized
        assert "actions" in serialized
        assert "rewards_extrinsic" in serialized
        assert "rewards_intrinsic" in serialized
        assert "next_observations" in serialized
        assert "dones" in serialized

    def test_replay_buffer_restores_correctly(self, device):
        """Buffer should restore exact contents after serialize/deserialize."""
        buffer = ReplayBuffer(capacity=100, device=device)

        # Add transitions
        expected_actions = []
        for i in range(10):
            obs = torch.randn(1, 40)
            action = torch.tensor([i % 5])
            expected_actions.append(action.item())
            reward_ext = torch.tensor([float(i)])
            reward_int = torch.tensor([float(i * 2)])
            next_obs = torch.randn(1, 40)
            done = torch.tensor([False])

            buffer.push(obs, action, reward_ext, reward_int, next_obs, done)

        # Serialize
        serialized = buffer.serialize()

        # Create new buffer and restore
        new_buffer = ReplayBuffer(capacity=100, device=device)
        new_buffer.load_from_serialized(serialized)

        # Verify size matches
        assert len(new_buffer) == len(buffer)

        # Verify actions match
        for i, expected_action in enumerate(expected_actions):
            assert new_buffer.actions[i].item() == expected_action

    def test_sequential_buffer_serialize_method_exists(self, device):
        """SequentialReplayBuffer should have serialize method."""
        buffer = SequentialReplayBuffer(capacity=1000, device=device)
        assert hasattr(buffer, "serialize")
        assert callable(buffer.serialize)

    def test_sequential_buffer_deserialize_method_exists(self, device):
        """SequentialReplayBuffer should have load_from_serialized method."""
        buffer = SequentialReplayBuffer(capacity=1000, device=device)
        assert hasattr(buffer, "load_from_serialized")
        assert callable(buffer.load_from_serialized)

    def test_sequential_buffer_serializes_episodes(self, device):
        """Sequential buffer should serialize complete episodes."""
        buffer = SequentialReplayBuffer(capacity=1000, device=device)

        # Add an episode
        episode = {
            "observations": torch.stack([torch.randn(40) for _ in range(5)]),
            "actions": torch.stack([torch.tensor(i % 5) for i in range(5)]),
            "rewards_extrinsic": torch.stack([torch.tensor(float(i)) for i in range(5)]),
            "rewards_intrinsic": torch.stack([torch.tensor(float(i * 2)) for i in range(5)]),
            "dones": torch.stack([torch.tensor(False) for _ in range(5)]),
        }
        buffer.store_episode(episode)

        serialized = buffer.serialize()

        assert "num_transitions" in serialized
        assert serialized["num_transitions"] == 5
        assert "episodes" in serialized
        assert len(serialized["episodes"]) == 1

    def test_sequential_buffer_restores_correctly(self, device):
        """Sequential buffer should restore episodes after serialize/deserialize."""
        buffer = SequentialReplayBuffer(capacity=1000, device=device)

        # Add episode
        episode = {
            "observations": torch.stack([torch.randn(40) for _ in range(5)]),
            "actions": torch.stack([torch.tensor(i % 5) for i in range(5)]),
            "rewards_extrinsic": torch.stack([torch.tensor(float(i)) for i in range(5)]),
            "rewards_intrinsic": torch.stack([torch.tensor(float(i * 2)) for i in range(5)]),
            "dones": torch.stack([torch.tensor(False) for _ in range(5)]),
        }
        buffer.store_episode(episode)

        original_actions = episode["actions"].clone()

        # Serialize
        serialized = buffer.serialize()

        # Create new buffer and restore
        new_buffer = SequentialReplayBuffer(capacity=1000, device=device)
        new_buffer.load_from_serialized(serialized)

        # Verify episode count
        assert len(new_buffer) == 1
        assert new_buffer.num_transitions == 5

        # Verify actions match
        restored_episode = new_buffer.episodes[0]
        assert torch.allclose(restored_episode["actions"], original_actions)


class TestEnvironmentLayoutCheckpointing:
    """Test affordance positions are saved and restored."""

    def test_env_get_affordance_positions_method_exists(self, simple_env):
        """Environment should have get_affordance_positions method."""
        assert hasattr(simple_env, "get_affordance_positions")
        assert callable(simple_env.get_affordance_positions)

    def test_env_set_affordance_positions_method_exists(self, simple_env):
        """Environment should have set_affordance_positions method."""
        assert hasattr(simple_env, "set_affordance_positions")
        assert callable(simple_env.set_affordance_positions)

    def test_get_affordance_positions_returns_dict(self, simple_env):
        """get_affordance_positions should return dict of positions."""
        simple_env.reset()
        data = simple_env.get_affordance_positions()

        assert isinstance(data, dict)
        positions = data.get("positions", data)
        assert isinstance(positions, dict)
        assert len(positions) > 0  # Should have affordances

        # Check a known affordance
        assert "Bed" in positions
        assert isinstance(positions["Bed"], list)
        assert len(positions["Bed"]) == 2  # [row, col]

    def test_set_affordance_positions_restores_layout(self, simple_env):
        """set_affordance_positions should restore exact positions."""
        simple_env.reset()

        # Get original positions
        original_data = simple_env.get_affordance_positions()
        original_positions = original_data.get("positions", original_data)
        original_bed_pos = original_positions["Bed"].copy()

        # Randomize layout
        simple_env.randomize_affordance_positions()
        new_data = simple_env.get_affordance_positions()
        new_positions = new_data.get("positions", new_data)

        # Verify layout changed
        assert new_positions["Bed"] != original_bed_pos

        # Restore original layout
        simple_env.set_affordance_positions(original_data)
        restored_data = simple_env.get_affordance_positions()
        restored_positions = restored_data.get("positions", restored_data)

        # Verify exact restoration
        assert restored_positions["Bed"] == original_bed_pos

    def test_affordance_positions_survive_reset(self, simple_env):
        """Set positions should persist across env.reset()."""
        simple_env.reset()

        # Set specific positions
        custom_data = simple_env.get_affordance_positions()
        positions = custom_data.get("positions", custom_data)
        positions["Bed"] = [1, 1]
        simple_env.set_affordance_positions(custom_data)

        # Reset environment
        simple_env.reset()

        # Verify positions persisted
        restored_data = simple_env.get_affordance_positions()
        restored_positions = restored_data.get("positions", restored_data)
        assert restored_positions["Bed"] == [1, 1]
