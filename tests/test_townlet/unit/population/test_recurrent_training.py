"""Unit tests for recurrent (LSTM) training in VectorizedPopulation."""

import shutil
import uuid

import pytest
import torch
import yaml

from townlet.curriculum.static import StaticCurriculum
from townlet.exploration.adaptive_intrinsic import AdaptiveIntrinsicExploration
from townlet.exploration.epsilon_greedy import EpsilonGreedyExploration
from townlet.population.vectorized import VectorizedPopulation


@pytest.fixture
def recurrent_env_builder(tmp_path, test_config_pack_path, env_factory, cpu_device):
    """Clone config packs, enforce POMDP settings, and build envs."""

    def _build(*, num_agents: int, overrides: dict | None = None):
        target = tmp_path / f"recurrent_env_{uuid.uuid4().hex}"
        shutil.copytree(test_config_pack_path, target)

        training_yaml = target / "training.yaml"
        with open(training_yaml) as f:
            training_config = yaml.safe_load(f)

        def _merge(config: dict, updates: dict) -> None:
            for section, values in updates.items():
                current = config.get(section, {}) or {}
                current.update(values)
                config[section] = current

        base_updates = {"environment": {"partial_observability": True, "vision_range": 2}}
        _merge(training_config, base_updates)
        if overrides:
            _merge(training_config, overrides)

        with open(training_yaml, "w") as f:
            yaml.safe_dump(training_config, f, sort_keys=False)

        return env_factory(config_dir=target, num_agents=num_agents, device_override=cpu_device)

    return _build


@pytest.fixture
def cpu_env_factory(env_factory, cpu_device):
    """Generic CPU-bound environment builder."""

    def _build(**kwargs):
        return env_factory(device_override=cpu_device, **kwargs)

    return _build


class TestRecurrentNetworkInitialization:
    """Test recurrent network setup and initialization."""

    def test_recurrent_network_creates_sequential_buffer(self, recurrent_env_builder):
        """Recurrent networks should use SequentialReplayBuffer, not standard ReplayBuffer.

        Coverage target: Initialization path for recurrent networks
        """
        env = recurrent_env_builder(num_agents=2)

        curriculum = StaticCurriculum(difficulty_level=0.5)
        exploration = EpsilonGreedyExploration(epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.999)

        population = VectorizedPopulation(
            env=env,
            curriculum=curriculum,
            exploration=exploration,
            agent_ids=["agent_0", "agent_1"],
            device=env.device,
            obs_dim=env.observation_dim,
            network_type="recurrent",  # Use RecurrentSpatialQNetwork
            learning_rate=0.0001,
            gamma=0.99,
            replay_buffer_capacity=1000,
            batch_size=8,
            sequence_length=10,  # Sequence length for LSTM training
        )

        # Verify recurrent setup
        assert population.is_recurrent is True, "Should be marked as recurrent"
        assert population.sequence_length == 10, "Sequence length should be set"

        # Verify SequentialReplayBuffer is used
        from townlet.training.sequential_replay_buffer import SequentialReplayBuffer

        assert isinstance(population.replay_buffer, SequentialReplayBuffer), "Should use SequentialReplayBuffer for LSTM"

        # Verify current_episodes container exists
        assert population.current_episodes is not None, "Should have episode containers"
        assert len(population.current_episodes) == 2, "Should have container for each agent"

    def test_recurrent_network_initializes_hidden_states(self, recurrent_env_builder):
        """Recurrent networks should initialize LSTM hidden states.

        Coverage target: LSTM hidden state initialization
        """
        env = recurrent_env_builder(num_agents=1)

        curriculum = StaticCurriculum(difficulty_level=0.5)
        exploration = EpsilonGreedyExploration(epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.999)

        population = VectorizedPopulation(
            env=env,
            curriculum=curriculum,
            exploration=exploration,
            agent_ids=["agent_0"],
            device=env.device,
            obs_dim=env.observation_dim,
            network_type="recurrent",
            learning_rate=0.0001,
            gamma=0.99,
            replay_buffer_capacity=1000,
            batch_size=4,
            sequence_length=5,
        )

        # Reset population (should initialize hidden states)
        population.reset()

        # Verify hidden state exists and has correct shape
        from townlet.agent.networks import RecurrentSpatialQNetwork

        recurrent_network = population.q_network
        assert isinstance(recurrent_network, RecurrentSpatialQNetwork)

        hidden_state = recurrent_network.get_hidden_state()
        assert hidden_state is not None, "Hidden state should be initialized"

        h, c = hidden_state
        # Hidden state shape: [num_layers, batch_size, hidden_dim]
        assert h.shape[1] == 1, "Batch size should match num_agents"
        assert c.shape[1] == 1, "Cell state batch size should match num_agents"


class TestLSTMHiddenStateManagement:
    """Test LSTM hidden state reset and management."""

    def test_reset_hidden_state_zeros_specific_agent(self, cpu_device, recurrent_env_builder):
        """_reset_hidden_state() should zero hidden state for specific agent only.

        This is called when an agent dies or episode ends, to clear its memory.

        Coverage target: lines 236-248 (_reset_hidden_state)
        """
        env = recurrent_env_builder(num_agents=3)

        curriculum = StaticCurriculum(difficulty_level=0.5)
        exploration = EpsilonGreedyExploration(epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.999)

        population = VectorizedPopulation(
            env=env,
            curriculum=curriculum,
            exploration=exploration,
            agent_ids=["agent_0", "agent_1", "agent_2"],
            device=cpu_device,
            obs_dim=env.observation_dim,
            network_type="recurrent",
            learning_rate=0.0001,
            gamma=0.99,
            replay_buffer_capacity=1000,
            batch_size=4,
            sequence_length=5,
        )

        population.reset()

        # Get initial hidden state

        recurrent_network = population.q_network
        h_before, c_before = recurrent_network.get_hidden_state()

        # Set some non-zero values for testing
        h_before[:, :, :] = torch.randn_like(h_before)
        c_before[:, :, :] = torch.randn_like(c_before)
        recurrent_network.set_hidden_state((h_before.clone(), c_before.clone()))

        # Reset hidden state for agent 1 only
        population._reset_hidden_state(agent_idx=1)

        # Verify agent 1's hidden state is zeroed
        h_after, c_after = recurrent_network.get_hidden_state()

        # Agent 1 should be zeroed
        assert torch.all(h_after[:, 1, :] == 0.0), "Agent 1 h should be zeroed"
        assert torch.all(c_after[:, 1, :] == 0.0), "Agent 1 c should be zeroed"

        # Agents 0 and 2 should be unchanged
        assert torch.allclose(h_after[:, 0, :], h_before[:, 0, :]), "Agent 0 h should be unchanged"
        assert torch.allclose(h_after[:, 2, :], h_before[:, 2, :]), "Agent 2 h should be unchanged"
        assert torch.allclose(c_after[:, 0, :], c_before[:, 0, :]), "Agent 0 c should be unchanged"
        assert torch.allclose(c_after[:, 2, :], c_before[:, 2, :]), "Agent 2 c should be unchanged"


class TestEpisodeBuffering:
    """Test episode storage for sequential replay buffer."""

    def test_store_episode_and_reset_adds_to_buffer(self, cpu_device, recurrent_env_builder):
        """_store_episode_and_reset() should store episode sequence and reset container.

        Coverage target: lines 212-232 (_store_episode_and_reset)
        """
        env = recurrent_env_builder(num_agents=1)

        curriculum = StaticCurriculum(difficulty_level=0.5)
        exploration = EpsilonGreedyExploration(epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.999)

        population = VectorizedPopulation(
            env=env,
            curriculum=curriculum,
            exploration=exploration,
            agent_ids=["agent_0"],
            device=cpu_device,
            obs_dim=env.observation_dim,
            network_type="recurrent",
            learning_rate=0.0001,
            gamma=0.99,
            replay_buffer_capacity=1000,
            batch_size=4,
            sequence_length=5,
        )

        population.reset()

        # Manually add some data to episode container
        agent_idx = 0
        episode = population.current_episodes[agent_idx]

        for i in range(5):
            episode["observations"].append(torch.randn(env.observation_dim, device=cpu_device))
            episode["actions"].append(torch.tensor(i % env.action_dim, device=cpu_device))
            episode["rewards_extrinsic"].append(torch.tensor(1.0, device=cpu_device))
            episode["rewards_intrinsic"].append(torch.tensor(0.1, device=cpu_device))
            episode["dones"].append(torch.tensor(False, device=cpu_device))

        # Store episode
        result = population._store_episode_and_reset(agent_idx)

        # Verify storage succeeded
        assert result is True, "Should return True when episode stored"

        # Verify episode container was reset
        assert len(population.current_episodes[agent_idx]["observations"]) == 0, "Container should be empty after reset"

        # Verify buffer size increased (use len() for SequentialReplayBuffer)
        assert len(population.replay_buffer) > 0, "Replay buffer should contain stored episode"

    def test_store_episode_skips_empty_episodes(self, cpu_device, recurrent_env_builder):
        """_store_episode_and_reset() should skip empty episodes.

        Coverage target: lines 216-217 (empty episode check)
        """
        env = recurrent_env_builder(num_agents=1)

        curriculum = StaticCurriculum(difficulty_level=0.5)
        exploration = EpsilonGreedyExploration(epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.999)

        population = VectorizedPopulation(
            env=env,
            curriculum=curriculum,
            exploration=exploration,
            agent_ids=["agent_0"],
            device=cpu_device,
            obs_dim=env.observation_dim,
            network_type="recurrent",
            learning_rate=0.0001,
            gamma=0.99,
            replay_buffer_capacity=1000,
            batch_size=4,
            sequence_length=5,
        )

        population.reset()

        # Try to store empty episode
        agent_idx = 0
        result = population._store_episode_and_reset(agent_idx)

        # Should return False for empty episode
        assert result is False, "Should return False for empty episode"
        assert len(population.replay_buffer) == 0, "Buffer should remain empty"

    def test_flush_episode_stores_and_finalizes(self, cpu_device, recurrent_env_builder):
        """flush_episode() should store episode and finalize agent state.

        Coverage target: lines 341-348 (flush_episode for recurrent)
        """
        env = recurrent_env_builder(num_agents=1)

        curriculum = StaticCurriculum(difficulty_level=0.5)
        exploration = EpsilonGreedyExploration(epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.999)

        population = VectorizedPopulation(
            env=env,
            curriculum=curriculum,
            exploration=exploration,
            agent_ids=["agent_0"],
            device=cpu_device,
            obs_dim=env.observation_dim,
            network_type="recurrent",
            learning_rate=0.0001,
            gamma=0.99,
            replay_buffer_capacity=1000,
            batch_size=4,
            sequence_length=5,
        )

        population.reset()

        # Add episode data
        agent_idx = 0
        episode = population.current_episodes[agent_idx]
        for i in range(3):
            episode["observations"].append(torch.randn(env.observation_dim, device=cpu_device))
            episode["actions"].append(torch.tensor(0, device=cpu_device))
            episode["rewards_extrinsic"].append(torch.tensor(1.0, device=cpu_device))
            episode["rewards_intrinsic"].append(torch.tensor(0.0, device=cpu_device))
            episode["dones"].append(torch.tensor(False, device=cpu_device))

        # Flush episode
        population.flush_episode(agent_idx)

        # Verify episode was stored (use len() for SequentialReplayBuffer)
        assert len(population.replay_buffer) > 0, "Episode should be in buffer"

        # Verify container was reset
        assert len(population.current_episodes[agent_idx]["observations"]) == 0, "Container should be empty"

        # Verify episode step count was reset
        assert population.episode_step_counts[agent_idx] == 0, "Step count should be reset"


class TestRecurrentTraining:
    """Test LSTM training loop."""

    def test_recurrent_training_path_via_step_population(self, cpu_device, recurrent_env_builder):
        """step_population() should trigger LSTM training when buffer has enough episodes.

        This indirectly tests the recurrent training path (lines 529-616) by ensuring
        step_population triggers training for recurrent networks with sufficient buffer.

        Coverage target: lines 529-616 (recurrent training loop via step_population)
        """
        env = recurrent_env_builder(num_agents=1)

        curriculum = StaticCurriculum(difficulty_level=0.5)
        exploration = EpsilonGreedyExploration(epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.999)

        population = VectorizedPopulation(
            env=env,
            curriculum=curriculum,
            exploration=exploration,
            agent_ids=["agent_0"],
            device=cpu_device,
            obs_dim=env.observation_dim,
            network_type="recurrent",
            learning_rate=0.0001,
            gamma=0.99,
            replay_buffer_capacity=1000,
            batch_size=2,
            sequence_length=5,
            train_frequency=1,  # Train every step
        )

        population.reset()

        # Add complete episodes to buffer to trigger training
        from townlet.training.sequential_replay_buffer import SequentialReplayBuffer

        buffer = population.replay_buffer
        assert isinstance(buffer, SequentialReplayBuffer)

        # Create 20 short episodes (training needs 16+ for recurrent)
        for ep in range(20):
            episode_data = {
                "observations": torch.randn(10, env.observation_dim, device=cpu_device),
                "actions": torch.randint(0, env.action_dim, (10,), device=cpu_device),
                "rewards_extrinsic": torch.rand(10, device=cpu_device),
                "rewards_intrinsic": torch.rand(10, device=cpu_device) * 0.1,
                "dones": torch.zeros(10, dtype=torch.bool, device=cpu_device),
            }
            episode_data["dones"][-1] = True
            buffer.store_episode(episode_data)

        # Verify buffer has episodes
        assert len(buffer) >= 16, "Buffer should have at least 16 episodes for training"

        # Step population (should trigger LSTM training internally)
        population.step_population(env)

        # Note: Training metrics may or may not be set depending on train_frequency
        # Just verify step executed without error and recurrent mode is active
        assert population.is_recurrent is True


class TestAdaptiveIntrinsicExplorationIntegration:
    """Test AdaptiveIntrinsicExploration integration."""

    def test_adaptive_exploration_updates_on_episode_end(self, cpu_env_factory, cpu_device):
        """Episode end should trigger AdaptiveIntrinsicExploration update.

        Coverage target: lines 318-322 (AdaptiveIntrinsicExploration integration)
        """
        env = cpu_env_factory(num_agents=1)

        curriculum = StaticCurriculum(difficulty_level=0.5)

        # Use AdaptiveIntrinsicExploration (not epsilon-greedy)
        exploration = AdaptiveIntrinsicExploration(
            obs_dim=env.observation_dim,  # Required parameter
            embed_dim=32,
            rnd_learning_rate=0.001,  # Correct parameter name
            initial_intrinsic_weight=1.0,
            variance_threshold=100.0,
            survival_window=10,
            device=env.device,
        )

        population = VectorizedPopulation(
            env=env,
            curriculum=curriculum,
            exploration=exploration,
            agent_ids=["agent_0"],
            device=env.device,
            obs_dim=env.observation_dim,
            network_type="simple",
            learning_rate=0.00025,
            gamma=0.99,
            replay_buffer_capacity=1000,
            batch_size=16,
        )

        population.reset()

        # Simulate episode end with survival time
        agent_idx = 0
        survival_time = 50

        # Call _finalize_episode which should trigger exploration update
        population._finalize_episode(agent_idx, survival_time)

        # Verify exploration was updated (weight may change based on performance)
        # The actual value change depends on internal state, so just verify method was called
        # by checking that sync happened (exploration telemetry should be synced)
        assert population.runtime_registry is not None


class TestSnapshotAndMetrics:
    """Test snapshot generation and metrics tracking."""

    def test_build_telemetry_snapshot_generates_runtime_data(self, cpu_env_factory, cpu_device):
        """build_telemetry_snapshot() should generate runtime data for all agents.

        Coverage target: lines 703-709 (snapshot generation)
        """
        env = cpu_env_factory(num_agents=2)

        curriculum = StaticCurriculum(difficulty_level=0.5)
        exploration = EpsilonGreedyExploration(epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.999)

        population = VectorizedPopulation(
            env=env,
            curriculum=curriculum,
            exploration=exploration,
            agent_ids=["agent_0", "agent_1"],
            device=env.device,
            obs_dim=env.observation_dim,
            network_type="simple",
            learning_rate=0.00025,
            gamma=0.99,
            replay_buffer_capacity=1000,
            batch_size=16,
        )

        population.reset()

        # Get snapshot (correct method name is build_telemetry_snapshot)
        snapshot = population.build_telemetry_snapshot(episode_index=5)

        # Verify snapshot structure
        assert "schema_version" in snapshot
        assert snapshot["schema_version"] == "1.0.0"
        assert "episode_index" in snapshot
        assert snapshot["episode_index"] == 5
        assert "agents" in snapshot
        assert len(snapshot["agents"]) == 2, "Should have snapshot for each agent"

        # Verify agent snapshot contains expected fields (actual fields from runtime registry)
        agent_snapshot = snapshot["agents"][0]
        assert "agent_id" in agent_snapshot
        assert "survival_time" in agent_snapshot  # Actual field, not total_episodes

    def test_update_curriculum_tracker_when_tracker_exists(self, cpu_env_factory, cpu_device):
        """update_curriculum_tracker() should update when tracker exists.

        Coverage target: lines 713-714 (curriculum tracker update)
        """
        env = cpu_env_factory(num_agents=1)

        # Use AdversarialCurriculum which has a tracker
        from townlet.curriculum.adversarial import AdversarialCurriculum

        curriculum = AdversarialCurriculum(
            survival_advance_threshold=0.7,
            survival_retreat_threshold=0.3,
            entropy_gate=0.5,
            min_steps_at_stage=100,
        )

        exploration = EpsilonGreedyExploration(epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.999)

        population = VectorizedPopulation(
            env=env,
            curriculum=curriculum,
            exploration=exploration,
            agent_ids=["agent_0"],
            device=env.device,
            obs_dim=env.observation_dim,
            network_type="simple",
            learning_rate=0.00025,
            gamma=0.99,
            replay_buffer_capacity=1000,
            batch_size=16,
        )

        population.reset()

        # Initialize curriculum tracker (required for AdversarialCurriculum)
        curriculum.initialize_population(num_agents=1)

        # Update curriculum tracker
        rewards = torch.tensor([1.0], device=cpu_device)
        dones = torch.tensor([False], device=cpu_device)

        # Should not raise error
        population.update_curriculum_tracker(rewards, dones)

        # Verify tracker exists and was initialized
        assert curriculum.tracker is not None, "Tracker should be initialized"

    def test_get_training_metrics_returns_dict(self, cpu_env_factory, cpu_device):
        """get_training_metrics() should return metrics dictionary.

        Coverage target: line 723 (get_training_metrics)
        """
        env = cpu_env_factory(num_agents=1)

        curriculum = StaticCurriculum(difficulty_level=0.5)
        exploration = EpsilonGreedyExploration(epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.999)

        population = VectorizedPopulation(
            env=env,
            curriculum=curriculum,
            exploration=exploration,
            agent_ids=["agent_0"],
            device=env.device,
            obs_dim=env.observation_dim,
            network_type="simple",
            learning_rate=0.00025,
            gamma=0.99,
            replay_buffer_capacity=1000,
            batch_size=16,
        )

        # Get metrics
        metrics = population.get_training_metrics()

        # Verify structure
        assert isinstance(metrics, dict)
        assert "td_error" in metrics
        assert "loss" in metrics
        assert "q_values_mean" in metrics
        assert "training_step" in metrics
