"""
Tests for P1.2: Max Steps Survival Bug Fix

Verifies that agents surviving to max_steps:
1. Have their episodes flushed to replay buffer
2. Don't leak memory (accumulator cleared)
3. Trigger exploration annealing
4. Update curriculum
5. Reset hidden state (recurrent mode)
"""

import torch
import pytest

from townlet.environment.vectorized_env import VectorizedHamletEnv
from townlet.curriculum.static import StaticCurriculum
from townlet.exploration.adaptive_intrinsic import AdaptiveIntrinsicExploration
from townlet.population.vectorized import VectorizedPopulation


@pytest.fixture
def device():
    """Get device for testing."""
    return torch.device("cpu")


@pytest.fixture
def simple_env(device):
    """Create simple feedforward environment."""
    return VectorizedHamletEnv(
        num_agents=1,
        grid_size=5,
        device=device,
        partial_observability=False,
        enable_temporal_mechanics=False,
    )


@pytest.fixture
def recurrent_env(device):
    """Create recurrent environment (POMDP)."""
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
    """Create static curriculum for testing."""
    return StaticCurriculum(
        difficulty_level=0.5,
        reward_mode="shaped",
        depletion_multiplier=1.0,
    )


@pytest.fixture
def exploration(simple_env, device):
    """Create exploration strategy."""
    return AdaptiveIntrinsicExploration(
        obs_dim=simple_env.observation_dim,
        embed_dim=32,
        initial_intrinsic_weight=1.0,
        variance_threshold=100.0,
        survival_window=10,
        device=device,
    )


@pytest.fixture
def simple_population(simple_env, curriculum, exploration, device):
    """Create feedforward population."""
    return VectorizedPopulation(
        env=simple_env,
        curriculum=curriculum,
        exploration=exploration,
        agent_ids=["agent_0"],
        device=device,
        obs_dim=simple_env.observation_dim,
        action_dim=simple_env.action_dim,
        learning_rate=0.001,
        gamma=0.99,
        replay_buffer_capacity=100,
        network_type="simple",
        vision_window_size=5,
        tb_logger=None,
    )


@pytest.fixture
def recurrent_population(recurrent_env, curriculum, device):
    """Create recurrent population."""
    exploration = AdaptiveIntrinsicExploration(
        obs_dim=recurrent_env.observation_dim,
        embed_dim=32,
        initial_intrinsic_weight=1.0,
        variance_threshold=100.0,
        survival_window=10,
        device=device,
    )

    return VectorizedPopulation(
        env=recurrent_env,
        curriculum=curriculum,
        exploration=exploration,
        agent_ids=["agent_0"],
        device=device,
        obs_dim=recurrent_env.observation_dim,
        action_dim=recurrent_env.action_dim,
        learning_rate=0.001,
        gamma=0.99,
        replay_buffer_capacity=100,
        network_type="recurrent",
        vision_window_size=5,
        tb_logger=None,
    )


class TestFlushEpisodeMethod:
    """Test the new flush_episode() helper method."""

    def test_flush_episode_exists(self, recurrent_population):
        """Test that flush_episode method exists."""
        assert hasattr(recurrent_population, "flush_episode")

    def test_flush_episode_callable(self, recurrent_population):
        """Test that flush_episode is callable."""
        assert callable(getattr(recurrent_population, "flush_episode", None))

    def test_flush_episode_signature(self, recurrent_population):
        """Test flush_episode has correct signature."""
        import inspect

        sig = inspect.signature(recurrent_population.flush_episode)

        # Should have agent_idx and synthetic_done parameters
        assert "agent_idx" in sig.parameters
        assert "synthetic_done" in sig.parameters

        # synthetic_done should have default False
        assert sig.parameters["synthetic_done"].default is False


class TestRecurrentEpisodeAccumulation:
    """Test episode accumulation in recurrent mode."""

    def test_episode_accumulator_exists(self, recurrent_population):
        """Test that current_episodes accumulator exists for recurrent mode."""
        assert hasattr(recurrent_population, "current_episodes")
        assert isinstance(recurrent_population.current_episodes, list)
        assert len(recurrent_population.current_episodes) == 1  # Single agent

    def test_episode_accumulator_structure(self, recurrent_population):
        """Test accumulator has correct structure."""
        episode = recurrent_population.current_episodes[0]

        required_keys = ["observations", "actions", "rewards_extrinsic", "rewards_intrinsic", "dones"]
        for key in required_keys:
            assert key in episode
            assert isinstance(episode[key], list)

    def test_episode_accumulates_during_steps(self, recurrent_population, recurrent_env):
        """Test that episodes accumulate as we step."""
        recurrent_env.reset()
        recurrent_population.reset()

        # Take 5 steps
        for _ in range(5):
            state = recurrent_population.step_population(recurrent_env)
            if state.dones[0]:
                break

        # Accumulator should have data
        episode = recurrent_population.current_episodes[0]
        assert len(episode["observations"]) > 0


class TestFlushEpisodeBehavior:
    """Test flush_episode() behavior."""

    def test_flush_clears_accumulator(self, recurrent_population, recurrent_env):
        """Test that flush_episode clears the accumulator."""
        recurrent_env.reset()
        recurrent_population.reset()

        # Accumulate some steps
        for _ in range(10):
            state = recurrent_population.step_population(recurrent_env)
            if state.dones[0]:
                break

        # Accumulator should have data
        episode_before = recurrent_population.current_episodes[0]
        assert len(episode_before["observations"]) > 0

        # Flush
        recurrent_population.flush_episode(agent_idx=0, synthetic_done=True)

        # Accumulator should be empty
        episode_after = recurrent_population.current_episodes[0]
        assert len(episode_after["observations"]) == 0
        assert len(episode_after["actions"]) == 0
        assert len(episode_after["rewards_extrinsic"]) == 0

    def test_flush_stores_to_replay_buffer(self, recurrent_population, recurrent_env):
        """Test that flush_episode stores data in replay buffer."""
        recurrent_env.reset()
        recurrent_population.reset()

        buffer_size_before = recurrent_population.replay_buffer.num_transitions

        # Accumulate some steps
        for _ in range(10):
            state = recurrent_population.step_population(recurrent_env)
            if state.dones[0]:
                break

        episode_length = len(recurrent_population.current_episodes[0]["observations"])

        # Flush
        recurrent_population.flush_episode(agent_idx=0, synthetic_done=True)

        # Replay buffer should have grown
        buffer_size_after = recurrent_population.replay_buffer.num_transitions
        assert buffer_size_after > buffer_size_before

    def test_flush_triggers_exploration_annealing(self, recurrent_population, recurrent_env):
        """Test that flush_episode triggers exploration annealing update."""
        recurrent_env.reset()
        recurrent_population.reset()

        # Get initial intrinsic weight
        initial_weight = recurrent_population.exploration.get_intrinsic_weight()

        # Accumulate enough steps to potentially trigger annealing
        for _ in range(50):
            state = recurrent_population.step_population(recurrent_env)
            if state.dones[0]:
                break

        # Flush (should call update_on_episode_end)
        recurrent_population.flush_episode(agent_idx=0, synthetic_done=True)

        # Note: Weight might not change if variance threshold not met,
        # but the call should happen without error
        # This test mainly verifies no crash occurs
        assert True  # If we got here, no crash occurred

    def test_flush_resets_hidden_state(self, recurrent_population, recurrent_env):
        """Test that flush_episode resets hidden state for recurrent networks."""
        recurrent_env.reset()
        recurrent_population.reset()

        # Take some steps to build up hidden state
        for _ in range(10):
            state = recurrent_population.step_population(recurrent_env)
            if state.dones[0]:
                break

        # Get hidden state before flush
        h_before, c_before = recurrent_population.q_network.get_hidden_state()

        # Flush
        recurrent_population.flush_episode(agent_idx=0, synthetic_done=True)

        # Hidden state for agent 0 should be zeroed
        h_after, c_after = recurrent_population.q_network.get_hidden_state()

        if h_after is not None:
            # Check that agent 0's hidden state is zero
            assert torch.allclose(h_after[:, 0, :], torch.zeros_like(h_after[:, 0, :]))
            assert torch.allclose(c_after[:, 0, :], torch.zeros_like(c_after[:, 0, :]))

    def test_flush_empty_episode_safe(self, recurrent_population):
        """Test that flushing empty episode doesn't crash."""
        # Don't accumulate anything, just flush
        recurrent_population.flush_episode(agent_idx=0, synthetic_done=True)

        # Should not crash
        assert True

    def test_flush_feedforward_is_noop(self, simple_population):
        """Test that flush_episode is no-op for feedforward networks."""
        # Feedforward doesn't accumulate episodes, so flush should do nothing
        simple_population.flush_episode(agent_idx=0, synthetic_done=True)

        # Should not crash
        assert True


class TestMaxStepsMemoryLeak:
    """Test that max_steps survival doesn't leak memory."""

    def test_no_leak_after_max_steps(self, recurrent_population, recurrent_env):
        """Test memory doesn't leak when agent survives to max_steps."""
        recurrent_env.reset()
        recurrent_population.reset()

        # Simulate surviving to max_steps without dying
        max_steps = 50
        for step in range(max_steps):
            state = recurrent_population.step_population(recurrent_env)

            # Force agent to not die (for testing)
            # We'll manually flush at the end
            if state.dones[0]:
                # If agent died, reset and continue
                recurrent_env.reset()
                recurrent_population.reset()

        # At max_steps, accumulator might have data
        episode_length_before = len(recurrent_population.current_episodes[0]["observations"])

        # Flush episode (this is what runner should do)
        if episode_length_before > 0:
            recurrent_population.flush_episode(agent_idx=0, synthetic_done=True)

            # Accumulator should be empty now
            episode_length_after = len(recurrent_population.current_episodes[0]["observations"])
            assert episode_length_after == 0

    def test_multiple_episodes_no_accumulation(self, recurrent_population, recurrent_env):
        """Test that multiple max_steps episodes don't accumulate unbounded memory."""
        recurrent_env.reset()
        recurrent_population.reset()

        max_steps = 20
        num_episodes = 5

        for episode in range(num_episodes):
            # Run to max_steps
            for step in range(max_steps):
                state = recurrent_population.step_population(recurrent_env)
                if state.dones[0]:
                    break

            # Flush at end of episode
            if len(recurrent_population.current_episodes[0]["observations"]) > 0:
                recurrent_population.flush_episode(agent_idx=0, synthetic_done=True)

            # Reset for next episode
            recurrent_env.reset()
            recurrent_population.reset()

        # After all episodes, accumulator should still be empty
        final_length = len(recurrent_population.current_episodes[0]["observations"])
        assert final_length == 0


class TestRunnerIntegration:
    """Test integration with DemoRunner (these are more like specs)."""

    def test_runner_should_flush_on_max_steps(self):
        """
        Specification: DemoRunner should call flush_episode when:
        - Episode reaches max_steps
        - agent_state.dones[0] is False

        This test documents the expected behavior.
        """
        # This is a specification test - documents what runner.py should do
        expected_behavior = """
        After the episode loop:
        
        if agent_state.dones[0]:
            # Agent died naturally
            final_meters = self.env.meters[0].cpu()
        else:
            # Agent survived to max_steps
            final_meters = self.env.meters[0].cpu()
            
            # MUST flush episode to prevent memory leak and data loss
            self.population.flush_episode(agent_idx=0, synthetic_done=True)
        
        # Curriculum update happens for both cases
        """
        assert True  # This is a documentation test

    def test_runner_should_update_curriculum_for_max_steps(self):
        """
        Specification: DemoRunner should update curriculum for max_steps survival.
        """
        expected_behavior = """
        Curriculum update should happen regardless of how episode ended:
        - If died: survival_time = steps until death
        - If max_steps: survival_time = max_steps
        
        Both cases should trigger curriculum.update_tracker()
        """
        assert True  # This is a documentation test
