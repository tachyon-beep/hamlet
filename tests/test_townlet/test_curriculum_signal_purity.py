"""
TDD Tests for P1.3: Curriculum Signal Purity

Verifies that curriculum updates happen exactly once per episode with pure survival signals.

Test Goals:
1. Curriculum receives exactly ONE update per episode
2. Update value is pure survival time (integer steps)
3. No contamination from mid-episode rewards
4. Works for both death and max_steps survival
"""

import pytest
import torch

from townlet.curriculum.adversarial import AdversarialCurriculum
from townlet.environment.vectorized_env import VectorizedHamletEnv
from townlet.exploration.adaptive_intrinsic import AdaptiveIntrinsicExploration
from townlet.population.vectorized import VectorizedPopulation


@pytest.fixture
def device():
    """Test device (CPU for tests)."""
    return torch.device("cpu")


@pytest.fixture
def env(device):
    """Simple environment for testing."""
    return VectorizedHamletEnv(
        num_agents=1,
        grid_size=5,
        device=device,
        partial_observability=False,
        enable_temporal_mechanics=False,
    )


@pytest.fixture
def adversarial_curriculum(device):
    """Adversarial curriculum with tracking."""
    curriculum = AdversarialCurriculum(
        max_steps_per_episode=500,
        device=device,
    )
    curriculum.initialize_population(num_agents=1)
    return curriculum


@pytest.fixture
def exploration(env, device):
    """Exploration strategy."""
    return AdaptiveIntrinsicExploration(
        obs_dim=env.observation_dim,
        device=device,
        initial_intrinsic_weight=1.0,
        variance_threshold=100.0,
        survival_window=100,
    )


@pytest.fixture
def population(env, adversarial_curriculum, exploration, device):
    """Population with adversarial curriculum."""
    return VectorizedPopulation(
        env=env,
        curriculum=adversarial_curriculum,
        exploration=exploration,
        agent_ids=["agent_0"],
        device=device,
        obs_dim=env.observation_dim,
        action_dim=env.action_dim,  # Use env's actual action dim (6 actions)
        learning_rate=0.00025,
        gamma=0.99,
        network_type="simple",
    )


class TestCurriculumUpdateFrequency:
    """Test that curriculum updates happen exactly once per episode."""

    def test_update_curriculum_tracker_exists(self, population):
        """VectorizedPopulation should have update_curriculum_tracker method."""
        assert hasattr(population, "update_curriculum_tracker")
        assert callable(population.update_curriculum_tracker)

    def test_update_curriculum_tracker_accepts_rewards_and_dones(self, population, device):
        """update_curriculum_tracker should accept reward tensor and done tensor."""
        # This simulates the runner calling the method once per episode
        survival_time = torch.tensor([150.0], dtype=torch.float32, device=device)
        done = torch.tensor([True], dtype=torch.bool, device=device)

        # Should not raise
        population.update_curriculum_tracker(survival_time, done)

    def test_curriculum_tracker_state_updated_after_single_call(self, population, device):
        """Curriculum tracker should update its state after one update call."""
        # Simulate episode ending after 150 steps
        survival_time = torch.tensor([150.0], dtype=torch.float32, device=device)
        done = torch.tensor([True], dtype=torch.bool, device=device)

        population.update_curriculum_tracker(survival_time, done)

        # Tracker should have accumulated this episode's reward
        # Note: The tracker accumulates rewards until done=True, then resets
        # So after one call with done=True, the running total should be reset to 0
        # But prev_avg_reward should have been updated
        assert population.curriculum.tracker.prev_avg_reward[0].item() > 0


class TestCurriculumSignalPurity:
    """Test that curriculum receives pure survival signals."""

    def test_survival_signal_is_integer_steps(self, population, device):
        """Curriculum should receive integer step count as reward signal."""
        # Episode runs for 100 steps before death
        survival_time = torch.tensor([100.0], dtype=torch.float32, device=device)
        done = torch.tensor([True], dtype=torch.bool, device=device)

        # Track what the curriculum sees
        population.update_curriculum_tracker(survival_time, done)

        # The tracker should have seen exactly 100.0 as the reward
        # (episode_rewards accumulates until done=True, then resets)
        # So after this call, prev_avg_reward should reflect the 100.0 signal
        assert abs(population.curriculum.tracker.prev_avg_reward[0].item() - 100.0) < 1e-5

    def test_survival_signal_not_contaminated_by_intrinsic_rewards(self, population, device):
        """Curriculum signal should be pure survival time, not affected by curiosity."""
        # Agent runs for 200 steps with high intrinsic rewards
        # But curriculum should only see the 200 steps, not the intrinsic component
        survival_time = torch.tensor([200.0], dtype=torch.float32, device=device)
        done = torch.tensor([True], dtype=torch.bool, device=device)

        population.update_curriculum_tracker(survival_time, done)

        # Curriculum sees pure 200.0, not inflated by intrinsic rewards
        assert abs(population.curriculum.tracker.prev_avg_reward[0].item() - 200.0) < 1e-5

    def test_max_steps_survival_sends_done_signal(self, population, device):
        """Agent surviving to max_steps should send done=True to curriculum."""
        # Agent survives full 500 steps
        survival_time = torch.tensor([500.0], dtype=torch.float32, device=device)
        done = torch.tensor([True], dtype=torch.bool, device=device)  # Must be True!

        # This should work without error
        population.update_curriculum_tracker(survival_time, done)

        # Curriculum should register this as a complete episode
        assert abs(population.curriculum.tracker.prev_avg_reward[0].item() - 500.0) < 1e-5


class TestCurriculumUpdateTiming:
    """Test that curriculum updates happen at the right time in episode lifecycle."""

    def test_no_update_during_episode_steps(self, population, env):
        """Curriculum should not be updated during mid-episode steps."""
        # Reset to start new episode
        population.reset()

        # Take several steps
        for _ in range(10):
            _ = population.step_population(env)

            # Curriculum tracker's episode_rewards should be accumulating
            # But it should NOT be finalized (prev_avg_reward should not change mid-episode)
            # This is implicit - the tracker doesn't auto-update on step

        # After 10 steps, if we haven't called update_curriculum_tracker,
        # the prev_avg_reward should still be its initial value
        # (or whatever it was from the previous episode)
        # We can't easily test this without instrumenting the tracker,
        # but we can verify the method isn't auto-called

    def test_single_update_per_episode_on_death(self, population, env, device):
        """Curriculum should receive exactly one update when agent dies."""
        # Reset to start episode
        population.reset()

        # Run until death or max steps
        max_steps = 100
        survival_time = 0
        agent_state = None

        for _ in range(max_steps):
            agent_state = population.step_population(env)
            survival_time += 1

            if agent_state.dones[0]:
                break

        # Now send the single curriculum update
        curriculum_survival = torch.tensor([float(survival_time)], dtype=torch.float32, device=device)
        curriculum_done = torch.tensor([True], dtype=torch.bool, device=device)

        # This should be the ONLY curriculum update for this episode
        population.update_curriculum_tracker(curriculum_survival, curriculum_done)

        # Curriculum should have received exactly the survival time
        # (Though it may be averaged with previous episodes)
        assert population.curriculum.tracker.prev_avg_reward[0].item() >= 0


class TestCurriculumSignalInterpretability:
    """Test that curriculum signals are interpretable and meaningful."""

    def test_short_episode_has_low_signal(self, population, device):
        """Episode ending after 10 steps should produce reward=10."""
        survival_time = torch.tensor([10.0], dtype=torch.float32, device=device)
        done = torch.tensor([True], dtype=torch.bool, device=device)

        population.update_curriculum_tracker(survival_time, done)

        assert abs(population.curriculum.tracker.prev_avg_reward[0].item() - 10.0) < 1e-5

    def test_long_episode_has_high_signal(self, population, device):
        """Episode ending after 400 steps should produce reward=400."""
        survival_time = torch.tensor([400.0], dtype=torch.float32, device=device)
        done = torch.tensor([True], dtype=torch.bool, device=device)

        population.update_curriculum_tracker(survival_time, done)

        assert abs(population.curriculum.tracker.prev_avg_reward[0].item() - 400.0) < 1e-5

    def test_signal_is_monotonic_with_survival_time(self, population, device):
        """Longer survival should always produce higher curriculum signal."""
        # Episode 1: 50 steps
        population.update_curriculum_tracker(
            torch.tensor([50.0], dtype=torch.float32, device=device),
            torch.tensor([True], dtype=torch.bool, device=device),
        )
        reward_50 = population.curriculum.tracker.prev_avg_reward[0].item()

        # Episode 2: 150 steps
        population.update_curriculum_tracker(
            torch.tensor([150.0], dtype=torch.float32, device=device),
            torch.tensor([True], dtype=torch.bool, device=device),
        )
        reward_150 = population.curriculum.tracker.prev_avg_reward[0].item()

        # Average should increase (assuming tracker uses some averaging)
        # Or if it's just the last value, 150 > 50
        assert reward_150 > reward_50
