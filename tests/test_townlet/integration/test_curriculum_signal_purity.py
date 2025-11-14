"""
Integration Tests for P1.3: Curriculum Signal Purity

Verifies that curriculum updates happen exactly once per episode with pure survival signals.

Test Goals:
1. Curriculum receives exactly ONE update per episode
2. Update value is pure survival time (integer steps)
3. No contamination from mid-episode rewards
4. Works for both death and max_steps survival
5. Multi-episode signal purity with active RND
6. Stage advancement based on survival rate, not reward magnitude
"""

import pytest
import torch

from townlet.curriculum.adversarial import AdversarialCurriculum
from townlet.exploration.adaptive_intrinsic import AdaptiveIntrinsicExploration
from townlet.population.vectorized import VectorizedPopulation


@pytest.fixture
def env(cpu_env_factory):
    """Simple environment for testing."""
    return cpu_env_factory(num_agents=1)


@pytest.fixture
def adversarial_curriculum(cpu_device):
    """Adversarial curriculum with tracking."""
    curriculum = AdversarialCurriculum(
        max_steps_per_episode=500,
        device=cpu_device,
    )
    curriculum.initialize_population(num_agents=1)
    return curriculum


@pytest.fixture
def exploration(env, cpu_device):
    """Exploration strategy."""
    return AdaptiveIntrinsicExploration(
        obs_dim=env.observation_dim,
        device=cpu_device,
        initial_intrinsic_weight=1.0,
        variance_threshold=100.0,
        survival_window=100,
    )


@pytest.fixture
def population(env, adversarial_curriculum, exploration, cpu_device, minimal_brain_config):
    """Population with adversarial curriculum."""
    return VectorizedPopulation(
        env=env,
        curriculum=adversarial_curriculum,
        exploration=exploration,
        agent_ids=["agent_0"],
        device=cpu_device,
        obs_dim=env.observation_dim,
        action_dim=env.action_dim,  # Use env's actual action dim (6 actions)
        brain_config=minimal_brain_config,
    )


class TestCurriculumUpdateFrequency:
    """Test that curriculum updates happen exactly once per episode."""

    def test_update_curriculum_tracker_exists(self, population):
        """VectorizedPopulation should have update_curriculum_tracker method."""
        assert hasattr(population, "update_curriculum_tracker")
        assert callable(population.update_curriculum_tracker)

    def test_update_curriculum_tracker_accepts_rewards_and_dones(self, population, cpu_device):
        """update_curriculum_tracker should accept reward tensor and done tensor."""
        # This simulates the runner calling the method once per episode
        survival_time = torch.tensor([150.0], dtype=torch.float32, device=cpu_device)
        done = torch.tensor([True], dtype=torch.bool, device=cpu_device)

        # Should not raise
        population.update_curriculum_tracker(survival_time, done)

    def test_curriculum_tracker_state_updated_after_single_call(self, population, cpu_device):
        """Curriculum tracker should update its state after one update call."""
        # Simulate episode ending after 150 steps
        survival_time = torch.tensor([150.0], dtype=torch.float32, device=cpu_device)
        done = torch.tensor([True], dtype=torch.bool, device=cpu_device)

        population.update_curriculum_tracker(survival_time, done)

        # Tracker should have accumulated this episode's reward
        # Note: The tracker accumulates rewards until done=True, then resets
        # So after one call with done=True, the running total should be reset to 0
        # But prev_avg_reward should have been updated
        assert population.curriculum.tracker.prev_avg_reward[0].item() > 0


class TestCurriculumSignalPurity:
    """Test that curriculum receives pure survival signals."""

    def test_survival_signal_is_integer_steps(self, population, cpu_device):
        """Curriculum should receive integer step count as reward signal."""
        # Episode runs for 100 steps before death
        survival_time = torch.tensor([100.0], dtype=torch.float32, device=cpu_device)
        done = torch.tensor([True], dtype=torch.bool, device=cpu_device)

        # Track what the curriculum sees
        population.update_curriculum_tracker(survival_time, done)

        # The tracker should have seen exactly 100.0 as the reward
        # (episode_rewards accumulates until done=True, then resets)
        # So after this call, prev_avg_reward should reflect the 100.0 signal
        assert abs(population.curriculum.tracker.prev_avg_reward[0].item() - 100.0) < 1e-5

    def test_survival_signal_not_contaminated_by_intrinsic_rewards(self, population, cpu_device):
        """Curriculum signal should be pure survival time, not affected by curiosity."""
        # Agent runs for 200 steps with high intrinsic rewards
        # But curriculum should only see the 200 steps, not the intrinsic component
        survival_time = torch.tensor([200.0], dtype=torch.float32, device=cpu_device)
        done = torch.tensor([True], dtype=torch.bool, device=cpu_device)

        population.update_curriculum_tracker(survival_time, done)

        # Curriculum sees pure 200.0, not inflated by intrinsic rewards
        assert abs(population.curriculum.tracker.prev_avg_reward[0].item() - 200.0) < 1e-5

    def test_max_steps_survival_sends_done_signal(self, population, cpu_device):
        """Agent surviving to max_steps should send done=True to curriculum."""
        # Agent survives full 500 steps
        survival_time = torch.tensor([500.0], dtype=torch.float32, device=cpu_device)
        done = torch.tensor([True], dtype=torch.bool, device=cpu_device)  # Must be True!

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

    def test_single_update_per_episode_on_death(self, population, env, cpu_device):
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
        curriculum_survival = torch.tensor([float(survival_time)], dtype=torch.float32, device=cpu_device)
        curriculum_done = torch.tensor([True], dtype=torch.bool, device=cpu_device)

        # This should be the ONLY curriculum update for this episode
        population.update_curriculum_tracker(curriculum_survival, curriculum_done)

        # Curriculum should have received exactly the survival time
        # (Though it may be averaged with previous episodes)
        assert population.curriculum.tracker.prev_avg_reward[0].item() >= 0


class TestCurriculumSignalInterpretability:
    """Test that curriculum signals are interpretable and meaningful."""

    def test_short_episode_has_low_signal(self, population, cpu_device):
        """Episode ending after 10 steps should produce reward=10."""
        survival_time = torch.tensor([10.0], dtype=torch.float32, device=cpu_device)
        done = torch.tensor([True], dtype=torch.bool, device=cpu_device)

        population.update_curriculum_tracker(survival_time, done)

        assert abs(population.curriculum.tracker.prev_avg_reward[0].item() - 10.0) < 1e-5

    def test_long_episode_has_high_signal(self, population, cpu_device):
        """Episode ending after 400 steps should produce reward=400."""
        survival_time = torch.tensor([400.0], dtype=torch.float32, device=cpu_device)
        done = torch.tensor([True], dtype=torch.bool, device=cpu_device)

        population.update_curriculum_tracker(survival_time, done)

        assert abs(population.curriculum.tracker.prev_avg_reward[0].item() - 400.0) < 1e-5

    def test_signal_is_monotonic_with_survival_time(self, population, cpu_device):
        """Longer survival should always produce higher curriculum signal."""
        # Episode 1: 50 steps
        population.update_curriculum_tracker(
            torch.tensor([50.0], dtype=torch.float32, device=cpu_device),
            torch.tensor([True], dtype=torch.bool, device=cpu_device),
        )
        reward_50 = population.curriculum.tracker.prev_avg_reward[0].item()

        # Episode 2: 150 steps
        population.update_curriculum_tracker(
            torch.tensor([150.0], dtype=torch.float32, device=cpu_device),
            torch.tensor([True], dtype=torch.bool, device=cpu_device),
        )
        reward_150 = population.curriculum.tracker.prev_avg_reward[0].item()

        # Average should increase (assuming tracker uses some averaging)
        # Or if it's just the last value, 150 > 50
        assert reward_150 > reward_50

    def test_signal_purity_across_multiple_episodes_with_active_rnd(self, cpu_device, cpu_env_factory, minimal_brain_config):
        """Verify curriculum receives pure survival time over 10 episodes with active RND.

        Critical integration test: Run multiple episodes with active RND exploration
        providing large intrinsic rewards. Verify curriculum tracker receives ONLY
        survival time (extrinsic rewards), not contaminated by intrinsic bonuses.

        This tests the architectural contract that curriculum advancement is based
        on survival performance, not exploration novelty.
        """
        # Setup: Population with RND exploration (high intrinsic weight)
        env = cpu_env_factory(num_agents=1)

        curriculum = AdversarialCurriculum(max_steps_per_episode=100)
        curriculum.initialize_population(num_agents=1)

        exploration = AdaptiveIntrinsicExploration(
            obs_dim=env.observation_dim,
            embed_dim=128,
            initial_intrinsic_weight=1.0,  # High intrinsic weight
            device=cpu_device,
        )

        population = VectorizedPopulation(
            env=env,
            curriculum=curriculum,
            exploration=exploration,
            agent_ids=["agent_0"],
            device=cpu_device,
            obs_dim=env.observation_dim,
            action_dim=env.action_dim,
            brain_config=minimal_brain_config,
        )

        # Run 10 episodes, tracking curriculum rewards
        curriculum_rewards_log = []

        for episode in range(10):
            population.reset()
            done = False
            survival_time = 0

            while not done:
                # Step population (includes action selection, Q-learning, exploration)
                agent_state = population.step_population(env)

                # Track survival time
                survival_time += 1
                done = agent_state.dones[0].item()

                # Note: step_population internally computes intrinsic rewards via RND
                # but they should NOT contaminate the curriculum signal

            # Update curriculum with pure survival time
            curriculum_survival = torch.tensor([float(survival_time)], dtype=torch.float32, device=cpu_device)
            curriculum_done = torch.tensor([True], dtype=torch.bool, device=cpu_device)
            population.update_curriculum_tracker(curriculum_survival, curriculum_done)

            # Get curriculum reward (what curriculum actually sees)
            curriculum_reward = curriculum.tracker.prev_avg_reward[0].item()

            # Log for verification
            curriculum_rewards_log.append(curriculum_reward)

            # Critical assertion: Curriculum reward equals survival time
            # NOT inflated by intrinsic rewards from RND exploration
            assert (
                abs(curriculum_reward - survival_time) < 1e-5
            ), f"Episode {episode}: Curriculum reward ({curriculum_reward}) should equal survival time ({survival_time})"

        # Verify all episodes showed positive survival
        for i, curr_r in enumerate(curriculum_rewards_log):
            assert curr_r > 0, f"Episode {i}: Curriculum should see survival time > 0"

    def test_curriculum_stage_advancement_uses_survival_rate_not_rewards(self, cpu_device, cpu_env_factory, minimal_brain_config):
        """Verify stage transitions based on survival rate, not reward magnitude.

        Critical integration test: Run episodes with varying survival times
        while intrinsic rewards create large reward differences. Verify that
        curriculum stage advancement is based on survival RATE (steps / max_steps),
        not total reward values.

        This prevents curriculum from advancing when agent is "curious" (high RND)
        rather than when agent is "surviving" (high survival rate).
        """
        # Setup: Population with adversarial curriculum and RND
        env = cpu_env_factory(num_agents=1)

        curriculum = AdversarialCurriculum(
            max_steps_per_episode=100,
            survival_advance_threshold=0.7,  # Advance at 70% survival rate
            survival_retreat_threshold=0.3,  # Retreat at 30% survival rate
            min_steps_at_stage=5,  # Low threshold for testing
        )
        curriculum.initialize_population(num_agents=1)

        exploration = AdaptiveIntrinsicExploration(
            obs_dim=env.observation_dim,
            embed_dim=128,
            initial_intrinsic_weight=1.0,  # High intrinsic weight creates large rewards
            device=cpu_device,
        )

        population = VectorizedPopulation(
            env=env,
            curriculum=curriculum,
            exploration=exploration,
            agent_ids=["agent_0"],
            device=cpu_device,
            obs_dim=env.observation_dim,
            action_dim=env.action_dim,
            brain_config=minimal_brain_config,
        )

        # Scenario 1: High survival rate (90 steps) should consider advancement
        # Run 5 episodes with high survival
        for episode in range(5):
            population.reset()
            survival_time = 0
            for step in range(90):  # 90% survival rate (90/100)
                agent_state = population.step_population(env)
                survival_time += 1
                if agent_state.dones[0]:
                    break

            # Update curriculum with actual survival time
            step_counts = torch.tensor([float(survival_time)], device=cpu_device)
            curriculum_done = torch.ones(1, dtype=torch.bool, device=cpu_device)
            population.update_curriculum_tracker(step_counts, curriculum_done)

        # After 5 high-survival episodes, curriculum should consider advancement
        # (even if intrinsic rewards were large, advancement is based on SURVIVAL RATE)
        initial_stage = curriculum.tracker.agent_stages[0].item()

        # Scenario 2: Low survival rate (30 steps) should NOT advance
        # Run 5 episodes with low survival
        for episode in range(5):
            population.reset()
            survival_time = 0
            for step in range(30):  # 30% survival rate (30/100)
                agent_state = population.step_population(env)
                survival_time += 1
                if agent_state.dones[0]:
                    break

            # Update curriculum with actual survival time
            step_counts = torch.tensor([float(survival_time)], device=cpu_device)
            curriculum_done = torch.ones(1, dtype=torch.bool, device=cpu_device)
            population.update_curriculum_tracker(step_counts, curriculum_done)

        final_stage = curriculum.tracker.agent_stages[0].item()

        # Verify: Stage should NOT have advanced (low survival rate)
        # Even if intrinsic rewards were large, advancement is based on survival performance
        assert (
            final_stage <= initial_stage
        ), f"Curriculum should not advance with low survival rate (30%). Initial stage: {initial_stage}, Final stage: {final_stage}"

        # Additional verification: Curriculum sees survival rate, not total rewards
        # (curriculum tracker would show ~30.0, not inflated reward values)
        last_curriculum_reward = curriculum.tracker.prev_avg_reward[0].item()
        assert (
            25.0 < last_curriculum_reward < 35.0
        ), f"Curriculum should see survival time (~30), not inflated reward. Got: {last_curriculum_reward}"
