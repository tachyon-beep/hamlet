"""
Tests for Adversarial Curriculum Manager.

Focus on business logic: stage progression, entropy gating, retreat conditions.
"""

import torch

from townlet.curriculum.adversarial import AdversarialCurriculum, PerformanceTracker
from townlet.training.state import BatchedAgentState


class TestPerformanceTracker:
    """Test performance metrics tracking."""

    def test_update_step_accumulates_rewards_and_steps(self):
        """update_step should accumulate rewards and steps until episode ends."""
        tracker = PerformanceTracker(num_agents=2, device=torch.device("cpu"))

        # Step 1
        tracker.update_step(
            rewards=torch.tensor([1.0, 2.0]),
            dones=torch.tensor([False, False]),
        )
        assert abs(tracker.episode_rewards[0] - 1.0) < 1e-6
        assert abs(tracker.episode_rewards[1] - 2.0) < 1e-6
        assert abs(tracker.episode_steps[0] - 1.0) < 1e-6
        assert abs(tracker.episode_steps[1] - 1.0) < 1e-6

        # Step 2
        tracker.update_step(
            rewards=torch.tensor([1.5, 2.5]),
            dones=torch.tensor([False, False]),
        )
        assert abs(tracker.episode_rewards[0] - 2.5) < 1e-6
        assert abs(tracker.episode_rewards[1] - 4.5) < 1e-6
        assert abs(tracker.episode_steps[0] - 2.0) < 1e-6
        assert abs(tracker.episode_steps[1] - 2.0) < 1e-6

    def test_update_step_resets_on_done(self):
        """update_step should reset counters when episode ends."""
        tracker = PerformanceTracker(num_agents=2, device=torch.device("cpu"))

        # Accumulate
        tracker.update_step(
            rewards=torch.tensor([5.0, 10.0]),
            dones=torch.tensor([False, False]),
        )
        tracker.update_step(
            rewards=torch.tensor([3.0, 8.0]),
            dones=torch.tensor([False, False]),
        )

        # Agent 0 dies
        tracker.update_step(
            rewards=torch.tensor([2.0, 5.0]),
            dones=torch.tensor([True, False]),
        )

        # Agent 0 should be reset, agent 1 continues
        assert abs(tracker.episode_rewards[0] - 0.0) < 1e-6
        assert abs(tracker.episode_steps[0] - 0.0) < 1e-6
        assert abs(tracker.episode_rewards[1] - 23.0) < 1e-6  # 10 + 8 + 5
        assert abs(tracker.episode_steps[1] - 3.0) < 1e-6

    def test_get_survival_rate_calculation(self):
        """get_survival_rate should return steps / max_steps."""
        tracker = PerformanceTracker(num_agents=3, device=torch.device("cpu"))

        tracker.episode_steps = torch.tensor([50.0, 100.0, 250.0])

        survival_rates = tracker.get_survival_rate(max_steps=500)

        assert abs(survival_rates[0] - 0.1) < 1e-6
        assert abs(survival_rates[1] - 0.2) < 1e-6
        assert abs(survival_rates[2] - 0.5) < 1e-6

    def test_get_learning_progress_positive(self):
        """get_learning_progress should show positive when improving."""
        tracker = PerformanceTracker(num_agents=1, device=torch.device("cpu"))

        # Baseline: average reward = 2.0
        tracker.prev_avg_reward[0] = 2.0

        # Current: 10 reward over 2 steps = 5.0 average
        tracker.episode_rewards[0] = 10.0
        tracker.episode_steps[0] = 2.0

        progress = tracker.get_learning_progress()

        assert abs(progress[0] - 3.0) < 1e-6

    def test_get_learning_progress_negative(self):
        """get_learning_progress should show negative when regressing."""
        tracker = PerformanceTracker(num_agents=1, device=torch.device("cpu"))

        # Baseline: average reward = 5.0
        tracker.prev_avg_reward[0] = 5.0

        # Current: 4 reward over 2 steps = 2.0 average
        tracker.episode_rewards[0] = 4.0
        tracker.episode_steps[0] = 2.0

        progress = tracker.get_learning_progress()

        assert abs(progress[0] - (-3.0)) < 1e-6

    def test_update_baseline_stores_current_average(self):
        """update_baseline should update prev_avg_reward to current average."""
        tracker = PerformanceTracker(num_agents=1, device=torch.device("cpu"))

        tracker.episode_rewards[0] = 15.0
        tracker.episode_steps[0] = 3.0

        tracker.update_baseline()

        # Should store 15.0 / 3.0 = 5.0
        assert abs(tracker.prev_avg_reward[0] - 5.0) < 1e-6


class TestStageAdvancement:
    """Test stage advancement conditions."""

    def test_should_not_advance_without_minimum_steps(self):
        """Should not advance before min_steps_at_stage reached."""
        curriculum = AdversarialCurriculum(
            max_steps_per_episode=500,
            min_steps_at_stage=1000,
            device=torch.device("cpu"),
        )
        curriculum.initialize_population(num_agents=1)

        # Only 500 steps at stage
        curriculum.tracker.steps_at_stage[0] = 500

        # Even with perfect metrics
        assert not curriculum._should_advance(0, entropy=0.0)

    def test_should_not_advance_from_stage_5(self):
        """Should not advance beyond stage 5 (max stage)."""
        curriculum = AdversarialCurriculum(
            max_steps_per_episode=500,
            min_steps_at_stage=100,
            device=torch.device("cpu"),
        )
        curriculum.initialize_population(num_agents=1)

        curriculum.tracker.agent_stages[0] = 5
        curriculum.tracker.steps_at_stage[0] = 2000

        # Even with perfect metrics
        assert not curriculum._should_advance(0, entropy=0.0)

    def test_should_advance_with_all_conditions_met(self):
        """Should advance when survival, learning, and entropy conditions met."""
        curriculum = AdversarialCurriculum(
            max_steps_per_episode=500,
            survival_advance_threshold=0.7,
            entropy_gate=0.5,
            min_steps_at_stage=100,
            device=torch.device("cpu"),
        )
        curriculum.initialize_population(num_agents=1)

        # Setup: at stage 2, enough steps
        curriculum.tracker.agent_stages[0] = 2
        curriculum.tracker.steps_at_stage[0] = 1000

        # High survival: 400 / 500 = 0.8 > 0.7
        # Positive learning: current (10/400=0.025) > prev (0.01)
        curriculum.tracker.episode_steps[0] = 400
        curriculum.tracker.episode_rewards[0] = 10.0
        curriculum.tracker.prev_avg_reward[0] = 0.01

        # Low entropy (converged): 0.3 < 0.5
        assert curriculum._should_advance(0, entropy=0.3)

    def test_should_not_advance_with_low_survival(self):
        """Should not advance with low survival rate."""
        curriculum = AdversarialCurriculum(
            max_steps_per_episode=500,
            survival_advance_threshold=0.7,
            min_steps_at_stage=100,
            device=torch.device("cpu"),
        )
        curriculum.initialize_population(num_agents=1)

        curriculum.tracker.agent_stages[0] = 2
        curriculum.tracker.steps_at_stage[0] = 1000

        # Low survival: 100 / 500 = 0.2 < 0.7
        curriculum.tracker.episode_steps[0] = 100

        # Good learning and entropy
        curriculum.tracker.episode_rewards[0] = 10.0
        curriculum.tracker.prev_avg_reward[0] = 1.0

        assert not curriculum._should_advance(0, entropy=0.3)

    def test_should_not_advance_with_negative_learning(self):
        """Should not advance with negative learning progress."""
        curriculum = AdversarialCurriculum(
            max_steps_per_episode=500,
            survival_advance_threshold=0.7,
            min_steps_at_stage=100,
            device=torch.device("cpu"),
        )
        curriculum.initialize_population(num_agents=1)

        curriculum.tracker.agent_stages[0] = 2
        curriculum.tracker.steps_at_stage[0] = 1000

        # High survival: 400 / 500 = 0.8
        # Negative learning: current (2/400=0.005) < prev (0.01)
        curriculum.tracker.episode_steps[0] = 400
        curriculum.tracker.episode_rewards[0] = 2.0
        curriculum.tracker.prev_avg_reward[0] = 0.01

        assert not curriculum._should_advance(0, entropy=0.3)

    def test_should_not_advance_with_high_entropy(self):
        """Should not advance with high entropy (not converged)."""
        curriculum = AdversarialCurriculum(
            max_steps_per_episode=500,
            survival_advance_threshold=0.7,
            entropy_gate=0.5,
            min_steps_at_stage=100,
            device=torch.device("cpu"),
        )
        curriculum.initialize_population(num_agents=1)

        curriculum.tracker.agent_stages[0] = 2
        curriculum.tracker.steps_at_stage[0] = 1000

        # High survival and positive learning
        curriculum.tracker.episode_steps[0] = 400
        curriculum.tracker.episode_rewards[0] = 10.0
        curriculum.tracker.prev_avg_reward[0] = 1.0

        # High entropy: 0.8 > 0.5 (still exploring)
        assert not curriculum._should_advance(0, entropy=0.8)


class TestStageRetreat:
    """Test stage retreat conditions."""

    def test_should_not_retreat_from_stage_1(self):
        """Should not retreat below stage 1 (min stage)."""
        curriculum = AdversarialCurriculum(
            max_steps_per_episode=500,
            min_steps_at_stage=100,
            device=torch.device("cpu"),
        )
        curriculum.initialize_population(num_agents=1)

        curriculum.tracker.agent_stages[0] = 1
        curriculum.tracker.steps_at_stage[0] = 2000

        # Even with terrible performance
        curriculum.tracker.episode_steps[0] = 10  # Low survival
        curriculum.tracker.prev_avg_reward[0] = 10.0  # Negative learning expected

        assert not curriculum._should_retreat(0)

    def test_should_not_retreat_without_minimum_steps(self):
        """Should not retreat before min_steps_at_stage reached."""
        curriculum = AdversarialCurriculum(
            max_steps_per_episode=500,
            min_steps_at_stage=1000,
            device=torch.device("cpu"),
        )
        curriculum.initialize_population(num_agents=1)

        curriculum.tracker.agent_stages[0] = 3
        curriculum.tracker.steps_at_stage[0] = 500

        # Even with terrible performance
        assert not curriculum._should_retreat(0)

    def test_should_retreat_with_low_survival(self):
        """Should retreat when survival < threshold."""
        curriculum = AdversarialCurriculum(
            max_steps_per_episode=500,
            survival_retreat_threshold=0.3,
            min_steps_at_stage=100,
            device=torch.device("cpu"),
        )
        curriculum.initialize_population(num_agents=1)

        curriculum.tracker.agent_stages[0] = 3
        curriculum.tracker.steps_at_stage[0] = 1000

        # Low survival: 100 / 500 = 0.2 < 0.3
        curriculum.tracker.episode_steps[0] = 100

        assert curriculum._should_retreat(0)

    def test_should_retreat_with_negative_learning(self):
        """Should retreat when learning progress is negative."""
        curriculum = AdversarialCurriculum(
            max_steps_per_episode=500,
            survival_retreat_threshold=0.3,
            min_steps_at_stage=100,
            device=torch.device("cpu"),
        )
        curriculum.initialize_population(num_agents=1)

        curriculum.tracker.agent_stages[0] = 3
        curriculum.tracker.steps_at_stage[0] = 1000

        # Good survival: 400 / 500 = 0.8
        # Negative learning: current (1/400=0.0025) < prev (0.01)
        curriculum.tracker.episode_steps[0] = 400
        curriculum.tracker.episode_rewards[0] = 1.0
        curriculum.tracker.prev_avg_reward[0] = 0.01

        assert curriculum._should_retreat(0)


class TestEntropyCalculation:
    """Test action entropy calculation from Q-values."""

    def test_entropy_zero_for_deterministic_policy(self):
        """Entropy should be ~0 for deterministic (peaked) Q-values."""
        curriculum = AdversarialCurriculum(device=torch.device("cpu"))

        # One action dominant
        q_values = torch.tensor([[100.0, 0.0, 0.0, 0.0, 0.0]])

        entropy = curriculum._calculate_action_entropy(q_values)

        # Should be near 0 (deterministic)
        assert entropy[0] < 0.1

    def test_entropy_high_for_uniform_policy(self):
        """Entropy should be ~1.0 for uniform (exploring) Q-values."""
        curriculum = AdversarialCurriculum(device=torch.device("cpu"))

        # All actions equally likely
        q_values = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0]])

        entropy = curriculum._calculate_action_entropy(q_values)

        # Should be near 1.0 (maximum entropy, normalized)
        assert entropy[0] > 0.9

    def test_entropy_batch_processing(self):
        """Should calculate entropy for each agent in batch."""
        curriculum = AdversarialCurriculum(device=torch.device("cpu"))

        q_values = torch.tensor(
            [
                [100.0, 0.0, 0.0, 0.0, 0.0],  # Deterministic
                [1.0, 1.0, 1.0, 1.0, 1.0],  # Uniform
            ]
        )

        entropy = curriculum._calculate_action_entropy(q_values)

        assert entropy[0] < 0.1  # Deterministic
        assert entropy[1] > 0.9  # Uniform


class TestBatchDecisions:
    """Test curriculum decision generation."""

    def test_get_batch_decisions_advances_stage(self):
        """get_batch_decisions_with_qvalues should advance stage when conditions met."""
        curriculum = AdversarialCurriculum(
            max_steps_per_episode=500,
            survival_advance_threshold=0.7,
            entropy_gate=0.5,
            min_steps_at_stage=100,
            device=torch.device("cpu"),
        )
        curriculum.initialize_population(num_agents=1)

        # Setup for advancement
        curriculum.tracker.agent_stages[0] = 2
        curriculum.tracker.steps_at_stage[0] = 1000
        # High survival: 400/500 = 0.8 > 0.7
        # Positive learning: current (100/400=0.25) > prev (0.1)
        curriculum.tracker.episode_steps[0] = 400
        curriculum.tracker.episode_rewards[0] = 100.0
        curriculum.tracker.prev_avg_reward[0] = 0.1

        # Low entropy Q-values (converged)
        q_values = torch.tensor([[100.0, 0.0, 0.0, 0.0, 0.0]])

        device = torch.device("cpu")
        agent_states = BatchedAgentState(
            observations=torch.zeros((1, 10)),
            actions=torch.zeros(1, dtype=torch.long),
            rewards=torch.zeros(1),
            dones=torch.zeros(1, dtype=torch.bool),
            epsilons=torch.zeros(1),
            intrinsic_rewards=torch.zeros(1),
            survival_times=torch.zeros(1),
            curriculum_difficulties=torch.zeros(1),
            device=device,
        )

        decisions = curriculum.get_batch_decisions_with_qvalues(agent_states, ["agent_0"], q_values)

        # Should have advanced from stage 2 to 3
        assert curriculum.tracker.agent_stages[0] == 3
        assert decisions[0].reward_mode == "shaped"
        assert "Stage 3" in decisions[0].reason

    def test_get_batch_decisions_retreats_stage(self):
        """get_batch_decisions_with_qvalues should retreat stage when struggling."""
        curriculum = AdversarialCurriculum(
            max_steps_per_episode=500,
            survival_retreat_threshold=0.3,
            min_steps_at_stage=100,
            device=torch.device("cpu"),
        )
        curriculum.initialize_population(num_agents=1)

        # Setup for retreat
        curriculum.tracker.agent_stages[0] = 3
        curriculum.tracker.steps_at_stage[0] = 1000
        curriculum.tracker.episode_steps[0] = 100  # 0.2 survival < 0.3

        q_values = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0]])

        device = torch.device("cpu")
        agent_states = BatchedAgentState(
            observations=torch.zeros((1, 10)),
            actions=torch.zeros(1, dtype=torch.long),
            rewards=torch.zeros(1),
            dones=torch.zeros(1, dtype=torch.bool),
            epsilons=torch.zeros(1),
            intrinsic_rewards=torch.zeros(1),
            survival_times=torch.zeros(1),
            curriculum_difficulties=torch.zeros(1),
            device=device,
        )

        decisions = curriculum.get_batch_decisions_with_qvalues(agent_states, ["agent_0"], q_values)

        # Should have retreated from stage 3 to 2
        assert curriculum.tracker.agent_stages[0] == 2
        assert "Stage 2" in decisions[0].reason

    def test_get_batch_decisions_maps_stage_to_difficulty(self):
        """Should map stage (1-5) to difficulty_level (0.0-1.0)."""
        curriculum = AdversarialCurriculum(device=torch.device("cpu"))
        curriculum.initialize_population(num_agents=5)

        # Set different stages
        curriculum.tracker.agent_stages = torch.tensor([1, 2, 3, 4, 5])

        q_values = torch.ones(5, 5)

        device = torch.device("cpu")
        agent_states = BatchedAgentState(
            observations=torch.zeros((5, 10)),
            actions=torch.zeros(5, dtype=torch.long),
            rewards=torch.zeros(5),
            dones=torch.zeros(5, dtype=torch.bool),
            epsilons=torch.zeros(5),
            intrinsic_rewards=torch.zeros(5),
            survival_times=torch.zeros(5),
            curriculum_difficulties=torch.zeros(5),
            device=device,
        )

        decisions = curriculum.get_batch_decisions_with_qvalues(
            agent_states, ["a0", "a1", "a2", "a3", "a4"], q_values
        )

        # Stage 1 -> 0.0, Stage 2 -> 0.25, Stage 3 -> 0.5, Stage 4 -> 0.75, Stage 5 -> 1.0
        assert abs(decisions[0].difficulty_level - 0.0) < 1e-6
        assert abs(decisions[1].difficulty_level - 0.25) < 1e-6
        assert abs(decisions[2].difficulty_level - 0.5) < 1e-6
        assert abs(decisions[3].difficulty_level - 0.75) < 1e-6
        assert abs(decisions[4].difficulty_level - 1.0) < 1e-6

    def test_stage_5_is_sparse_rewards(self):
        """Stage 5 should use sparse rewards (graduation!)."""
        curriculum = AdversarialCurriculum(device=torch.device("cpu"))
        curriculum.initialize_population(num_agents=1)

        curriculum.tracker.agent_stages[0] = 5

        q_values = torch.ones(1, 5)

        device = torch.device("cpu")
        agent_states = BatchedAgentState(
            observations=torch.zeros((1, 10)),
            actions=torch.zeros(1, dtype=torch.long),
            rewards=torch.zeros(1),
            dones=torch.zeros(1, dtype=torch.bool),
            epsilons=torch.zeros(1),
            intrinsic_rewards=torch.zeros(1),
            survival_times=torch.zeros(1),
            curriculum_difficulties=torch.zeros(1),
            device=device,
        )

        decisions = curriculum.get_batch_decisions_with_qvalues(agent_states, ["agent_0"], q_values)

        assert decisions[0].reward_mode == "sparse"
        assert "SPARSE" in decisions[0].reason


class TestStatePersistence:
    """Test checkpoint and restore functionality."""

    def test_checkpoint_state_includes_all_tracker_state(self):
        """checkpoint_state should include all performance tracker state."""
        curriculum = AdversarialCurriculum(device=torch.device("cpu"))
        curriculum.initialize_population(num_agents=2)

        # Modify state
        curriculum.tracker.agent_stages = torch.tensor([2, 3])
        curriculum.tracker.episode_rewards = torch.tensor([10.0, 20.0])
        curriculum.tracker.episode_steps = torch.tensor([5.0, 8.0])
        curriculum.tracker.prev_avg_reward = torch.tensor([1.5, 2.5])
        curriculum.tracker.steps_at_stage = torch.tensor([500.0, 800.0])

        state = curriculum.checkpoint_state()

        assert torch.equal(state["agent_stages"], torch.tensor([2, 3]))
        assert torch.allclose(state["episode_rewards"], torch.tensor([10.0, 20.0]))
        assert torch.allclose(state["episode_steps"], torch.tensor([5.0, 8.0]))
        assert torch.allclose(state["prev_avg_reward"], torch.tensor([1.5, 2.5]))
        assert torch.allclose(state["steps_at_stage"], torch.tensor([500.0, 800.0]))

    def test_load_state_restores_all_state(self):
        """load_state should restore all performance tracker state."""
        curriculum = AdversarialCurriculum(device=torch.device("cpu"))
        curriculum.initialize_population(num_agents=2)

        state = {
            "agent_stages": torch.tensor([4, 5]),
            "episode_rewards": torch.tensor([15.0, 25.0]),
            "episode_steps": torch.tensor([7.0, 9.0]),
            "prev_avg_reward": torch.tensor([2.0, 3.0]),
            "steps_at_stage": torch.tensor([1200.0, 1500.0]),
        }

        curriculum.load_state(state)

        assert torch.equal(curriculum.tracker.agent_stages, torch.tensor([4, 5]))
        assert torch.allclose(curriculum.tracker.episode_rewards, torch.tensor([15.0, 25.0]))
        assert torch.allclose(curriculum.tracker.episode_steps, torch.tensor([7.0, 9.0]))
        assert torch.allclose(curriculum.tracker.prev_avg_reward, torch.tensor([2.0, 3.0]))
        assert torch.allclose(curriculum.tracker.steps_at_stage, torch.tensor([1200.0, 1500.0]))
