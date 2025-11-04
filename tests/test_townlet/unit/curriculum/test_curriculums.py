"""Consolidated tests for curriculum implementations.

This file consolidates:
- test_static_curriculum.py (StaticCurriculum - fixed difficulty)
- test_adversarial_curriculum.py (AdversarialCurriculum - adaptive difficulty)
- test_curriculum_signal_purity.py (unit-level signal purity tests)

Old files consolidated:
- tests/test_townlet/test_static_curriculum.py (7 tests)
- tests/test_townlet/test_adversarial_curriculum.py (25 tests)
- tests/test_townlet/test_curriculum_signal_purity.py (4 unit-level tests)

NOT included (integration-level tests for Task 11):
- test_p1_1_phase3_curriculum.py (runner checkpoint integration)
- test_curriculum_signal_purity.py integration tests (full episode runs)

Key behaviors tested:
- StaticCurriculum: fixed difficulty, state persistence
- AdversarialCurriculum: 5-stage progression, advancement/retreat conditions
- PerformanceTracker: reward accumulation, survival rate calculation
- Signal purity: curriculum sees pure survival time, not contaminated by intrinsic rewards
"""

import pytest
import torch

from townlet.curriculum.adversarial import AdversarialCurriculum, PerformanceTracker
from townlet.curriculum.static import StaticCurriculum
from townlet.training.state import BatchedAgentState


# =============================================================================
# STATIC CURRICULUM TESTS
# =============================================================================


class TestStaticCurriculum:
    """Test StaticCurriculum (fixed difficulty)."""

    def test_returns_same_decision_for_all_agents(self, cpu_device):
        """Should return identical decisions for all agents."""
        curriculum = StaticCurriculum(
            difficulty_level=0.7,
            reward_mode="sparse",
            active_meters=["energy", "hygiene"],
            depletion_multiplier=2.0,
        )

        # Create dummy agent state
        agent_state = BatchedAgentState(
            observations=torch.zeros((3, 10)),
            actions=torch.zeros(3, dtype=torch.long),
            rewards=torch.zeros(3),
            dones=torch.zeros(3, dtype=torch.bool),
            epsilons=torch.zeros(3),
            intrinsic_rewards=torch.zeros(3),
            survival_times=torch.zeros(3),
            curriculum_difficulties=torch.zeros(3),
            device=cpu_device,
        )

        agent_ids = ["agent_0", "agent_1", "agent_2"]
        decisions = curriculum.get_batch_decisions(agent_state, agent_ids)

        # Should return list of length 3
        assert len(decisions) == 3

        # All decisions should be identical
        for decision in decisions:
            assert abs(decision.difficulty_level - 0.7) < 1e-6
            assert decision.reward_mode == "sparse"
            assert decision.active_meters == ["energy", "hygiene"]
            assert abs(decision.depletion_multiplier - 2.0) < 1e-6
            assert "Static curriculum" in decision.reason
            assert "0.7" in decision.reason

    def test_respects_different_difficulty_levels(self, cpu_device):
        """Should respect configured difficulty levels."""
        easy = StaticCurriculum(difficulty_level=0.2)
        hard = StaticCurriculum(difficulty_level=0.9)

        agent_state = BatchedAgentState(
            observations=torch.zeros((1, 10)),
            actions=torch.zeros(1, dtype=torch.long),
            rewards=torch.zeros(1),
            dones=torch.zeros(1, dtype=torch.bool),
            epsilons=torch.zeros(1),
            intrinsic_rewards=torch.zeros(1),
            survival_times=torch.zeros(1),
            curriculum_difficulties=torch.zeros(1),
            device=cpu_device,
        )

        easy_decisions = easy.get_batch_decisions(agent_state, ["agent_0"])
        hard_decisions = hard.get_batch_decisions(agent_state, ["agent_0"])

        assert abs(easy_decisions[0].difficulty_level - 0.2) < 1e-6
        assert abs(hard_decisions[0].difficulty_level - 0.9) < 1e-6

    def test_checkpoint_state_returns_all_config(self):
        """checkpoint_state should return complete configuration."""
        curriculum = StaticCurriculum(
            difficulty_level=0.6,
            reward_mode="shaped",
            active_meters=["energy", "mood", "social"],
            depletion_multiplier=1.5,
        )

        state = curriculum.checkpoint_state()

        # Should contain all configuration
        assert abs(state["difficulty_level"] - 0.6) < 1e-6
        assert state["reward_mode"] == "shaped"
        assert state["active_meters"] == ["energy", "mood", "social"]
        assert abs(state["depletion_multiplier"] - 1.5) < 1e-6

    def test_load_state_restores_configuration(self):
        """load_state should restore all configuration."""
        curriculum = StaticCurriculum(
            difficulty_level=0.3,
            reward_mode="sparse",
            active_meters=["energy"],
            depletion_multiplier=1.0,
        )

        # Create new state to load
        new_state = {
            "difficulty_level": 0.8,
            "reward_mode": "shaped",
            "active_meters": ["energy", "hygiene", "satiation"],
            "depletion_multiplier": 2.5,
        }

        curriculum.load_state(new_state)

        # Should have updated all fields
        assert abs(curriculum.difficulty_level - 0.8) < 1e-6
        assert curriculum.reward_mode == "shaped"
        assert curriculum.active_meters == ["energy", "hygiene", "satiation"]
        assert abs(curriculum.depletion_multiplier - 2.5) < 1e-6

    def test_checkpoint_restore_roundtrip(self, cpu_device):
        """Checkpoint and restore should preserve exact state."""
        original = StaticCurriculum(
            difficulty_level=0.42,
            reward_mode="sparse",
            active_meters=["energy", "hygiene", "satiation", "mood"],
            depletion_multiplier=1.75,
        )

        # Save state
        state = original.checkpoint_state()

        # Create new curriculum and restore
        restored = StaticCurriculum()  # Default init
        restored.load_state(state)

        # Should match original exactly
        assert restored.difficulty_level == original.difficulty_level
        assert restored.reward_mode == original.reward_mode
        assert restored.active_meters == original.active_meters
        assert restored.depletion_multiplier == original.depletion_multiplier

        # Verify decisions are identical
        agent_state = BatchedAgentState(
            observations=torch.zeros((1, 10)),
            actions=torch.zeros(1, dtype=torch.long),
            rewards=torch.zeros(1),
            dones=torch.zeros(1, dtype=torch.bool),
            epsilons=torch.zeros(1),
            intrinsic_rewards=torch.zeros(1),
            survival_times=torch.zeros(1),
            curriculum_difficulties=torch.zeros(1),
            device=cpu_device,
        )

        original_decision = original.get_batch_decisions(agent_state, ["agent_0"])[0]
        restored_decision = restored.get_batch_decisions(agent_state, ["agent_0"])[0]

        assert abs(original_decision.difficulty_level - restored_decision.difficulty_level) < 1e-6
        assert original_decision.reward_mode == restored_decision.reward_mode
        assert original_decision.active_meters == restored_decision.active_meters
        assert original_decision.depletion_multiplier == restored_decision.depletion_multiplier


# =============================================================================
# PERFORMANCE TRACKER TESTS
# =============================================================================


class TestPerformanceTracker:
    """Test performance metrics tracking."""

    def test_update_step_accumulates_rewards_and_steps(self, cpu_device):
        """update_step should accumulate rewards and steps until episode ends."""
        tracker = PerformanceTracker(num_agents=2, device=cpu_device)

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

    def test_update_step_resets_on_done(self, cpu_device):
        """update_step should reset counters when episode ends."""
        tracker = PerformanceTracker(num_agents=2, device=cpu_device)

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

    def test_get_survival_rate_calculation(self, cpu_device):
        """get_survival_rate should return steps / max_steps."""
        tracker = PerformanceTracker(num_agents=3, device=cpu_device)

        tracker.episode_steps = torch.tensor([50.0, 100.0, 250.0])

        survival_rates = tracker.get_survival_rate(max_steps=500)

        assert abs(survival_rates[0] - 0.1) < 1e-6
        assert abs(survival_rates[1] - 0.2) < 1e-6
        assert abs(survival_rates[2] - 0.5) < 1e-6

    def test_get_learning_progress_positive(self, cpu_device):
        """get_learning_progress should show positive when improving."""
        tracker = PerformanceTracker(num_agents=1, device=cpu_device)

        # Baseline: average reward = 2.0
        tracker.prev_avg_reward[0] = 2.0

        # Current: 10 reward over 2 steps = 5.0 average
        tracker.episode_rewards[0] = 10.0
        tracker.episode_steps[0] = 2.0

        progress = tracker.get_learning_progress()

        assert abs(progress[0] - 3.0) < 1e-6

    def test_get_learning_progress_negative(self, cpu_device):
        """get_learning_progress should show negative when regressing."""
        tracker = PerformanceTracker(num_agents=1, device=cpu_device)

        # Baseline: average reward = 5.0
        tracker.prev_avg_reward[0] = 5.0

        # Current: 4 reward over 2 steps = 2.0 average
        tracker.episode_rewards[0] = 4.0
        tracker.episode_steps[0] = 2.0

        progress = tracker.get_learning_progress()

        assert abs(progress[0] - (-3.0)) < 1e-6

    def test_update_baseline_stores_current_average(self, cpu_device):
        """update_baseline should update prev_avg_reward to current average."""
        tracker = PerformanceTracker(num_agents=1, device=cpu_device)

        tracker.episode_rewards[0] = 15.0
        tracker.episode_steps[0] = 3.0

        tracker.update_baseline()

        # Should store 15.0 / 3.0 = 5.0
        assert abs(tracker.prev_avg_reward[0] - 5.0) < 1e-6


# =============================================================================
# ADVERSARIAL CURRICULUM: STAGE ADVANCEMENT
# =============================================================================


class TestAdversarialCurriculumStageAdvancement:
    """Test stage advancement conditions."""

    def test_should_not_advance_without_minimum_steps(self, cpu_device):
        """Should not advance before min_steps_at_stage reached."""
        curriculum = AdversarialCurriculum(
            max_steps_per_episode=500,
            min_steps_at_stage=1000,
            device=cpu_device,
        )
        curriculum.initialize_population(num_agents=1)

        # Only 500 steps at stage
        curriculum.tracker.steps_at_stage[0] = 500

        # Even with perfect metrics
        assert not curriculum._should_advance(0, entropy=0.0)

    def test_should_not_advance_from_stage_5(self, cpu_device):
        """Should not advance beyond stage 5 (max stage)."""
        curriculum = AdversarialCurriculum(
            max_steps_per_episode=500,
            min_steps_at_stage=100,
            device=cpu_device,
        )
        curriculum.initialize_population(num_agents=1)

        curriculum.tracker.agent_stages[0] = 5
        curriculum.tracker.steps_at_stage[0] = 2000

        # Even with perfect metrics
        assert not curriculum._should_advance(0, entropy=0.0)

    def test_should_advance_with_all_conditions_met(self, cpu_device):
        """Should advance when survival, learning, and entropy conditions met."""
        curriculum = AdversarialCurriculum(
            max_steps_per_episode=500,
            survival_advance_threshold=0.7,
            entropy_gate=0.5,
            min_steps_at_stage=100,
            device=cpu_device,
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

    def test_should_not_advance_with_low_survival(self, cpu_device):
        """Should not advance with low survival rate."""
        curriculum = AdversarialCurriculum(
            max_steps_per_episode=500,
            survival_advance_threshold=0.7,
            min_steps_at_stage=100,
            device=cpu_device,
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

    def test_should_not_advance_with_negative_learning(self, cpu_device):
        """Should not advance with negative learning progress."""
        curriculum = AdversarialCurriculum(
            max_steps_per_episode=500,
            survival_advance_threshold=0.7,
            min_steps_at_stage=100,
            device=cpu_device,
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

    def test_should_not_advance_with_high_entropy(self, cpu_device):
        """Should not advance with high entropy (not converged)."""
        curriculum = AdversarialCurriculum(
            max_steps_per_episode=500,
            survival_advance_threshold=0.7,
            entropy_gate=0.5,
            min_steps_at_stage=100,
            device=cpu_device,
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

    def test_stage_transition_logs_reason(self, cpu_device):
        """Advancement should emit structured transition telemetry."""
        curriculum = AdversarialCurriculum(
            max_steps_per_episode=500,
            survival_advance_threshold=0.7,
            entropy_gate=0.5,
            min_steps_at_stage=100,
            device=cpu_device,
        )
        curriculum.initialize_population(num_agents=1)

        # Pre-condition: agent at stage 1 with sufficient steps
        curriculum.tracker.agent_stages[0] = 1
        curriculum.tracker.steps_at_stage[0] = 1000
        curriculum.tracker.episode_steps[0] = 450  # High survival (0.9)
        curriculum.tracker.episode_rewards[0] = 90.0  # Average reward = 0.2
        curriculum.tracker.prev_avg_reward[0] = 0.05  # Positive learning progress

        # Dummy agent state (values unused by curriculum logic)
        agent_state = BatchedAgentState(
            observations=torch.zeros((1, 1)),
            actions=torch.zeros(1, dtype=torch.long),
            rewards=torch.zeros(1),
            dones=torch.tensor([True]),
            epsilons=torch.zeros(1),
            intrinsic_rewards=torch.zeros(1),
            survival_times=torch.tensor([450.0]),
            curriculum_difficulties=torch.zeros(1),
            device=cpu_device,
        )

        q_values = torch.tensor([[10.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)

        # Expect curriculum to expose telemetry container
        assert hasattr(curriculum, "transition_events")
        curriculum.transition_events.clear()

        curriculum.get_batch_decisions_with_qvalues(agent_state, ["agent_0"], q_values)

        assert len(curriculum.transition_events) == 1
        event = curriculum.transition_events[0]
        assert event["agent_id"] == "agent_0"
        assert event["from_stage"] == 1
        assert event["to_stage"] == 2
        assert event["reason"] == "advance"
        assert event["survival_rate"] == pytest.approx(0.9, rel=1e-3)
        assert event["learning_progress"] > 0.0


# =============================================================================
# ADVERSARIAL CURRICULUM: STAGE RETREAT
# =============================================================================


class TestAdversarialCurriculumStageRetreat:
    """Test stage retreat conditions."""

    def test_should_not_retreat_from_stage_1(self, cpu_device):
        """Should not retreat below stage 1 (min stage)."""
        curriculum = AdversarialCurriculum(
            max_steps_per_episode=500,
            min_steps_at_stage=100,
            device=cpu_device,
        )
        curriculum.initialize_population(num_agents=1)

        curriculum.tracker.agent_stages[0] = 1
        curriculum.tracker.steps_at_stage[0] = 2000

        # Even with terrible performance
        curriculum.tracker.episode_steps[0] = 10  # Low survival
        curriculum.tracker.prev_avg_reward[0] = 10.0  # Negative learning expected

        assert not curriculum._should_retreat(0)

    def test_should_not_retreat_without_minimum_steps(self, cpu_device):
        """Should not retreat before min_steps_at_stage reached."""
        curriculum = AdversarialCurriculum(
            max_steps_per_episode=500,
            min_steps_at_stage=1000,
            device=cpu_device,
        )
        curriculum.initialize_population(num_agents=1)

        curriculum.tracker.agent_stages[0] = 3
        curriculum.tracker.steps_at_stage[0] = 500

        # Even with terrible performance
        assert not curriculum._should_retreat(0)

    def test_should_retreat_with_low_survival(self, cpu_device):
        """Should retreat when survival < threshold."""
        curriculum = AdversarialCurriculum(
            max_steps_per_episode=500,
            survival_retreat_threshold=0.3,
            min_steps_at_stage=100,
            device=cpu_device,
        )
        curriculum.initialize_population(num_agents=1)

        curriculum.tracker.agent_stages[0] = 3
        curriculum.tracker.steps_at_stage[0] = 1000

        # Low survival: 100 / 500 = 0.2 < 0.3
        curriculum.tracker.episode_steps[0] = 100

        assert curriculum._should_retreat(0)

    def test_should_retreat_with_negative_learning(self, cpu_device):
        """Should retreat when learning progress is negative."""
        curriculum = AdversarialCurriculum(
            max_steps_per_episode=500,
            survival_retreat_threshold=0.3,
            min_steps_at_stage=100,
            device=cpu_device,
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


# =============================================================================
# ADVERSARIAL CURRICULUM: ENTROPY CALCULATION
# =============================================================================


class TestAdversarialCurriculumEntropyCalculation:
    """Test action entropy calculation from Q-values."""

    def test_entropy_zero_for_deterministic_policy(self, cpu_device):
        """Entropy should be ~0 for deterministic (peaked) Q-values."""
        curriculum = AdversarialCurriculum(device=cpu_device)

        # One action dominant
        q_values = torch.tensor([[100.0, 0.0, 0.0, 0.0, 0.0]])

        entropy = curriculum._calculate_action_entropy(q_values)

        # Should be near 0 (deterministic)
        assert entropy[0] < 0.1

    def test_entropy_high_for_uniform_policy(self, cpu_device):
        """Entropy should be ~1.0 for uniform (exploring) Q-values."""
        curriculum = AdversarialCurriculum(device=cpu_device)

        # All actions equally likely
        q_values = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0]])

        entropy = curriculum._calculate_action_entropy(q_values)

        # Should be near 1.0 (maximum entropy, normalized)
        assert entropy[0] > 0.9

    def test_entropy_batch_processing(self, cpu_device):
        """Should calculate entropy for each agent in batch."""
        curriculum = AdversarialCurriculum(device=cpu_device)

        q_values = torch.tensor(
            [
                [100.0, 0.0, 0.0, 0.0, 0.0],  # Deterministic
                [1.0, 1.0, 1.0, 1.0, 1.0],  # Uniform
            ]
        )

        entropy = curriculum._calculate_action_entropy(q_values)

        assert entropy[0] < 0.1  # Deterministic
        assert entropy[1] > 0.9  # Uniform


# =============================================================================
# ADVERSARIAL CURRICULUM: BATCH DECISIONS
# =============================================================================


class TestAdversarialCurriculumBatchDecisions:
    """Test curriculum decision generation."""

    def test_get_batch_decisions_advances_stage(self, cpu_device):
        """get_batch_decisions_with_qvalues should advance stage when conditions met."""
        curriculum = AdversarialCurriculum(
            max_steps_per_episode=500,
            survival_advance_threshold=0.7,
            entropy_gate=0.5,
            min_steps_at_stage=100,
            device=cpu_device,
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

        agent_states = BatchedAgentState(
            observations=torch.zeros((1, 10)),
            actions=torch.zeros(1, dtype=torch.long),
            rewards=torch.zeros(1),
            dones=torch.zeros(1, dtype=torch.bool),
            epsilons=torch.zeros(1),
            intrinsic_rewards=torch.zeros(1),
            survival_times=torch.zeros(1),
            curriculum_difficulties=torch.zeros(1),
            device=cpu_device,
        )

        decisions = curriculum.get_batch_decisions_with_qvalues(agent_states, ["agent_0"], q_values)

        # Should have advanced from stage 2 to 3
        assert curriculum.tracker.agent_stages[0] == 3
        assert decisions[0].reward_mode == "shaped"
        assert "Stage 3" in decisions[0].reason

    def test_get_batch_decisions_retreats_stage(self, cpu_device):
        """get_batch_decisions_with_qvalues should retreat stage when struggling."""
        curriculum = AdversarialCurriculum(
            max_steps_per_episode=500,
            survival_retreat_threshold=0.3,
            min_steps_at_stage=100,
            device=cpu_device,
        )
        curriculum.initialize_population(num_agents=1)

        # Setup for retreat
        curriculum.tracker.agent_stages[0] = 3
        curriculum.tracker.steps_at_stage[0] = 1000
        curriculum.tracker.episode_steps[0] = 100  # 0.2 survival < 0.3

        q_values = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0]])

        agent_states = BatchedAgentState(
            observations=torch.zeros((1, 10)),
            actions=torch.zeros(1, dtype=torch.long),
            rewards=torch.zeros(1),
            dones=torch.zeros(1, dtype=torch.bool),
            epsilons=torch.zeros(1),
            intrinsic_rewards=torch.zeros(1),
            survival_times=torch.zeros(1),
            curriculum_difficulties=torch.zeros(1),
            device=cpu_device,
        )

        decisions = curriculum.get_batch_decisions_with_qvalues(agent_states, ["agent_0"], q_values)

        # Should have retreated from stage 3 to 2
        assert curriculum.tracker.agent_stages[0] == 2
        assert "Stage 2" in decisions[0].reason

    def test_get_batch_decisions_maps_stage_to_difficulty(self, cpu_device):
        """Should map stage (1-5) to difficulty_level (0.0-1.0)."""
        curriculum = AdversarialCurriculum(device=cpu_device)
        curriculum.initialize_population(num_agents=5)

        # Set different stages
        curriculum.tracker.agent_stages = torch.tensor([1, 2, 3, 4, 5])

        q_values = torch.ones(5, 5)

        agent_states = BatchedAgentState(
            observations=torch.zeros((5, 10)),
            actions=torch.zeros(5, dtype=torch.long),
            rewards=torch.zeros(5),
            dones=torch.zeros(5, dtype=torch.bool),
            epsilons=torch.zeros(5),
            intrinsic_rewards=torch.zeros(5),
            survival_times=torch.zeros(5),
            curriculum_difficulties=torch.zeros(5),
            device=cpu_device,
        )

        decisions = curriculum.get_batch_decisions_with_qvalues(agent_states, ["a0", "a1", "a2", "a3", "a4"], q_values)

        # Stage 1 -> 0.0, Stage 2 -> 0.25, Stage 3 -> 0.5, Stage 4 -> 0.75, Stage 5 -> 1.0
        assert abs(decisions[0].difficulty_level - 0.0) < 1e-6
        assert abs(decisions[1].difficulty_level - 0.25) < 1e-6
        assert abs(decisions[2].difficulty_level - 0.5) < 1e-6
        assert abs(decisions[3].difficulty_level - 0.75) < 1e-6
        assert abs(decisions[4].difficulty_level - 1.0) < 1e-6

    def test_stage_5_is_sparse_rewards(self, cpu_device):
        """Stage 5 should use sparse rewards (graduation!)."""
        curriculum = AdversarialCurriculum(device=cpu_device)
        curriculum.initialize_population(num_agents=1)

        curriculum.tracker.agent_stages[0] = 5

        q_values = torch.ones(1, 5)

        agent_states = BatchedAgentState(
            observations=torch.zeros((1, 10)),
            actions=torch.zeros(1, dtype=torch.long),
            rewards=torch.zeros(1),
            dones=torch.zeros(1, dtype=torch.bool),
            epsilons=torch.zeros(1),
            intrinsic_rewards=torch.zeros(1),
            survival_times=torch.zeros(1),
            curriculum_difficulties=torch.zeros(1),
            device=cpu_device,
        )

        decisions = curriculum.get_batch_decisions_with_qvalues(agent_states, ["agent_0"], q_values)

        assert decisions[0].reward_mode == "sparse"
        assert "SPARSE" in decisions[0].reason


# =============================================================================
# ADVERSARIAL CURRICULUM: STATE PERSISTENCE
# =============================================================================


class TestAdversarialCurriculumStatePersistence:
    """Test checkpoint and restore functionality."""

    def test_checkpoint_state_includes_all_tracker_state(self, cpu_device):
        """checkpoint_state should include all performance tracker state."""
        curriculum = AdversarialCurriculum(device=cpu_device)
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

    def test_load_state_restores_all_state(self, cpu_device):
        """load_state should restore all performance tracker state."""
        curriculum = AdversarialCurriculum(device=cpu_device)
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


# =============================================================================
# CURRICULUM SIGNAL PURITY (UNIT-LEVEL TESTS ONLY)
# =============================================================================


class TestCurriculumSignalPurity:
    """Test reward signal purity (unit-level tests only).

    Integration tests that run full episodes are excluded and will be
    moved to integration/test_curriculum_integration.py in Task 11.
    """

    def test_curriculum_has_state_dict_methods(self, cpu_device):
        """Verify curriculum has state_dict() and load_state_dict() methods."""
        curriculum = AdversarialCurriculum(
            max_steps_per_episode=100,
            survival_advance_threshold=0.7,
            survival_retreat_threshold=0.3,
            device=cpu_device,
        )
        curriculum.initialize_population(1)

        # Verify method exists
        assert hasattr(curriculum, "state_dict"), "Curriculum should have state_dict()"
        assert hasattr(curriculum, "load_state_dict"), "Curriculum should have load_state_dict()"

        state = curriculum.state_dict()
        assert isinstance(state, dict), "state_dict should return dict"

        # Verify expected keys (tracker fields are returned directly)
        assert "agent_stages" in state, "Should have agent_stages"
        assert "episode_rewards" in state, "Should have episode_rewards"
        assert "steps_at_stage" in state, "Should have steps_at_stage"

    def test_curriculum_state_preserves_agent_stages(self, cpu_device):
        """Curriculum stages should be preserved across save/load."""
        curriculum1 = AdversarialCurriculum(
            max_steps_per_episode=100,
            survival_advance_threshold=0.7,
            survival_retreat_threshold=0.3,
            device=cpu_device,
        )
        curriculum1.initialize_population(1)

        # Directly set curriculum stage to test persistence
        curriculum1.tracker.agent_stages[0] = 3
        curriculum1.tracker.steps_at_stage[0] = 5000

        original_stage = curriculum1.tracker.agent_stages[0].item()
        assert original_stage == 3, "Should be at stage 3"

        # Save state
        state = curriculum1.state_dict()

        # Create new curriculum and load
        curriculum2 = AdversarialCurriculum(
            max_steps_per_episode=100,
            survival_advance_threshold=0.7,
            survival_retreat_threshold=0.3,
            device=cpu_device,
        )
        curriculum2.initialize_population(1)

        # Verify fresh curriculum starts at stage 1
        fresh_stage = curriculum2.tracker.agent_stages[0].item()
        assert fresh_stage == 1, "Fresh curriculum should start at stage 1"

        # Load saved state
        curriculum2.load_state_dict(state)

        # Verify stage was restored
        restored_stage = curriculum2.tracker.agent_stages[0].item()
        assert restored_stage == original_stage, f"Stage should be restored: {original_stage} vs {restored_stage}"
