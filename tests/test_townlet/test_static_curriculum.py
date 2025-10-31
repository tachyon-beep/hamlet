"""
Tests for Static Curriculum Manager.

Focus on business logic for static curriculum decisions and state persistence.
"""

import torch

from townlet.curriculum.static import StaticCurriculum
from townlet.training.state import BatchedAgentState


class TestStaticCurriculumDecisions:
    """Test curriculum decision generation."""

    def test_returns_same_decision_for_all_agents(self):
        """Should return identical decisions for all agents."""
        curriculum = StaticCurriculum(
            difficulty_level=0.7,
            reward_mode="sparse",
            active_meters=["energy", "hygiene"],
            depletion_multiplier=2.0,
        )

        # Create dummy agent state
        device = torch.device("cpu")
        agent_state = BatchedAgentState(
            observations=torch.zeros((3, 10)),
            actions=torch.zeros(3, dtype=torch.long),
            rewards=torch.zeros(3),
            dones=torch.zeros(3, dtype=torch.bool),
            epsilons=torch.zeros(3),
            intrinsic_rewards=torch.zeros(3),
            survival_times=torch.zeros(3),
            curriculum_difficulties=torch.zeros(3),
            device=device,
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

    def test_respects_different_difficulty_levels(self):
        """Should respect configured difficulty levels."""
        easy = StaticCurriculum(difficulty_level=0.2)
        hard = StaticCurriculum(difficulty_level=0.9)

        device = torch.device("cpu")
        agent_state = BatchedAgentState(
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

        easy_decisions = easy.get_batch_decisions(agent_state, ["agent_0"])
        hard_decisions = hard.get_batch_decisions(agent_state, ["agent_0"])

        assert abs(easy_decisions[0].difficulty_level - 0.2) < 1e-6
        assert abs(hard_decisions[0].difficulty_level - 0.9) < 1e-6


class TestStaticCurriculumStatePersistence:
    """Test checkpoint and restore functionality."""

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

    def test_checkpoint_restore_roundtrip(self):
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
        device = torch.device("cpu")
        agent_state = BatchedAgentState(
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

        original_decision = original.get_batch_decisions(agent_state, ["agent_0"])[0]
        restored_decision = restored.get_batch_decisions(agent_state, ["agent_0"])[0]

        assert abs(original_decision.difficulty_level - restored_decision.difficulty_level) < 1e-6
        assert original_decision.reward_mode == restored_decision.reward_mode
        assert original_decision.active_meters == restored_decision.active_meters
        assert original_decision.depletion_multiplier == restored_decision.depletion_multiplier
