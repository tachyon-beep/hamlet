"""
Tests for curriculum stage info API.

Tests the get_stage_info() method for recording criteria decisions.
"""

import pytest
import torch


class TestCurriculumStageInfo:
    """Test curriculum stage information API."""

    def test_get_stage_info_returns_current_stage(self):
        """get_stage_info should return current stage for agent."""
        from townlet.curriculum.adversarial import AdversarialCurriculum

        curriculum = AdversarialCurriculum(
            max_steps_per_episode=100,
            device=torch.device("cpu"),
        )
        curriculum.initialize_population(num_agents=3)

        # Default stage is 1
        info = curriculum.get_stage_info(agent_idx=0)

        assert "current_stage" in info
        assert info["current_stage"] == 1

    def test_get_stage_info_returns_episodes_at_stage(self):
        """get_stage_info should return episode count at current stage."""
        from townlet.curriculum.adversarial import AdversarialCurriculum

        curriculum = AdversarialCurriculum(
            max_steps_per_episode=100,
            min_steps_at_stage=1000,
            device=torch.device("cpu"),
        )
        curriculum.initialize_population(num_agents=1)

        # Track several episodes
        for _ in range(5):
            survival = torch.tensor([80.0])  # 80% survival
            done = torch.tensor([True])
            curriculum.tracker.update_step(survival, done)

        info = curriculum.get_stage_info(agent_idx=0)

        assert "episodes_at_stage" in info
        assert info["episodes_at_stage"] == 5

    def test_get_stage_info_returns_survival_rate(self):
        """get_stage_info should return recent survival rate."""
        from townlet.curriculum.adversarial import AdversarialCurriculum

        curriculum = AdversarialCurriculum(
            max_steps_per_episode=100,
            device=torch.device("cpu"),
        )
        curriculum.initialize_population(num_agents=1)

        # Simulate 70% survival
        survival = torch.tensor([70.0])
        done = torch.tensor([True])
        curriculum.tracker.update_step(survival, done)

        info = curriculum.get_stage_info(agent_idx=0)

        assert "survival_rate" in info
        assert info["survival_rate"] == pytest.approx(0.7, rel=0.01)

    def test_get_stage_info_predicts_likely_transition(self):
        """get_stage_info should predict if stage transition is likely soon."""
        from townlet.curriculum.adversarial import AdversarialCurriculum

        curriculum = AdversarialCurriculum(
            max_steps_per_episode=100,
            survival_advance_threshold=0.7,
            min_steps_at_stage=10,  # Low threshold for testing
            device=torch.device("cpu"),
        )
        curriculum.initialize_population(num_agents=1)

        # Track enough episodes to meet min_steps_at_stage
        for _ in range(15):
            survival = torch.tensor([75.0])  # Above advance threshold
            done = torch.tensor([True])
            curriculum.tracker.update_step(survival, done)

        info = curriculum.get_stage_info(agent_idx=0)

        assert "likely_transition_soon" in info
        # With high survival and enough episodes, transition is likely
        assert info["likely_transition_soon"] is True

    def test_get_stage_info_no_transition_when_below_threshold(self):
        """get_stage_info should show no transition when survival is low."""
        from townlet.curriculum.adversarial import AdversarialCurriculum

        curriculum = AdversarialCurriculum(
            max_steps_per_episode=100,
            survival_advance_threshold=0.7,
            min_steps_at_stage=10,
            device=torch.device("cpu"),
        )
        curriculum.initialize_population(num_agents=1)

        # Track episodes with low survival
        for _ in range(15):
            survival = torch.tensor([50.0])  # Below advance threshold
            done = torch.tensor([True])
            curriculum.tracker.update_step(survival, done)

        info = curriculum.get_stage_info(agent_idx=0)

        assert info["likely_transition_soon"] is False

    def test_get_stage_info_no_transition_when_insufficient_episodes(self):
        """get_stage_info should show no transition when not enough episodes."""
        from townlet.curriculum.adversarial import AdversarialCurriculum

        curriculum = AdversarialCurriculum(
            max_steps_per_episode=100,
            survival_advance_threshold=0.7,
            min_steps_at_stage=100,  # Needs many episodes
            device=torch.device("cpu"),
        )
        curriculum.initialize_population(num_agents=1)

        # Track only a few episodes
        for _ in range(5):
            survival = torch.tensor([80.0])  # High survival
            done = torch.tensor([True])
            curriculum.tracker.update_step(survival, done)

        info = curriculum.get_stage_info(agent_idx=0)

        # Not enough episodes yet
        assert info["likely_transition_soon"] is False

    def test_get_stage_info_multi_agent(self):
        """get_stage_info should work for multiple agents independently."""
        from townlet.curriculum.adversarial import AdversarialCurriculum

        curriculum = AdversarialCurriculum(
            max_steps_per_episode=100,
            device=torch.device("cpu"),
        )
        curriculum.initialize_population(num_agents=3)

        # Different performance for each agent
        survival = torch.tensor([90.0, 50.0, 70.0])
        done = torch.tensor([True, True, True])
        curriculum.tracker.update_step(survival, done)

        # Check each agent independently
        info0 = curriculum.get_stage_info(agent_idx=0)
        info1 = curriculum.get_stage_info(agent_idx=1)
        info2 = curriculum.get_stage_info(agent_idx=2)

        assert info0["survival_rate"] == pytest.approx(0.9, rel=0.01)
        assert info1["survival_rate"] == pytest.approx(0.5, rel=0.01)
        assert info2["survival_rate"] == pytest.approx(0.7, rel=0.01)

    def test_get_stage_info_at_max_stage(self):
        """get_stage_info should handle agents at maximum stage."""
        from townlet.curriculum.adversarial import AdversarialCurriculum

        curriculum = AdversarialCurriculum(
            max_steps_per_episode=100,
            device=torch.device("cpu"),
        )
        curriculum.initialize_population(num_agents=1)

        # Manually set to max stage (5)
        curriculum.tracker.agent_stages[0] = 5

        info = curriculum.get_stage_info(agent_idx=0)

        assert info["current_stage"] == 5
        # At max stage, no transition possible
        assert info["likely_transition_soon"] is False
