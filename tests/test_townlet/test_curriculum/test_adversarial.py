"""Tests for AdversarialCurriculum."""

import pytest
import torch
from townlet.curriculum.adversarial import AdversarialCurriculum
from townlet.training.state import BatchedAgentState


def test_adversarial_curriculum_construction():
    """AdversarialCurriculum should initialize with stage 1 defaults."""
    curriculum = AdversarialCurriculum(
        max_steps_per_episode=500,
        device=torch.device('cpu'),
    )

    assert curriculum.current_stage == 1
    assert curriculum.max_steps_per_episode == 500
    assert curriculum.device.type == 'cpu'

    # Stage 1 specs
    assert curriculum._get_active_meters(1) == ['energy', 'hygiene']
    assert curriculum._get_depletion_multiplier(1) == 0.2
    assert curriculum._get_reward_mode(1) == 'shaped'
