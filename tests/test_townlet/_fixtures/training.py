"""Training component fixtures."""

from __future__ import annotations

import pytest
import torch

from townlet.curriculum.adversarial import AdversarialCurriculum
from townlet.curriculum.static import StaticCurriculum
from townlet.environment.vectorized_env import VectorizedHamletEnv
from townlet.exploration.adaptive_intrinsic import AdaptiveIntrinsicExploration
from townlet.exploration.epsilon_greedy import EpsilonGreedyExploration
from townlet.training.replay_buffer import ReplayBuffer

__all__ = [
    "replay_buffer",
    "adversarial_curriculum",
    "static_curriculum",
    "epsilon_greedy_exploration",
    "adaptive_intrinsic_exploration",
]


@pytest.fixture
def replay_buffer(device: torch.device) -> ReplayBuffer:
    """Create a basic replay buffer."""

    return ReplayBuffer(capacity=1000, device=device)


@pytest.fixture
def adversarial_curriculum() -> AdversarialCurriculum:
    """Adversarial curriculum with short test-friendly parameters."""

    return AdversarialCurriculum(
        max_steps_per_episode=200,
        survival_advance_threshold=0.7,
        survival_retreat_threshold=0.3,
        entropy_gate=0.5,
        min_steps_at_stage=50,
    )


@pytest.fixture
def static_curriculum() -> StaticCurriculum:
    """Static curriculum with fixed difficulty."""

    return StaticCurriculum(max_steps_per_episode=200)


@pytest.fixture
def epsilon_greedy_exploration(device: torch.device) -> EpsilonGreedyExploration:
    """Baseline epsilon-greedy exploration strategy."""

    return EpsilonGreedyExploration(
        epsilon=1.0,
        epsilon_min=0.1,
        epsilon_decay=0.99,
    )


@pytest.fixture
def adaptive_intrinsic_exploration(basic_env: VectorizedHamletEnv, device: torch.device) -> AdaptiveIntrinsicExploration:
    """Adaptive intrinsic exploration strategy with RND."""

    obs_dim = basic_env.observation_dim
    return AdaptiveIntrinsicExploration(
        obs_dim=obs_dim,
        embed_dim=128,
        initial_intrinsic_weight=1.0,
        variance_threshold=100.0,
        survival_window=50,
        device=device,
    )
