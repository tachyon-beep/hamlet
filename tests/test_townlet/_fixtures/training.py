"""Training component fixtures."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
import torch

from tests.test_townlet._fixtures.constants import FULL_OBS_DIM_8X8
from townlet.curriculum.adversarial import AdversarialCurriculum
from townlet.curriculum.static import StaticCurriculum
from townlet.environment.vectorized_env import VectorizedHamletEnv
from townlet.exploration.adaptive_intrinsic import AdaptiveIntrinsicExploration
from townlet.exploration.epsilon_greedy import EpsilonGreedyExploration
from townlet.population.vectorized import VectorizedPopulation
from townlet.training.replay_buffer import ReplayBuffer
from townlet.universe.compiled import CompiledUniverse

__all__ = [
    "replay_buffer",
    "adversarial_curriculum",
    "static_curriculum",
    "epsilon_greedy_exploration",
    "adaptive_intrinsic_exploration",
    "vectorized_population",
    "non_training_recurrent_population",
]


@pytest.fixture
def replay_buffer(device: torch.device) -> ReplayBuffer:
    """Create a basic replay buffer."""

    return ReplayBuffer(capacity=1000, obs_dim=FULL_OBS_DIM_8X8, device=device)


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


@pytest.fixture
def vectorized_population(
    basic_env: VectorizedHamletEnv,
    adversarial_curriculum: AdversarialCurriculum,
    epsilon_greedy_exploration: EpsilonGreedyExploration,
    device: torch.device,
) -> VectorizedPopulation:
    """Vectorized population for standard training tests."""

    return VectorizedPopulation(
        env=basic_env,
        curriculum=adversarial_curriculum,
        exploration=epsilon_greedy_exploration,
        network_type="simple",
        learning_rate=0.00025,
        gamma=0.99,
        replay_buffer_capacity=1000,
        batch_size=32,
        device=device,
    )


@pytest.fixture
def non_training_recurrent_population(
    compile_universe: Callable[[Path | str], CompiledUniverse],
    cpu_device: torch.device,
) -> VectorizedPopulation:
    """Recurrent population with training disabled for deterministic tests."""

    pomdp_universe = compile_universe(Path("configs/L2_partial_observability"))
    env = VectorizedHamletEnv.from_universe(
        pomdp_universe,
        num_agents=1,
        device=cpu_device,
    )

    curriculum = StaticCurriculum()
    exploration = EpsilonGreedyExploration(
        epsilon=0.1,
        epsilon_min=0.1,
        epsilon_decay=1.0,
        device=cpu_device,
    )

    return VectorizedPopulation(
        env=env,
        curriculum=curriculum,
        exploration=exploration,
        agent_ids=["agent_0"],
        device=cpu_device,
        network_type="recurrent",
        learning_rate=0.00025,
        gamma=0.99,
        replay_buffer_capacity=1000,
        batch_size=8,
        train_frequency=10000,
        sequence_length=8,
    )
