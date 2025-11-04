"""Shared test fixtures for Townlet test suite.

This module provides common fixtures used across unit, integration, and e2e tests.
Fixtures are organized by component area for easy discovery.

Usage:
    from pathlib import Path

    def test_something(mock_config_path, basic_env):
        # Use fixtures directly in test functions
        assert basic_env.grid_size == 8
"""

import shutil
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

from townlet.agent.networks import RecurrentSpatialQNetwork, SimpleQNetwork
from townlet.curriculum.adversarial import AdversarialCurriculum
from townlet.curriculum.static import StaticCurriculum
from townlet.environment.vectorized_env import VectorizedHamletEnv
from townlet.exploration.adaptive_intrinsic import AdaptiveIntrinsicExploration
from townlet.exploration.epsilon_greedy import EpsilonGreedyExploration
from townlet.population.vectorized import VectorizedPopulation
from townlet.training.replay_buffer import ReplayBuffer

# =============================================================================
# CONFIGURATION FIXTURES
# =============================================================================


@pytest.fixture(scope="session")
def mock_config_path() -> Path:
    """Return path to frozen mock configuration for exact-value assertions.

    This configuration is LOCKED and should not be modified. Tests that
    assert exact values MUST use this config to ensure stability.

    Returns:
        Path to fixtures/mock_config.yaml
    """
    return Path(__file__).parent / "fixtures" / "mock_config.yaml"


@pytest.fixture(scope="session")
def test_config_pack_path() -> Path:
    """Return path to test configuration pack (configs/test).

    This is a lightweight config pack for integration testing.
    Unlike mock_config, this can evolve with the codebase.

    Returns:
        Path to configs/test directory
    """
    repo_root = Path(__file__).parent.parent.parent
    return repo_root / "configs" / "test"


@pytest.fixture
def temp_config_pack(tmp_path: Path, test_config_pack_path: Path) -> Path:
    """Create a temporary writable copy of the test config pack.

    Use this fixture when you need to modify config files during tests.

    Args:
        tmp_path: pytest's temporary directory
        test_config_pack_path: Path to source config pack

    Returns:
        Path to temporary config pack directory
    """
    target_pack = tmp_path / "config_pack"
    shutil.copytree(test_config_pack_path, target_pack)
    return target_pack


@pytest.fixture
def mock_config(mock_config_path: Path) -> dict[str, Any]:
    """Load mock configuration as a dictionary.

    Returns:
        Dict containing frozen mock configuration
    """
    with open(mock_config_path) as f:
        return yaml.safe_load(f)


# =============================================================================
# DEVICE FIXTURES
# =============================================================================


@pytest.fixture(scope="session")
def device() -> torch.device:
    """Return CUDA device if available, otherwise CPU.

    Returns:
        torch.device: Preferred device for tests
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def cpu_device() -> torch.device:
    """Force CPU device for tests that need deterministic behavior.

    Returns:
        torch.device: CPU device
    """
    return torch.device("cpu")


# =============================================================================
# ENVIRONMENT FIXTURES
# =============================================================================


@pytest.fixture
def basic_env(test_config_pack_path: Path, device: torch.device) -> VectorizedHamletEnv:
    """Create a basic environment with default test config.

    Configuration:
        - 1 agent
        - 8×8 grid
        - Full observability
        - No temporal mechanics
        - Device: CUDA if available, else CPU

    Returns:
        VectorizedHamletEnv instance
    """
    return VectorizedHamletEnv(
        num_agents=1,
        grid_size=8,
        partial_observability=False,
        vision_range=8,
        enable_temporal_mechanics=False,
        move_energy_cost=0.005,
        wait_energy_cost=0.001,
        interact_energy_cost=0.0,
        config_pack_path=test_config_pack_path,
        device=device,
    )


@pytest.fixture
def pomdp_env(test_config_pack_path: Path, device: torch.device) -> VectorizedHamletEnv:
    """Create a POMDP environment with partial observability.

    Configuration:
        - 1 agent
        - 8×8 grid
        - Partial observability (5×5 vision)
        - No temporal mechanics
        - Device: CUDA if available, else CPU

    Returns:
        VectorizedHamletEnv instance with POMDP
    """
    return VectorizedHamletEnv(
        num_agents=1,
        grid_size=8,
        partial_observability=True,
        vision_range=2,  # 5×5 window
        enable_temporal_mechanics=False,
        move_energy_cost=0.005,
        wait_energy_cost=0.001,
        interact_energy_cost=0.0,
        config_pack_path=test_config_pack_path,
        device=device,
    )


@pytest.fixture
def temporal_env(test_config_pack_path: Path, device: torch.device) -> VectorizedHamletEnv:
    """Create an environment with temporal mechanics enabled.

    Configuration:
        - 1 agent
        - 8×8 grid
        - Full observability
        - Temporal mechanics (24-hour cycle)
        - Device: CUDA if available, else CPU

    Returns:
        VectorizedHamletEnv instance with temporal mechanics
    """
    return VectorizedHamletEnv(
        num_agents=1,
        grid_size=8,
        partial_observability=False,
        vision_range=8,
        enable_temporal_mechanics=True,
        move_energy_cost=0.005,
        wait_energy_cost=0.001,
        interact_energy_cost=0.0,
        config_pack_path=test_config_pack_path,
        device=device,
    )


@pytest.fixture
def multi_agent_env(test_config_pack_path: Path, device: torch.device) -> VectorizedHamletEnv:
    """Create an environment with multiple agents.

    Configuration:
        - 4 agents (for batching tests)
        - 8×8 grid
        - Full observability
        - No temporal mechanics
        - Device: CUDA if available, else CPU

    Returns:
        VectorizedHamletEnv instance with 4 agents
    """
    return VectorizedHamletEnv(
        num_agents=4,
        grid_size=8,
        partial_observability=False,
        vision_range=8,
        enable_temporal_mechanics=False,
        move_energy_cost=0.005,
        wait_energy_cost=0.001,
        interact_energy_cost=0.0,
        config_pack_path=test_config_pack_path,
        device=device,
    )


# =============================================================================
# NETWORK FIXTURES
# =============================================================================


@pytest.fixture
def simple_qnetwork(basic_env: VectorizedHamletEnv, device: torch.device) -> SimpleQNetwork:
    """Create a SimpleQNetwork for full observability.

    Args:
        basic_env: Environment to get observation dimension
        device: Device to place network on

    Returns:
        SimpleQNetwork instance
    """
    obs_dim = basic_env.observation_builder.get_observation_dim()
    return SimpleQNetwork(obs_dim=obs_dim, action_dim=5).to(device)


@pytest.fixture
def recurrent_qnetwork(pomdp_env: VectorizedHamletEnv, device: torch.device) -> RecurrentSpatialQNetwork:
    """Create a RecurrentSpatialQNetwork for POMDP.

    Args:
        pomdp_env: POMDP environment to get observation dimension
        device: Device to place network on

    Returns:
        RecurrentSpatialQNetwork instance
    """
    return RecurrentSpatialQNetwork(
        grid_size=5,  # 5×5 local vision
        num_affordance_types=14,
        num_meters=8,
        action_dim=5,
        hidden_dim=256,
    ).to(device)


# =============================================================================
# TRAINING COMPONENT FIXTURES
# =============================================================================


@pytest.fixture
def replay_buffer(device: torch.device) -> ReplayBuffer:
    """Create a basic replay buffer.

    Configuration:
        - Capacity: 1000
        - Observation dimension: 72 (full observability)
        - Device: CUDA if available, else CPU

    Returns:
        ReplayBuffer instance
    """
    return ReplayBuffer(capacity=1000, obs_dim=72, device=device)


@pytest.fixture
def adversarial_curriculum() -> AdversarialCurriculum:
    """Create an adversarial curriculum with test parameters.

    Configuration:
        - Max steps: 200 (short for tests)
        - Survival advance: 70%
        - Survival retreat: 30%
        - Entropy gate: 0.5
        - Min steps at stage: 50

    Returns:
        AdversarialCurriculum instance
    """
    return AdversarialCurriculum(
        max_steps_per_episode=200,
        survival_advance_threshold=0.7,
        survival_retreat_threshold=0.3,
        entropy_gate=0.5,
        min_steps_at_stage=50,
    )


@pytest.fixture
def static_curriculum() -> StaticCurriculum:
    """Create a static curriculum with fixed difficulty.

    Configuration:
        - Max steps: 200

    Returns:
        StaticCurriculum instance
    """
    return StaticCurriculum(max_steps_per_episode=200)


@pytest.fixture
def epsilon_greedy_exploration(device: torch.device) -> EpsilonGreedyExploration:
    """Create an epsilon-greedy exploration strategy.

    Configuration:
        - Epsilon start: 1.0
        - Epsilon min: 0.1
        - Epsilon decay: 0.99
        - Device: CUDA if available, else CPU

    Returns:
        EpsilonGreedyExploration instance
    """
    return EpsilonGreedyExploration(
        epsilon_start=1.0,
        epsilon_min=0.1,
        epsilon_decay=0.99,
        device=device,
    )


@pytest.fixture
def adaptive_intrinsic_exploration(basic_env: VectorizedHamletEnv, device: torch.device) -> AdaptiveIntrinsicExploration:
    """Create an adaptive intrinsic exploration strategy with RND.

    Configuration:
        - Embed dimension: 128
        - Initial intrinsic weight: 1.0
        - Variance threshold: 100.0
        - Survival window: 50

    Returns:
        AdaptiveIntrinsicExploration instance
    """
    obs_dim = basic_env.observation_builder.get_observation_dim()
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
    """Create a vectorized population for training.

    Configuration:
        - Network type: Simple (MLP)
        - Learning rate: 0.00025
        - Gamma: 0.99
        - Replay buffer capacity: 1000
        - Batch size: 32

    Returns:
        VectorizedPopulation instance
    """
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


# =============================================================================
# UTILITY FIXTURES
# =============================================================================


@pytest.fixture
def sample_observations(basic_env: VectorizedHamletEnv, device: torch.device) -> torch.Tensor:
    """Generate sample observations from basic environment.

    Args:
        basic_env: Environment to generate observations from
        device: Device to place observations on

    Returns:
        Tensor of shape (num_agents, obs_dim)
    """
    obs = basic_env.reset()
    return obs.to(device)


@pytest.fixture
def sample_actions(device: torch.device) -> torch.Tensor:
    """Generate sample actions tensor.

    Returns:
        Tensor of shape (4,) with actions [0, 1, 2, 3]
    """
    return torch.tensor([0, 1, 2, 3], device=device, dtype=torch.long)


# =============================================================================
# PYTEST CONFIGURATION
# =============================================================================


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: mark test as slow (run with --runslow)")
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU (skipped if no CUDA)")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "e2e: mark test as end-to-end test")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to skip GPU tests if CUDA unavailable."""
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="CUDA not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
