"""Integration tests for DACEngine in environment."""

from pathlib import Path

import pytest
import torch

from townlet.environment.vectorized_env import VectorizedHamletEnv
from townlet.universe.compiler import UniverseCompiler


@pytest.fixture(scope="module")
def compiler():
    """Create a UniverseCompiler instance."""
    return UniverseCompiler()


@pytest.fixture(scope="module")
def configs_root():
    """Path to configs directory."""
    return Path(__file__).parent.parent.parent.parent / "configs"


def test_dac_engine_reward_calculation(compiler, configs_root):
    """Test that environment uses DACEngine for reward calculation."""
    # Compile L0_0_minimal universe
    config_dir = configs_root / "L0_0_minimal"
    universe = compiler.compile(config_dir)

    # Create environment
    env = VectorizedHamletEnv.from_universe(universe, num_agents=4, device=torch.device("cpu"))

    # Reset environment
    obs = env.reset()

    # Step once
    actions = torch.zeros(4, dtype=torch.long)  # All agents WAIT
    obs, rewards, dones, info = env.step(actions)

    # Verify rewards computed
    assert rewards.shape == (4,)
    assert torch.all(rewards >= 0.0)  # L0_0 uses multiplicative, should be positive

    # Verify DACEngine attribute exists
    assert hasattr(env, "dac_engine")
    assert env.dac_engine is not None

    # Verify no legacy reward_strategy attribute
    assert not hasattr(env, "reward_strategy")


def test_dac_engine_all_curriculum_levels(compiler, configs_root):
    """Test DACEngine integration across all curriculum levels."""
    config_dirs = [
        "L0_0_minimal",
        "L0_5_dual_resource",
        "L1_full_observability",
        "L2_partial_observability",
        "L3_temporal_mechanics",
    ]

    for config_name in config_dirs:
        config_dir = configs_root / config_name
        universe = compiler.compile(config_dir)
        env = VectorizedHamletEnv.from_universe(universe, num_agents=2, device=torch.device("cpu"))

        obs = env.reset()
        actions = torch.zeros(2, dtype=torch.long)
        obs, rewards, dones, info = env.step(actions)

        # Verify rewards computed
        assert rewards.shape == (2,)
        assert hasattr(env, "dac_engine")
