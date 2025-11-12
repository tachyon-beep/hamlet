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


def test_intrinsic_rewards_integration(compiler, configs_root):
    """Test that intrinsic rewards are properly wired from exploration module to DACEngine.

    CRITICAL BUG: Environment currently passes zeros instead of real intrinsic rewards.
    This test will FAIL until the bug is fixed.
    """
    from townlet.exploration.rnd import RNDExploration

    # Use L0_0_minimal which has intrinsic.base_weight = 0.1
    config_dir = configs_root / "L0_0_minimal"
    universe = compiler.compile(config_dir)

    # Create environment
    device = torch.device("cpu")
    env = VectorizedHamletEnv.from_universe(universe, num_agents=4, device=device)

    # Get observation dimension from environment
    obs = env.reset()
    obs_dim = obs.shape[1]

    # Create RND exploration module
    exploration = RNDExploration(
        obs_dim=obs_dim,
        embed_dim=128,
        learning_rate=1e-4,
        training_batch_size=128,
        epsilon_start=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        device=device,
    )

    # Wire exploration module to environment
    env.set_exploration_module(exploration)

    # Step environment
    actions = torch.zeros(4, dtype=torch.long)  # All agents WAIT
    obs, rewards, dones, info = env.step(actions)

    # Compute intrinsic rewards directly from exploration module
    expected_intrinsic = exploration.compute_intrinsic_rewards(obs, update_stats=False)

    # Verify intrinsic rewards are non-zero (RND should produce novelty signals)
    assert torch.any(expected_intrinsic != 0.0), "RND exploration should produce non-zero intrinsic rewards for novel states"

    # Verify environment actually used intrinsic rewards
    # Check if reward components were stored
    assert hasattr(env, "_last_reward_components"), "Environment should store reward components for debugging"

    components = env._last_reward_components
    assert "intrinsic" in components, "Reward components should include intrinsic values"

    # Verify intrinsic component is non-zero
    intrinsic_component = components["intrinsic"]
    assert torch.any(intrinsic_component != 0.0), (
        "CRITICAL BUG: Environment is passing zeros instead of real intrinsic rewards! "
        "Check vectorized_env.py:_calculate_shaped_rewards() line ~1203"
    )
