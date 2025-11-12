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


def test_affordance_history_tracking_last_action(compiler, configs_root):
    """Test that _get_last_action_affordances() returns affordance IDs after INTERACT.

    RED: This test should fail because tracking is not implemented yet.
    """
    # Use L0_0_minimal which has one affordance (Bed)
    config_dir = configs_root / "L0_0_minimal"
    universe = compiler.compile(config_dir)

    device = torch.device("cpu")
    env = VectorizedHamletEnv.from_universe(universe, num_agents=4, device=device)

    # Reset environment
    obs = env.reset()

    # Get the INTERACT action index (should be 4 for Grid2D: UP, DOWN, LEFT, RIGHT, INTERACT)
    interact_action = env.interact_action_idx

    # Find the Bed affordance position
    bed_pos = env.affordances.get("Bed")
    assert bed_pos is not None, "L0_0_minimal should have a Bed affordance"

    # Place agents on the bed position
    env.positions[0] = bed_pos.clone()
    env.positions[1] = bed_pos.clone()

    # Agents 0 and 1 INTERACT at Bed, agents 2 and 3 WAIT
    actions = torch.tensor([interact_action, interact_action, 5, 5], dtype=torch.long)  # 5 = WAIT
    obs, rewards, dones, info = env.step(actions)

    # Get last action affordances
    last_affordances = env._get_last_action_affordances()

    # Verify tracking
    assert len(last_affordances) == 4, "Should return list of 4 affordances"
    assert last_affordances[0] == "Bed", "Agent 0 interacted with Bed"
    assert last_affordances[1] == "Bed", "Agent 1 interacted with Bed"
    assert last_affordances[2] is None, "Agent 2 did not interact"
    assert last_affordances[3] is None, "Agent 3 did not interact"


def test_affordance_history_tracking_streaks(compiler, configs_root):
    """Test that _get_affordance_streaks() increments on consecutive uses.

    RED: This test should fail because tracking is not implemented yet.
    """
    config_dir = configs_root / "L0_0_minimal"
    universe = compiler.compile(config_dir)

    device = torch.device("cpu")
    env = VectorizedHamletEnv.from_universe(universe, num_agents=2, device=device)

    obs = env.reset()

    interact_action = env.interact_action_idx
    bed_pos = env.affordances.get("Bed")

    # Place agent 0 on Bed
    env.positions[0] = bed_pos.clone()

    # Agent 0 interacts with Bed twice in a row
    actions = torch.tensor([interact_action, 5], dtype=torch.long)  # Agent 0 interact, Agent 1 wait

    # First interaction
    obs, rewards, dones, info = env.step(actions)
    streaks = env._get_affordance_streaks()
    assert "Bed" in streaks, "Bed should appear in streaks"
    assert streaks["Bed"][0] == 1, "First interaction should have streak of 1"
    assert streaks["Bed"][1] == 0, "Agent 1 did not interact, streak should be 0"

    # Second consecutive interaction
    obs, rewards, dones, info = env.step(actions)
    streaks = env._get_affordance_streaks()
    assert streaks["Bed"][0] == 2, "Second consecutive interaction should increment streak to 2"
    assert streaks["Bed"][1] == 0, "Agent 1 still did not interact"


def test_affordance_history_tracking_unique_count(compiler, configs_root):
    """Test that _get_unique_affordances_used() counts unique affordances.

    RED: This test should fail because tracking is not implemented yet.
    """
    # Use L0_5_dual_resource which has 4 affordances
    config_dir = configs_root / "L0_5_dual_resource"
    universe = compiler.compile(config_dir)

    device = torch.device("cpu")
    env = VectorizedHamletEnv.from_universe(universe, num_agents=2, device=device)

    obs = env.reset()

    interact_action = env.interact_action_idx

    # Get positions of different affordances
    affordance_names = list(env.affordances.keys())
    assert len(affordance_names) >= 2, "L0_5 should have at least 2 affordances"

    first_aff = affordance_names[0]
    second_aff = affordance_names[1]

    first_pos = env.affordances[first_aff]
    second_pos = env.affordances[second_aff]

    # Agent 0 interacts with first affordance
    env.positions[0] = first_pos.clone()
    actions = torch.tensor([interact_action, 5], dtype=torch.long)
    obs, rewards, dones, info = env.step(actions)

    unique_counts = env._get_unique_affordances_used()
    assert unique_counts[0] == 1, "Agent 0 used 1 unique affordance"
    assert unique_counts[1] == 0, "Agent 1 used 0 unique affordances"

    # Agent 0 interacts with second affordance (different from first)
    env.positions[0] = second_pos.clone()
    obs, rewards, dones, info = env.step(actions)

    unique_counts = env._get_unique_affordances_used()
    assert unique_counts[0] == 2, "Agent 0 used 2 unique affordances"
    assert unique_counts[1] == 0, "Agent 1 still used 0 unique affordances"

    # Agent 0 interacts with first affordance again (should not increment unique count)
    env.positions[0] = first_pos.clone()
    obs, rewards, dones, info = env.step(actions)

    unique_counts = env._get_unique_affordances_used()
    assert unique_counts[0] == 2, "Agent 0 still used only 2 unique affordances"


def test_affordance_history_tracking_reset(compiler, configs_root):
    """Test that affordance tracking is reset when environment is reset.

    RED: This test should fail because tracking is not implemented yet.
    """
    config_dir = configs_root / "L0_0_minimal"
    universe = compiler.compile(config_dir)

    device = torch.device("cpu")
    env = VectorizedHamletEnv.from_universe(universe, num_agents=2, device=device)

    obs = env.reset()

    interact_action = env.interact_action_idx
    bed_pos = env.affordances.get("Bed")

    # Interact with Bed
    env.positions[0] = bed_pos.clone()
    actions = torch.tensor([interact_action, 5], dtype=torch.long)
    obs, rewards, dones, info = env.step(actions)

    # Verify tracking has data
    last_affordances = env._get_last_action_affordances()
    assert last_affordances[0] == "Bed", "Should have tracked interaction"

    # Reset environment
    env.reset()

    # Verify tracking is cleared
    last_affordances = env._get_last_action_affordances()
    assert all(aff is None for aff in last_affordances), "All tracking should be cleared after reset"

    unique_counts = env._get_unique_affordances_used()
    assert torch.all(unique_counts == 0), "Unique counts should be reset to zero"

    streaks = env._get_affordance_streaks()
    # Streaks dict should be empty or have all zeros
    for aff_name, streak_tensor in streaks.items():
        assert torch.all(streak_tensor == 0), f"Streak for {aff_name} should be reset to zero"
