"""
Oracle validation tests.

Compare Townlet implementation against Hamlet reference to verify correctness.
"""

import pytest
import torch
import numpy as np


@pytest.mark.slow
def test_shaped_rewards_match_hamlet():
    """Townlet shaped rewards should match Hamlet within tolerance."""
    from townlet.environment.vectorized_env import VectorizedHamletEnv
    from hamlet.environment.hamlet_env import HamletEnv
    from hamlet.training.config import EnvironmentConfig

    # Create both environments
    townlet_env = VectorizedHamletEnv(
        num_agents=1,
        grid_size=8,
        device=torch.device('cpu'),
    )

    hamlet_env = HamletEnv(config=EnvironmentConfig(grid_width=8, grid_height=8))

    # Reset both
    townlet_obs = townlet_env.reset()
    hamlet_obs = hamlet_env.reset()

    # Set same initial state (agent at position [4, 4])
    townlet_env.positions[0] = torch.tensor([4, 4])
    hamlet_agent = hamlet_env.agents["agent_0"]
    hamlet_agent.x = 4
    hamlet_agent.y = 4

    # Set same meter values
    # NOTE: money=0.75 in Townlet corresponds to money=50 in Hamlet (range [-100, 100])
    initial_meters = [1.0, 1.0, 1.0, 0.75, 1.0, 0.5]
    townlet_env.meters[0] = torch.tensor(initial_meters)
    hamlet_agent.meters.meters["energy"].value = 100.0
    hamlet_agent.meters.meters["hygiene"].value = 100.0
    hamlet_agent.meters.meters["satiation"].value = 100.0
    hamlet_agent.meters.meters["money"].value = 50.0
    hamlet_agent.meters.meters["mood"].value = 100.0
    hamlet_agent.meters.meters["social"].value = 50.0

    # Take same action sequence
    actions = [0, 0, 1, 3, 2, 4]  # UP, UP, DOWN, RIGHT, LEFT, INTERACT

    townlet_rewards = []
    hamlet_rewards = []

    for action in actions:
        # Townlet step
        _, townlet_reward, _, _ = townlet_env.step(torch.tensor([action]))
        townlet_rewards.append(townlet_reward[0].item())

        # Hamlet step
        _, hamlet_reward, _, _ = hamlet_env.step(action)
        hamlet_rewards.append(hamlet_reward)

    # Compare rewards
    townlet_rewards = np.array(townlet_rewards)
    hamlet_rewards = np.array(hamlet_rewards)

    # Should match within 1e-3 tolerance
    np.testing.assert_allclose(
        townlet_rewards,
        hamlet_rewards,
        rtol=1e-3,
        atol=1e-3,
        err_msg="Townlet rewards diverged from Hamlet oracle"
    )


@pytest.mark.slow
def test_meter_depletion_matches_hamlet():
    """Meter depletion should match Hamlet."""
    from townlet.environment.vectorized_env import VectorizedHamletEnv
    from hamlet.environment.hamlet_env import HamletEnv
    from hamlet.training.config import EnvironmentConfig

    townlet_env = VectorizedHamletEnv(num_agents=1, grid_size=8, device=torch.device('cpu'))
    hamlet_env = HamletEnv(config=EnvironmentConfig(grid_width=8, grid_height=8))

    townlet_env.reset()
    hamlet_env.reset()

    # Take 50 random steps
    np.random.seed(42)
    for _ in range(50):
        action = np.random.randint(0, 4)  # Movement only

        townlet_env.step(torch.tensor([action]))
        hamlet_env.step(action)

    # Compare meter values (use normalize() to handle different ranges)
    townlet_meters = townlet_env.meters[0].cpu().numpy()
    hamlet_agent = hamlet_env.agents["agent_0"]
    hamlet_meters = np.array([
        hamlet_agent.meters.meters["energy"].normalize(),
        hamlet_agent.meters.meters["hygiene"].normalize(),
        hamlet_agent.meters.meters["satiation"].normalize(),
        hamlet_agent.meters.meters["money"].normalize(),
        hamlet_agent.meters.meters["mood"].normalize(),
        hamlet_agent.meters.meters["social"].normalize(),
    ])

    # Should match within 1e-2 tolerance
    np.testing.assert_allclose(
        townlet_meters,
        hamlet_meters,
        rtol=1e-2,
        atol=1e-2,
        err_msg="Townlet meter depletion diverged from Hamlet"
    )


def test_vectorized_env_deterministic_seed():
    """Same seed should produce same trajectory."""
    from townlet.environment.vectorized_env import VectorizedHamletEnv

    def run_trajectory(seed: int):
        torch.manual_seed(seed)
        env = VectorizedHamletEnv(num_agents=1, grid_size=8, device=torch.device('cpu'))
        env.reset()

        rewards = []
        for _ in range(20):
            _, reward, _, _ = env.step(torch.tensor([0]))  # Always UP
            rewards.append(reward[0].item())

        return rewards

    rewards1 = run_trajectory(seed=123)
    rewards2 = run_trajectory(seed=123)

    assert rewards1 == rewards2, "Same seed should produce same trajectory"
