#!/usr/bin/env python3
"""
Test script to demonstrate improved money reward valuation.

Compares reward signals at different money levels to show that the
new reward function encourages proactive buffer maintenance.
"""

from src.hamlet.environment.hamlet_env import HamletEnv
from src.hamlet.environment.entities import Agent
from src.hamlet.training.config import EnvironmentConfig


def test_reward_at_money_level(money_level: float, energy: float = 0.8, hygiene: float = 0.8, satiation: float = 0.8):
    """
    Test reward function at specific money level.

    Args:
        money_level: Money meter normalized (0.0 to 1.0)
        energy, hygiene, satiation: Other meters (normalized)

    Returns:
        Gradient reward for this state
    """
    # Create env and reset to initialize agents
    config = EnvironmentConfig()
    env = HamletEnv(config=config)
    env.reset()
    agent = env.agents["agent_0"]

    # Set meter values (de-normalize from 0-1 to actual values)
    agent.meters.get("energy").value = energy * 100
    agent.meters.get("hygiene").value = hygiene * 100
    agent.meters.get("satiation").value = satiation * 100
    agent.meters.get("money").value = money_level * 100

    # Calculate shaped reward (just gradient portion)
    prev_meters = {
        "energy": energy,
        "hygiene": hygiene,
        "satiation": satiation,
        "money": money_level
    }

    reward = env._calculate_shaped_reward(agent, prev_meters, interaction_affordance=None)

    return reward


def test_job_interaction_reward(money_level: float):
    """
    Test reward for working at Job at different money levels.

    Shows that new reward function encourages working earlier (50% money)
    rather than waiting until desperate (20% money).
    """
    config = EnvironmentConfig()
    env = HamletEnv(config=config)
    env.reset()
    agent = env.agents["agent_0"]

    # Set money to test level
    agent.meters.get("money").value = money_level * 100

    # Find Job affordance
    job = None
    for affordance in env.affordances:
        if affordance.name == "Job":
            job = affordance
            break

    # Calculate interaction reward
    prev_meters = {"money": money_level}
    interaction_reward = env._calculate_need_based_interaction_reward(
        agent, job, prev_meters
    )

    return interaction_reward


if __name__ == "__main__":
    print("=" * 70)
    print("IMPROVED MONEY REWARD VALUATION TEST")
    print("=" * 70)
    print()

    print("1. GRADIENT REWARDS (passive feedback for maintaining money buffer)")
    print("-" * 70)
    print(f"{'Money Level':<15} {'Normalized':<12} {'Gradient Reward':<20} {'Interpretation'}")
    print("-" * 70)

    test_levels = [
        (80, "High buffer"),
        (60, "Comfortable"),
        (50, "Adequate"),
        (40, "Low - work soon"),
        (30, "Concerning"),
        (20, "Critical!"),
        (10, "Emergency!"),
    ]

    for money_pct, interpretation in test_levels:
        money_norm = money_pct / 100.0
        reward = test_reward_at_money_level(money_norm)
        print(f"${money_pct:<14} {money_norm:<12.2f} {reward:<20.1f} {interpretation}")

    print()
    print("Key insight: Agent now gets continuous feedback about money levels")
    print("  - 60%+ money → +0.5 reward (maintain this!)")
    print("  - 40-60% → +0.2 reward (adequate)")
    print("  - 20-40% → -0.5 penalty (work soon!)")
    print("  - <20% → -2.0 penalty (critical!)")
    print()

    print()
    print("2. INTERACTION REWARDS (active reward for working at Job)")
    print("-" * 70)
    print(f"{'Money Level':<15} {'Normalized':<12} {'Job Reward':<20} {'Interpretation'}")
    print("-" * 70)

    for money_pct, interpretation in test_levels:
        money_norm = money_pct / 100.0
        job_reward = test_job_interaction_reward(money_norm)
        print(f"${money_pct:<14} {money_norm:<12.2f} {job_reward:<20.2f} {interpretation}")

    print()
    print("Key insight: 1.5x money multiplier makes working at 50% rewarding")
    print("  - Old: Working at 50% money → 0.6 reward")
    print("  - New: Working at 50% money → 0.9 reward (50% increase!)")
    print("  - Agent learns to work proactively, not desperately")
    print()

    print()
    print("3. PROXIMITY SHAPING (guides agent toward Job when money low)")
    print("-" * 70)
    print("New behavior:")
    print("  - When money < 50%, agent gets proximity rewards for moving toward Job")
    print("  - Same mechanism that guides to Bed/Shower/Fridge for biological needs")
    print("  - Agent learns: 'Low money → go to work location'")
    print()

    print()
    print("=" * 70)
    print("EXPECTED BEHAVIOR CHANGES")
    print("=" * 70)
    print()
    print("OLD AGENT (money undervalued):")
    print("  1. Ignores money until critical")
    print("  2. Uses services → money drops")
    print("  3. Multiple meters critical → death spiral")
    print("  4. Too late to recover")
    print()
    print("NEW AGENT (money properly valued):")
    print("  1. Maintains 40-60% money buffer (gets gradient reward)")
    print("  2. Works proactively when money hits 50% (amplified need)")
    print("  3. Gets guided toward Job when money low (proximity)")
    print("  4. Sustainable cycle: work → buffer → spend → work")
    print()
    print("Training a new agent will test this hypothesis.")
    print("=" * 70)
