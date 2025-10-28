"""
Tests for HamletEnv PettingZoo environment.
"""

import pytest
import numpy as np
from hamlet.environment.hamlet_env import HamletEnv
from hamlet.training.config import EnvironmentConfig


def test_env_initialization():
    """Test that environment initializes correctly."""
    env = HamletEnv()
    assert env.grid is not None
    assert env.agents is not None
    assert env.affordances is not None


def test_env_reset():
    """Test environment reset."""
    env = HamletEnv()
    obs = env.reset()

    assert obs is not None
    assert "agent_0" in env.agents
    assert env.current_step == 0


def test_env_action_space():
    """Test that action space is defined correctly."""
    env = HamletEnv()
    # Actions: 0=up, 1=down, 2=left, 3=right, 4=interact
    assert env.num_actions == 5


def test_env_step_movement():
    """Test that agent can move."""
    env = HamletEnv()
    env.reset()

    agent = env.agents["agent_0"]
    initial_x, initial_y = agent.x, agent.y

    # Move right
    obs, reward, done, info = env.step(3)  # Action 3 = right

    assert agent.x == initial_x + 1 or agent.x == initial_x  # Moved or at edge
    assert env.current_step == 1


def test_env_step_depletes_meters():
    """Test that stepping depletes meters."""
    env = HamletEnv()
    env.reset()

    agent = env.agents["agent_0"]
    initial_energy = agent.meters.get("energy").value

    # Take a step (movement depletes meters)
    env.step(0)  # Move up

    assert agent.meters.get("energy").value < initial_energy


def test_env_interact_action():
    """Test interact action with affordance."""
    env = HamletEnv()
    env.reset()

    agent = env.agents["agent_0"]

    # Find a job affordance
    job_pos = None
    for affordance in env.affordances:
        if affordance.name == "Job":
            job_pos = (affordance.x, affordance.y)
            break

    # Move agent to job position
    agent.x, agent.y = job_pos
    env.grid.add_entity(agent, agent.x, agent.y)

    initial_money = agent.meters.get("money").value

    # Interact with job
    obs, reward, done, info = env.step(4)  # Action 4 = interact

    # Should have earned money
    assert agent.meters.get("money").value > initial_money


def test_env_observe():
    """Test observation generation."""
    env = HamletEnv()
    env.reset()

    obs = env.observe("agent_0")

    assert "position" in obs
    assert "meters" in obs
    assert "grid" in obs


def test_env_reward_calculation():
    """Test that rewards are calculated."""
    env = HamletEnv()
    env.reset()

    obs, reward, done, info = env.step(0)

    assert reward is not None
    assert isinstance(reward, (int, float))


def test_env_termination_meter_zero():
    """Test that episode ends when meter hits zero."""
    env = HamletEnv()
    env.reset()

    agent = env.agents["agent_0"]

    # Force a meter to zero
    agent.meters.get("energy").value = 0.0

    obs, reward, done, info = env.step(0)

    assert done is True
    assert reward < 0  # Death penalty


def test_env_survival_reward():
    """Test that agent gets reward for surviving."""
    env = HamletEnv()
    env.reset()

    agent = env.agents["agent_0"]

    # Keep meters healthy
    for meter_name in ["energy", "hygiene", "satiation"]:
        agent.meters.get(meter_name).value = 80.0

    obs, reward, done, info = env.step(0)

    assert reward > 0  # Survival bonus


def test_env_multiple_steps():
    """Test running multiple steps."""
    env = HamletEnv()
    env.reset()

    for i in range(10):
        obs, reward, done, info = env.step(i % 5)  # Cycle through actions
        if done:
            break

    assert env.current_step > 0


def test_env_render():
    """Test rendering returns state dict."""
    env = HamletEnv()
    env.reset()

    state = env.render()

    assert state is not None
    assert "agents" in state or "grid" in state


def test_env_with_config():
    """Test environment with custom configuration."""
    config = EnvironmentConfig(grid_width=10, grid_height=10)
    env = HamletEnv(config=config)
    env.reset()

    assert env.grid.width == 10
    assert env.grid.height == 10


def test_agent_cant_afford_service_no_termination():
    """Test that being broke doesn't immediately end episode."""
    env = HamletEnv()
    env.reset()

    agent = env.agents["agent_0"]
    agent.meters.get("money").value = 0.0

    obs, reward, done, info = env.step(0)

    # Should not terminate just for being broke
    # Only terminates if broke AND biological meters are critical
    assert done is False or agent.meters.is_any_critical()


def test_critical_meters_give_penalty():
    """Test that critical meters result in negative rewards."""
    env = HamletEnv()
    env.reset()

    agent = env.agents["agent_0"]

    # Make energy critical
    agent.meters.get("energy").value = 10.0

    obs, reward, done, info = env.step(0)

    # Should get penalty for critical meter
    assert reward < 1.0  # Less than survival bonus


def test_full_episode():
    """Test running a full episode until termination."""
    env = HamletEnv()
    env.reset()

    steps = 0
    max_steps = 1000

    while steps < max_steps:
        action = steps % 5  # Simple policy
        obs, reward, done, info = env.step(action)
        steps += 1

        if done:
            break

    # Should eventually terminate
    assert steps < max_steps or env.agents["agent_0"].meters.is_any_critical()
