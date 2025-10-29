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
    assert info.get("failure_reason") is None


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
    assert info.get("failure_reason") == "energy_depleted"


def test_env_survival_reward():
    """Test that agent gets reward for surviving."""
    env = HamletEnv()
    env.reset()

    agent = env.agents["agent_0"]

    # Keep meters healthy
    for meter_name in ["energy", "hygiene", "satiation", "mood", "social"]:
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


def test_env_initial_meter_config_respected():
    """Initial meter values and bounds honour EnvironmentConfig."""
    config = EnvironmentConfig(
        initial_energy=75.0,
        initial_hygiene=55.0,
        initial_satiation=65.0,
        initial_money=12.0,
        initial_mood=90.0,
        initial_social=30.0,
        money_min=-25.0,
    )

    env = HamletEnv(config=config)
    env.reset()

    agent = env.agents["agent_0"]
    assert agent.meters.get("energy").value == pytest.approx(75.0)
    assert agent.meters.get("hygiene").value == pytest.approx(55.0)
    assert agent.meters.get("satiation").value == pytest.approx(65.0)
    assert agent.meters.get("money").value == pytest.approx(12.0)
    assert agent.meters.get("mood").value == pytest.approx(90.0)
    assert agent.meters.get("social").value == pytest.approx(30.0)
    assert agent.meters.get("money").min_value == pytest.approx(-25.0)


def test_env_meter_depletion_config_respected():
    """Meter depletion rates can be configured from EnvironmentConfig."""
    config = EnvironmentConfig(
        energy_depletion=1.5,
        hygiene_depletion=0.8,
        satiation_depletion=1.2,
        mood_depletion=0.4,
        social_depletion=0.9,
        money_depletion=0.2,
    )

    env = HamletEnv(config=config)
    env.reset()

    agent = env.agents["agent_0"]
    assert agent.meters.get("energy").depletion_rate == pytest.approx(1.5)
    assert agent.meters.get("hygiene").depletion_rate == pytest.approx(0.8)
    assert agent.meters.get("satiation").depletion_rate == pytest.approx(1.2)
    assert agent.meters.get("mood").depletion_rate == pytest.approx(0.4)
    assert agent.meters.get("social").depletion_rate == pytest.approx(0.9)
    assert agent.meters.get("money").depletion_rate == pytest.approx(0.2)


def test_mood_penalty_increases_when_social_low():
    """Loneliness should amplify mood decline."""
    env = HamletEnv()

    # Baseline mood drop with high social
    env.reset()
    agent = env.agents["agent_0"]
    agent.meters.get("mood").value = 100.0
    agent.meters.get("social").value = 100.0
    env.step(HamletEnv.ACTION_RIGHT)
    baseline_mood = agent.meters.get("mood").value

    # Mood drop with low social should be larger
    env.reset()
    agent = env.agents["agent_0"]
    agent.meters.get("mood").value = 100.0
    agent.meters.get("social").value = 0.0
    env.step(HamletEnv.ACTION_RIGHT)
    lonely_mood = agent.meters.get("mood").value

    assert lonely_mood < baseline_mood


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
