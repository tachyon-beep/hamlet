"""
Integration tests for DRL Agent.
"""

import pytest
import torch
import numpy as np
import tempfile
import os
from hamlet.agent.drl_agent import DRLAgent
from hamlet.agent.replay_buffer import ReplayBuffer
from hamlet.environment.hamlet_env import HamletEnv


def test_drl_agent_initialization():
    """Test that DRL agent initializes correctly."""
    agent = DRLAgent("test_agent", state_dim=70, action_dim=5)

    assert agent.agent_id == "test_agent"
    assert agent.state_dim == 70
    assert agent.action_dim == 5
    assert agent.q_network is not None
    assert agent.target_network is not None
    assert agent.optimizer is not None
    assert agent.device is not None


def test_drl_agent_networks_initialized():
    """Test that both networks are properly initialized."""
    agent = DRLAgent("test_agent", state_dim=10, action_dim=3)

    # Check networks have parameters
    assert list(agent.q_network.parameters())
    assert list(agent.target_network.parameters())

    # Check target network initially has same weights as Q-network
    for q_param, target_param in zip(agent.q_network.parameters(), agent.target_network.parameters()):
        assert torch.allclose(q_param, target_param)


def test_drl_agent_select_action_from_dict_observation():
    """Test action selection from dict observation."""
    agent = DRLAgent("test_agent", state_dim=70, action_dim=5, device="cpu")

    # Create dict observation matching HamletEnv format
    obs = {
        "position": np.array([4.0, 3.0], dtype=np.float32),
        "meters": {
            "energy": 0.8,
            "hygiene": 0.6,
            "satiation": 0.9,
            "money": 0.5,
        },
        "grid": np.random.rand(8, 8).astype(np.float32),
    }

    action = agent.select_action(obs)

    assert isinstance(action, int)
    assert 0 <= action < 5


def test_drl_agent_select_action_from_array():
    """Test action selection from preprocessed array."""
    agent = DRLAgent("test_agent", state_dim=70, action_dim=5, device="cpu")

    state = np.random.randn(70).astype(np.float32)
    action = agent.select_action(state)

    assert isinstance(action, int)
    assert 0 <= action < 5


def test_drl_agent_epsilon_greedy_exploration():
    """Test that epsilon controls exploration."""
    agent = DRLAgent("test_agent", state_dim=10, action_dim=3, epsilon=1.0, device="cpu")

    state = np.zeros(10, dtype=np.float32)

    # With epsilon=1.0, should always explore (random actions)
    actions = [agent.select_action(state, explore=True) for _ in range(20)]

    # Should have some variety in actions
    assert len(set(actions)) > 1


def test_drl_agent_greedy_no_exploration():
    """Test greedy action selection without exploration."""
    agent = DRLAgent("test_agent", state_dim=10, action_dim=3, device="cpu")

    state = np.ones(10, dtype=np.float32)

    # With explore=False, should always return same action
    actions = [agent.select_action(state, explore=False) for _ in range(10)]

    # All actions should be the same (greedy)
    assert len(set(actions)) == 1


def test_drl_agent_learn_from_batch():
    """Test learning from a batch of experiences."""
    agent = DRLAgent("test_agent", state_dim=10, action_dim=3, device="cpu")

    # Create a simple batch
    batch_size = 4
    states = [np.random.randn(10).astype(np.float32) for _ in range(batch_size)]
    actions = [np.random.randint(3) for _ in range(batch_size)]
    rewards = [np.random.randn() for _ in range(batch_size)]
    next_states = [np.random.randn(10).astype(np.float32) for _ in range(batch_size)]
    dones = [False, False, True, False]

    batch = (states, actions, rewards, next_states, dones)

    # Get initial Q-values
    initial_q_values = agent.q_network(torch.FloatTensor(states[0])).detach().clone()

    # Learn from batch
    loss = agent.learn(batch)

    assert isinstance(loss, float)
    assert loss >= 0

    # Q-values should have changed after learning
    final_q_values = agent.q_network(torch.FloatTensor(states[0]))
    assert not torch.allclose(initial_q_values, final_q_values)


def test_drl_agent_update_target_network():
    """Test updating target network."""
    agent = DRLAgent("test_agent", state_dim=10, action_dim=3, device="cpu")

    # Modify Q-network weights
    with torch.no_grad():
        for param in agent.q_network.parameters():
            param.add_(torch.randn_like(param) * 0.1)

    # Networks should be different now
    q_params = list(agent.q_network.parameters())[0]
    target_params = list(agent.target_network.parameters())[0]
    assert not torch.allclose(q_params, target_params)

    # Update target network
    agent.update_target_network()

    # Now they should be the same
    for q_param, target_param in zip(agent.q_network.parameters(), agent.target_network.parameters()):
        assert torch.allclose(q_param, target_param)


def test_drl_agent_epsilon_decay():
    """Test epsilon decay."""
    agent = DRLAgent(
        "test_agent",
        state_dim=10,
        action_dim=3,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.95,
        device="cpu"
    )

    assert agent.epsilon == 1.0

    agent.decay_epsilon()
    assert agent.epsilon == pytest.approx(0.95)

    agent.decay_epsilon()
    assert agent.epsilon == pytest.approx(0.9025)

    # Decay many times
    for _ in range(100):
        agent.decay_epsilon()

    # Should not go below epsilon_min
    assert agent.epsilon >= 0.01


def test_drl_agent_save_and_load():
    """Test saving and loading agent."""
    agent1 = DRLAgent("test_agent", state_dim=10, action_dim=3, device="cpu")

    # Train a bit to modify weights
    state = np.random.randn(10).astype(np.float32)
    for _ in range(5):
        batch = (
            [state],
            [1],
            [1.0],
            [state],
            [False]
        )
        agent1.learn(batch)

    agent1.decay_epsilon()
    saved_epsilon = agent1.epsilon

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as f:
        filepath = f.name

    try:
        agent1.save(filepath)

        # Create new agent and load
        agent2 = DRLAgent("test_agent", state_dim=10, action_dim=3, device="cpu")
        agent2.load(filepath)

        # Check epsilon was loaded
        assert agent2.epsilon == pytest.approx(saved_epsilon)

        # Check networks produce same output
        test_state = torch.FloatTensor(state)
        with torch.no_grad():
            q1 = agent1.q_network(test_state)
            q2 = agent2.q_network(test_state)
            assert torch.allclose(q1, q2)

    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


def test_drl_agent_with_hamlet_env():
    """Test agent can interact with HamletEnv."""
    env = HamletEnv()
    obs = env.reset()

    agent = DRLAgent("test_agent", state_dim=70, action_dim=5, device="cpu")

    # Agent should be able to select action
    action = agent.select_action(obs)
    assert isinstance(action, int)
    assert 0 <= action < 5

    # Execute action in environment
    next_obs, reward, done, info = env.step(action)

    # Agent should be able to select action from next_obs
    next_action = agent.select_action(next_obs)
    assert isinstance(next_action, int)


def test_drl_agent_full_episode_with_learning():
    """Test agent can run full episode and learn."""
    env = HamletEnv()
    agent = DRLAgent("test_agent", state_dim=70, action_dim=5, device="cpu")
    buffer = ReplayBuffer(capacity=1000)

    obs = env.reset()

    # Collect some experiences
    for step in range(50):
        action = agent.select_action(obs)
        next_obs, reward, done, info = env.step(action)

        # Store in buffer
        state = np.concatenate([
            obs["position"] / 8,
            [obs["meters"]["energy"], obs["meters"]["hygiene"],
             obs["meters"]["satiation"], obs["meters"]["money"]],
            obs["grid"].flatten()
        ]).astype(np.float32)

        next_state = np.concatenate([
            next_obs["position"] / 8,
            [next_obs["meters"]["energy"], next_obs["meters"]["hygiene"],
             next_obs["meters"]["satiation"], next_obs["meters"]["money"]],
            next_obs["grid"].flatten()
        ]).astype(np.float32)

        buffer.push(state, action, reward, next_state, done)

        if done:
            break

        obs = next_obs

    # Learn from buffer
    if buffer.is_ready(32):
        batch = buffer.sample(32)
        loss = agent.learn(batch)
        assert isinstance(loss, float)


def test_drl_agent_different_devices():
    """Test agent works on different devices."""
    # CPU
    agent_cpu = DRLAgent("test_agent", state_dim=10, action_dim=3, device="cpu")
    assert agent_cpu.device.type == "cpu"

    state = np.random.randn(10).astype(np.float32)
    action = agent_cpu.select_action(state)
    assert isinstance(action, int)


def test_drl_agent_learns_reduces_loss():
    """Test that repeated learning reduces loss."""
    agent = DRLAgent("test_agent", state_dim=10, action_dim=3, device="cpu", learning_rate=1e-2)

    # Create fixed batch
    state = np.ones(10, dtype=np.float32)
    batch = (
        [state] * 32,
        [1] * 32,
        [1.0] * 32,
        [state] * 32,
        [False] * 32
    )

    # Learn multiple times
    losses = []
    for _ in range(20):
        loss = agent.learn(batch)
        losses.append(loss)

    # Loss should generally decrease (though not strictly monotonic due to target network)
    avg_first_5 = np.mean(losses[:5])
    avg_last_5 = np.mean(losses[-5:])
    assert avg_last_5 < avg_first_5
