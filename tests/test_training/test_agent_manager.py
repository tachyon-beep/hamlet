"""
Tests for AgentManager.
"""

import pytest
import numpy as np
from hamlet.training.agent_manager import AgentManager
from hamlet.training.config import AgentConfig


def test_agent_manager_initialization():
    """Test that agent manager initializes correctly."""
    manager = AgentManager(buffer_size=1000, buffer_threshold=10)

    assert manager.buffer_size == 1000
    assert manager.buffer_threshold == 10
    assert manager.buffer_mode == "per_agent"
    assert manager.num_agents() == 0
    assert manager.shared_buffer is None
    assert len(manager.per_agent_buffers) == 0


def test_add_single_agent():
    """Test adding a single agent."""
    manager = AgentManager()
    config = AgentConfig(agent_id="agent_0", algorithm="dqn")

    agent = manager.add_agent(config)

    assert agent is not None
    assert agent.agent_id == "agent_0"
    assert manager.num_agents() == 1
    assert "agent_0" in manager.get_agent_ids()
    assert manager.buffer_mode == "per_agent"


def test_add_multiple_agents():
    """Test adding multiple agents."""
    manager = AgentManager()

    for i in range(5):
        config = AgentConfig(agent_id=f"agent_{i}", algorithm="dqn")
        manager.add_agent(config)

    assert manager.num_agents() == 5
    assert manager.buffer_mode == "per_agent"
    assert len(manager.get_agent_ids()) == 5


def test_buffer_mode_switches_to_shared():
    """Test that buffer mode switches to shared when threshold exceeded."""
    manager = AgentManager(buffer_threshold=3)

    # Add 2 agents - should stay per_agent
    for i in range(2):
        config = AgentConfig(agent_id=f"agent_{i}", algorithm="dqn")
        manager.add_agent(config)

    assert manager.buffer_mode == "per_agent"

    # Add 3rd agent - should switch to shared
    config = AgentConfig(agent_id="agent_2", algorithm="dqn")
    manager.add_agent(config)

    assert manager.buffer_mode == "shared"
    assert manager.shared_buffer is not None


def test_buffer_mode_switches_to_per_agent():
    """Test that buffer mode switches back to per_agent when below threshold."""
    manager = AgentManager(buffer_threshold=3)

    # Add 4 agents - mode should be shared
    for i in range(4):
        config = AgentConfig(agent_id=f"agent_{i}", algorithm="dqn")
        manager.add_agent(config)

    assert manager.buffer_mode == "shared"

    # Remove 2 agents - should switch to per_agent
    manager.remove_agent("agent_2")
    manager.remove_agent("agent_3")

    assert manager.buffer_mode == "per_agent"
    assert manager.num_agents() == 2


def test_get_agent():
    """Test getting specific agent."""
    manager = AgentManager()
    config = AgentConfig(agent_id="test_agent", algorithm="dqn")
    added_agent = manager.add_agent(config)

    retrieved_agent = manager.get_agent("test_agent")

    assert retrieved_agent is added_agent
    assert retrieved_agent.agent_id == "test_agent"


def test_get_agent_not_found():
    """Test getting non-existent agent raises error."""
    manager = AgentManager()

    with pytest.raises(ValueError, match="Agent unknown not found"):
        manager.get_agent("unknown")


def test_remove_agent():
    """Test removing agent."""
    manager = AgentManager()
    config = AgentConfig(agent_id="remove_me", algorithm="dqn")
    manager.add_agent(config)

    assert manager.num_agents() == 1

    manager.remove_agent("remove_me")

    assert manager.num_agents() == 0
    assert "remove_me" not in manager.get_agent_ids()


def test_remove_agent_not_found():
    """Test removing non-existent agent raises error."""
    manager = AgentManager()

    with pytest.raises(ValueError, match="Agent unknown not found"):
        manager.remove_agent("unknown")


def test_store_experience_per_agent_mode():
    """Test storing experience in per-agent mode."""
    manager = AgentManager()
    config = AgentConfig(agent_id="agent_0", algorithm="dqn", state_dim=4, action_dim=2)
    manager.add_agent(config)

    state = np.array([1.0, 2.0, 3.0, 4.0])
    next_state = np.array([1.1, 2.1, 3.1, 4.1])

    manager.store_experience("agent_0", state, 0, 1.0, next_state, False)

    assert "agent_0" in manager.per_agent_buffers
    assert len(manager.per_agent_buffers["agent_0"]) == 1


def test_store_experience_shared_mode():
    """Test storing experience in shared mode."""
    manager = AgentManager(buffer_threshold=2)

    # Add 2 agents to trigger shared mode
    for i in range(2):
        config = AgentConfig(agent_id=f"agent_{i}", algorithm="dqn", state_dim=4)
        manager.add_agent(config)

    assert manager.buffer_mode == "shared"

    state = np.array([1.0, 2.0, 3.0, 4.0])
    next_state = np.array([1.1, 2.1, 3.1, 4.1])

    manager.store_experience("agent_0", state, 0, 1.0, next_state, False)

    assert manager.shared_buffer is not None
    assert len(manager.shared_buffer) == 1


def test_sample_batch_per_agent_mode():
    """Test sampling batch in per-agent mode."""
    manager = AgentManager()
    config = AgentConfig(agent_id="agent_0", algorithm="dqn", state_dim=4)
    manager.add_agent(config)

    # Store multiple experiences
    for i in range(10):
        state = np.array([float(i)] * 4)
        next_state = np.array([float(i + 1)] * 4)
        manager.store_experience("agent_0", state, 0, 1.0, next_state, False)

    batch = manager.sample_batch(5, agent_id="agent_0")

    assert batch is not None
    states, actions, rewards, next_states, dones = batch
    assert len(states) == 5
    assert len(actions) == 5


def test_sample_batch_shared_mode():
    """Test sampling batch in shared mode."""
    manager = AgentManager(buffer_threshold=2)

    # Add 2 agents to trigger shared mode
    for i in range(2):
        config = AgentConfig(agent_id=f"agent_{i}", algorithm="dqn", state_dim=4)
        manager.add_agent(config)

    # Store experiences
    for i in range(10):
        state = np.array([float(i)] * 4)
        next_state = np.array([float(i + 1)] * 4)
        manager.store_experience("agent_0", state, 0, 1.0, next_state, False)

    batch = manager.sample_batch(5)

    assert batch is not None
    states, actions, rewards, next_states, dones = batch
    assert len(states) == 5


def test_sample_batch_insufficient_experiences():
    """Test sampling returns None when insufficient experiences."""
    manager = AgentManager()
    config = AgentConfig(agent_id="agent_0", algorithm="dqn", state_dim=4)
    manager.add_agent(config)

    # Store only 3 experiences
    for i in range(3):
        state = np.array([float(i)] * 4)
        next_state = np.array([float(i + 1)] * 4)
        manager.store_experience("agent_0", state, 0, 1.0, next_state, False)

    # Try to sample 5
    batch = manager.sample_batch(5, agent_id="agent_0")

    assert batch is None


def test_can_sample():
    """Test can_sample check."""
    manager = AgentManager()
    config = AgentConfig(agent_id="agent_0", algorithm="dqn", state_dim=4)
    manager.add_agent(config)

    # Initially can't sample
    assert not manager.can_sample(5, agent_id="agent_0")

    # Store experiences
    for i in range(10):
        state = np.array([float(i)] * 4)
        next_state = np.array([float(i + 1)] * 4)
        manager.store_experience("agent_0", state, 0, 1.0, next_state, False)

    # Now can sample
    assert manager.can_sample(5, agent_id="agent_0")
    assert not manager.can_sample(15, agent_id="agent_0")


def test_get_buffer_info_per_agent():
    """Test getting buffer info in per-agent mode."""
    manager = AgentManager()
    config = AgentConfig(agent_id="agent_0", algorithm="dqn", state_dim=4)
    manager.add_agent(config)

    # Store some experiences
    for i in range(5):
        state = np.array([float(i)] * 4)
        next_state = np.array([float(i + 1)] * 4)
        manager.store_experience("agent_0", state, 0, 1.0, next_state, False)

    info = manager.get_buffer_info()

    assert info["mode"] == "per_agent"
    assert info["num_agents"] == 1
    assert "per_agent_buffers" in info
    assert info["per_agent_buffers"]["agent_0"] == 5


def test_get_buffer_info_shared():
    """Test getting buffer info in shared mode."""
    manager = AgentManager(buffer_threshold=2)

    # Add 2 agents for shared mode
    for i in range(2):
        config = AgentConfig(agent_id=f"agent_{i}", algorithm="dqn", state_dim=4)
        manager.add_agent(config)

    # Store experiences
    for i in range(10):
        state = np.array([float(i)] * 4)
        next_state = np.array([float(i + 1)] * 4)
        manager.store_experience("agent_0", state, 0, 1.0, next_state, False)

    info = manager.get_buffer_info()

    assert info["mode"] == "shared"
    assert info["num_agents"] == 2
    assert "shared_buffer_size" in info
    assert info["shared_buffer_size"] == 10


def test_migrate_to_shared():
    """Test migration from per-agent to shared buffers."""
    manager = AgentManager(buffer_threshold=3)

    # Add 2 agents in per-agent mode
    for i in range(2):
        config = AgentConfig(agent_id=f"agent_{i}", algorithm="dqn", state_dim=4)
        manager.add_agent(config)

    # Store experiences for both agents
    for agent_idx in range(2):
        for i in range(5):
            state = np.array([float(i)] * 4)
            next_state = np.array([float(i + 1)] * 4)
            manager.store_experience(f"agent_{agent_idx}", state, 0, 1.0, next_state, False)

    assert manager.buffer_mode == "per_agent"
    assert len(manager.per_agent_buffers) == 2

    # Add 3rd agent - should trigger migration to shared
    config = AgentConfig(agent_id="agent_2", algorithm="dqn", state_dim=4)
    manager.add_agent(config)

    assert manager.buffer_mode == "shared"
    assert manager.shared_buffer is not None
    assert len(manager.shared_buffer) == 10  # 5 from each of 2 agents
    assert len(manager.per_agent_buffers) == 0


def test_migrate_to_per_agent():
    """Test migration from shared to per-agent buffers."""
    manager = AgentManager(buffer_threshold=3)

    # Add 3 agents to trigger shared mode
    for i in range(3):
        config = AgentConfig(agent_id=f"agent_{i}", algorithm="dqn", state_dim=4)
        manager.add_agent(config)

    assert manager.buffer_mode == "shared"

    # Store experiences in shared buffer
    for i in range(12):
        state = np.array([float(i)] * 4)
        next_state = np.array([float(i + 1)] * 4)
        manager.store_experience("agent_0", state, 0, 1.0, next_state, False)

    assert len(manager.shared_buffer) == 12

    # Remove agent to drop below threshold
    manager.remove_agent("agent_2")

    assert manager.buffer_mode == "per_agent"
    assert manager.shared_buffer is None
    assert len(manager.per_agent_buffers) == 2
    # Experiences should be distributed (12 / 2 = 6 each)
    assert len(manager.per_agent_buffers["agent_0"]) == 6
    assert len(manager.per_agent_buffers["agent_1"]) == 6


def test_get_all_agents():
    """Test getting all agents."""
    manager = AgentManager()

    # Add multiple agents
    for i in range(3):
        config = AgentConfig(agent_id=f"agent_{i}", algorithm="dqn")
        manager.add_agent(config)

    all_agents = manager.get_all_agents()

    assert len(all_agents) == 3
    agent_ids = [agent.agent_id for agent in all_agents]
    assert "agent_0" in agent_ids
    assert "agent_1" in agent_ids
    assert "agent_2" in agent_ids


def test_unknown_algorithm():
    """Test that unknown algorithm raises error."""
    manager = AgentManager()
    config = AgentConfig(agent_id="agent_0", algorithm="unknown_algo")

    with pytest.raises(ValueError, match="Unknown algorithm"):
        manager.add_agent(config)
