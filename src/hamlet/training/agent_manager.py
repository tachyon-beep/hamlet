"""
Agent management for training.

Handles multiple agents with automatic buffer mode switching:
- <10 agents: Per-agent buffers for learning isolation
- 10+ agents: Shared buffer for memory efficiency
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

from hamlet.agent.base_algorithm import BaseAlgorithm
from hamlet.agent.drl_agent import DRLAgent
from hamlet.agent.replay_buffer import ReplayBuffer
from hamlet.agent.observation_utils import get_state_dim
from hamlet.training.config import AgentConfig


class AgentManager:
    """
    Manages multiple agents with intelligent buffer management.

    Automatically switches between per-agent and shared buffer modes
    based on agent count for optimal performance.
    """

    def __init__(
        self,
        buffer_size: int = 10000,
        buffer_threshold: int = 10,
    ):
        """
        Initialize agent manager.

        Args:
            buffer_size: Size of replay buffer(s)
            buffer_threshold: Agent count threshold for shared buffer (default 10)
        """
        self.buffer_size = buffer_size
        self.buffer_threshold = buffer_threshold

        # Agent storage
        self.agents: Dict[str, BaseAlgorithm] = {}

        # Buffer management
        self.shared_buffer: Optional[ReplayBuffer] = None
        self.per_agent_buffers: Dict[str, ReplayBuffer] = {}
        self.buffer_mode: str = "per_agent"  # "per_agent" or "shared"

    def add_agent(self, config: AgentConfig) -> BaseAlgorithm:
        """
        Add agent from configuration.

        Args:
            config: Agent configuration

        Returns:
            Created agent instance
        """
        # Create agent based on algorithm type
        if config.algorithm.lower() == "dqn":
            expected_state_dim = get_state_dim(config.grid_size)
            state_dim = config.state_dim
            if state_dim != expected_state_dim:
                state_dim = expected_state_dim
            agent = DRLAgent(
                agent_id=config.agent_id,
                state_dim=state_dim,
                action_dim=config.action_dim,
                learning_rate=config.learning_rate,
                gamma=config.gamma,
                epsilon=config.epsilon,
                epsilon_min=config.epsilon_min,
                epsilon_decay=config.epsilon_decay,
                device=config.device if config.device != "auto" else None,
                network_type=config.network_type,
                grid_size=config.grid_size,
            )
        else:
            raise ValueError(f"Unknown algorithm: {config.algorithm}")

        # Store agent
        self.agents[config.agent_id] = agent

        # Update buffer mode if needed
        self._update_buffer_mode()

        return agent

    def remove_agent(self, agent_id: str):
        """
        Remove agent and clean up buffers.

        Args:
            agent_id: ID of agent to remove
        """
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found")

        # Remove agent
        del self.agents[agent_id]

        # Remove per-agent buffer if exists
        if agent_id in self.per_agent_buffers:
            del self.per_agent_buffers[agent_id]

        # Update buffer mode if needed
        self._update_buffer_mode()

    def get_agent(self, agent_id: str) -> BaseAlgorithm:
        """
        Get agent by ID.

        Args:
            agent_id: Agent identifier

        Returns:
            Agent instance
        """
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found")
        return self.agents[agent_id]

    def get_all_agents(self) -> List[BaseAlgorithm]:
        """Get all agents."""
        return list(self.agents.values())

    def get_agent_ids(self) -> List[str]:
        """Get all agent IDs."""
        return list(self.agents.keys())

    def num_agents(self) -> int:
        """Get number of agents."""
        return len(self.agents)

    def store_experience(
        self,
        agent_id: str,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """
        Store experience in appropriate buffer.

        Args:
            agent_id: Agent that generated experience
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Episode termination flag
        """
        if self.buffer_mode == "shared":
            if self.shared_buffer is None:
                self.shared_buffer = ReplayBuffer(self.buffer_size)
            self.shared_buffer.push(state, action, reward, next_state, done)
        else:
            if agent_id not in self.per_agent_buffers:
                self.per_agent_buffers[agent_id] = ReplayBuffer(self.buffer_size)
            self.per_agent_buffers[agent_id].push(state, action, reward, next_state, done)

    def sample_batch(
        self,
        batch_size: int,
        agent_id: Optional[str] = None,
    ) -> Tuple[List, List, List, List, List]:
        """
        Sample batch from appropriate buffer(s).

        Args:
            batch_size: Number of experiences to sample
            agent_id: Specific agent to sample for (optional)

        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        if self.buffer_mode == "shared":
            if self.shared_buffer is None or len(self.shared_buffer) < batch_size:
                return None
            return self.shared_buffer.sample(batch_size)
        else:
            # Per-agent mode
            if agent_id is None:
                raise ValueError("agent_id required in per_agent buffer mode")

            if agent_id not in self.per_agent_buffers:
                return None

            buffer = self.per_agent_buffers[agent_id]
            if len(buffer) < batch_size:
                return None

            return buffer.sample(batch_size)

    def can_sample(self, batch_size: int, agent_id: Optional[str] = None) -> bool:
        """
        Check if sufficient experiences available for sampling.

        Args:
            batch_size: Required batch size
            agent_id: Specific agent (optional)

        Returns:
            True if can sample, False otherwise
        """
        if self.buffer_mode == "shared":
            return self.shared_buffer is not None and len(self.shared_buffer) >= batch_size
        else:
            if agent_id is None:
                return False
            if agent_id not in self.per_agent_buffers:
                return False
            return len(self.per_agent_buffers[agent_id]) >= batch_size

    def get_buffer_info(self) -> dict:
        """
        Get buffer statistics.

        Returns:
            Dictionary with buffer mode and sizes
        """
        info = {
            "mode": self.buffer_mode,
            "num_agents": self.num_agents(),
        }

        if self.buffer_mode == "shared":
            info["shared_buffer_size"] = (
                len(self.shared_buffer) if self.shared_buffer else 0
            )
        else:
            info["per_agent_buffers"] = {
                agent_id: len(buffer)
                for agent_id, buffer in self.per_agent_buffers.items()
            }

        return info

    def _update_buffer_mode(self):
        """
        Update buffer mode based on agent count.

        Switches to shared buffer when agent count exceeds threshold.
        """
        new_mode = (
            "shared" if self.num_agents() >= self.buffer_threshold else "per_agent"
        )

        if new_mode != self.buffer_mode:
            # Mode switch - migrate experiences if needed
            if new_mode == "shared":
                self._migrate_to_shared()
            else:
                self._migrate_to_per_agent()

            self.buffer_mode = new_mode

    def _migrate_to_shared(self):
        """Migrate per-agent buffers to shared buffer."""
        # Create shared buffer
        self.shared_buffer = ReplayBuffer(self.buffer_size)

        if not self.per_agent_buffers:
            return

        # Migrate all experiences from per-agent buffers
        for agent_id, buffer in self.per_agent_buffers.items():
            # Get all experiences from buffer
            if len(buffer) > 0:
                for exp in buffer.buffer:
                    self.shared_buffer.push(
                        exp.state, exp.action, exp.reward, exp.next_state, exp.done
                    )

        # Clear per-agent buffers
        self.per_agent_buffers.clear()

    def _migrate_to_per_agent(self):
        """Migrate shared buffer to per-agent buffers."""
        if self.shared_buffer is None or len(self.shared_buffer) == 0:
            return

        # Can't directly attribute experiences to agents, so we distribute evenly
        num_agents = self.num_agents()
        experiences_per_agent = len(self.shared_buffer) // num_agents

        agent_ids = list(self.agents.keys())

        # Create per-agent buffers and distribute experiences
        idx = 0
        for agent_id in agent_ids:
            self.per_agent_buffers[agent_id] = ReplayBuffer(self.buffer_size)

            # Distribute experiences evenly
            for _ in range(experiences_per_agent):
                if idx < len(self.shared_buffer):
                    exp = self.shared_buffer.buffer[idx]
                    self.per_agent_buffers[agent_id].push(
                        exp.state, exp.action, exp.reward, exp.next_state, exp.done
                    )
                    idx += 1

        # Clear shared buffer
        self.shared_buffer = None
