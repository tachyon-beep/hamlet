"""
Vectorized population manager.

Coordinates multiple agents with shared curriculum and exploration strategies.
Manages Q-networks, replay buffers, and training loops.
"""

from typing import List, TYPE_CHECKING
import torch
import torch.nn as nn

from townlet.population.base import PopulationManager
from townlet.training.state import BatchedAgentState, PopulationCheckpoint
from townlet.curriculum.base import CurriculumManager
from townlet.exploration.base import ExplorationStrategy

if TYPE_CHECKING:
    from townlet.environment.vectorized_env import VectorizedHamletEnv


class SimpleQNetwork(nn.Module):
    """Simple MLP Q-network for Phase 1."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VectorizedPopulation(PopulationManager):
    """
    Vectorized population manager.

    Coordinates training for num_agents parallel agents with shared
    curriculum and exploration strategies.
    """

    def __init__(
        self,
        env: 'VectorizedHamletEnv',
        curriculum: CurriculumManager,
        exploration: ExplorationStrategy,
        agent_ids: List[str],
        device: torch.device,
        obs_dim: int = 70,
        action_dim: int = 5,
    ):
        """
        Initialize vectorized population.

        Args:
            env: Vectorized environment
            curriculum: Curriculum manager
            exploration: Exploration strategy
            agent_ids: List of agent identifiers
            device: PyTorch device
            obs_dim: Observation dimension
            action_dim: Action dimension
        """
        self.env = env
        self.curriculum = curriculum
        self.exploration = exploration
        self.agent_ids = agent_ids
        self.num_agents = len(agent_ids)
        self.device = device

        # Q-network (shared across all agents for now)
        self.q_network = SimpleQNetwork(obs_dim, action_dim).to(device)

        # Current state
        self.current_obs: torch.Tensor = None
        self.current_epsilons: torch.Tensor = None

    def reset(self) -> None:
        """Reset all environments and state."""
        self.current_obs = self.env.reset()
        self.current_epsilons = torch.full(
            (self.num_agents,), self.exploration.epsilon, device=self.device
        )

    def step_population(
        self,
        envs: 'VectorizedHamletEnv',
    ) -> BatchedAgentState:
        """
        Execute one training step for entire population.

        Args:
            envs: Vectorized environment (same as self.env)

        Returns:
            BatchedAgentState with all agent data after step
        """
        # 1. Get Q-values from network
        with torch.no_grad():
            q_values = self.q_network(self.current_obs)

        # 2. Create temporary agent state for action selection
        temp_state = BatchedAgentState(
            observations=self.current_obs,
            actions=torch.zeros(self.num_agents, dtype=torch.long, device=self.device),
            rewards=torch.zeros(self.num_agents, device=self.device),
            dones=torch.zeros(self.num_agents, dtype=torch.bool, device=self.device),
            epsilons=self.current_epsilons,
            intrinsic_rewards=torch.zeros(self.num_agents, device=self.device),
            survival_times=envs.step_counts.clone(),
            curriculum_difficulties=torch.zeros(self.num_agents, device=self.device),
            device=self.device,
        )

        # 3. Select actions via exploration strategy
        actions = self.exploration.select_actions(q_values, temp_state)

        # 4. Step environment
        next_obs, rewards, dones, info = envs.step(actions)

        # 5. Compute intrinsic rewards
        intrinsic_rewards = self.exploration.compute_intrinsic_rewards(next_obs)

        # 6. Update current state
        self.current_obs = next_obs

        # 7. Construct BatchedAgentState
        state = BatchedAgentState(
            observations=next_obs,
            actions=actions,
            rewards=rewards,
            dones=dones,
            epsilons=self.current_epsilons,
            intrinsic_rewards=intrinsic_rewards,
            survival_times=info['step_counts'],
            curriculum_difficulties=torch.zeros(self.num_agents, device=self.device),
            device=self.device,
        )

        return state

    def get_checkpoint(self) -> PopulationCheckpoint:
        """
        Return Pydantic checkpoint.

        Returns:
            PopulationCheckpoint DTO
        """
        return PopulationCheckpoint(
            generation=0,
            num_agents=self.num_agents,
            agent_ids=self.agent_ids,
            curriculum_states={'global': self.curriculum.checkpoint_state()},
            exploration_states={'global': self.exploration.checkpoint_state()},
            pareto_frontier=[],
            metrics_summary={},
        )
