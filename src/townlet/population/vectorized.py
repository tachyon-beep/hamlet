"""
Vectorized population manager.

Coordinates multiple agents with shared curriculum and exploration strategies.
Manages Q-networks, replay buffers, and training loops.
"""

from typing import List, TYPE_CHECKING
import torch
import torch.nn as nn
import torch.nn.functional as F

from townlet.population.base import PopulationManager
from townlet.training.state import BatchedAgentState, PopulationCheckpoint
from townlet.curriculum.base import CurriculumManager
from townlet.exploration.base import ExplorationStrategy
from townlet.training.replay_buffer import ReplayBuffer
from townlet.exploration.rnd import RNDExploration
from townlet.exploration.adaptive_intrinsic import AdaptiveIntrinsicExploration

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
        learning_rate: float = 0.00025,
        gamma: float = 0.99,
        replay_buffer_capacity: int = 10000,
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
            learning_rate: Learning rate for Q-network optimizer
            gamma: Discount factor
            replay_buffer_capacity: Maximum number of transitions in replay buffer
        """
        self.env = env
        self.curriculum = curriculum
        self.exploration = exploration
        self.agent_ids = agent_ids
        self.num_agents = len(agent_ids)
        self.device = device
        self.gamma = gamma

        # Q-network (shared across all agents for now)
        self.q_network = SimpleQNetwork(obs_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=replay_buffer_capacity, device=device)

        # Training counters
        self.total_steps = 0
        self.train_frequency = 4  # Train Q-network every N steps

        # Episode step counters (reset on done)
        self.episode_step_counts = torch.zeros(self.num_agents, dtype=torch.long, device=device)

        # Current state
        self.current_obs: torch.Tensor = None
        self.current_epsilons: torch.Tensor = None
        self.current_curriculum_decisions: List = []  # Store curriculum decisions

    def reset(self) -> None:
        """Reset all environments and state."""
        self.current_obs = self.env.reset()

        # Get epsilon from exploration strategy (handle both direct and composed)
        if isinstance(self.exploration, AdaptiveIntrinsicExploration):
            epsilon = self.exploration.rnd.epsilon
        else:
            epsilon = self.exploration.epsilon

        self.current_epsilons = torch.full(
            (self.num_agents,), epsilon, device=self.device
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

        # 2. Create temporary agent state for curriculum decision
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

        # 3. Get curriculum decisions (pass Q-values if curriculum supports it)
        if hasattr(self.curriculum, 'get_batch_decisions_with_qvalues'):
            # AdversarialCurriculum - pass Q-values for entropy calculation
            self.current_curriculum_decisions = self.curriculum.get_batch_decisions_with_qvalues(
                temp_state,
                self.agent_ids,
                q_values,
            )
        else:
            # StaticCurriculum or other curricula - no Q-values needed
            self.current_curriculum_decisions = self.curriculum.get_batch_decisions(
                temp_state,
                self.agent_ids,
            )

        # 4. Select actions via exploration strategy
        actions = self.exploration.select_actions(q_values, temp_state)

        # 5. Step environment
        next_obs, rewards, dones, info = envs.step(actions)

        # 6. Compute intrinsic rewards (if RND-based exploration)
        intrinsic_rewards = torch.zeros_like(rewards)
        if isinstance(self.exploration, (RNDExploration, AdaptiveIntrinsicExploration)):
            intrinsic_rewards = self.exploration.compute_intrinsic_rewards(self.current_obs)

        # 7. Store transition in replay buffer
        self.replay_buffer.push(
            observations=self.current_obs,
            actions=actions,
            rewards_extrinsic=rewards,
            rewards_intrinsic=intrinsic_rewards,
            next_observations=next_obs,
            dones=dones,
        )

        # 8. Train RND predictor (if applicable)
        if isinstance(self.exploration, (RNDExploration, AdaptiveIntrinsicExploration)):
            rnd = self.exploration.rnd if isinstance(self.exploration, AdaptiveIntrinsicExploration) else self.exploration
            # Accumulate observations in RND buffer
            for i in range(self.num_agents):
                rnd.obs_buffer.append(self.current_obs[i].cpu())
            # Train predictor if buffer is full
            loss = rnd.update_predictor()

        # 9. Train Q-network from replay buffer (every train_frequency steps)
        self.total_steps += 1
        if self.total_steps % self.train_frequency == 0 and len(self.replay_buffer) >= 64:
            intrinsic_weight = (
                self.exploration.get_intrinsic_weight()
                if isinstance(self.exploration, AdaptiveIntrinsicExploration)
                else 1.0
            )
            batch = self.replay_buffer.sample(batch_size=64, intrinsic_weight=intrinsic_weight)

            # Standard DQN update (simplified, no target network for now)
            q_pred = self.q_network(batch['observations']).gather(1, batch['actions'].unsqueeze(1)).squeeze()

            with torch.no_grad():
                q_next = self.q_network(batch['next_observations']).max(1)[0]
                q_target = batch['rewards'] + self.gamma * q_next * (~batch['dones']).float()

            loss = F.mse_loss(q_pred, q_target)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
            self.optimizer.step()

        # 10. Update current state
        self.current_obs = next_obs

        # Track episode steps
        self.episode_step_counts += 1

        # 11. Handle episode resets (for adaptive intrinsic annealing)
        if dones.any():
            reset_indices = torch.where(dones)[0]
            for idx in reset_indices:
                # Update adaptive intrinsic annealing
                if isinstance(self.exploration, AdaptiveIntrinsicExploration):
                    survival_time = self.episode_step_counts[idx].item()
                    self.exploration.update_on_episode_end(survival_time=survival_time)
                # Reset episode counter
                self.episode_step_counts[idx] = 0

        # 12. Construct BatchedAgentState (use combined rewards for curriculum tracking)
        total_rewards = rewards + intrinsic_rewards * (
            self.exploration.get_intrinsic_weight()
            if isinstance(self.exploration, AdaptiveIntrinsicExploration)
            else 1.0
        )

        state = BatchedAgentState(
            observations=next_obs,
            actions=actions,
            rewards=total_rewards,
            dones=dones,
            epsilons=self.current_epsilons,
            intrinsic_rewards=intrinsic_rewards,
            survival_times=info['step_counts'],
            curriculum_difficulties=torch.zeros(self.num_agents, device=self.device),
            device=self.device,
        )

        return state

    def update_curriculum_tracker(self, rewards: torch.Tensor, dones: torch.Tensor):
        """Update curriculum performance tracking after step.

        Call this after step_population if using AdversarialCurriculum.

        Args:
            rewards: Rewards from environment step
            dones: Done flags from environment step
        """
        if hasattr(self.curriculum, 'tracker') and self.curriculum.tracker is not None:
            self.curriculum.tracker.update_step(rewards, dones)

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
