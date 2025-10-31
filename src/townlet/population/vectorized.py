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
from townlet.agent.networks import SimpleQNetwork, RecurrentSpatialQNetwork

if TYPE_CHECKING:
    from townlet.environment.vectorized_env import VectorizedHamletEnv


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
        network_type: str = "simple",
        vision_window_size: int = 5,
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
            network_type: Network architecture ('simple' or 'recurrent')
            vision_window_size: Size of local vision window for recurrent networks (5 for 5×5)
        """
        self.env = env
        self.curriculum = curriculum
        self.exploration = exploration
        self.agent_ids = agent_ids
        self.num_agents = len(agent_ids)
        self.device = device
        self.gamma = gamma
        self.network_type = network_type
        self.is_recurrent = (network_type == "recurrent")

        # Q-network (shared across all agents for now)
        if network_type == "recurrent":
            self.q_network = RecurrentSpatialQNetwork(
                action_dim=action_dim,
                window_size=vision_window_size,
                num_meters=8,
            ).to(device)
        else:
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

        # Reset recurrent network hidden state (if applicable)
        if self.is_recurrent:
            self.q_network.reset_hidden_state(batch_size=self.num_agents, device=self.device)

        # Get epsilon from exploration strategy (handle both direct and composed)
        if isinstance(self.exploration, AdaptiveIntrinsicExploration):
            epsilon = self.exploration.rnd.epsilon
        else:
            epsilon = self.exploration.epsilon

        self.current_epsilons = torch.full(
            (self.num_agents,), epsilon, device=self.device
        )

    def select_greedy_actions(self, env: 'VectorizedHamletEnv') -> torch.Tensor:
        """
        Select greedy actions with action masking for inference.

        This is the canonical way to select actions during inference.
        Uses the same action masking logic as training to prevent boundary violations.

        Args:
            env: Environment to get action masks from

        Returns:
            actions: [num_agents] tensor of selected actions
        """
        with torch.no_grad():
            # Get Q-values from network
            q_output = self.q_network(self.current_obs)
            # Recurrent networks return (q_values, hidden_state)
            q_values = q_output[0] if isinstance(q_output, tuple) else q_output

            # Get action masks from environment
            action_masks = env.get_action_masks()

            # Mask invalid actions with -inf before argmax
            masked_q_values = q_values.clone()
            masked_q_values[~action_masks] = float('-inf')

            # Select best valid action
            actions = masked_q_values.argmax(dim=1)

        return actions

    def select_epsilon_greedy_actions(
        self,
        env: 'VectorizedHamletEnv',
        epsilon: float
    ) -> torch.Tensor:
        """
        Select epsilon-greedy actions with action masking.

        With probability epsilon, select random valid action.
        With probability (1-epsilon), select greedy action.

        Args:
            env: Environment to get action masks from
            epsilon: Exploration rate [0, 1]

        Returns:
            actions: [num_agents] tensor of selected actions
        """
        with torch.no_grad():
            # Get Q-values from network
            q_output = self.q_network(self.current_obs)
            # Recurrent networks return (q_values, hidden_state)
            q_values = q_output[0] if isinstance(q_output, tuple) else q_output

            # Get action masks from environment
            action_masks = env.get_action_masks()

            # Mask invalid actions with -inf before argmax
            masked_q_values = q_values.clone()
            masked_q_values[~action_masks] = float('-inf')

            # Select best valid action (greedy)
            greedy_actions = masked_q_values.argmax(dim=1)

            # Epsilon-greedy exploration
            num_agents = q_values.shape[0]
            actions = torch.zeros(num_agents, dtype=torch.long, device=q_values.device)

            for i in range(num_agents):
                if torch.rand(1).item() < epsilon:
                    # Random action from valid actions
                    valid_actions = torch.where(action_masks[i])[0]
                    random_idx = torch.randint(0, len(valid_actions), (1,)).item()
                    actions[i] = valid_actions[random_idx]
                else:
                    # Greedy action
                    actions[i] = greedy_actions[i]

        return actions

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
            if self.is_recurrent:
                q_values, new_hidden = self.q_network(self.current_obs)
                # Update hidden state for next step (episode rollout memory)
                self.q_network.set_hidden_state(new_hidden)
            else:
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

        # 4. Get action masks from environment
        action_masks = envs.get_action_masks()

        # 5. Select actions via exploration strategy (with action masking)
        actions = self.exploration.select_actions(q_values, temp_state, action_masks)

        # 6. Step environment
        next_obs, rewards, dones, info = envs.step(actions)

        # 7. Compute intrinsic rewards (if RND-based exploration)
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
            if self.is_recurrent:
                # Reset hidden states for batch training (treat transitions independently)
                batch_size = batch['observations'].shape[0]
                self.q_network.reset_hidden_state(batch_size=batch_size, device=self.device)

                q_values, _ = self.q_network(batch['observations'])
                q_pred = q_values.gather(1, batch['actions'].unsqueeze(1)).squeeze()

                with torch.no_grad():
                    self.q_network.reset_hidden_state(batch_size=batch_size, device=self.device)
                    q_next_values, _ = self.q_network(batch['next_observations'])
                    q_next = q_next_values.max(1)[0]
                    q_target = batch['rewards'] + self.gamma * q_next * (~batch['dones']).float()
            else:
                q_pred = self.q_network(batch['observations']).gather(1, batch['actions'].unsqueeze(1)).squeeze()

                with torch.no_grad():
                    q_next = self.q_network(batch['next_observations']).max(1)[0]
                    q_target = batch['rewards'] + self.gamma * q_next * (~batch['dones']).float()

            loss = F.mse_loss(q_pred, q_target)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
            self.optimizer.step()

            # Reset hidden state back to episode batch size after training
            if self.is_recurrent:
                self.q_network.reset_hidden_state(batch_size=self.num_agents, device=self.device)

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

            # Reset hidden states for agents that terminated (if using recurrent network)
            if self.is_recurrent:
                # Get current hidden state
                h, c = self.q_network.get_hidden_state()
                # Zero out hidden states for terminated agents
                h[:, reset_indices, :] = 0.0
                c[:, reset_indices, :] = 0.0
                self.q_network.set_hidden_state((h, c))

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
