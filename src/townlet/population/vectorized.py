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
from townlet.training.sequential_replay_buffer import SequentialReplayBuffer
from townlet.exploration.rnd import RNDExploration
from townlet.exploration.adaptive_intrinsic import AdaptiveIntrinsicExploration
from townlet.exploration.action_selection import epsilon_greedy_action_selection
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
        env: "VectorizedHamletEnv",
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
        self.is_recurrent = network_type == "recurrent"

        # Q-network (shared across all agents for now)
        if network_type == "recurrent":
            self.q_network = RecurrentSpatialQNetwork(
                action_dim=action_dim,
                window_size=vision_window_size,
                num_meters=8,
                num_affordance_types=env.num_affordance_types,
                enable_temporal_features=env.enable_temporal_mechanics,
            ).to(device)
        else:
            self.q_network = SimpleQNetwork(obs_dim, action_dim).to(device)

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Replay buffer (dual system: sequential for recurrent, standard for feedforward)
        if self.is_recurrent:
            self.replay_buffer = SequentialReplayBuffer(
                capacity=replay_buffer_capacity, device=device
            )
            # Episode tracking for sequential buffer
            self.current_episodes = [
                {
                    "observations": [],
                    "actions": [],
                    "rewards_extrinsic": [],
                    "rewards_intrinsic": [],
                    "dones": [],
                }
                for _ in range(self.num_agents)
            ]
        else:
            self.replay_buffer = ReplayBuffer(capacity=replay_buffer_capacity, device=device)

        # Training counters
        self.total_steps = 0
        self.train_frequency = 4  # Train Q-network every N steps
        self.sequence_length = 8  # Length of sequences for LSTM training

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

        self.current_epsilons = torch.full((self.num_agents,), epsilon, device=self.device)

    def select_greedy_actions(self, env: "VectorizedHamletEnv") -> torch.Tensor:
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
            masked_q_values[~action_masks] = float("-inf")

            # Select best valid action
            actions = masked_q_values.argmax(dim=1)

        return actions

    def select_epsilon_greedy_actions(
        self, env: "VectorizedHamletEnv", epsilon: float
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

            # Use shared epsilon-greedy action selection
            epsilons = torch.full(
                (self.num_agents,), epsilon, device=self.device, dtype=torch.float32
            )
            return epsilon_greedy_action_selection(
                q_values=q_values,
                epsilons=epsilons,
                action_masks=action_masks,
            )

    def step_population(
        self,
        envs: "VectorizedHamletEnv",
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
        if hasattr(self.curriculum, "get_batch_decisions_with_qvalues"):
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
        if self.is_recurrent:
            # For recurrent networks: accumulate episodes
            for i in range(self.num_agents):
                self.current_episodes[i]["observations"].append(self.current_obs[i].cpu())
                self.current_episodes[i]["actions"].append(actions[i].cpu())
                self.current_episodes[i]["rewards_extrinsic"].append(rewards[i].cpu())
                self.current_episodes[i]["rewards_intrinsic"].append(intrinsic_rewards[i].cpu())
                self.current_episodes[i]["dones"].append(dones[i].cpu())
        else:
            # For feedforward networks: store individual transitions
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
            rnd = (
                self.exploration.rnd
                if isinstance(self.exploration, AdaptiveIntrinsicExploration)
                else self.exploration
            )
            # Accumulate observations in RND buffer
            for i in range(self.num_agents):
                rnd.obs_buffer.append(self.current_obs[i].cpu())
            # Train predictor if buffer is full
            loss = rnd.update_predictor()

        # 9. Train Q-network from replay buffer (every train_frequency steps)
        self.total_steps += 1
        # For recurrent: need enough episodes (16+) for sequence sampling
        # For feedforward: need enough transitions (64+) for batch sampling
        min_buffer_size = 16 if self.is_recurrent else 64
        if (
            self.total_steps % self.train_frequency == 0
            and len(self.replay_buffer) >= min_buffer_size
        ):
            intrinsic_weight = (
                self.exploration.get_intrinsic_weight()
                if isinstance(self.exploration, AdaptiveIntrinsicExploration)
                else 1.0
            )

            if self.is_recurrent:
                # Sequential LSTM training: sample sequences and unroll through time
                batch = self.replay_buffer.sample_sequences(
                    batch_size=16,  # Fewer sequences (each has multiple timesteps)
                    seq_len=self.sequence_length,
                    intrinsic_weight=intrinsic_weight,
                )
                # batch["observations"]: [batch_size, seq_len, obs_dim]
                # batch["actions"]: [batch_size, seq_len]
                # batch["rewards"]: [batch_size, seq_len]
                # batch["dones"]: [batch_size, seq_len]

                batch_size = batch["observations"].shape[0]
                seq_len = batch["observations"].shape[1]

                # Initialize hidden state for sequence processing
                self.q_network.reset_hidden_state(batch_size=batch_size, device=self.device)

                # Unroll through sequence, maintaining hidden state
                q_pred_list = []
                q_target_list = []

                for t in range(seq_len):
                    # Current timestep Q-values
                    q_values, _ = self.q_network(batch["observations"][:, t, :])
                    q_pred = q_values.gather(1, batch["actions"][:, t].unsqueeze(1)).squeeze()
                    q_pred_list.append(q_pred)

                    # Target Q-values (using next observation from sequence)
                    with torch.no_grad():
                        if t < seq_len - 1:
                            # Use next obs from sequence
                            next_obs_batch = batch["observations"][:, t + 1, :]
                        else:
                            # Last timestep - use current (will be masked by done anyway)
                            next_obs_batch = batch["observations"][:, t, :]

                        # Compute target with separate forward pass (no gradient)
                        self.q_network.reset_hidden_state(batch_size=batch_size, device=self.device)
                        q_next_values, _ = self.q_network(next_obs_batch)
                        q_next = q_next_values.max(1)[0]
                        q_target = (
                            batch["rewards"][:, t]
                            + self.gamma * q_next * (~batch["dones"][:, t]).float()
                        )
                        q_target_list.append(q_target)

                # Concatenate all timesteps and compute loss
                q_pred_all = torch.stack(q_pred_list, dim=1)  # [batch, seq_len]
                q_target_all = torch.stack(q_target_list, dim=1)  # [batch, seq_len]
                loss = F.mse_loss(q_pred_all, q_target_all)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
                self.optimizer.step()

                # Reset hidden state back to episode batch size after training
                self.q_network.reset_hidden_state(batch_size=self.num_agents, device=self.device)
            else:
                # Standard feedforward DQN training (unchanged)
                batch = self.replay_buffer.sample(batch_size=64, intrinsic_weight=intrinsic_weight)

                q_pred = (
                    self.q_network(batch["observations"])
                    .gather(1, batch["actions"].unsqueeze(1))
                    .squeeze()
                )

                with torch.no_grad():
                    q_next = self.q_network(batch["next_observations"]).max(1)[0]
                    q_target = batch["rewards"] + self.gamma * q_next * (~batch["dones"]).float()

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

                # Store complete episode in sequential buffer (for recurrent networks)
                if self.is_recurrent and len(self.current_episodes[idx]["observations"]) > 0:
                    episode = {
                        "observations": torch.stack(self.current_episodes[idx]["observations"]),
                        "actions": torch.stack(self.current_episodes[idx]["actions"]),
                        "rewards_extrinsic": torch.stack(
                            self.current_episodes[idx]["rewards_extrinsic"]
                        ),
                        "rewards_intrinsic": torch.stack(
                            self.current_episodes[idx]["rewards_intrinsic"]
                        ),
                        "dones": torch.stack(self.current_episodes[idx]["dones"]),
                    }
                    self.replay_buffer.store_episode(episode)
                    # Reset episode accumulator
                    self.current_episodes[idx] = {
                        "observations": [],
                        "actions": [],
                        "rewards_extrinsic": [],
                        "rewards_intrinsic": [],
                        "dones": [],
                    }

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
            survival_times=info["step_counts"],
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
        if hasattr(self.curriculum, "tracker") and self.curriculum.tracker is not None:
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
            curriculum_states={"global": self.curriculum.checkpoint_state()},
            exploration_states={"global": self.exploration.checkpoint_state()},
            pareto_frontier=[],
            metrics_summary={},
        )
