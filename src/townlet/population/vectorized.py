"""
Vectorized population manager.

Coordinates multiple agents with shared curriculum and exploration strategies.
Manages Q-networks, replay buffers, and training loops.
"""

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from townlet.agent.networks import RecurrentSpatialQNetwork, SimpleQNetwork
from townlet.curriculum.base import CurriculumManager
from townlet.exploration.action_selection import epsilon_greedy_action_selection
from townlet.exploration.adaptive_intrinsic import AdaptiveIntrinsicExploration
from townlet.exploration.base import ExplorationStrategy
from townlet.exploration.rnd import RNDExploration
from townlet.population.base import PopulationManager
from townlet.training.replay_buffer import ReplayBuffer
from townlet.training.sequential_replay_buffer import SequentialReplayBuffer
from townlet.training.state import BatchedAgentState, PopulationCheckpoint

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
        agent_ids: list[str],
        device: torch.device,
        obs_dim: int = 70,
        action_dim: int = 5,
        learning_rate: float = 0.00025,
        gamma: float = 0.99,
        replay_buffer_capacity: int = 10000,
        network_type: str = "simple",
        vision_window_size: int = 5,
        tb_logger=None,
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
        self.tb_logger = tb_logger

        # Training metrics (for TensorBoard logging)
        self.last_td_error = 0.0
        self.last_loss = 0.0
        self.last_q_values_mean = 0.0
        self.last_training_step = 0

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

        # Target network (for stable LSTM training with temporal dependencies)
        if self.is_recurrent:
            self.target_network = RecurrentSpatialQNetwork(
                action_dim=action_dim,
                window_size=vision_window_size,
                num_meters=8,
                num_affordance_types=env.num_affordance_types,
                enable_temporal_features=env.enable_temporal_mechanics,
            ).to(device)
            # Initialize target network with same weights as Q-network
            self.target_network.load_state_dict(self.q_network.state_dict())
            self.target_network.eval()  # Target network always in eval mode
            self.target_update_frequency = 100  # Update target every N training steps
            self.training_step_counter = 0
        else:
            self.target_network = None

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
        self.current_curriculum_decisions: list = []  # Store curriculum decisions

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
                # Sequential LSTM training with target network for temporal dependencies
                batch = self.replay_buffer.sample_sequences(
                    batch_size=16,
                    seq_len=self.sequence_length,
                    intrinsic_weight=intrinsic_weight,
                )

                batch_size = batch["observations"].shape[0]
                seq_len = batch["observations"].shape[1]

                # PASS 1: Collect Q-predictions from online network
                # Unroll through sequence, maintaining hidden state for gradient computation
                self.q_network.reset_hidden_state(batch_size=batch_size, device=self.device)
                q_pred_list = []

                for t in range(seq_len):
                    q_values, _ = self.q_network(batch["observations"][:, t, :])
                    q_pred = q_values.gather(1, batch["actions"][:, t].unsqueeze(1)).squeeze()
                    q_pred_list.append(q_pred)

                # PASS 2: Collect Q-targets from target network
                # Unroll through sequence with target network to maintain hidden state
                with torch.no_grad():
                    self.target_network.reset_hidden_state(
                        batch_size=batch_size, device=self.device
                    )
                    q_values_list = []

                    # First, unroll through entire sequence to collect Q-values
                    for t in range(seq_len):
                        q_values, _ = self.target_network(batch["observations"][:, t, :])
                        q_values_list.append(q_values)

                    # Now compute targets using Q-values from next timestep
                    q_target_list = []
                    for t in range(seq_len):
                        if t < seq_len - 1:
                            # Use Q-values from t+1 (computed with hidden state from t)
                            q_next = q_values_list[t + 1].max(1)[0]
                            q_target = (
                                batch["rewards"][:, t]
                                + self.gamma * q_next * (~batch["dones"][:, t]).float()
                            )
                        else:
                            # Terminal state: no next observation
                            q_target = batch["rewards"][:, t]

                        q_target_list.append(q_target)

                # Compute loss across all timesteps
                q_pred_all = torch.stack(q_pred_list, dim=1)  # [batch, seq_len]
                q_target_all = torch.stack(q_target_list, dim=1)  # [batch, seq_len]
                loss = F.mse_loss(q_pred_all, q_target_all)

                # Store training metrics
                with torch.no_grad():
                    self.last_td_error = (q_target_all - q_pred_all).abs().mean().item()
                    self.last_loss = loss.item()
                    self.last_q_values_mean = q_pred_all.mean().item()
                    self.last_training_step = self.total_steps

                # Backprop and optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
                self.optimizer.step()

                # Log network statistics to TensorBoard (every 100 training steps)
                if self.tb_logger is not None and self.total_steps % 100 == 0:
                    for name, param in self.q_network.named_parameters():
                        self.tb_logger.writer.add_histogram(
                            f"Network/Weights/{name}", param.data, self.total_steps
                        )
                        if param.grad is not None:
                            self.tb_logger.writer.add_histogram(
                                f"Network/Gradients/{name}", param.grad, self.total_steps
                            )

                # Update target network periodically
                self.training_step_counter += 1
                if self.training_step_counter % self.target_update_frequency == 0:
                    self.target_network.load_state_dict(self.q_network.state_dict())

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

                # Store training metrics
                with torch.no_grad():
                    self.last_td_error = (q_target - q_pred).abs().mean().item()
                    self.last_loss = loss.item()
                    self.last_q_values_mean = q_pred.mean().item()
                    self.last_training_step = self.total_steps

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
                self.optimizer.step()

                # Log network statistics to TensorBoard (every 100 training steps)
                if self.tb_logger is not None and self.total_steps % 100 == 0:
                    for name, param in self.q_network.named_parameters():
                        self.tb_logger.writer.add_histogram(
                            f"Network/Weights/{name}", param.data, self.total_steps
                        )
                        if param.grad is not None:
                            self.tb_logger.writer.add_histogram(
                                f"Network/Gradients/{name}", param.grad, self.total_steps
                            )

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

    def update_curriculum_tracker(self, rewards: torch.Tensor, dones: torch.Tensor) -> None:
        """Update curriculum tracker with episode rewards/dones."""
        if hasattr(self.curriculum, "tracker") and self.curriculum.tracker is not None:
            self.curriculum.tracker.update_step(rewards, dones)

    def get_training_metrics(self) -> dict:
        """Get recent training metrics for logging.

        Returns:
            Dictionary with TD error, loss, Q-values mean, and training step.
            Returns None values if no training has occurred yet.
        """
        return {
            "td_error": self.last_td_error,
            "loss": self.last_loss,
            "q_values_mean": self.last_q_values_mean,
            "training_step": self.last_training_step,
        }

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
