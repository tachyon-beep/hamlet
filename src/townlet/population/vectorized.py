"""
Vectorized population manager.

Coordinates multiple agents with shared curriculum and exploration strategies.
Manages Q-networks, replay buffers, and training loops.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from townlet.agent.brain_config import BrainConfig
from townlet.agent.loss_factory import LossFactory
from townlet.agent.network_factory import NetworkFactory
from townlet.agent.networks import RecurrentSpatialQNetwork, SimpleQNetwork, StructuredQNetwork
from townlet.agent.optimizer_factory import OptimizerFactory
from townlet.curriculum.base import CurriculumManager
from townlet.exploration.action_selection import epsilon_greedy_action_selection
from townlet.exploration.adaptive_intrinsic import AdaptiveIntrinsicExploration
from townlet.exploration.base import ExplorationStrategy
from townlet.exploration.rnd import RNDExploration
from townlet.population.base import PopulationManager
from townlet.population.runtime_registry import AgentRuntimeRegistry
from townlet.training.replay_buffer import ReplayBuffer
from townlet.training.sequential_replay_buffer import SequentialReplayBuffer
from townlet.training.state import BatchedAgentState, CurriculumDecision, PopulationCheckpoint

if TYPE_CHECKING:
    from townlet.environment.vectorized_env import VectorizedHamletEnv

EpisodeContainer = dict[str, list[torch.Tensor]]


class VectorizedPopulation(PopulationManager):
    """
    Vectorized population manager.

    Coordinates training for num_agents parallel agents with shared
    curriculum and exploration strategies.
    """

    def __init__(
        self,
        env: VectorizedHamletEnv,
        curriculum: CurriculumManager,
        exploration: ExplorationStrategy,
        agent_ids: list[str],
        device: torch.device,
        obs_dim: int = 70,
        action_dim: int | None = None,
        learning_rate: float = 0.00025,
        gamma: float = 0.99,
        replay_buffer_capacity: int = 10000,
        network_type: str = "simple",
        vision_window_size: int = 5,
        tb_logger=None,
        train_frequency: int = 4,
        target_update_frequency: int = 100,
        batch_size: int | None = None,
        sequence_length: int = 8,
        max_grad_norm: float = 10.0,
        use_double_dqn: bool = False,
        brain_config: BrainConfig | None = None,
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
            action_dim: Action dimension (defaults to env.action_dim if not specified)
            learning_rate: Learning rate for Q-network optimizer
            gamma: Discount factor
            replay_buffer_capacity: Maximum number of transitions in replay buffer
            network_type: Network architecture ('simple' or 'recurrent')
            vision_window_size: Size of local vision window for recurrent networks (5 for 5×5)
            tb_logger: Optional TensorBoard logger
            train_frequency: Train Q-network every N steps (default: 4)
            target_update_frequency: Update target network every N training steps (default: 100)
            batch_size: Batch size for experience replay (default: 64 for feedforward, 16 for recurrent)
            sequence_length: Length of sequences for LSTM training (default: 8, recurrent only)
            max_grad_norm: Gradient clipping threshold (default: 10.0)
            use_double_dqn: Use Double DQN algorithm (default: False for vanilla DQN)
            brain_config: Optional BrainConfig for architecture, optimizer, loss (TASK-005 Phase 1)
        """
        self.env = env
        self.curriculum = curriculum
        self.exploration = exploration
        self.agent_ids = agent_ids
        self.num_agents = len(agent_ids)
        self.device = device
        self.network_type = network_type
        self.is_recurrent = network_type == "recurrent"
        self.tb_logger = tb_logger
        self.brain_config = brain_config

        # TASK-005 Phase 1: Validate brain_config constraints
        if brain_config is not None:
            if network_type != "simple":
                raise ValueError(
                    f"brain_config requires network_type='simple' in Phase 1. "
                    f"Got network_type='{network_type}'. "
                    f"Recurrent networks will be supported in Phase 2."
                )
            # Override Q-learning parameters from brain_config
            self.gamma = brain_config.q_learning.gamma
            self.use_double_dqn = brain_config.q_learning.use_double_dqn
            target_update_frequency = brain_config.q_learning.target_update_frequency
        else:
            # Use constructor parameters when no brain_config
            self.gamma = gamma
            self.use_double_dqn = use_double_dqn

        # Default action_dim to env.action_dim if not specified (TASK-002B Phase 4.1)
        if action_dim is None:
            action_dim = env.action_dim
        self.action_dim = action_dim

        # Agent runtime metrics (telemetry + reward baseline source of truth)
        self.runtime_registry = AgentRuntimeRegistry(agent_ids=agent_ids, device=device)
        self.env.attach_runtime_registry(self.runtime_registry)

        # Wire exploration module to environment for intrinsic reward computation
        self.env.set_exploration_module(exploration)

        # Training metrics (for TensorBoard logging)
        self.last_td_error = 0.0
        self.last_loss = 0.0
        self.last_q_values_mean = 0.0
        self.last_training_step = 0
        self.last_rnd_loss = 0.0  # RND predictor loss (for monitoring intrinsic exploration)

        # Q-network (shared across all agents for now)
        self.q_network: nn.Module
        if brain_config is not None:
            # TASK-005 Phase 1: Build network from brain_config using NetworkFactory
            assert brain_config.architecture.type == "feedforward", "Phase 1 only supports feedforward"
            assert brain_config.architecture.feedforward is not None, "feedforward config must be present"
            self.q_network = NetworkFactory.build_feedforward(
                config=brain_config.architecture.feedforward,
                obs_dim=obs_dim,
                action_dim=action_dim,
            ).to(device)
        elif network_type == "recurrent":
            self.q_network = RecurrentSpatialQNetwork(
                action_dim=action_dim,
                window_size=vision_window_size,
                position_dim=env.substrate.position_dim,  # Dynamic: 2 for Grid2D, 3 for Grid3D, 0 for Aspatial
                num_meters=env.meter_count,  # TASK-001: Use dynamic meter count from environment
                num_affordance_types=env.num_affordance_types,
                enable_temporal_features=env.enable_temporal_mechanics,
                hidden_dim=256,  # TODO(BRAIN_AS_CODE): Should come from config
            ).to(device)
        elif network_type == "structured":
            self.q_network = StructuredQNetwork(
                obs_dim=obs_dim,
                action_dim=action_dim,
                observation_activity=env.observation_activity,
                group_embed_dim=32,  # TODO(BRAIN_AS_CODE): Should come from config
                q_head_hidden_dim=128,  # TODO(BRAIN_AS_CODE): Should come from config
            ).to(device)
        else:
            self.q_network = SimpleQNetwork(obs_dim, action_dim, hidden_dim=128).to(device)  # TODO(BRAIN_AS_CODE): Should come from config

        # Target network (stabilises training for both feed-forward and recurrent agents)
        self.target_network: nn.Module
        if brain_config is not None:
            # TASK-005 Phase 1: Build target network from brain_config using NetworkFactory
            assert brain_config.architecture.type == "feedforward", "Phase 1 only supports feedforward"
            assert brain_config.architecture.feedforward is not None, "feedforward config must be present"
            self.target_network = NetworkFactory.build_feedforward(
                config=brain_config.architecture.feedforward,
                obs_dim=obs_dim,
                action_dim=action_dim,
            ).to(device)
        elif self.is_recurrent:
            self.target_network = RecurrentSpatialQNetwork(
                action_dim=action_dim,
                window_size=vision_window_size,
                position_dim=env.substrate.position_dim,  # Dynamic: 2 for Grid2D, 3 for Grid3D, 0 for Aspatial
                num_meters=env.meter_count,  # TASK-001: Use dynamic meter count from environment
                num_affordance_types=env.num_affordance_types,
                enable_temporal_features=env.enable_temporal_mechanics,
                hidden_dim=256,  # TODO(BRAIN_AS_CODE): Should come from config
            ).to(device)
        elif network_type == "structured":
            self.target_network = StructuredQNetwork(
                obs_dim=obs_dim,
                action_dim=action_dim,
                observation_activity=env.observation_activity,
                group_embed_dim=32,  # TODO(BRAIN_AS_CODE): Should come from config
                q_head_hidden_dim=128,  # TODO(BRAIN_AS_CODE): Should come from config
            ).to(device)
        else:
            self.target_network = SimpleQNetwork(obs_dim, action_dim, hidden_dim=128).to(
                device
            )  # TODO(BRAIN_AS_CODE): Should come from config

        # Initialize common target network state
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Target network always in eval mode
        self.target_update_frequency = target_update_frequency
        self.training_step_counter = 0

        # Optimizer (from config or hardcoded)
        if brain_config is not None:
            # TASK-005 Phase 1: Build optimizer from brain_config using OptimizerFactory
            self.optimizer = OptimizerFactory.build(
                config=brain_config.optimizer,
                parameters=self.q_network.parameters(),
            )
        else:
            self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Loss function (from config or hardcoded)
        # Note: Currently not used in train_batch() but stored for future use
        if brain_config is not None:
            # TASK-005 Phase 1: Build loss function from brain_config using LossFactory
            self.loss_fn = LossFactory.build(config=brain_config.loss)
        else:
            self.loss_fn = nn.MSELoss()  # Default to MSE (matches current hardcoded behavior)

        # Replay buffer (dual system: sequential for recurrent, standard for feedforward)
        self.replay_buffer: ReplayBuffer | SequentialReplayBuffer
        self.current_episodes: list[EpisodeContainer] = []
        if self.is_recurrent:
            self.replay_buffer = SequentialReplayBuffer(capacity=replay_buffer_capacity, device=device)
            # Episode tracking for sequential buffer
            self.current_episodes = [self._new_episode_container() for _ in range(self.num_agents)]
        else:
            self.replay_buffer = ReplayBuffer(capacity=replay_buffer_capacity, device=device)

        # Training hyperparameters (configurable)
        self.total_steps = 0
        self.train_frequency = train_frequency
        self.sequence_length = sequence_length
        self.max_grad_norm = max_grad_norm
        # Default batch_size based on network type if not specified
        if batch_size is None:
            self.batch_size = 16 if self.is_recurrent else 64
        else:
            self.batch_size = batch_size

        # Episode step counters (reset on done)
        self.episode_step_counts = torch.zeros(self.num_agents, dtype=torch.long, device=device)

        # Current state
        self.current_obs = torch.zeros((self.num_agents, obs_dim), dtype=torch.float32, device=device)
        self.current_epsilons = torch.zeros(self.num_agents, dtype=torch.float32, device=device)
        self.current_curriculum_decisions: list[CurriculumDecision] = []  # Store curriculum decisions
        self.current_depletion_multiplier: float = 1.0  # Track curriculum difficulty

    def reset(self) -> None:
        """Reset all environments and state."""
        self.current_obs = self.env.reset()

        # Reset recurrent network hidden state (if applicable)
        if self.is_recurrent:
            recurrent_network = cast(RecurrentSpatialQNetwork, self.q_network)
            recurrent_network.reset_hidden_state(batch_size=self.num_agents, device=self.device)

        # Get epsilon from exploration strategy (handle both direct and composed)
        # Sync telemetry + exploration metrics (initial epsilon / stage)
        self.sync_exploration_metrics()
        self._sync_curriculum_metrics()

    # ------------------------------------------------------------------ #
    # Episode lifecycle helpers
    # ------------------------------------------------------------------ #
    def _new_episode_container(self) -> EpisodeContainer:
        """Create a fresh container for accumulating episode data."""
        return {
            "observations": [],
            "actions": [],
            "rewards_extrinsic": [],
            "rewards_intrinsic": [],
            "dones": [],
        }

    def _store_episode_and_reset(self, agent_idx: int) -> bool:
        """Store accumulated episode for agent and reset buffers."""
        if not self.is_recurrent or not self.current_episodes:
            return False

        episode = self.current_episodes[agent_idx]
        if len(episode["observations"]) == 0:
            return False

        sequential_buffer = cast(SequentialReplayBuffer, self.replay_buffer)

        sequential_buffer.store_episode(
            {
                "observations": torch.stack(episode["observations"]),
                "actions": torch.stack(episode["actions"]),
                "rewards_extrinsic": torch.stack(episode["rewards_extrinsic"]),
                "rewards_intrinsic": torch.stack(episode["rewards_intrinsic"]),
                "dones": torch.stack(episode["dones"]),
            }
        )

        self.current_episodes[agent_idx] = self._new_episode_container()
        return True

    def _reset_hidden_state(self, agent_idx: int) -> None:
        """Zero recurrent hidden state for a single agent."""
        if not self.is_recurrent:
            return

        recurrent_network = cast(RecurrentSpatialQNetwork, self.q_network)
        hidden = recurrent_network.get_hidden_state()
        if hidden is None:
            return

        h, c = hidden

        h[:, agent_idx, :] = 0.0
        c[:, agent_idx, :] = 0.0
        recurrent_network.set_hidden_state((h, c))

    # ------------------------------------------------------------------ #
    # Telemetry synchronisation helpers
    # ------------------------------------------------------------------ #
    def _get_current_epsilon_value(self) -> float:
        """Return current exploration epsilon (global for now)."""
        if isinstance(self.exploration, AdaptiveIntrinsicExploration):
            return float(self.exploration.rnd.epsilon)
        if hasattr(self.exploration, "epsilon"):
            return float(self.exploration.epsilon)
        return 0.0

    def _get_current_intrinsic_weight_value(self) -> float:
        """Return current intrinsic reward weight."""
        if hasattr(self.exploration, "get_intrinsic_weight"):
            return float(self.exploration.get_intrinsic_weight())
        return 0.0

    def _sync_curriculum_metrics(self) -> None:
        """
        Write curriculum metadata (stage) into the runtime registry.

        Prefers the most recent curriculum decisions; falls back to tracker state
        when decisions are unavailable (e.g. before the first population step).
        """
        if self.current_curriculum_decisions:
            for idx, decision in enumerate(self.current_curriculum_decisions):
                stage_value = self._difficulty_to_stage(float(decision.difficulty_level))
                self.runtime_registry.set_curriculum_stage(agent_idx=idx, stage=stage_value)
            return

        tracker = getattr(self.curriculum, "tracker", None)
        if tracker is None or not hasattr(tracker, "agent_stages"):
            return

        for idx in range(self.num_agents):
            stage_value = int(tracker.agent_stages[idx].item())
            self.runtime_registry.set_curriculum_stage(agent_idx=idx, stage=stage_value)

    @staticmethod
    def _difficulty_to_stage(difficulty_level: float) -> int:
        """Convert curriculum difficulty (0.0-1.0) to discrete stage (1-5)."""
        stage = int(round(difficulty_level * 4.0)) + 1
        return max(1, min(5, stage))

    def sync_exploration_metrics(self) -> None:
        """
        Synchronise exploration parameters (epsilon, intrinsic weight) to registry.

        Also refreshes current_epsilons to keep action selection in sync with telemetry.
        """
        epsilon_tensor = torch.full(
            (self.num_agents,),
            self._get_current_epsilon_value(),
            dtype=torch.float32,
            device=self.device,
        )
        self.current_epsilons = epsilon_tensor

        intrinsic_weight = self._get_current_intrinsic_weight_value()

        for idx in range(self.num_agents):
            self.runtime_registry.set_epsilon(agent_idx=idx, epsilon=epsilon_tensor[idx])
            self.runtime_registry.set_intrinsic_weight(agent_idx=idx, weight=intrinsic_weight)

    def _finalize_episode(self, agent_idx: int, survival_time: int) -> None:
        """Finalize episode metadata and bookkeeping after store."""
        self.runtime_registry.record_survival_time(agent_idx=agent_idx, steps=survival_time)

        if isinstance(self.exploration, AdaptiveIntrinsicExploration):
            self.exploration.update_on_episode_end(survival_time=survival_time)

        # Sync exploration telemetry after any annealing/decay changes
        self.sync_exploration_metrics()

        self.episode_step_counts[agent_idx] = 0
        self._reset_hidden_state(agent_idx)

    def flush_episode(self, agent_idx: int) -> None:
        """
        Flush current episode for an agent to replay buffer.

        Used when agent dies or episode hits max_steps.
        This prevents memory leaks and ensures successful episodes reach the replay buffer.

        Args:
            agent_idx: Index of agent to flush
        """
        if not self.is_recurrent:
            # Feedforward mode: transitions already in buffer, nothing to flush
            return

        episode = self.current_episodes[agent_idx]
        if len(episode["observations"]) == 0:
            # Nothing to flush
            return

        survival_time = len(episode["observations"])
        self._store_episode_and_reset(agent_idx)
        self._finalize_episode(agent_idx, survival_time)

    def select_greedy_actions(self, env: VectorizedHamletEnv) -> torch.Tensor:
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
            actions: torch.Tensor = masked_q_values.argmax(dim=1)

        return actions

    def select_epsilon_greedy_actions(self, env: VectorizedHamletEnv, epsilon: float) -> torch.Tensor:
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
            epsilons = torch.full((self.num_agents,), epsilon, device=self.device, dtype=torch.float32)
            return epsilon_greedy_action_selection(
                q_values=q_values,
                epsilons=epsilons,
                action_masks=action_masks,
            )

    def step_population(
        self,
        envs: VectorizedHamletEnv,
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
                recurrent_network = cast(RecurrentSpatialQNetwork, self.q_network)
                q_values, new_hidden = recurrent_network(self.current_obs)
                # Update hidden state for next step (episode rollout memory)
                recurrent_network.set_hidden_state(new_hidden)
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

        # 3.5 Sync curriculum metrics to registry
        self._sync_curriculum_metrics()

        # 4. Get action masks from environment
        action_masks = envs.get_action_masks()

        # 5. Select actions via exploration strategy (with action masking)
        actions = self.exploration.select_actions(q_values, temp_state, action_masks)

        # 6. Extract curriculum difficulty multiplier
        depletion_multiplier = 1.0
        if self.current_curriculum_decisions:
            depletion_multiplier = self.current_curriculum_decisions[0].depletion_multiplier

        # 7. Step environment with curriculum difficulty
        # Note: rewards from environment already contain full DAC composition:
        #   rewards = extrinsic + (intrinsic * base_weight * modifiers) + shaping
        next_obs, rewards, dones, info = envs.step(actions, depletion_multiplier)

        # 7. Compute intrinsic rewards for logging/tracking only (not added to rewards)
        # DAC engine already includes intrinsic in the rewards tensor above
        intrinsic_rewards = torch.zeros_like(rewards)
        if isinstance(self.exploration, RNDExploration | AdaptiveIntrinsicExploration):
            intrinsic_rewards = self.exploration.compute_intrinsic_rewards(self.current_obs, update_stats=True)

        # 7. Store transition in replay buffer
        # Note: Since DAC already composed rewards, we store total rewards and zero intrinsic
        # to avoid double-counting during replay buffer sampling
        if self.is_recurrent:
            # For recurrent networks: accumulate episodes
            for i in range(self.num_agents):
                self.current_episodes[i]["observations"].append(self.current_obs[i].cpu())
                self.current_episodes[i]["actions"].append(actions[i].cpu())
                self.current_episodes[i]["rewards_extrinsic"].append(rewards[i].cpu())  # Actually total from DAC
                self.current_episodes[i]["rewards_intrinsic"].append(torch.zeros_like(rewards[i]).cpu())  # Zero to avoid double-counting
                self.current_episodes[i]["dones"].append(dones[i].cpu())
        else:
            # For feedforward networks: store individual transitions
            standard_buffer = cast(ReplayBuffer, self.replay_buffer)
            standard_buffer.push(
                observations=self.current_obs,
                actions=actions,
                rewards_extrinsic=rewards,  # Actually total rewards from DAC
                rewards_intrinsic=torch.zeros_like(rewards),  # Zero to avoid double-counting
                next_observations=next_obs,
                dones=dones,
            )

        # 8. Train RND predictor (if applicable)
        if isinstance(self.exploration, RNDExploration | AdaptiveIntrinsicExploration):
            rnd = self.exploration.rnd if isinstance(self.exploration, AdaptiveIntrinsicExploration) else self.exploration
            # Accumulate observations in RND buffer
            for i in range(self.num_agents):
                rnd.obs_buffer.append(self.current_obs[i].cpu())
            # Train predictor if buffer is full
            rnd_loss = rnd.update_predictor()
            # Track RND loss for monitoring (similar to Q-network loss)
            self.last_rnd_loss = rnd_loss

        # 9. Train Q-network from replay buffer (every train_frequency steps)
        self.total_steps += 1
        # For recurrent: need enough episodes (16+) for sequence sampling
        # For feedforward: need enough transitions (>= batch_size) for batch sampling
        min_buffer_size = 16 if self.is_recurrent else self.batch_size
        if self.total_steps % self.train_frequency == 0 and len(self.replay_buffer) >= min_buffer_size:
            # Note: intrinsic_weight is 1.0 because DAC already composed rewards.
            # The replay buffer stores total rewards (not separate extrinsic/intrinsic).
            intrinsic_weight = 1.0

            if self.is_recurrent:
                # Sequential LSTM training with target network for temporal dependencies
                sequential_buffer = cast(SequentialReplayBuffer, self.replay_buffer)
                batch = sequential_buffer.sample_sequences(
                    batch_size=self.batch_size,
                    seq_len=self.sequence_length,
                    intrinsic_weight=intrinsic_weight,
                )

                batch_size = batch["observations"].shape[0]
                seq_len = batch["observations"].shape[1]

                # PASS 1: Collect Q-predictions from online network
                # Unroll through sequence, maintaining hidden state for gradient computation
                recurrent_network = cast(RecurrentSpatialQNetwork, self.q_network)
                recurrent_network.reset_hidden_state(batch_size=batch_size, device=self.device)
                q_pred_list = []

                for t in range(seq_len):
                    q_values, _ = recurrent_network(batch["observations"][:, t, :])
                    q_pred = q_values.gather(1, batch["actions"][:, t].unsqueeze(1)).squeeze()
                    q_pred_list.append(q_pred)

                # PASS 2: Collect Q-targets from target network
                # Unroll through sequence with target network to maintain hidden state
                with torch.no_grad():
                    target_recurrent = cast(RecurrentSpatialQNetwork, self.target_network)
                    target_recurrent.reset_hidden_state(batch_size=batch_size, device=self.device)

                    if self.use_double_dqn:
                        # Double DQN: Use online network for action selection
                        online_recurrent = cast(RecurrentSpatialQNetwork, self.q_network)
                        online_recurrent.reset_hidden_state(batch_size=batch_size, device=self.device)

                        # First pass: Get action selections from online network
                        next_action_list = []
                        for t in range(seq_len):
                            q_values_online, _ = online_recurrent(batch["observations"][:, t, :])
                            next_actions = q_values_online.argmax(1)
                            next_action_list.append(next_actions)

                        # Second pass: Evaluate those actions with target network
                        q_values_list = []
                        for t in range(seq_len):
                            q_values_target, _ = target_recurrent(batch["observations"][:, t, :])
                            q_values_list.append(q_values_target)

                        # Compute targets using selected actions
                        q_target_list = []
                        for t in range(seq_len):
                            if t < seq_len - 1:
                                # Use Q-values from t+1, evaluated at actions selected by online network
                                next_actions = next_action_list[t + 1]
                                q_next = q_values_list[t + 1].gather(1, next_actions.unsqueeze(1)).squeeze()
                                q_target = batch["rewards"][:, t] + self.gamma * q_next * (~batch["dones"][:, t]).float()
                            else:
                                # Terminal state: no next observation
                                q_target = batch["rewards"][:, t]
                            q_target_list.append(q_target)
                    else:
                        # Vanilla DQN: Use target network for both selection and evaluation
                        q_values_list = []

                        # First, unroll through entire sequence to collect Q-values
                        for t in range(seq_len):
                            q_values, _ = target_recurrent(batch["observations"][:, t, :])
                            q_values_list.append(q_values)

                        # Now compute targets using Q-values from next timestep
                        q_target_list = []
                        for t in range(seq_len):
                            if t < seq_len - 1:
                                # Use Q-values from t+1 (computed with hidden state from t)
                                q_next = q_values_list[t + 1].max(1)[0]
                                q_target = batch["rewards"][:, t] + self.gamma * q_next * (~batch["dones"][:, t]).float()
                            else:
                                # Terminal state: no next observation
                                q_target = batch["rewards"][:, t]

                            q_target_list.append(q_target)

                # Compute loss across all timesteps with post-terminal masking (P2.2)
                q_pred_all = torch.stack(q_pred_list, dim=1)  # [batch, seq_len]
                q_target_all = torch.stack(q_target_list, dim=1)  # [batch, seq_len]

                # P2.2: Apply mask to prevent gradients from post-terminal garbage
                # TASK-005 Phase 1: Use configured loss type (element-wise for masking)
                if self.brain_config is not None and self.brain_config.loss.type == "huber":
                    losses = F.huber_loss(
                        q_pred_all,
                        q_target_all,
                        reduction="none",
                        delta=self.brain_config.loss.huber_delta,
                    )
                elif self.brain_config is not None and self.brain_config.loss.type == "smooth_l1":
                    losses = F.smooth_l1_loss(q_pred_all, q_target_all, reduction="none")
                else:
                    # MSE or legacy (no brain_config)
                    losses = F.mse_loss(q_pred_all, q_target_all, reduction="none")
                mask = batch["mask"].float()  # [batch, seq_len] - True for valid timesteps
                masked_loss = (losses * mask).sum() / mask.sum().clamp_min(1)
                loss: torch.Tensor = masked_loss

                # Store training metrics
                with torch.no_grad():
                    # Compute metrics only on valid (masked) timesteps
                    valid_errors = ((q_target_all - q_pred_all).abs() * mask).sum() / mask.sum().clamp_min(1)
                    self.last_td_error = valid_errors.item()
                    self.last_loss = loss.item()
                    # Q-values mean across valid timesteps only
                    valid_q_mean = (q_pred_all * mask).sum() / mask.sum().clamp_min(1)
                    self.last_q_values_mean = valid_q_mean.item()
                    self.last_training_step = self.total_steps

                # Backprop and optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=self.max_grad_norm)
                self.optimizer.step()

                # Log network statistics to TensorBoard (every 100 training steps)
                if self.tb_logger is not None and self.total_steps % 100 == 0:
                    for name, param in self.q_network.named_parameters():
                        self.tb_logger.writer.add_histogram(f"Network/Weights/{name}", param.data, self.total_steps)
                        if param.grad is not None:
                            self.tb_logger.writer.add_histogram(f"Network/Gradients/{name}", param.grad, self.total_steps)

                # Update target network periodically
                self.training_step_counter += 1
                if self.training_step_counter % self.target_update_frequency == 0:
                    self.target_network.load_state_dict(self.q_network.state_dict())

                # Reset hidden state back to episode batch size after training
                recurrent_network = cast(RecurrentSpatialQNetwork, self.q_network)
                recurrent_network.reset_hidden_state(batch_size=self.num_agents, device=self.device)
            else:
                # Standard feedforward DQN training
                standard_buffer = cast(ReplayBuffer, self.replay_buffer)
                batch = standard_buffer.sample(batch_size=self.batch_size, intrinsic_weight=intrinsic_weight)

                # Compute Q-predictions from online network
                q_pred = self.q_network(batch["observations"]).gather(1, batch["actions"].unsqueeze(1)).squeeze()

                # Compute Q-targets (vanilla DQN vs Double DQN)
                with torch.no_grad():
                    if self.use_double_dqn:
                        # Double DQN: Use online network for action selection, target network for evaluation
                        next_actions = self.q_network(batch["next_observations"]).argmax(1)
                        q_next = self.target_network(batch["next_observations"]).gather(1, next_actions.unsqueeze(1)).squeeze()
                    else:
                        # Vanilla DQN: Use target network for both selection and evaluation
                        q_next = self.target_network(batch["next_observations"]).max(1)[0]

                    q_target = batch["rewards"] + self.gamma * q_next * (~batch["dones"]).float()

                # TASK-005 Phase 1: Use configured loss function
                loss = self.loss_fn(q_pred, q_target)

                # Store training metrics
                with torch.no_grad():
                    self.last_td_error = (q_target - q_pred).abs().mean().item()
                    self.last_loss = loss.item()
                    self.last_q_values_mean = q_pred.mean().item()
                    self.last_training_step = self.total_steps

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=self.max_grad_norm)
                self.optimizer.step()

                # Periodically sync target network for stability
                self.training_step_counter += 1
                if self.training_step_counter % self.target_update_frequency == 0:
                    self.target_network.load_state_dict(self.q_network.state_dict())

                # Log network statistics to TensorBoard (every 100 training steps)
                if self.tb_logger is not None and self.total_steps % 100 == 0:
                    for name, param in self.q_network.named_parameters():
                        self.tb_logger.writer.add_histogram(f"Network/Weights/{name}", param.data, self.total_steps)
                        if param.grad is not None:
                            self.tb_logger.writer.add_histogram(f"Network/Gradients/{name}", param.grad, self.total_steps)

        # 10. Update current state
        self.current_obs = next_obs

        # Track episode steps
        self.episode_step_counts += 1

        # 11. Handle episode resets (for adaptive intrinsic annealing)
        if dones.any():
            reset_indices = torch.where(dones)[0]
            for idx in reset_indices:
                survival_time = int(self.episode_step_counts[idx].item())
                if self.is_recurrent:
                    self._store_episode_and_reset(idx)
                self._finalize_episode(idx, survival_time)

        # 12. Construct BatchedAgentState
        # Note: rewards from environment already contain full DAC composition including intrinsic.
        # We do NOT add intrinsic_rewards again to avoid double-counting.
        # intrinsic_rewards is kept for logging/tracking purposes only.
        total_rewards = rewards  # Already contains: extrinsic + intrinsic + shaping from DAC

        # 10. Construct and return batched agent state
        # Add Q-values to info for recording (clone to CPU to avoid GPU memory issues)
        info["q_values"] = [q_values[i].cpu().tolist() for i in range(self.num_agents)]

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
            info=info,  # Pass environment info (includes successful_interactions, q_values)
        )

        return state

    def build_telemetry_snapshot(self, episode_index: int | None = None) -> dict:
        """
        Construct JSON-safe telemetry snapshot for all agents.

        Args:
            episode_index: Optional episode index to include in payload.

        Returns:
            Dict with schema version, episode index, and per-agent telemetry.
        """
        agents = [self.runtime_registry.get_snapshot_for_agent(i).to_dict() for i in range(self.num_agents)]
        payload = {
            "schema_version": "1.0.0",
            "episode_index": int(episode_index) if episode_index is not None else None,
            "agents": agents,
        }
        return payload

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

    def get_checkpoint_state(self) -> dict:
        """
        Get complete checkpoint state for saving (P1.1 complete checkpointing).

        Returns comprehensive state dict including:
        - Version number
        - Q-network weights
        - Target network weights (if recurrent)
        - Optimizer state
        - Training counters
        - Replay buffer contents
        - Exploration strategy state
        - Curriculum state
        - Universe metadata (meter count, names) for validation (TASK-001)

        Returns:
            Complete checkpoint state dictionary
        """
        checkpoint = {
            "version": 2,  # Checkpoint format version
            "q_network": self.q_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
            "exploration_state": self.exploration.checkpoint_state(),
        }

        # Universe metadata for compatibility validation (TASK-001)
        # This allows detecting meter count mismatches when loading checkpoints
        bars_config = self.env.bars_config
        checkpoint["universe_metadata"] = {
            "meter_count": bars_config.meter_count,
            "meter_names": bars_config.meter_names,
            "version": bars_config.version,
            "obs_dim": self.env.observation_dim,
            "action_dim": self.action_dim,  # From environment action space (TASK-002B Phase 4.1)
        }

        # Target network (recurrent mode only)
        if self.target_network is not None:
            checkpoint["target_network"] = self.target_network.state_dict()
            checkpoint["training_step_counter"] = self.training_step_counter
        else:
            checkpoint["target_network"] = None
            checkpoint["training_step_counter"] = 0

        # Replay buffer
        checkpoint["replay_buffer"] = self.replay_buffer.serialize()

        return checkpoint

    def load_checkpoint_state(self, checkpoint: dict) -> None:
        """
        Restore population state from checkpoint (P1.1 complete checkpointing).

        Args:
            checkpoint: State dictionary from get_checkpoint_state()

        Raises:
            ValueError: If checkpoint universe metadata doesn't match current environment
        """
        # Validate universe compatibility
        if "universe_metadata" not in checkpoint:
            raise ValueError(
                "Checkpoint missing 'universe_metadata' field.\n"
                "This checkpoint format is no longer supported.\n"
                "Please retrain from scratch."
            )

        metadata = checkpoint["universe_metadata"]
        bars_config = self.env.bars_config
        current_meter_count = bars_config.meter_count

        # Validate meter count matches
        checkpoint_meter_count = metadata.get("meter_count")
        if checkpoint_meter_count != current_meter_count:
            raise ValueError(
                f"Checkpoint meter count mismatch: checkpoint has {checkpoint_meter_count} meters, "
                f"but current environment has {current_meter_count} meters. "
                f"Cannot load checkpoint trained on different universe configuration."
            )

        # Validate obs_dim matches (secondary check)
        checkpoint_obs_dim = metadata.get("obs_dim")
        current_obs_dim = self.env.observation_dim
        if checkpoint_obs_dim != current_obs_dim:
            import warnings

            warnings.warn(
                f"Checkpoint obs_dim mismatch: checkpoint has {checkpoint_obs_dim}, "
                f"current env has {current_obs_dim}. This may indicate grid size or "
                f"observability mode differences. Proceeding with caution.",
                UserWarning,
            )

        # Restore Q-network
        self.q_network.load_state_dict(checkpoint["q_network"])

        # Restore optimizer
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        # Restore training counters
        self.total_steps = checkpoint.get("total_steps", 0)

        # Restore target network (if exists)
        if "target_network" in checkpoint and checkpoint["target_network"] is not None:
            if self.target_network is not None:
                self.target_network.load_state_dict(checkpoint["target_network"])
                self.training_step_counter = checkpoint.get("training_step_counter", 0)

        # Restore replay buffer
        if "replay_buffer" in checkpoint:
            self.replay_buffer.load_from_serialized(checkpoint["replay_buffer"])

        # Restore exploration state
        if "exploration_state" in checkpoint:
            self.exploration.load_state(checkpoint["exploration_state"])
