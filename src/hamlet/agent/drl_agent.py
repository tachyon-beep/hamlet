"""
DRL agent implementation for Hamlet.

Implements Deep Q-Network (DQN) algorithm for learning survival behavior.
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from .base_agent import BaseAgent
from .base_algorithm import BaseAlgorithm
from .networks import QNetwork, DuelingQNetwork, SpatialQNetwork, SpatialDuelingQNetwork, RelationalQNetwork, RecurrentSpatialQNetwork
from .observation_utils import preprocess_observation


class DRLAgent(BaseAgent, BaseAlgorithm):
    """
    Deep Q-Network (DQN) agent.

    Learns to maximize survival time by balancing meter management.
    Uses experience replay and target network for stability.
    """

    def __init__(
        self,
        agent_id: str,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.00025,  # Reduced from 1e-3 for stability
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        device: str = None,
        network_type: str = "qnetwork",  # 'qnetwork', 'dueling', 'spatial', 'spatial_dueling'
        grid_size: int = 8,  # For spatial networks
    ):
        """
        Initialize DRL agent with selectable network architecture.

        Args:
            agent_id: Unique identifier
            state_dim: Observation space dimension
            action_dim: Number of possible actions
            learning_rate: Learning rate for optimizer (default 0.00025, Atari DQN standard)
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Epsilon decay per episode
            device: Device to run on ('cpu' or 'cuda', auto-detect if None)
            network_type: Network architecture ('qnetwork', 'dueling', 'spatial', 'spatial_dueling')
            grid_size: Grid size for spatial networks (default 8)
        """
        BaseAgent.__init__(self, agent_id)
        BaseAlgorithm.__init__(self, agent_id)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.network_type = network_type
        self.grid_size = grid_size
        self.is_recurrent = (network_type == "recurrent")

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Initialize networks based on type
        network_class = self._get_network_class(network_type)
        self.q_network = self._create_network(network_class, state_dim, action_dim, grid_size).to(self.device)
        self.target_network = self._create_network(network_class, state_dim, action_dim, grid_size).to(self.device)

        # Copy weights to target network
        self.update_target_network()

        # Initialize optimizer with gradient clipping
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.max_grad_norm = 10.0  # Gradient clipping threshold

    def _get_network_class(self, network_type: str):
        """Get network class based on type string."""
        network_map = {
            "qnetwork": QNetwork,
            "dueling": DuelingQNetwork,
            "spatial": SpatialQNetwork,
            "spatial_dueling": SpatialDuelingQNetwork,
            "relational": RelationalQNetwork,
            "recurrent": RecurrentSpatialQNetwork,
        }

        if network_type not in network_map:
            raise ValueError(
                f"Unknown network type: {network_type}. "
                f"Choose from: {list(network_map.keys())}"
            )

        return network_map[network_type]

    def _create_network(self, network_class, state_dim: int, action_dim: int, grid_size: int):
        """Create network instance with appropriate arguments."""
        if network_class == RecurrentSpatialQNetwork:
            # Recurrent network needs action_dim, window_size, num_meters
            # grid_size is the local window size (5x5 for Level 2)
            return network_class(action_dim=action_dim, window_size=grid_size, num_meters=8)
        elif network_class in [SpatialQNetwork, SpatialDuelingQNetwork, RelationalQNetwork]:
            # Spatial and relational networks need grid_size
            return network_class(state_dim, action_dim, grid_size)
        else:
            # Standard networks don't need grid_size
            return network_class(state_dim, action_dim)

    def select_action(self, observation, explore: bool = True):
        """
        Select action using epsilon-greedy policy with action masking.

        Args:
            observation: Current environment observation (dict or preprocessed array)
            explore: Whether to use epsilon-greedy exploration (False for greedy)

        Returns:
            Action index (int)
        """
        # Extract action mask if available
        action_mask = None
        if isinstance(observation, dict) and 'action_mask' in observation:
            action_mask = observation['action_mask']
            valid_actions = np.where(action_mask)[0]
        else:
            valid_actions = np.arange(self.action_dim)

        # Epsilon-greedy action selection (only from valid actions)
        if explore and np.random.random() < self.epsilon:
            return np.random.choice(valid_actions)

        # Greedy action from Q-network (only from valid actions)
        with torch.no_grad():
            if self.is_recurrent:
                # Recurrent networks expect observation dict with tensors
                if isinstance(observation, dict):
                    obs_dict = {
                        'grid': torch.FloatTensor(observation['grid']).to(self.device),
                        'position': torch.FloatTensor(observation['position']).to(self.device),
                        'meters': torch.FloatTensor(observation['meters']).to(self.device),
                    }
                else:
                    raise ValueError("Recurrent network requires observation dict, not preprocessed array")

                # Forward pass returns (q_values, hidden_state)
                q_values, new_hidden = self.q_network.forward(obs_dict)
                # Update hidden state for next step
                self.q_network.set_hidden_state(new_hidden)
            else:
                # Standard networks use preprocessed array
                if isinstance(observation, dict):
                    state = preprocess_observation(observation)
                else:
                    state = observation
                state_tensor = torch.FloatTensor(state).to(self.device)
                q_values = self.q_network(state_tensor)

            # Mask invalid actions by setting their Q-values to -inf
            if action_mask is not None:
                q_values_np = q_values.cpu().numpy()
                q_values_np[~action_mask] = -np.inf
                action = np.argmax(q_values_np)
            else:
                action = q_values.argmax().item()

        return action

    def learn(self, batch):
        """
        Learn from a batch of experiences using DQN algorithm.

        Args:
            batch: Tuple of (states, actions, rewards, next_states, dones)
                Each element is a list of experiences

        Note:
            For recurrent networks, this treats each transition independently
            (resets hidden state per transition). Temporal memory is only
            leveraged during episode rollouts, not during batch training.
            Full sequential training would require episode replay buffer.
        """
        states, actions, rewards, next_states, dones = batch

        if self.is_recurrent:
            # Recurrent networks: batch of observation dicts
            # Convert list of dicts to dict of batched tensors
            batch_size = len(states)

            # Extract and batch each component
            grids = np.array([s['grid'] for s in states])
            positions = np.array([s['position'] for s in states])
            meters = np.array([s['meters'] for s in states])

            next_grids = np.array([s['grid'] for s in next_states])
            next_positions = np.array([s['position'] for s in next_states])
            next_meters = np.array([s['meters'] for s in next_states])

            states_dict = {
                'grid': torch.FloatTensor(grids).to(self.device),
                'position': torch.FloatTensor(positions).to(self.device),
                'meters': torch.FloatTensor(meters).to(self.device),
            }

            next_states_dict = {
                'grid': torch.FloatTensor(next_grids).to(self.device),
                'position': torch.FloatTensor(next_positions).to(self.device),
                'meters': torch.FloatTensor(next_meters).to(self.device),
            }

            actions_tensor = torch.LongTensor(actions).to(self.device)
            rewards_tensor = torch.FloatTensor(rewards).to(self.device)
            dones_tensor = torch.FloatTensor(dones).to(self.device)

            # Reset hidden states for batch training (treat transitions independently)
            self.q_network.reset_hidden_state(batch_size=batch_size, device=self.device)
            self.target_network.reset_hidden_state(batch_size=batch_size, device=self.device)

            # Current Q-values
            current_q_values, _ = self.q_network(states_dict)
            current_q_values = current_q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

            # Target Q-values
            with torch.no_grad():
                next_q_values, _ = self.target_network(next_states_dict)
                max_next_q_values = next_q_values.max(1)[0]
                target_q_values = rewards_tensor + (1 - dones_tensor) * self.gamma * max_next_q_values

        else:
            # Standard networks: preprocessed arrays
            states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
            actions_tensor = torch.LongTensor(actions).to(self.device)
            rewards_tensor = torch.FloatTensor(rewards).to(self.device)
            next_states_tensor = torch.FloatTensor(np.array(next_states)).to(self.device)
            dones_tensor = torch.FloatTensor(dones).to(self.device)

            # Current Q-values
            current_q_values = self.q_network(states_tensor)
            current_q_values = current_q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

            # Target Q-values
            with torch.no_grad():
                next_q_values = self.target_network(next_states_tensor)
                max_next_q_values = next_q_values.max(1)[0]
                target_q_values = rewards_tensor + (1 - dones_tensor) * self.gamma * max_next_q_values

        # Compute loss (same for both network types)
        loss = F.mse_loss(current_q_values, target_q_values)

        # Optimize with gradient clipping
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """Copy weights from Q-network to target network."""
        network_class = self._get_network_class(self.network_type)
        new_target = self._create_network(
            network_class,
            self.state_dim,
            self.action_dim,
            self.grid_size,
        ).to(self.device)
        new_target.load_state_dict(self.q_network.state_dict())
        self.target_network = new_target

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def reset_episode(self):
        """
        Reset agent state for new episode.

        For recurrent networks, this resets the LSTM hidden state.
        For standard networks, this is a no-op.
        """
        if self.is_recurrent:
            self.q_network.reset_hidden_state(batch_size=1, device=self.device)
            # Note: target_network hidden state is reset during learn()

    def save(self, filepath: str):
        """
        Save agent networks and parameters.

        Args:
            filepath: Path to save checkpoint
        """
        checkpoint = {
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
        }
        torch.save(checkpoint, filepath)

    @staticmethod
    def detect_network_type(filepath: str) -> str:
        """
        Detect network architecture type from checkpoint file.

        Args:
            filepath: Path to checkpoint file

        Returns:
            Network type string ('qnetwork', 'dueling', 'spatial', 'spatial_dueling', 'relational', 'recurrent')
        """
        checkpoint = torch.load(filepath, map_location='cpu')
        q_network_keys = set(checkpoint["q_network"].keys())

        # Check for RecurrentSpatialQNetwork (LSTM-based)
        if "lstm.weight_ih_l0" in q_network_keys or "vision_encoder.0.weight" in q_network_keys:
            return "recurrent"

        # Check for RelationalQNetwork (attention-based)
        if "meter_embeddings.0.0.weight" in q_network_keys:
            return "relational"

        # Check for Spatial networks (CNN-based)
        if any("conv" in key for key in q_network_keys):
            # Check if it's dueling
            if "value_stream.0.weight" in q_network_keys:
                return "spatial_dueling"
            return "spatial"

        # Check for Dueling network
        if "value_stream.0.weight" in q_network_keys:
            return "dueling"

        # Default to basic QNetwork
        return "qnetwork"

    @staticmethod
    def detect_config(filepath: str) -> dict:
        """
        Detect agent configuration from checkpoint file.

        Args:
            filepath: Path to checkpoint file

        Returns:
            Dictionary with network_type, state_dim, action_dim
        """
        checkpoint = torch.load(filepath, map_location='cpu')

        # Detect network type
        network_type = DRLAgent.detect_network_type(filepath)

        # Get dimensions from checkpoint if available
        state_dim = checkpoint.get("state_dim", 72)  # Default to 72 (8x8 grid)
        action_dim = checkpoint.get("action_dim", 5)  # Default to 5

        return {
            "network_type": network_type,
            "state_dim": state_dim,
            "action_dim": action_dim
        }

    def load(self, filepath: str):
        """
        Load agent networks and parameters.

        Args:
            filepath: Path to load checkpoint from
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]

    def get_config(self):
        """Get DQN configuration."""
        config = super().get_config()
        config.update({
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "device": str(self.device),
            "network_type": self.network_type,
            "grid_size": self.grid_size,
        })
        return config
