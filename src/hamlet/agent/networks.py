"""
Neural network architectures for Hamlet agents.

Defines PyTorch networks for Q-value estimation.
"""

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """
    Q-Network for DQN agent.

    Maps observation states to Q-values for each action.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list = None):
        """
        Initialize Q-Network.

        Args:
            state_dim: Input observation dimension
            action_dim: Number of possible actions (output dimension)
            hidden_dims: List of hidden layer sizes (default: [128, 128])
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        if hidden_dims is None:
            hidden_dims = [128, 128]

        # Build fully-connected layers
        layers = []
        input_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        # Output layer (no activation - raw Q-values)
        layers.append(nn.Linear(input_dim, action_dim))

        self.layers = nn.Sequential(*layers)

    def forward(self, state):
        """
        Forward pass through network.

        Args:
            state: Input state tensor of shape (batch_size, state_dim) or (state_dim,)

        Returns:
            Q-values for each action of shape (batch_size, action_dim) or (action_dim,)
        """
        return self.layers(state)


class DuelingQNetwork(nn.Module):
    """
    Dueling Q-Network architecture.

    Separates value and advantage streams for better learning.
    Particularly effective for Hamlet where many states have similar value
    but different action advantages (e.g., high health states).
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list = None):
        """
        Initialize Dueling Q-Network.

        Args:
            state_dim: Input observation dimension
            action_dim: Number of possible actions
            hidden_dims: List of hidden layer sizes (default: [256, 256, 128])
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        if hidden_dims is None:
            hidden_dims = [256, 256, 128]

        # Shared feature extractor
        feature_layers = []
        input_dim = state_dim
        for i, hidden_dim in enumerate(hidden_dims):
            feature_layers.append(nn.Linear(input_dim, hidden_dim))
            feature_layers.append(nn.ReLU())
            input_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*feature_layers)
        final_feature_dim = hidden_dims[-1]

        # Value stream: V(s) - scalar value of being in state s
        self.value_stream = nn.Sequential(
            nn.Linear(final_feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # Advantage stream: A(s,a) - advantage of each action in state s
        self.advantage_stream = nn.Sequential(
            nn.Linear(final_feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, state):
        """
        Forward pass through dueling network.

        Args:
            state: Input state tensor of shape (batch_size, state_dim) or (state_dim,)

        Returns:
            Q-values computed as Q(s,a) = V(s) + (A(s,a) - mean(A(s)))

        The mean-centering ensures identifiability: given Q-values, we can
        uniquely recover V and A. This helps separate "state goodness" from
        "action preference".
        """
        features = self.feature_extractor(state)

        value = self.value_stream(features)
        advantages = self.advantage_stream(features)

        # Q(s,a) = V(s) + (A(s,a) - mean(A(s)))
        # Mean-centering makes the advantages relative
        q_values = value + (advantages - advantages.mean(dim=-1, keepdim=True))

        return q_values


class SpatialQNetwork(nn.Module):
    """
    Hybrid spatial Q-Network for grid-based environments.

    Uses CNN for spatial processing of grid and MLP for meter processing.
    Combines both streams for final Q-value prediction.

    Architecture:
    - Grid (8×8) → CNN → 64 features
    - Position + Meters (8 dims) → MLP → 32 features
    - Combined (96 dims) → MLP → Q-values
    """

    def __init__(self, state_dim: int, action_dim: int, grid_size: int = 8):
        """
        Initialize Spatial Q-Network.

        Args:
            state_dim: Total input dimension (should be 2 + 6 + grid_size^2)
            action_dim: Number of possible actions
            grid_size: Size of grid (default 8 for 8×8)
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.grid_size = grid_size

        # Verify state_dim matches expected
        expected_dim = 2 + 6 + (grid_size * grid_size)  # pos + 6 meters + grid
        assert state_dim == expected_dim, f"Expected state_dim={expected_dim}, got {state_dim}"

        # CNN for spatial grid processing
        # Input: 1×8×8 (single channel grid)
        # Output: 64 features
        self.grid_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # 1×8×8 → 16×8×8
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # 16×8×8 → 32×8×8
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  # 32×8×8 → 32×8×8
            nn.ReLU(),
            nn.Flatten(),  # 32×8×8 → 2048
            nn.Linear(32 * grid_size * grid_size, 64),
            nn.ReLU()
        )

        # MLP for position and meter processing
        # Input: 2 (position) + 6 (meters) = 8 dims
        # Output: 32 features
        self.meter_mlp = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # Combined network
        # Input: 64 (grid) + 32 (meters) = 96 features
        # Output: Q-values for each action
        self.combined = nn.Sequential(
            nn.Linear(64 + 32, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, state):
        """
        Forward pass through spatial network.

        Args:
            state: Input state tensor of shape (batch_size, state_dim) or (state_dim,)
                   Format: [pos_x, pos_y, energy, hygiene, satiation, money, mood, social, grid...]

        Returns:
            Q-values for each action of shape (batch_size, action_dim) or (action_dim,)
        """
        # Handle both batched and unbatched inputs
        is_batched = state.dim() == 2
        if not is_batched:
            state = state.unsqueeze(0)  # Add batch dimension

        batch_size = state.shape[0]

        # Split state into components
        # Position: [0:2], Meters: [2:8], Grid: [8:]
        position = state[:, 0:2]
        meters = state[:, 2:8]
        grid = state[:, 8:]

        # Process position + meters through MLP
        meter_features = torch.cat([position, meters], dim=1)
        meter_output = self.meter_mlp(meter_features)

        # Process grid through CNN
        # Reshape grid: (batch, 64) → (batch, 1, 8, 8)
        grid_2d = grid.view(batch_size, 1, self.grid_size, self.grid_size)
        grid_output = self.grid_cnn(grid_2d)

        # Combine both streams
        combined_features = torch.cat([grid_output, meter_output], dim=1)
        q_values = self.combined(combined_features)

        # Remove batch dimension if input was unbatched
        if not is_batched:
            q_values = q_values.squeeze(0)

        return q_values


class SpatialDuelingQNetwork(nn.Module):
    """
    Combines spatial processing (CNN) with dueling architecture.

    Best of both worlds: spatial understanding + value/advantage separation.
    This is the most advanced architecture for Hamlet's grid world.
    """

    def __init__(self, state_dim: int, action_dim: int, grid_size: int = 8):
        """
        Initialize Spatial Dueling Q-Network.

        Args:
            state_dim: Total input dimension
            action_dim: Number of possible actions
            grid_size: Size of grid (default 8)
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.grid_size = grid_size

        expected_dim = 2 + 6 + (grid_size * grid_size)
        assert state_dim == expected_dim, f"Expected state_dim={expected_dim}, got {state_dim}"

        # CNN for grid
        self.grid_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * grid_size * grid_size, 64),
            nn.ReLU()
        )

        # MLP for meters
        self.meter_mlp = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(64 + 32, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, state):
        """
        Forward pass through spatial dueling network.

        Args:
            state: Input state tensor

        Returns:
            Q-values computed as V(s) + (A(s,a) - mean(A(s)))
        """
        # Handle batching
        is_batched = state.dim() == 2
        if not is_batched:
            state = state.unsqueeze(0)

        batch_size = state.shape[0]

        # Split and process components
        position = state[:, 0:2]
        meters = state[:, 2:8]
        grid = state[:, 8:]

        meter_features = torch.cat([position, meters], dim=1)
        meter_output = self.meter_mlp(meter_features)

        grid_2d = grid.view(batch_size, 1, self.grid_size, self.grid_size)
        grid_output = self.grid_cnn(grid_2d)

        # Combine and extract features
        combined = torch.cat([grid_output, meter_output], dim=1)
        features = self.feature_extractor(combined)

        # Dueling streams
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)

        # Combine: Q(s,a) = V(s) + (A(s,a) - mean(A(s)))
        q_values = value + (advantages - advantages.mean(dim=-1, keepdim=True))

        if not is_batched:
            q_values = q_values.squeeze(0)

        return q_values


class RelationalQNetwork(nn.Module):
    """
    Relational Q-Network with attention for learning cross-meter dependencies.

    Designed specifically for Hamlet's complex indirect relationships:
    - Job payment depends on energy AND hygiene
    - Food choice depends on position (location-aware decisions)
    - Bar visit requires money buffer (economic planning)

    Architecture:
    - Meters → Embeddings → Multi-Head Attention → Relational features
    - Position + Grid → Spatial CNN → Spatial features
    - Combined → Dueling streams → Q-values

    The attention mechanism helps the network learn which meters affect
    which decisions without hardcoding these relationships.
    """

    def __init__(self, state_dim: int, action_dim: int, grid_size: int = 8, num_heads: int = 4):
        """
        Initialize Relational Q-Network.

        Args:
            state_dim: Total input dimension (should be 2 + 6 + grid_size^2)
            action_dim: Number of possible actions
            grid_size: Size of grid (default 8)
            num_heads: Number of attention heads (default 4)
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.grid_size = grid_size
        self.num_heads = num_heads

        expected_dim = 2 + 6 + (grid_size * grid_size)
        assert state_dim == expected_dim, f"Expected state_dim={expected_dim}, got {state_dim}"

        # Meter embedding dimension (must be divisible by num_heads)
        self.meter_embed_dim = 64  # 64 = 4 heads × 16 dims per head

        # Embed each meter into higher-dimensional space
        # 6 meters: energy, hygiene, satiation, money, mood, social
        self.meter_embeddings = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, 32),
                nn.ReLU(),
                nn.Linear(32, self.meter_embed_dim)
            )
            for _ in range(6)
        ])

        # Multi-head self-attention for meter relationships
        # This lets the network learn: "job payment depends on energy AND hygiene"
        self.meter_attention = nn.MultiheadAttention(
            embed_dim=self.meter_embed_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # Layer norm after attention
        self.attention_norm = nn.LayerNorm(self.meter_embed_dim)

        # Feed-forward after attention
        self.meter_ffn = nn.Sequential(
            nn.Linear(self.meter_embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # Position embedding
        self.position_embed = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )

        # CNN for grid
        self.grid_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * grid_size * grid_size, 64),
            nn.ReLU()
        )

        # Combined feature extractor
        # Input: 64 (meter_features) + 32 (position) + 64 (grid) = 160 dims
        combined_dim = 64 + 32 + 64
        self.feature_extractor = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        # Dueling streams
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, state):
        """
        Forward pass through relational network.

        Args:
            state: Input state tensor of shape (batch_size, state_dim) or (state_dim,)
                   Format: [pos_x, pos_y, energy, hygiene, satiation, money, mood, social, grid...]

        Returns:
            Q-values for each action
        """
        # Handle batching
        is_batched = state.dim() == 2
        if not is_batched:
            state = state.unsqueeze(0)

        batch_size = state.shape[0]

        # Split state: Position [0:2], Meters [2:8], Grid [8:]
        position = state[:, 0:2]
        meters = state[:, 2:8]
        grid = state[:, 8:]

        # === METER PROCESSING WITH ATTENTION ===
        # Embed each meter separately: [batch, 6, embed_dim]
        meter_embeds = []
        for i in range(6):
            meter_val = meters[:, i:i+1]  # [batch, 1]
            embed = self.meter_embeddings[i](meter_val)  # [batch, embed_dim]
            meter_embeds.append(embed)

        meter_embeds = torch.stack(meter_embeds, dim=1)  # [batch, 6, embed_dim]

        # Multi-head self-attention: learn meter relationships
        # Query: "Which meters should I pay attention to for this decision?"
        # Key/Value: All meter states
        attn_output, attn_weights = self.meter_attention(
            meter_embeds, meter_embeds, meter_embeds
        )  # [batch, 6, embed_dim]

        # Residual connection + layer norm
        meter_embeds = self.attention_norm(meter_embeds + attn_output)

        # Feed-forward network
        meter_features_per = self.meter_ffn(meter_embeds)  # [batch, 6, 64]

        # Pool across meters: mean pooling
        meter_features = meter_features_per.mean(dim=1)  # [batch, 64]

        # === SPATIAL PROCESSING ===
        # Position embedding
        position_features = self.position_embed(position)  # [batch, 32]

        # Grid CNN
        grid_2d = grid.view(batch_size, 1, self.grid_size, self.grid_size)
        grid_features = self.grid_cnn(grid_2d)  # [batch, 64]

        # === COMBINE ALL FEATURES ===
        combined = torch.cat([meter_features, position_features, grid_features], dim=1)
        features = self.feature_extractor(combined)

        # === DUELING STREAMS ===
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)

        # Q(s,a) = V(s) + (A(s,a) - mean(A(s)))
        q_values = value + (advantages - advantages.mean(dim=-1, keepdim=True))

        if not is_batched:
            q_values = q_values.squeeze(0)

        return q_values


class RecurrentSpatialQNetwork(nn.Module):
    """
    Recurrent Spatial Q-Network for partial observability (Level 2 POMDP).

    Architecture:
    - Vision Encoder: CNN for 5×5 local window → 128 features
    - Position Encoder: (x, y) → 32 features
    - Meter Encoder: 8 meters → 32 features
    - LSTM: Maintain memory across timesteps (256 hidden)
    - Q-Head: Combined features → 5 action Q-values

    Total params: ~600K

    Key design decisions:
    - Uses LSTM to build implicit spatial memory of environment
    - Absolute position helps agent understand where it's been
    - Recurrent state persists across episode to remember affordance locations
    """

    def __init__(self, action_dim: int = 5, window_size: int = 5, num_meters: int = 8):
        """
        Initialize Recurrent Spatial Q-Network.

        Args:
            action_dim: Number of possible actions (default 5)
            window_size: Size of local observation window (default 5 for 5×5)
            num_meters: Number of agent meters (default 8)
        """
        super().__init__()
        self.action_dim = action_dim
        self.window_size = window_size
        self.num_meters = num_meters

        # === VISION ENCODER (CNN for 5×5 window) ===
        self.vision_encoder = nn.Sequential(
            # Input: 1×5×5 (single channel grid)
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # 16×5×5
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # 32×5×5
            nn.ReLU(),
            nn.Flatten(),  # 32*5*5 = 800
            nn.Linear(800, 128),
            nn.ReLU(),
        )

        # === POSITION ENCODER ===
        self.position_encoder = nn.Sequential(
            nn.Linear(2, 32),  # (x, y) → 32 features
            nn.ReLU(),
        )

        # === METER ENCODER ===
        self.meter_encoder = nn.Sequential(
            nn.Linear(num_meters, 32),  # 8 meters → 32 features
            nn.ReLU(),
        )

        # === LSTM (maintains temporal memory) ===
        # Input: 128 (vision) + 32 (position) + 32 (meters) = 192 features
        self.lstm_input_dim = 128 + 32 + 32
        self.lstm_hidden_dim = 256
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_dim,
            hidden_size=self.lstm_hidden_dim,
            num_layers=1,
            batch_first=True
        )

        # === Q-HEAD (outputs Q-values for actions) ===
        self.q_head = nn.Sequential(
            nn.Linear(self.lstm_hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

        # LSTM hidden state (maintained across steps within episode)
        self.hidden_state = None

    def forward(self, obs, hidden=None):
        """
        Forward pass through recurrent network.

        Args:
            obs: Observation dict with 'grid' (5×5), 'position' (x,y), 'meters' (8 values)
            hidden: Optional LSTM hidden state tuple (h, c). If None, uses internal state.

        Returns:
            Tuple of (q_values, hidden_state)
            - q_values: shape (batch_size, action_dim) or (action_dim,)
            - hidden_state: tuple (h, c) for next timestep
        """
        # Handle both batched and unbatched inputs
        is_batched = len(obs['grid'].shape) == 4  # (batch, channel, h, w)

        if not is_batched:
            # Add batch and channel dimensions: (h, w) → (1, 1, h, w)
            grid = obs['grid'].unsqueeze(0).unsqueeze(0)
            position = obs['position'].unsqueeze(0)
            meters = obs['meters'].unsqueeze(0)
            batch_size = 1
        else:
            # Add channel dimension if needed: (batch, h, w) → (batch, 1, h, w)
            if len(obs['grid'].shape) == 3:
                grid = obs['grid'].unsqueeze(1)
            else:
                grid = obs['grid']
            position = obs['position']
            meters = obs['meters']
            batch_size = grid.shape[0]

        # === ENCODE FEATURES ===
        vision_features = self.vision_encoder(grid)  # (batch, 128)
        position_features = self.position_encoder(position)  # (batch, 32)
        meter_features = self.meter_encoder(meters)  # (batch, 32)

        # Combine all features
        combined = torch.cat([vision_features, position_features, meter_features], dim=1)  # (batch, 192)

        # Add sequence dimension for LSTM: (batch, seq_len=1, features)
        lstm_input = combined.unsqueeze(1)

        # === LSTM FORWARD ===
        if hidden is None:
            hidden = self.hidden_state

        lstm_output, new_hidden = self.lstm(lstm_input, hidden)  # (batch, 1, 256), ((1, batch, 256), (1, batch, 256))

        # Remove sequence dimension: (batch, 1, 256) → (batch, 256)
        lstm_output = lstm_output.squeeze(1)

        # === Q-VALUES ===
        q_values = self.q_head(lstm_output)  # (batch, action_dim)

        if not is_batched:
            q_values = q_values.squeeze(0)  # (action_dim,)

        return q_values, new_hidden

    def reset_hidden_state(self, batch_size=1, device='cpu'):
        """
        Reset LSTM hidden state (call at episode start).

        Args:
            batch_size: Batch size for hidden state
            device: Device to create tensors on
        """
        self.hidden_state = (
            torch.zeros(1, batch_size, self.lstm_hidden_dim, device=device),
            torch.zeros(1, batch_size, self.lstm_hidden_dim, device=device)
        )

    def get_hidden_state(self):
        """Get current LSTM hidden state."""
        return self.hidden_state

    def set_hidden_state(self, hidden):
        """Set LSTM hidden state."""
        self.hidden_state = hidden
