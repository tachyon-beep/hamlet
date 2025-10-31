"""Neural network architectures for townlet agents."""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class SimpleQNetwork(nn.Module):
    """Simple MLP Q-network."""

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
        """Forward pass.

        Args:
            x: [batch, obs_dim] observations

        Returns:
            q_values: [batch, action_dim]
        """
        return self.net(x)


class RecurrentSpatialQNetwork(nn.Module):
    """
    Recurrent Spatial Q-Network for partial observability (Level 2 POMDP).

    Architecture:
    - Vision Encoder: CNN for local window → 128 features
    - Position Encoder: (x, y) → 32 features
    - Meter Encoder: 8 meters → 32 features
    - Affordance Encoder: 15 affordance types → 32 features
    - LSTM: 224 input → 256 hidden
    - Q-Head: 256 → 128 → action_dim

    Handles partial observations:
    - Grid: [batch, window_size²] flattened local window (25 for 5×5)
    - Position: [batch, 2] normalized (x, y)
    - Meters: [batch, 8] normalized meter values
    - Affordance: [batch, 15] one-hot affordance type (14 types + "none")
    """

    def __init__(
        self,
        action_dim: int = 5,
        window_size: int = 5,
        num_meters: int = 8,
        num_affordance_types: int = 15,
        enable_temporal_features: bool = False,
        hidden_dim: int = 256,
    ):
        """
        Initialize recurrent spatial Q-network.

        Args:
            action_dim: Number of actions
            window_size: Size of local vision window (5 for 5×5)
            num_meters: Number of meter values (8)
            num_affordance_types: Number of affordance types (15 by default)
            enable_temporal_features: Whether to expect temporal features (time_of_day, interaction_progress)
            hidden_dim: LSTM hidden dimension (256)
        """
        super().__init__()
        self.action_dim = action_dim
        self.window_size = window_size
        self.num_meters = num_meters
        self.num_affordance_types = num_affordance_types
        self.enable_temporal_features = enable_temporal_features
        self.hidden_dim = hidden_dim

        # Calculate affordance encoding dimension (types + 1 for "none")
        self.num_affordance_dims = num_affordance_types + 1

        # Vision Encoder: CNN for local window → 128 features
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # 16×window_size×window_size
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # 32×window_size×window_size
            nn.ReLU(),
            nn.Flatten(),  # 32 * window_size * window_size
            nn.Linear(32 * window_size * window_size, 128),
            nn.ReLU(),
        )

        # Position Encoder: (x, y) → 32 features
        self.position_encoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
        )

        # Meter Encoder: 8 meters → 32 features
        self.meter_encoder = nn.Sequential(
            nn.Linear(num_meters, 32),
            nn.ReLU(),
        )

        # Affordance Encoder: dynamic size based on num_affordance_dims
        self.affordance_encoder = nn.Sequential(
            nn.Linear(self.num_affordance_dims, 32),
            nn.ReLU(),
        )

        # LSTM: 224 input (128 + 32 + 32 + 32) → hidden_dim
        self.lstm_input_dim = 128 + 32 + 32 + 32
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True
        )

        # Q-Head: hidden_dim → 128 → action_dim
        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.ReLU(), nn.Linear(128, action_dim)
        )

        # Hidden state (initialized per episode)
        self.hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def forward(
        self, obs: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with LSTM memory.

        Args:
            obs: [batch, obs_dim] observations where:
                - obs[:, :window_size²] = local grid
                - obs[:, window_size²:window_size²+2] = position (2)
                - obs[:, window_size²+2:window_size²+2+num_meters] = meters
                - obs[:, window_size²+2+num_meters:window_size²+2+num_meters+num_affordance_dims] = affordance
                - obs[:, window_size²+2+num_meters+num_affordance_dims:] = temporal (if enabled)
            hidden: Optional LSTM hidden state (h, c), each [1, batch, hidden_dim]

        Returns:
            q_values: [batch, action_dim]
            new_hidden: Tuple of (h, c) hidden states
        """
        batch_size = obs.shape[0]

        # Split observation components with dynamic indices
        grid_size_flat = self.window_size * self.window_size
        idx = 0

        # Extract grid
        grid = obs[:, idx : idx + grid_size_flat]
        idx += grid_size_flat

        # Extract position
        position = obs[:, idx : idx + 2]
        idx += 2

        # Extract meters
        meters = obs[:, idx : idx + self.num_meters]
        idx += self.num_meters

        # Extract affordance
        affordance = obs[:, idx : idx + self.num_affordance_dims]
        idx += self.num_affordance_dims

        # Temporal features are ignored (we encode spatial + meter state)
        # If enable_temporal_features is True, there will be 2 extra dims at the end
        # but we don't need to process them separately for now

        # Reshape grid for CNN: [batch, 1, window_size, window_size]
        grid_2d = grid.view(batch_size, 1, self.window_size, self.window_size)

        # Encode components
        vision_features = self.vision_encoder(grid_2d)  # [batch, 128]
        position_features = self.position_encoder(position)  # [batch, 32]
        meter_features = self.meter_encoder(meters)  # [batch, 32]
        affordance_features = self.affordance_encoder(affordance)  # [batch, 32]

        # Concatenate features
        combined = torch.cat(
            [vision_features, position_features, meter_features, affordance_features], dim=1
        )  # [batch, 224]

        # LSTM expects [batch, seq_len, input_dim]
        combined = combined.unsqueeze(1)  # [batch, 1, 224]

        # Use provided hidden state or self.hidden_state
        if hidden is None:
            hidden = self.hidden_state

        # If still None, initialize with zeros
        if hidden is None:
            h = torch.zeros(1, batch_size, self.hidden_dim, device=obs.device)
            c = torch.zeros(1, batch_size, self.hidden_dim, device=obs.device)
            hidden = (h, c)

        # LSTM forward
        lstm_out, new_hidden = self.lstm(combined, hidden)  # lstm_out: [batch, 1, hidden_dim]
        lstm_out = lstm_out.squeeze(1)  # [batch, hidden_dim]

        # Q-values
        q_values = self.q_head(lstm_out)  # [batch, action_dim]

        return q_values, new_hidden

    def reset_hidden_state(self, batch_size: int = 1, device: torch.device = None) -> None:
        """
        Reset LSTM hidden state (call at episode start).

        Args:
            batch_size: Batch size for hidden state
            device: Device for tensors
        """
        if device is None:
            device = torch.device("cpu")

        h = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        c = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        self.hidden_state = (h, c)

    def set_hidden_state(self, hidden: Tuple[torch.Tensor, torch.Tensor]) -> None:
        """
        Set LSTM hidden state (for episode rollouts).

        Args:
            hidden: Tuple of (h, c) hidden states
        """
        self.hidden_state = hidden

    def get_hidden_state(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Get current LSTM hidden state."""
        return self.hidden_state
