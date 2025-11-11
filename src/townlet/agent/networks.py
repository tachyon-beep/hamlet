"""Neural network architectures for townlet agents."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from townlet.universe.dto import ObservationActivity


class SimpleQNetwork(nn.Module):
    """Simple MLP Q-network with LayerNorm."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int):
        """
        Initialize simple MLP Q-network.

        Args:
            obs_dim: Observation dimension
            action_dim: Number of actions
            hidden_dim: Hidden layer dimension (typically 128-256)

        Note (PDR-002):
            All network architecture parameters must be explicitly specified.
            No BAC (BRAIN_AS_CODE) defaults allowed.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
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
        return cast(torch.Tensor, self.net(x))


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
        action_dim: int,
        window_size: int,
        position_dim: int,
        num_meters: int,
        num_affordance_types: int,
        enable_temporal_features: bool,
        hidden_dim: int,
    ):
        """
        Initialize recurrent spatial Q-network.

        Args:
            action_dim: Number of actions
            window_size: Size of local vision window (5 for 5×5)
            position_dim: Dimensionality of position (2 for Grid2D, 3 for Grid3D, 0 for Aspatial)
            num_meters: Number of meter values
            num_affordance_types: Number of affordance types
            enable_temporal_features: Whether to expect temporal features
            hidden_dim: LSTM hidden dimension (typically 256)

        Note (PDR-002):
            All network architecture parameters must be explicitly specified.
            No BAC (BRAIN_AS_CODE) defaults allowed.

        Future (BRAIN_AS_CODE):
            These parameters should come from network config YAML.
        """
        super().__init__()
        self.action_dim = action_dim
        self.window_size = window_size
        self.position_dim = position_dim
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
            nn.LayerNorm(128),
            nn.ReLU(),
        )

        # Position Encoder: position_dim → 32 features (conditional on position_dim > 0)
        self.position_encoder: nn.Sequential | None
        if position_dim > 0:
            self.position_encoder = nn.Sequential(
                nn.Linear(position_dim, 32),
                nn.ReLU(),
            )
            position_features = 32
        else:
            # Aspatial: no position encoding
            self.position_encoder = None
            position_features = 0

        # Meter Encoder: num_meters → 32 features
        self.meter_encoder = nn.Sequential(
            nn.Linear(num_meters, 32),
            nn.ReLU(),
        )

        # Affordance Encoder: dynamic size based on num_affordance_dims
        self.affordance_encoder = nn.Sequential(
            nn.Linear(self.num_affordance_dims, 32),
            nn.ReLU(),
        )

        # LSTM: variable input → hidden_dim
        # Input size: 128 (vision) + position_features (0 or 32) + 32 (meters) + 32 (affordance)
        self.lstm_input_dim = 128 + position_features + 32 + 32
        self.lstm = nn.LSTM(input_size=self.lstm_input_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)

        # LayerNorm for LSTM output
        self.lstm_norm = nn.LayerNorm(hidden_dim)

        # Q-Head: hidden_dim → 128 → action_dim
        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

        # Hidden state (initialized per episode)
        self.hidden_state: tuple[torch.Tensor, torch.Tensor] | None = None

    def forward(
        self, obs: torch.Tensor, hidden: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with LSTM memory.

        Args:
            obs: [batch, obs_dim] observations where:
                - obs[:, :window_size²] = local grid
                - obs[:, window_size²:window_size²+position_dim] = position (position_dim)
                - obs[:, window_size²+position_dim:window_size²+position_dim+num_meters] = meters
                - obs[:, window_size²+position_dim+num_meters:window_size²+position_dim+num_meters+num_affordance_dims] = affordance
                - obs[:, window_size²+position_dim+num_meters+num_affordance_dims:] = temporal (if enabled)
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

        # Extract position (if position_dim > 0)
        if self.position_dim > 0:
            position = obs[:, idx : idx + self.position_dim]
            idx += self.position_dim
        else:
            position = None

        # Extract meters
        meters = obs[:, idx : idx + self.num_meters]
        idx += self.num_meters

        # Extract affordance
        affordance = obs[:, idx : idx + self.num_affordance_dims]
        idx += self.num_affordance_dims

        # Temporal features are ignored (we encode spatial + meter state)
        # If enable_temporal_features is True, there will be 3 extra dims (sin(time), cos(time), progress)
        # but we don't need to process them separately for now

        # Reshape grid for CNN: [batch, 1, window_size, window_size]
        grid_2d = grid.view(batch_size, 1, self.window_size, self.window_size)

        # Encode components
        vision_features = self.vision_encoder(grid_2d)  # [batch, 128]

        if self.position_encoder is not None:
            position_features = self.position_encoder(position)  # [batch, 32]
        else:
            # Aspatial: no position features
            position_features = None

        meter_features = self.meter_encoder(meters)  # [batch, 32]
        affordance_features = self.affordance_encoder(affordance)  # [batch, 32]

        # Concatenate features (conditionally include position)
        if position_features is not None:
            combined = torch.cat([vision_features, position_features, meter_features, affordance_features], dim=1)
        else:
            combined = torch.cat([vision_features, meter_features, affordance_features], dim=1)

        # LSTM expects [batch, seq_len, input_dim]
        combined = combined.unsqueeze(1)  # [batch, 1, lstm_input_dim]

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

        # Apply LayerNorm to LSTM output
        lstm_out = self.lstm_norm(lstm_out)  # [batch, hidden_dim]

        # Q-values
        q_values = self.q_head(lstm_out)  # [batch, action_dim]

        return q_values, new_hidden

    def reset_hidden_state(self, batch_size: int, device: torch.device | None = None) -> None:
        """
        Reset LSTM hidden state (call at episode start).

        Args:
            batch_size: Batch size for hidden state
            device: Device for tensors (default: cpu). Infrastructure default - PDR-002 exemption.
        """
        if device is None:
            device = torch.device("cpu")

        h = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        c = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        self.hidden_state = (h, c)

    def set_hidden_state(self, hidden: tuple[torch.Tensor, torch.Tensor]) -> None:
        """
        Set LSTM hidden state (for episode rollouts).

        Args:
            hidden: Tuple of (h, c) hidden states
        """
        self.hidden_state = hidden

    def get_hidden_state(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Get current LSTM hidden state."""
        return self.hidden_state


class StructuredQNetwork(nn.Module):
    """
    Structured Q-Network with group encoders for semantic observation groups.

    Uses ObservationActivity to identify semantic groups (spatial, bars, affordances, temporal, custom)
    and processes each group with its own encoder MLP before combining for Q-value prediction.

    Architecture:
    - Group Encoders: Each semantic group → embedding_dim features (default 32)
    - Concatenation: All group embeddings → combined_dim
    - Q-Head: combined_dim → hidden_dim → action_dim

    This architecture leverages observation structure for better inductive bias compared to
    SimpleQNetwork which treats all observation dimensions uniformly.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        observation_activity: ObservationActivity,
        group_embed_dim: int = 32,
        q_head_hidden_dim: int = 128,
    ):
        """
        Initialize structured Q-network with group encoders.

        Args:
            obs_dim: Total observation dimension
            action_dim: Number of actions
            observation_activity: ObservationActivity with group_slices
            group_embed_dim: Embedding dimension for each group encoder (default 32)
            q_head_hidden_dim: Hidden dimension for final Q-head MLP (default 128)

        Note (PDR-002):
            Architecture parameters explicitly specified, no BAC defaults.
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.observation_activity = observation_activity
        self.group_embed_dim = group_embed_dim

        # Create encoder for each semantic group
        self.group_encoders = nn.ModuleDict()
        total_embed_dim = 0

        for group_name, group_slice in observation_activity.group_slices.items():
            group_size = group_slice.stop - group_slice.start

            # Skip empty groups
            if group_size <= 0:
                continue

            # Create encoder: group_size → group_embed_dim
            encoder = nn.Sequential(
                nn.Linear(group_size, group_embed_dim),
                nn.LayerNorm(group_embed_dim),
                nn.ReLU(),
            )
            self.group_encoders[group_name] = encoder
            total_embed_dim += group_embed_dim

        # Q-head: combined embeddings → hidden → action_dim
        self.q_head = nn.Sequential(
            nn.Linear(total_embed_dim, q_head_hidden_dim),
            nn.LayerNorm(q_head_hidden_dim),
            nn.ReLU(),
            nn.Linear(q_head_hidden_dim, action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with structured group processing.

        Args:
            obs: [batch, obs_dim] observations

        Returns:
            q_values: [batch, action_dim]
        """
        # Extract and encode each group
        group_embeddings = []

        for group_name, encoder in self.group_encoders.items():
            group_slice = self.observation_activity.group_slices[group_name]
            group_obs = obs[:, group_slice]
            group_embed = encoder(group_obs)
            group_embeddings.append(group_embed)

        # Concatenate all group embeddings
        combined = torch.cat(group_embeddings, dim=1)  # [batch, total_embed_dim]

        # Q-values
        q_values = self.q_head(combined)  # [batch, action_dim]

        return cast(torch.Tensor, q_values)
