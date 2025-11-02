"""
Sequential Replay Buffer for LSTM Training.

Unlike standard replay buffers that sample individual transitions,
this buffer stores complete episodes and samples sequences of consecutive
transitions to maintain temporal structure for recurrent networks.
"""

import torch


class SequentialReplayBuffer:
    """
    Replay buffer that maintains temporal structure for LSTM training.

    Stores complete episodes and samples sequences of consecutive transitions.
    This is essential for training recurrent networks which need temporal context.

    Attributes:
        capacity: Maximum number of transitions to store
        device: Device to store tensors on
        episodes: List of stored episodes (each is a dict of tensors)
        num_transitions: Total number of transitions stored
    """

    def __init__(self, capacity: int, device: torch.device):
        """
        Initialize sequential replay buffer.

        Args:
            capacity: Maximum number of transitions to store
            device: Device to store tensors on (CPU or CUDA)
        """
        self.capacity = capacity
        self.device = device
        self.episodes = []
        self.num_transitions = 0

    def __len__(self) -> int:
        """Return number of episodes stored."""
        return len(self.episodes)

    def store_episode(self, episode: dict[str, torch.Tensor]) -> None:
        """
        Store a complete episode.

        Args:
            episode: Dict with keys:
                - 'observations': [seq_len, obs_dim]
                - 'actions': [seq_len]
                - 'rewards' OR ('rewards_extrinsic' AND 'rewards_intrinsic'): [seq_len]
                - 'dones': [seq_len]

        Raises:
            ValueError: If episode structure is invalid
        """
        # Validate episode structure
        required_keys = {"observations", "actions", "dones"}

        missing_keys = required_keys - set(episode.keys())
        if missing_keys:
            raise ValueError(f"Missing required keys: {missing_keys}")

        # Check for reward keys (either 'rewards' or both extrinsic/intrinsic)
        has_rewards = "rewards" in episode
        has_split_rewards = "rewards_extrinsic" in episode and "rewards_intrinsic" in episode

        if not has_rewards and not has_split_rewards:
            raise ValueError("Episode must have 'rewards' or both 'rewards_extrinsic' and 'rewards_intrinsic'")

        # Validate all tensors have same length
        seq_len = len(episode["observations"])
        for key, tensor in episode.items():
            if len(tensor) != seq_len:
                raise ValueError(f"Episode tensor length mismatch: observations has {seq_len} steps, but {key} has {len(tensor)} steps")

        # Move episode to correct device
        episode_on_device = {key: tensor.to(self.device) for key, tensor in episode.items()}

        # Add episode
        self.episodes.append(episode_on_device)
        self.num_transitions += seq_len

        # Evict oldest episodes if over capacity
        while self.num_transitions > self.capacity and len(self.episodes) > 0:
            oldest_episode = self.episodes.pop(0)
            self.num_transitions -= len(oldest_episode["observations"])

    def sample_sequences(self, batch_size: int, seq_len: int, intrinsic_weight: float = 1.0) -> dict[str, torch.Tensor]:
        """
        Sample a batch of sequential transitions.

        Args:
            batch_size: Number of sequences to sample
            seq_len: Length of each sequence
            intrinsic_weight: Weight for intrinsic rewards (if using dual rewards)

        Returns:
            Dict with keys:
                - 'observations': [batch_size, seq_len, obs_dim]
                - 'actions': [batch_size, seq_len]
                - 'rewards': [batch_size, seq_len]
                - 'dones': [batch_size, seq_len]
                - 'mask': [batch_size, seq_len] bool - True for valid timesteps,
                          False after terminal (for post-terminal masking in loss)

        Raises:
            ValueError: If not enough data to sample
        """
        # Check if we have enough data
        if len(self.episodes) == 0:
            raise ValueError("Cannot sample: buffer is empty (not enough data)")

        # Find episodes long enough for the requested sequence length
        valid_episodes = [(i, ep) for i, ep in enumerate(self.episodes) if len(ep["observations"]) >= seq_len]

        if len(valid_episodes) == 0:
            raise ValueError(f"Cannot sample: no episodes long enough for seq_len={seq_len} (not enough data)")

        # Sample batch_size sequences
        sampled_sequences = []

        for _ in range(batch_size):
            # Randomly select an episode
            ep_idx, episode = valid_episodes[torch.randint(len(valid_episodes), (1,)).item()]

            # Randomly select a starting position (ensuring we can get seq_len transitions)
            ep_len = len(episode["observations"])
            start_idx = torch.randint(0, ep_len - seq_len + 1, (1,)).item()
            end_idx = start_idx + seq_len

            # Extract sequence
            sequence = {
                "observations": episode["observations"][start_idx:end_idx],
                "actions": episode["actions"][start_idx:end_idx],
                "dones": episode["dones"][start_idx:end_idx],
            }

            # Handle rewards (combine if using dual rewards)
            if "rewards" in episode:
                sequence["rewards"] = episode["rewards"][start_idx:end_idx]
            else:
                # Combine extrinsic and intrinsic rewards
                extrinsic = episode["rewards_extrinsic"][start_idx:end_idx]
                intrinsic = episode["rewards_intrinsic"][start_idx:end_idx]
                sequence["rewards"] = extrinsic + intrinsic * intrinsic_weight

            # Create validity mask (P2.2: Post-terminal masking)
            # Mask is True up to and including terminal, False after
            dones_seq = episode["dones"][start_idx:end_idx]
            mask = torch.ones(seq_len, dtype=torch.bool, device=self.device)

            # Find first terminal in sequence
            terminal_indices = torch.where(dones_seq)[0]
            if len(terminal_indices) > 0:
                terminal_idx = terminal_indices[0].item()
                # Mask out everything AFTER terminal (terminal itself is valid)
                if terminal_idx < seq_len - 1:
                    mask[terminal_idx + 1 :] = False

            sequence["mask"] = mask

            sampled_sequences.append(sequence)

        # Stack sequences into batch
        batch = {
            "observations": torch.stack([s["observations"] for s in sampled_sequences]),
            "actions": torch.stack([s["actions"] for s in sampled_sequences]),
            "rewards": torch.stack([s["rewards"] for s in sampled_sequences]),
            "dones": torch.stack([s["dones"] for s in sampled_sequences]),
            "mask": torch.stack([s["mask"] for s in sampled_sequences]),
        }

        return batch

    def serialize(self) -> dict:
        """
        Serialize episode buffer for checkpointing (P1.1).

        Returns:
            Dictionary with all episodes on CPU for saving
        """
        if len(self.episodes) == 0:
            return {
                "num_transitions": 0,
                "episodes": [],
                "capacity": self.capacity,
            }

        # Convert episodes to CPU tensors
        serialized_episodes = []
        for episode in self.episodes:
            serialized_episodes.append(
                {
                    "observations": episode["observations"].cpu(),
                    "actions": episode["actions"].cpu(),
                    "rewards_extrinsic": episode["rewards_extrinsic"].cpu(),
                    "rewards_intrinsic": episode["rewards_intrinsic"].cpu(),
                    "dones": episode["dones"].cpu(),
                }
            )

        return {
            "num_transitions": self.num_transitions,
            "episodes": serialized_episodes,
            "capacity": self.capacity,
        }

    def load_from_serialized(self, state: dict) -> None:
        """
        Restore episode buffer from serialized state (P1.1).

        Args:
            state: Dictionary from serialize()
        """
        self.num_transitions = state["num_transitions"]
        self.episodes = []

        # Restore episodes to device
        for ep_state in state["episodes"]:
            self.episodes.append(
                {
                    "observations": ep_state["observations"].to(self.device),
                    "actions": ep_state["actions"].to(self.device),
                    "rewards_extrinsic": ep_state["rewards_extrinsic"].to(self.device),
                    "rewards_intrinsic": ep_state["rewards_intrinsic"].to(self.device),
                    "dones": ep_state["dones"].to(self.device),
                }
            )
