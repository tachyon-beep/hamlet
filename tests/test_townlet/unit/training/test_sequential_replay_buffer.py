"""Unit tests for SequentialReplayBuffer edge cases."""

from __future__ import annotations

import pytest
import torch

from townlet.training.sequential_replay_buffer import SequentialReplayBuffer


def make_episode(length: int, *, with_split_rewards: bool = False) -> dict[str, torch.Tensor]:
    obs = torch.arange(length * 2, dtype=torch.float32).view(length, 2)
    actions = torch.arange(length, dtype=torch.long)
    dones = torch.zeros(length, dtype=torch.bool)
    dones[-1] = True

    episode: dict[str, torch.Tensor] = {
        "observations": obs,
        "actions": actions,
        "dones": dones,
    }

    if with_split_rewards:
        episode["rewards_extrinsic"] = torch.ones(length)
        episode["rewards_intrinsic"] = torch.full((length,), 0.5)
    else:
        episode["rewards"] = torch.linspace(0.0, 1.0, steps=length)

    return episode


class TestSequentialReplayBuffer:
    def test_store_episode_eviction(self):
        buffer = SequentialReplayBuffer(capacity=5, device=torch.device("cpu"))

        buffer.store_episode(make_episode(3))
        buffer.store_episode(make_episode(4))

        # Capacity 5 → first episode evicted (3 transitions) when second (4) inserted
        assert len(buffer) == 1
        assert buffer.num_transitions == 4

    def test_sample_sequences_masks_post_terminal(self, monkeypatch):
        buffer = SequentialReplayBuffer(capacity=20, device=torch.device("cpu"))
        buffer.store_episode(make_episode(6, with_split_rewards=True))

        # Force deterministic choice
        monkeypatch.setattr("random.choice", lambda seq: seq[0])
        monkeypatch.setattr("random.randint", lambda a, b: 0)

        batch = buffer.sample_sequences(batch_size=1, seq_len=4, intrinsic_weight=2.0)
        mask = batch["mask"][0]

        # Terminal at final timestep (index 3) → mask all True
        assert mask[-1]
        assert torch.equal(mask, torch.ones_like(mask))

        rewards = batch["rewards"][0]
        # Combined extrinsic + 2*intrinsic = 1 + 2*0.5 = 2
        assert torch.allclose(rewards, torch.full((4,), 2.0))

    def test_sample_sequences_requires_long_enough_episode(self):
        buffer = SequentialReplayBuffer(capacity=20, device=torch.device("cpu"))
        buffer.store_episode(make_episode(3))

        with pytest.raises(ValueError):  # type: ignore[name-defined]
            buffer.sample_sequences(batch_size=1, seq_len=5)

    def test_serialize_and_restore(self):
        buffer = SequentialReplayBuffer(capacity=10, device=torch.device("cpu"))
        buffer.store_episode(make_episode(4, with_split_rewards=True))

        state = buffer.serialize()

        restored = SequentialReplayBuffer(capacity=10, device=torch.device("cpu"))
        restored.load_from_serialized(state)

        assert len(restored) == 1
        batch = restored.sample_sequences(batch_size=1, seq_len=2)
        assert batch["observations"].shape == (1, 2, 2)


class TestSequentialReplayBufferClearAPI:
    """Test clear() method for sequential buffer."""

    def test_clear_resets_episodes(self):
        """clear() should remove all episodes."""
        buffer = SequentialReplayBuffer(capacity=100, device=torch.device("cpu"))

        buffer.store_episode(make_episode(5, with_split_rewards=True))
        buffer.store_episode(make_episode(7, with_split_rewards=True))

        assert len(buffer) == 2
        assert buffer.num_transitions == 12

        buffer.clear()

        assert len(buffer) == 0
        assert buffer.num_transitions == 0
        assert len(buffer.episodes) == 0

    def test_clear_idempotence(self):
        """Calling clear() multiple times should be safe."""
        buffer = SequentialReplayBuffer(capacity=100, device=torch.device("cpu"))

        # Clear empty buffer
        buffer.clear()
        assert len(buffer) == 0

        # Add episodes and clear
        buffer.store_episode(make_episode(5, with_split_rewards=True))
        buffer.clear()

        # Clear again
        buffer.clear()
        assert len(buffer) == 0
        assert buffer.num_transitions == 0

    def test_buffer_works_after_clear(self):
        """Buffer should work normally after clear()."""
        buffer = SequentialReplayBuffer(capacity=100, device=torch.device("cpu"))

        # Fill buffer
        buffer.store_episode(make_episode(10, with_split_rewards=True))
        buffer.clear()

        # Should be able to store episodes again
        buffer.store_episode(make_episode(5, with_split_rewards=True))
        assert len(buffer) == 1
        assert buffer.num_transitions == 5

        # Should be able to sample
        batch = buffer.sample_sequences(batch_size=1, seq_len=3)
        assert batch["observations"].shape == (1, 3, 2)


class TestSequentialReplayBufferStatsAPI:
    """Test stats() method for sequential buffer."""

    def test_stats_empty_buffer(self):
        """stats() should work on empty buffer."""
        buffer = SequentialReplayBuffer(capacity=100, device=torch.device("cpu"))

        stats = buffer.stats()

        assert stats["size"] == 0  # num_transitions
        assert stats["capacity"] == 100
        assert stats["occupancy_ratio"] == 0.0
        assert stats["memory_bytes"] == 0
        assert stats["device"] == "cpu"
        assert stats["num_episodes"] == 0
        assert stats["num_transitions"] == 0

    def test_stats_with_episodes(self):
        """stats() should report episode and transition counts."""
        buffer = SequentialReplayBuffer(capacity=100, device=torch.device("cpu"))

        buffer.store_episode(make_episode(5, with_split_rewards=True))
        buffer.store_episode(make_episode(7, with_split_rewards=True))
        buffer.store_episode(make_episode(3, with_split_rewards=True))

        stats = buffer.stats()

        assert stats["size"] == 15  # total transitions
        assert stats["capacity"] == 100
        assert stats["occupancy_ratio"] == 0.15
        assert stats["num_episodes"] == 3
        assert stats["num_transitions"] == 15
        assert stats["memory_bytes"] > 0

    def test_stats_memory_calculation(self):
        """stats() should calculate memory across all episodes."""
        buffer = SequentialReplayBuffer(capacity=100, device=torch.device("cpu"))

        # Empty buffer
        assert buffer.stats()["memory_bytes"] == 0

        # Add one episode (length 5, obs_dim=2)
        buffer.store_episode(make_episode(5, with_split_rewards=True))

        stats = buffer.stats()

        # Each episode has: observations [5,2], actions [5], rewards_extrinsic [5],
        # rewards_intrinsic [5], dones [5]
        expected_bytes = (
            5 * 2 * 4  # observations (float32)
            + 5 * 8  # actions (int64)
            + 5 * 4  # rewards_extrinsic (float32)
            + 5 * 4  # rewards_intrinsic (float32)
            + 5 * 1  # dones (bool)
        )

        assert stats["memory_bytes"] == expected_bytes

    def test_stats_after_clear(self):
        """stats() should show empty buffer after clear()."""
        buffer = SequentialReplayBuffer(capacity=100, device=torch.device("cpu"))

        buffer.store_episode(make_episode(10, with_split_rewards=True))
        buffer.clear()

        stats = buffer.stats()

        assert stats["size"] == 0
        assert stats["num_episodes"] == 0
        assert stats["num_transitions"] == 0
        assert stats["occupancy_ratio"] == 0.0
        assert stats["memory_bytes"] == 0
