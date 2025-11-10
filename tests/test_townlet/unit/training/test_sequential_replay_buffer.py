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
