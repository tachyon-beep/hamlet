"""
Tests for post-terminal masking in SequentialReplayBuffer (P2.2).

Ensures that sequences sampled from the buffer include a mask that marks
valid timesteps (before terminal) vs invalid timesteps (after terminal).
This prevents gradients from flowing through post-terminal garbage data.
"""

import pytest
import torch

from townlet.training.sequential_replay_buffer import SequentialReplayBuffer


@pytest.fixture
def device():
    """PyTorch device for testing."""
    return torch.device("cpu")


@pytest.fixture
def buffer(device):
    """Empty sequential replay buffer."""
    return SequentialReplayBuffer(capacity=10000, device=device)


class TestPostTerminalMasking:
    """Test that sampled sequences include validity masks."""

    def test_sample_returns_mask(self, buffer, device):
        """Sample should return a mask field."""
        # Store episode with terminal in middle
        episode = {
            "observations": torch.randn(10, 72, device=device),
            "actions": torch.randint(0, 6, (10,), device=device),
            "rewards_extrinsic": torch.randn(10, device=device),
            "rewards_intrinsic": torch.randn(10, device=device),
            "dones": torch.tensor([False] * 5 + [True] + [False] * 4, device=device),
        }
        buffer.store_episode(episode)

        batch = buffer.sample_sequences(batch_size=2, seq_len=5)

        assert "mask" in batch, "Sample should return mask field"
        assert batch["mask"].shape == (2, 5), f"Mask shape should be [batch, seq_len], got {batch['mask'].shape}"
        assert batch["mask"].dtype == torch.bool, f"Mask should be bool, got {batch['mask'].dtype}"

    def test_mask_all_true_when_no_terminal(self, buffer, device):
        """Mask should be all True when sequence has no terminal."""
        # Episode with no terminal
        episode = {
            "observations": torch.randn(10, 72, device=device),
            "actions": torch.randint(0, 6, (10,), device=device),
            "rewards_extrinsic": torch.randn(10, device=device),
            "rewards_intrinsic": torch.randn(10, device=device),
            "dones": torch.zeros(10, dtype=torch.bool, device=device),
        }
        buffer.store_episode(episode)

        batch = buffer.sample_sequences(batch_size=4, seq_len=5)

        # All masks should be True (no terminals)
        assert torch.all(batch["mask"]), "Mask should be all True when no terminal in sequence"

    def test_mask_false_after_terminal(self, buffer, device):
        """Mask should be False after terminal state."""
        # Episode: terminal at index 3
        episode = {
            "observations": torch.randn(10, 72, device=device),
            "actions": torch.randint(0, 6, (10,), device=device),
            "rewards_extrinsic": torch.randn(10, device=device),
            "rewards_intrinsic": torch.randn(10, device=device),
            "dones": torch.tensor([False, False, False, True, False, False, False, False, False, False], device=device),
        }
        buffer.store_episode(episode)

        # Sample sequence starting at index 0 (length 7)
        # Should have terminal at index 3
        batch = buffer.sample_sequences(batch_size=1, seq_len=7)

        mask = batch["mask"][0]  # First (only) sequence
        dones = batch["dones"][0]

        # Find terminal in sampled sequence
        if torch.any(dones):
            terminal_idx = torch.where(dones)[0][0].item()

            # Mask should be True up to and including terminal
            assert torch.all(mask[: terminal_idx + 1]), f"Mask should be True before terminal at {terminal_idx}"

            # Mask should be False after terminal
            if terminal_idx < len(mask) - 1:
                assert torch.all(~mask[terminal_idx + 1 :]), f"Mask should be False after terminal at {terminal_idx}"

    def test_mask_includes_terminal_timestep(self, buffer, device):
        """Mask should include the terminal timestep itself (True at terminal)."""
        # Episode with terminal at index 2
        episode = {
            "observations": torch.randn(5, 72, device=device),
            "actions": torch.randint(0, 6, (5,), device=device),
            "rewards_extrinsic": torch.randn(5, device=device),
            "rewards_intrinsic": torch.randn(5, device=device),
            "dones": torch.tensor([False, False, True, False, False], device=device),
        }
        buffer.store_episode(episode)

        # Sample full episode
        batch = buffer.sample_sequences(batch_size=1, seq_len=5)

        mask = batch["mask"][0]
        dones = batch["dones"][0]

        terminal_idx = torch.where(dones)[0][0].item()

        # Terminal timestep itself should be valid (True)
        assert mask[terminal_idx], f"Mask should be True at terminal timestep {terminal_idx}"

        # Only timesteps AFTER terminal should be False
        assert torch.all(~mask[terminal_idx + 1 :]), "Only timesteps after terminal should be False"

    def test_mask_with_multiple_sequences(self, buffer, device):
        """Each sequence in batch should have its own mask based on its terminal."""
        # Store multiple episodes with different terminal positions
        episodes = [
            {  # Terminal at index 2
                "observations": torch.randn(10, 72, device=device),
                "actions": torch.randint(0, 6, (10,), device=device),
                "rewards_extrinsic": torch.randn(10, device=device),
                "rewards_intrinsic": torch.randn(10, device=device),
                "dones": torch.tensor([False, False, True] + [False] * 7, device=device),
            },
            {  # Terminal at index 5
                "observations": torch.randn(10, 72, device=device),
                "actions": torch.randint(0, 6, (10,), device=device),
                "rewards_extrinsic": torch.randn(10, device=device),
                "rewards_intrinsic": torch.randn(10, device=device),
                "dones": torch.tensor([False] * 5 + [True] + [False] * 4, device=device),
            },
            {  # No terminal
                "observations": torch.randn(10, 72, device=device),
                "actions": torch.randint(0, 6, (10,), device=device),
                "rewards_extrinsic": torch.randn(10, device=device),
                "rewards_intrinsic": torch.randn(10, device=device),
                "dones": torch.zeros(10, dtype=torch.bool, device=device),
            },
        ]

        for ep in episodes:
            buffer.store_episode(ep)

        # Sample batch
        batch = buffer.sample_sequences(batch_size=8, seq_len=6)

        # Check each sequence has valid mask
        for i in range(8):
            mask = batch["mask"][i]
            dones = batch["dones"][i]

            if torch.any(dones):
                terminal_idx = torch.where(dones)[0][0].item()
                # Should be True up to terminal, False after
                assert torch.all(mask[: terminal_idx + 1]), f"Sequence {i}: mask wrong before terminal"
                if terminal_idx < len(mask) - 1:
                    assert torch.all(~mask[terminal_idx + 1 :]), f"Sequence {i}: mask wrong after terminal"
            else:
                # No terminal -> all True
                assert torch.all(mask), f"Sequence {i}: should be all True (no terminal)"

    def test_mask_shape_matches_batch(self, buffer, device):
        """Mask shape should match [batch_size, seq_len]."""
        episode = {
            "observations": torch.randn(20, 72, device=device),
            "actions": torch.randint(0, 6, (20,), device=device),
            "rewards_extrinsic": torch.randn(20, device=device),
            "rewards_intrinsic": torch.randn(20, device=device),
            "dones": torch.zeros(20, dtype=torch.bool, device=device),
        }
        buffer.store_episode(episode)

        batch = buffer.sample_sequences(batch_size=16, seq_len=10)

        assert batch["mask"].shape == (16, 10), f"Mask shape should be [16, 10], got {batch['mask'].shape}"

    def test_mask_works_with_unified_rewards(self, buffer, device):
        """Mask should work when episode has unified 'rewards' instead of split."""
        episode = {
            "observations": torch.randn(10, 72, device=device),
            "actions": torch.randint(0, 6, (10,), device=device),
            "rewards": torch.randn(10, device=device),  # Unified rewards
            "dones": torch.tensor([False] * 4 + [True] + [False] * 5, device=device),
        }
        buffer.store_episode(episode)

        batch = buffer.sample_sequences(batch_size=2, seq_len=8)

        assert "mask" in batch, "Should have mask with unified rewards"
        assert batch["mask"].dtype == torch.bool, "Mask should be bool"


class TestMaskIntegrationWithLoss:
    """Test that mask can be used correctly in loss computation."""

    def test_mask_sum_counts_valid_timesteps(self, buffer, device):
        """Mask sum should equal number of valid timesteps."""
        # Episode with terminal at index 3 (4 valid timesteps: 0,1,2,3)
        episode = {
            "observations": torch.randn(10, 72, device=device),
            "actions": torch.randint(0, 6, (10,), device=device),
            "rewards_extrinsic": torch.randn(10, device=device),
            "rewards_intrinsic": torch.randn(10, device=device),
            "dones": torch.tensor([False, False, False, True] + [False] * 6, device=device),
        }
        buffer.store_episode(episode)

        # Sample sequence of length 10 starting at 0
        batch = buffer.sample_sequences(batch_size=1, seq_len=10)

        mask = batch["mask"][0]
        valid_count = mask.sum().item()

        # Should have 4 valid timesteps (0,1,2,3 - including terminal)
        assert valid_count == 4, f"Expected 4 valid timesteps, got {valid_count}"

    def test_masked_loss_example(self, buffer, device):
        """Demonstrate masked loss computation pattern."""
        episode = {
            "observations": torch.randn(10, 72, device=device),
            "actions": torch.randint(0, 6, (10,), device=device),
            "rewards_extrinsic": torch.randn(10, device=device),
            "rewards_intrinsic": torch.randn(10, device=device),
            "dones": torch.tensor([False] * 5 + [True] + [False] * 4, device=device),
        }
        buffer.store_episode(episode)

        batch = buffer.sample_sequences(batch_size=2, seq_len=8)

        # Simulate loss computation
        fake_losses = torch.randn(2, 8, device=device)  # [batch, seq_len]
        mask = batch["mask"]

        # Apply mask
        masked_losses = fake_losses * mask.float()
        total_loss = masked_losses.sum() / mask.sum().clamp_min(1)

        # Should be finite
        assert torch.isfinite(total_loss), "Masked loss should be finite"

        # Denominator should equal number of valid timesteps
        assert mask.sum() > 0, "Should have at least some valid timesteps"
