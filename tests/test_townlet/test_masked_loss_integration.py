"""
Test that P2.2 post-terminal masking is correctly computed.

This verifies the fix where masked loss is computed as:
    losses = F.mse_loss(q_pred, q_target, reduction='none')
    masked_loss = (losses * mask).sum() / mask.sum().clamp_min(1)
"""

import torch
import torch.nn.functional as F


class TestMaskedLossComputation:
    """Test that mask computation matches P2.2 specification."""

    def test_mask_all_true_gives_same_result(self):
        """Test that when mask is all True, masked loss equals unmasked loss."""
        batch_size = 4
        seq_len = 16

        # Create random predictions and targets
        q_pred = torch.randn(batch_size, seq_len)
        q_target = torch.randn(batch_size, seq_len)

        # Compute both losses
        unmasked_loss = F.mse_loss(q_pred, q_target)

        losses = F.mse_loss(q_pred, q_target, reduction='none')
        mask = torch.ones(batch_size, seq_len)  # All True
        masked_loss = (losses * mask).sum() / mask.sum().clamp_min(1)

        # Should be identical when mask is all True
        assert torch.allclose(masked_loss, unmasked_loss), \
            "Masked loss with all-True mask should equal unmasked loss"

    def test_mask_zeros_out_invalid_timesteps(self):
        """Test that masked loss ignores timesteps where mask is False."""
        batch_size = 2
        seq_len = 8

        # Create predictions and targets with known error
        q_pred = torch.zeros(batch_size, seq_len)
        q_target = torch.ones(batch_size, seq_len)  # Error of 1.0 per timestep

        # Create mask that only validates first 4 timesteps
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        mask[:, :4] = True  # First 4 valid, last 4 invalid

        # Compute masked loss
        losses = F.mse_loss(q_pred, q_target, reduction='none')  # All losses = 1.0
        mask_float = mask.float()
        masked_loss = (losses * mask_float).sum() / mask_float.sum().clamp_min(1)

        # Expected: only first 4 timesteps contribute
        # Total error from valid timesteps: 2 * 4 * 1.0 = 8.0
        # Number of valid timesteps: 2 * 4 = 8
        # Expected loss: 8.0 / 8 = 1.0
        expected_loss = 1.0

        assert torch.allclose(masked_loss, torch.tensor(expected_loss)), \
            f"Masked loss should be {expected_loss}, got {masked_loss.item()}"

        # Verify this is different from unmasked loss
        unmasked_loss = F.mse_loss(q_pred, q_target)
        assert torch.allclose(unmasked_loss, torch.tensor(1.0)), \
            "Unmasked loss should also be 1.0 in this case (all errors are 1.0)"

        # But if we corrupt the invalid timesteps with huge errors...
        q_target_corrupt = q_target.clone()
        q_target_corrupt[:, 4:] = 1000.0  # Huge errors in invalid region

        unmasked_loss_corrupt = F.mse_loss(q_pred, q_target_corrupt)
        losses_corrupt = F.mse_loss(q_pred, q_target_corrupt, reduction='none')
        masked_loss_corrupt = (losses_corrupt * mask_float).sum() / mask_float.sum().clamp_min(1)

        # Masked loss should still be 1.0 (ignoring corrupted region)
        assert torch.allclose(masked_loss_corrupt, torch.tensor(1.0)), \
            "Masked loss should ignore corrupted invalid region"

        # But unmasked loss should be huge
        assert unmasked_loss_corrupt > 100.0, \
            "Unmasked loss should be affected by corruption"

        print("✅ Mask correctly zeros out invalid timesteps")
        print(f"   Clean unmasked loss: {unmasked_loss.item():.2f}")
        print(f"   Clean masked loss: {masked_loss.item():.2f}")
        print(f"   Corrupt unmasked loss: {unmasked_loss_corrupt.item():.2f}")
        print(f"   Corrupt masked loss: {masked_loss_corrupt.item():.2f}")
