"""Tests for RND intrinsic reward normalization.

This test file demonstrates that RND raw MSE values need normalization
to be comparable in magnitude to extrinsic rewards.
"""

import torch

from townlet.exploration.rnd import RNDExploration


class TestRNDNormalization:
    """Test that RND intrinsic rewards are normalized properly."""

    def test_raw_mse_is_too_small_compared_to_extrinsic(self):
        """Demonstrate the bug: raw MSE is 10-100x smaller than extrinsic rewards.

        This test SHOULD FAIL initially, demonstrating the bug.
        After implementing normalization, it should pass.
        """
        obs_dim = 29
        embed_dim = 128
        device = torch.device("cpu")

        rnd = RNDExploration(
            obs_dim=obs_dim,
            embed_dim=embed_dim,
            device=device,
        )

        # Generate 100 random observations
        observations = torch.randn(100, obs_dim, device=device)

        # Compute intrinsic rewards with statistics update enabled
        intrinsic_rewards = rnd.compute_intrinsic_rewards(observations, update_stats=True)

        # Raw MSE is typically 0.001-0.1 range
        mean_intrinsic = intrinsic_rewards.mean().item()
        std_intrinsic = intrinsic_rewards.std().item()

        print("\\nRaw MSE statistics:")
        print(f"  Mean: {mean_intrinsic:.6f}")
        print(f"  Std:  {std_intrinsic:.6f}")
        print(f"  Min:  {intrinsic_rewards.min().item():.6f}")
        print(f"  Max:  {intrinsic_rewards.max().item():.6f}")

        # Typical extrinsic reward: 0.4-0.6 range (health * energy)
        typical_extrinsic = 0.5

        # Bug: Intrinsic is 10-100x smaller than extrinsic
        ratio = mean_intrinsic / typical_extrinsic
        print(f"  Ratio (intrinsic/extrinsic): {ratio:.6f}")

        # THIS ASSERTION SHOULD FAIL BEFORE FIX
        # After normalization, intrinsic should be comparable to extrinsic
        assert ratio > 0.1, (
            f"Intrinsic reward ({mean_intrinsic:.6f}) is {1 / ratio:.1f}x smaller than "
            f"extrinsic ({typical_extrinsic}). Should be normalized to comparable magnitude."
        )

    def test_normalized_rewards_have_unit_variance_after_warmup(self):
        """Test that normalized intrinsic rewards have ~unit variance after warmup.

        This test SHOULD FAIL initially (no normalization implemented).
        After fix, normalized rewards should have variance close to 1.0.
        """
        obs_dim = 29
        embed_dim = 128
        device = torch.device("cpu")

        rnd = RNDExploration(
            obs_dim=obs_dim,
            embed_dim=embed_dim,
            device=device,
        )

        # Warmup: collect statistics over 1000 observations
        for _ in range(10):
            observations = torch.randn(100, obs_dim, device=device)
            _ = rnd.compute_intrinsic_rewards(observations, update_stats=True)  # Warm up statistics

            # Simulate training: update predictor
            batch = {"observations": observations}
            rnd.update(batch)

        # After warmup, compute rewards on new batch
        test_observations = torch.randn(100, obs_dim, device=device)
        normalized_rewards = rnd.compute_intrinsic_rewards(test_observations)

        variance = normalized_rewards.var().item()

        print("\\nNormalized reward statistics:")
        print(f"  Mean: {normalized_rewards.mean().item():.6f}")
        print(f"  Variance: {variance:.6f}")
        print(f"  Std: {normalized_rewards.std().item():.6f}")

        # After normalization, variance should be reasonably close to 1.0
        # (Note: variance may be < 1.0 if predictor learns well during warmup)
        assert 0.3 < variance < 2.0, f"Normalized rewards should have variance ~1.0, got {variance:.6f}"

    def test_normalization_is_persistent_across_checkpoints(self):
        """Test that normalization statistics are saved/loaded correctly."""
        obs_dim = 29
        embed_dim = 128
        device = torch.device("cpu")

        # Create RND and warmup
        rnd1 = RNDExploration(obs_dim=obs_dim, embed_dim=embed_dim, device=device)

        for _ in range(5):
            observations = torch.randn(100, obs_dim, device=device)
            rnd1.compute_intrinsic_rewards(observations, update_stats=True)
            rnd1.update({"observations": observations})

        # Save checkpoint
        checkpoint = rnd1.checkpoint_state()

        # Create new RND and load checkpoint
        rnd2 = RNDExploration(obs_dim=obs_dim, embed_dim=embed_dim, device=device)
        rnd2.load_state(checkpoint)

        # Compute rewards on same observations
        test_obs = torch.randn(50, obs_dim, device=device)
        rewards1 = rnd1.compute_intrinsic_rewards(test_obs)
        rewards2 = rnd2.compute_intrinsic_rewards(test_obs)

        # Rewards should be identical (normalization stats preserved)
        # THIS ASSERTION SHOULD FAIL IF checkpoint_state/load_state don't include norm stats
        torch.testing.assert_close(rewards1, rewards2, rtol=1e-5, atol=1e-5, msg="Normalized rewards should match after checkpoint load")


class TestAdaptiveIntrinsicDoubleWeighting:
    """Test that AdaptiveIntrinsicExploration doesn't apply weight twice."""

    def test_adaptive_applies_weight_only_once(self):
        """Demonstrate double-weighting bug in AdaptiveIntrinsicExploration.

        This test SHOULD FAIL initially, showing weight is applied twice.
        After fix, weight should only be applied once (in replay buffer).
        """
        from townlet.exploration.adaptive_intrinsic import AdaptiveIntrinsicExploration

        obs_dim = 29
        embed_dim = 128
        device = torch.device("cpu")

        adaptive = AdaptiveIntrinsicExploration(
            obs_dim=obs_dim,
            embed_dim=embed_dim,
            initial_intrinsic_weight=0.7,
            device=device,
        )

        observations = torch.randn(10, obs_dim, device=device)

        # Get rewards from adaptive
        adaptive_rewards = adaptive.compute_intrinsic_rewards(observations)

        # Reset RND statistics to same state
        adaptive.rnd.reward_rms.mean = 0.0
        adaptive.rnd.reward_rms.var = 1.0
        adaptive.rnd.reward_rms.count = 1e-4

        # Get raw RND rewards (should be same because adaptive doesn't apply weight)
        raw_rnd_rewards = adaptive.rnd.compute_intrinsic_rewards(observations)

        current_weight = adaptive.get_intrinsic_weight()
        assert current_weight == 0.7, "Weight should be 0.7"

        # BUG: Before fix, adaptive_rewards would be raw_rnd * 0.7
        # FIX: After fix, adaptive_rewards should equal raw_rnd (weight applied in replay buffer)

        # THIS ASSERTION SHOULD FAIL BEFORE FIX (showing double weighting)
        torch.testing.assert_close(
            adaptive_rewards,
            raw_rnd_rewards,
            rtol=0.1,
            atol=0.1,
            msg="Adaptive should return raw RND rewards (weight applied in replay buffer)",
        )
