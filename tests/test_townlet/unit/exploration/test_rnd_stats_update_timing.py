"""Test that RND stats are updated during environment rollouts (BUG-22).

This test verifies that RND normalization statistics are updated in sync
with the observations being normalized, not lagging behind by multiple steps.
"""

import torch

from townlet.exploration.rnd import RNDExploration


class TestRNDStatsUpdateTiming:
    """Tests for RND running statistics update timing."""

    def test_stats_update_during_training_rollout(self):
        """Test that RND stats update when compute_intrinsic_rewards is called during training.

        BUG-22: Originally, the environment called compute_intrinsic_rewards(..., update_stats=False)
        during training rollouts, causing normalization stats to lag behind by 2+ steps.

        This test verifies that stats are updated synchronously when computing intrinsic rewards.
        """
        device = torch.device("cpu")
        obs_dim = 29
        rnd = RNDExploration(obs_dim=obs_dim, device=device)

        # Record initial variance
        initial_var = rnd.reward_rms.var

        # Create a batch of observations
        observations = torch.randn(32, obs_dim, device=device)

        # Compute intrinsic rewards WITH stats update (training mode)
        _ = rnd.compute_intrinsic_rewards(observations, update_stats=True)

        # Verify that stats have been updated
        updated_var = rnd.reward_rms.var

        # Stats should have changed after processing observations
        assert updated_var != initial_var, (
            "RND stats should update when update_stats=True. " f"Got var={updated_var}, expected != {initial_var}"
        )

    def test_stats_dont_update_during_eval(self):
        """Test that RND stats don't update when update_stats=False (eval mode)."""
        device = torch.device("cpu")
        obs_dim = 29
        rnd = RNDExploration(obs_dim=obs_dim, device=device)

        # Warmup: establish some statistics
        warmup_obs = torch.randn(100, obs_dim, device=device)
        _ = rnd.compute_intrinsic_rewards(warmup_obs, update_stats=True)

        # Record variance after warmup
        var_after_warmup = rnd.reward_rms.var

        # Compute intrinsic rewards WITHOUT stats update (eval mode)
        eval_obs = torch.randn(32, obs_dim, device=device)
        _ = rnd.compute_intrinsic_rewards(eval_obs, update_stats=False)

        # Verify that stats have NOT changed
        var_after_eval = rnd.reward_rms.var

        assert var_after_eval == var_after_warmup, (
            "RND stats should NOT update when update_stats=False. " f"Got var={var_after_eval}, expected {var_after_warmup}"
        )
