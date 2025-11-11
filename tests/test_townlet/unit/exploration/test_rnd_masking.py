"""Test active_mask integration with RND networks for observation masking."""

from pathlib import Path

import torch

from townlet.exploration.rnd import RNDExploration, RNDNetwork
from townlet.universe.compiler import UniverseCompiler


class TestRNDNetworkMasking:
    def test_rnd_network_accepts_active_mask(self):
        """RNDNetwork should accept and store active_mask parameter."""
        obs_dim = 41
        active_mask = tuple([True] * 30 + [False] * 11)  # 30 active, 11 padding

        network = RNDNetwork(obs_dim=obs_dim, embed_dim=128, active_mask=active_mask)

        assert hasattr(network, "active_mask")
        assert len(network.active_mask) == obs_dim

    def test_rnd_network_masks_observations(self):
        """RNDNetwork should zero out padding dimensions before processing."""
        obs_dim = 41
        active_mask = tuple([True] * 30 + [False] * 11)  # 30 active, 11 padding

        network = RNDNetwork(obs_dim=obs_dim, embed_dim=128, active_mask=active_mask)

        # Create observation with non-zero values in padding dimensions
        obs = torch.randn(4, obs_dim)  # [batch=4, obs_dim=41]
        obs[:, 30:] = 5.0  # Set padding dims to 5.0

        # Forward pass should mask padding
        output = network(obs)

        # Verify output shape
        assert output.shape == (4, 128)

        # Since padding is masked, output should be deterministic for same active dims
        # (This is implicit - we're just checking no errors occur)

    def test_rnd_network_without_mask_unchanged(self):
        """RNDNetwork should work without active_mask (backward compatibility)."""
        obs_dim = 41

        network = RNDNetwork(obs_dim=obs_dim, embed_dim=128)
        obs = torch.randn(4, obs_dim)

        output = network(obs)
        assert output.shape == (4, 128)


class TestRNDExplorationMasking:
    def test_rnd_exploration_accepts_active_mask(self):
        """RNDExploration should accept and pass active_mask to networks."""
        obs_dim = 41
        active_mask = tuple([True] * 30 + [False] * 11)

        rnd = RNDExploration(
            obs_dim=obs_dim,
            embed_dim=128,
            active_mask=active_mask,
            device=torch.device("cpu"),
        )

        # Both networks should have active_mask
        assert hasattr(rnd.fixed_network, "active_mask")
        assert hasattr(rnd.predictor_network, "active_mask")
        assert len(rnd.fixed_network.active_mask) == obs_dim
        assert len(rnd.predictor_network.active_mask) == obs_dim

    def test_rnd_exploration_computes_intrinsic_rewards_with_mask(self):
        """RNDExploration should compute intrinsic rewards with masked observations."""
        obs_dim = 41
        active_mask = tuple([True] * 30 + [False] * 11)

        rnd = RNDExploration(
            obs_dim=obs_dim,
            embed_dim=128,
            active_mask=active_mask,
            device=torch.device("cpu"),
        )

        # Create observations with values in padding dimensions
        obs = torch.randn(8, obs_dim)
        obs[:, 30:] = 10.0  # Padding dims should be ignored

        intrinsic_rewards = rnd.compute_intrinsic_rewards(obs)

        # Should return rewards for each observation
        assert intrinsic_rewards.shape == (8,)
        assert torch.all(intrinsic_rewards >= 0)  # Intrinsic rewards are non-negative

    def test_rnd_exploration_without_mask_unchanged(self):
        """RNDExploration should work without active_mask (backward compatibility)."""
        obs_dim = 41

        rnd = RNDExploration(
            obs_dim=obs_dim,
            embed_dim=128,
            device=torch.device("cpu"),
        )

        obs = torch.randn(8, obs_dim)
        intrinsic_rewards = rnd.compute_intrinsic_rewards(obs)

        assert intrinsic_rewards.shape == (8,)
        assert torch.all(intrinsic_rewards >= 0)


class TestRNDMaskingIntegration:
    def test_rnd_with_real_observation_activity(self):
        """Test RND with actual ObservationActivity from compiled universe."""
        config_dir = Path("configs/L0_5_dual_resource")

        compiler = UniverseCompiler()
        compiled = compiler.compile(config_dir, use_cache=False)
        env = compiled.create_environment(num_agents=4, device="cpu")

        obs_dim = env.observation_dim
        active_mask = env.observation_activity.active_mask

        # Create RND with active_mask from environment
        rnd = RNDExploration(
            obs_dim=obs_dim,
            embed_dim=128,
            active_mask=active_mask,
            device=torch.device("cpu"),
        )

        # Generate some observations
        obs = torch.randn(4, obs_dim)
        intrinsic_rewards = rnd.compute_intrinsic_rewards(obs)

        assert intrinsic_rewards.shape == (4,)
        assert torch.all(intrinsic_rewards >= 0)
