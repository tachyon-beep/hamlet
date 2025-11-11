"""Test StructuredQNetwork with group encoders for semantic observation groups."""

from pathlib import Path

import torch

from townlet.agent.networks import StructuredQNetwork
from townlet.universe.compiler import UniverseCompiler


class TestStructuredQNetworkBasics:
    def test_structured_qnetwork_accepts_observation_activity(self):
        """StructuredQNetwork should accept ObservationActivity parameter."""
        config_dir = Path("configs/L0_5_dual_resource")

        compiler = UniverseCompiler()
        compiled = compiler.compile(config_dir, use_cache=False)
        obs_activity = compiled.observation_activity

        network = StructuredQNetwork(
            obs_dim=compiled.metadata.observation_dim,
            action_dim=compiled.metadata.action_count,
            observation_activity=obs_activity,
        )

        assert network is not None
        assert hasattr(network, "observation_activity")

    def test_structured_qnetwork_forward_pass(self):
        """StructuredQNetwork should produce Q-values for batch of observations."""
        config_dir = Path("configs/L0_5_dual_resource")

        compiler = UniverseCompiler()
        compiled = compiler.compile(config_dir, use_cache=False)
        obs_activity = compiled.observation_activity

        network = StructuredQNetwork(
            obs_dim=compiled.metadata.observation_dim,
            action_dim=compiled.metadata.action_count,
            observation_activity=obs_activity,
        )

        batch_size = 4
        obs = torch.randn(batch_size, compiled.metadata.observation_dim)

        q_values = network(obs)

        assert q_values.shape == (batch_size, compiled.metadata.action_count)

    def test_structured_qnetwork_has_group_encoders(self):
        """StructuredQNetwork should create encoders for each semantic group."""
        config_dir = Path("configs/L0_5_dual_resource")

        compiler = UniverseCompiler()
        compiled = compiler.compile(config_dir, use_cache=False)
        obs_activity = compiled.observation_activity

        network = StructuredQNetwork(
            obs_dim=compiled.metadata.observation_dim,
            action_dim=compiled.metadata.action_count,
            observation_activity=obs_activity,
        )

        # Should have encoders attribute
        assert hasattr(network, "group_encoders")

        # Should have encoder for each group in observation_activity
        for group_name in obs_activity.group_slices.keys():
            assert group_name in network.group_encoders


class TestStructuredQNetworkArchitecture:
    def test_group_encoders_process_correct_slices(self):
        """Each group encoder should process only its semantic group dimensions."""
        config_dir = Path("configs/L0_5_dual_resource")

        compiler = UniverseCompiler()
        compiled = compiler.compile(config_dir, use_cache=False)
        obs_activity = compiled.observation_activity

        network = StructuredQNetwork(
            obs_dim=compiled.metadata.observation_dim,
            action_dim=compiled.metadata.action_count,
            observation_activity=obs_activity,
        )

        # Create observation
        obs = torch.randn(1, compiled.metadata.observation_dim)

        # Manually extract and encode spatial group
        spatial_slice = obs_activity.get_group_slice("spatial")
        if spatial_slice is not None:
            spatial_obs = obs[:, spatial_slice]
            spatial_encoder = network.group_encoders["spatial"]
            spatial_embedding = spatial_encoder(spatial_obs)

            # Should produce embedding
            assert spatial_embedding.shape[0] == 1  # batch size
            assert spatial_embedding.shape[1] > 0  # embedding dim

    def test_structured_qnetwork_output_matches_simple_qnetwork_shape(self):
        """StructuredQNetwork output should match SimpleQNetwork for compatibility."""
        from townlet.agent.networks import SimpleQNetwork

        config_dir = Path("configs/L0_0_minimal")

        compiler = UniverseCompiler()
        compiled = compiler.compile(config_dir, use_cache=False)
        obs_activity = compiled.observation_activity

        obs_dim = compiled.metadata.observation_dim
        action_dim = compiled.metadata.action_count

        simple_net = SimpleQNetwork(obs_dim, action_dim, hidden_dim=128)
        structured_net = StructuredQNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            observation_activity=obs_activity,
        )

        batch_size = 8
        obs = torch.randn(batch_size, obs_dim)

        simple_q = simple_net(obs)
        structured_q = structured_net(obs)

        # Same shape
        assert simple_q.shape == structured_q.shape
        assert structured_q.shape == (batch_size, action_dim)


class TestStructuredQNetworkIntegration:
    def test_structured_qnetwork_with_gradient_flow(self):
        """StructuredQNetwork should support gradient backpropagation."""
        config_dir = Path("configs/L0_5_dual_resource")

        compiler = UniverseCompiler()
        compiled = compiler.compile(config_dir, use_cache=False)
        obs_activity = compiled.observation_activity

        network = StructuredQNetwork(
            obs_dim=compiled.metadata.observation_dim,
            action_dim=compiled.metadata.action_count,
            observation_activity=obs_activity,
        )

        obs = torch.randn(4, compiled.metadata.observation_dim, requires_grad=True)
        q_values = network(obs)

        # Compute loss and backprop
        loss = q_values.mean()
        loss.backward()

        # Gradients should flow to input
        assert obs.grad is not None
        assert not torch.all(obs.grad == 0)

    def test_structured_qnetwork_on_gpu(self):
        """StructuredQNetwork should work on GPU if available."""
        if not torch.cuda.is_available():
            return  # Skip if no GPU

        config_dir = Path("configs/L0_0_minimal")

        compiler = UniverseCompiler()
        compiled = compiler.compile(config_dir, use_cache=False)
        obs_activity = compiled.observation_activity

        device = torch.device("cuda")
        network = StructuredQNetwork(
            obs_dim=compiled.metadata.observation_dim,
            action_dim=compiled.metadata.action_count,
            observation_activity=obs_activity,
        ).to(device)

        obs = torch.randn(4, compiled.metadata.observation_dim, device=device)
        q_values = network(obs)

        assert q_values.device.type == "cuda"
        assert q_values.shape == (4, compiled.metadata.action_count)
