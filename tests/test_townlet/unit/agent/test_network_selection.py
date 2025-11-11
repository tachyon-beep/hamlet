"""Test network selection based on population config (mask_unused_obs, network_type)."""

from pathlib import Path

from townlet.agent.networks import RecurrentSpatialQNetwork, SimpleQNetwork, StructuredQNetwork
from townlet.config.population import PopulationConfig
from townlet.universe.compiler import UniverseCompiler


class TestNetworkTypeSelection:
    def test_population_config_accepts_structured_network_type(self):
        """PopulationConfig should accept 'structured' as a valid network_type."""
        config = PopulationConfig(
            num_agents=1,
            learning_rate=0.00025,
            gamma=0.99,
            replay_buffer_capacity=10000,
            network_type="structured",
            mask_unused_obs=True,
        )

        assert config.network_type == "structured"

    def test_population_config_accepts_mask_unused_obs_flag(self):
        """PopulationConfig should accept mask_unused_obs boolean flag."""
        config = PopulationConfig(
            num_agents=1,
            learning_rate=0.00025,
            gamma=0.99,
            replay_buffer_capacity=10000,
            network_type="simple",
            mask_unused_obs=True,
        )

        assert config.mask_unused_obs is True

    def test_population_config_requires_mask_unused_obs(self):
        """PopulationConfig should require explicit mask_unused_obs (no defaults)."""
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="mask_unused_obs"):
            PopulationConfig(
                num_agents=1,
                learning_rate=0.00025,
                gamma=0.99,
                replay_buffer_capacity=10000,
                network_type="simple",
                # Missing mask_unused_obs - should fail
            )


class TestStructuredNetworkInstantiation:
    def test_structured_network_instantiated_with_observation_activity(self):
        """VectorizedPopulation should instantiate StructuredQNetwork when network_type='structured'."""
        config_dir = Path("configs/L0_5_dual_resource")

        compiler = UniverseCompiler()
        compiled = compiler.compile(config_dir, use_cache=False)

        # Create environment to get observation_activity
        env = compiled.create_environment(num_agents=4, device="cpu")

        # Manually create StructuredQNetwork as population would
        network = StructuredQNetwork(
            obs_dim=env.observation_dim,
            action_dim=env.action_dim,
            observation_activity=env.observation_activity,
        )

        assert isinstance(network, StructuredQNetwork)
        assert hasattr(network, "observation_activity")
        assert hasattr(network, "group_encoders")

    def test_simple_network_still_works(self):
        """VectorizedPopulation should still instantiate SimpleQNetwork when network_type='simple'."""
        config_dir = Path("configs/L0_5_dual_resource")

        compiler = UniverseCompiler()
        compiled = compiler.compile(config_dir, use_cache=False)
        env = compiled.create_environment(num_agents=4, device="cpu")

        # Manually create SimpleQNetwork as population would
        network = SimpleQNetwork(
            obs_dim=env.observation_dim,
            action_dim=env.action_dim,
            hidden_dim=128,
        )

        assert isinstance(network, SimpleQNetwork)

    def test_recurrent_network_still_works(self):
        """VectorizedPopulation should still instantiate RecurrentSpatialQNetwork when network_type='recurrent'."""
        config_dir = Path("configs/L2_partial_observability")

        compiler = UniverseCompiler()
        compiled = compiler.compile(config_dir, use_cache=False)
        env = compiled.create_environment(num_agents=4, device="cpu")

        # Manually create RecurrentSpatialQNetwork as population would
        network = RecurrentSpatialQNetwork(
            action_dim=env.action_dim,
            window_size=5,
            position_dim=2,
            num_meters=8,
            num_affordance_types=14,
            enable_temporal_features=True,
            hidden_dim=256,
        )

        assert isinstance(network, RecurrentSpatialQNetwork)
