"""Test ObservationActivity exposure in VectorizedHamletEnv."""

from pathlib import Path

from townlet.universe.compiler import UniverseCompiler


class TestEnvironmentObservationActivity:
    def test_env_exposes_observation_activity(self):
        """VectorizedHamletEnv should expose observation_activity from runtime."""
        config_dir = Path("configs/L0_5_dual_resource")

        compiler = UniverseCompiler()
        compiled = compiler.compile(config_dir, use_cache=False)
        env = compiled.create_environment(num_agents=4, device="cpu")

        assert hasattr(env, "observation_activity")
        assert env.observation_activity is not None

    def test_env_observation_activity_matches_runtime(self):
        """Environment's observation_activity should match RuntimeUniverse."""
        config_dir = Path("configs/L0_0_minimal")

        compiler = UniverseCompiler()
        compiled = compiler.compile(config_dir, use_cache=False)
        runtime = compiled.to_runtime()
        env = compiled.create_environment(num_agents=4, device="cpu")

        assert env.observation_activity is runtime.observation_activity

    def test_env_observation_activity_has_valid_mask(self):
        """Environment's observation_activity should have valid active_mask."""
        config_dir = Path("configs/L0_5_dual_resource")

        compiler = UniverseCompiler()
        compiled = compiler.compile(config_dir, use_cache=False)
        env = compiled.create_environment(num_agents=4, device="cpu")

        # active_mask length should match observation_dim
        assert len(env.observation_activity.active_mask) == env.observation_dim

    def test_env_observation_activity_has_group_slices(self):
        """Environment's observation_activity should have semantic group slices."""
        config_dir = Path("configs/L0_5_dual_resource")

        compiler = UniverseCompiler()
        compiled = compiler.compile(config_dir, use_cache=False)
        env = compiled.create_environment(num_agents=4, device="cpu")

        # Should have at least spatial and bars groups
        assert env.observation_activity.get_group_slice("spatial") is not None
        assert env.observation_activity.get_group_slice("bars") is not None
        assert len(env.observation_activity.group_slices) >= 2
