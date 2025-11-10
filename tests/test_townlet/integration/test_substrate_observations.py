"""Test observation encoding uses substrate methods."""

from pathlib import Path


def test_full_observation_uses_substrate(cpu_env_factory):
    """Full observability should use substrate.encode_observation()."""
    env = cpu_env_factory(config_dir=Path("configs/L1_full_observability"), num_agents=1)

    # Get observation
    obs = env.reset()

    # Observation should include substrate position encoding
    expected_dim = env.substrate.get_observation_dim() + env.meter_count + (env.num_affordance_types + 1) + 4
    assert obs.shape[1] == expected_dim, f"Expected {expected_dim}, got {obs.shape[1]}"


def test_partial_observation_uses_substrate(cpu_env_factory):
    """Partial observability should use substrate.encode_partial_observation()."""
    env = cpu_env_factory(config_dir=Path("configs/L2_partial_observability"), num_agents=1)

    # Get observation
    obs = env.reset()

    # Partial obs should use local window encoding
    # For 5×5 window: 25 (local grid) + 2 (normalized position) + 8 (meters) + 15 (affordance) + 4 (temporal) = 54
    window_size = 2 * env.vision_range + 1  # 5×5 for vision_range=2
    expected_dim = (
        window_size * window_size  # 25 for 5×5 window
        + env.substrate.position_dim  # 2 for Grid2D
        + env.meter_count
        + (env.num_affordance_types + 1)
        + 4
    )
    assert obs.shape[1] == expected_dim, f"Expected {expected_dim}, got {obs.shape[1]}"
    assert obs.shape[1] == env.metadata.observation_dim, (
        "L2 observation should match metadata: " f"expected {env.metadata.observation_dim}, got {obs.shape[1]}"
    )


def test_observation_dim_matches_actual_observation(cpu_env_factory, test_config_pack_path):
    """Environment's observation_dim should match actual observation shape."""
    env = cpu_env_factory(config_dir=test_config_pack_path, num_agents=1)

    obs = env.reset()

    # Observation dimension should match actual observation
    assert obs.shape[1] == env.observation_dim, f"observation_dim={env.observation_dim} doesn't match actual obs.shape[1]={obs.shape[1]}"


def test_partial_observation_dim_matches_actual(cpu_env_factory, test_config_pack_path):
    """POMDP observation_dim should match actual observation shape."""
    env = cpu_env_factory(config_dir=test_config_pack_path, num_agents=1)

    obs = env.reset()

    # Observation dimension should match actual observation
    assert obs.shape[1] == env.observation_dim, f"observation_dim={env.observation_dim} doesn't match actual obs.shape[1]={obs.shape[1]}"
