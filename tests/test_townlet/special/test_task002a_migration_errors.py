"""TASK-002A specific test: Migration error messages.

This test validates that users get helpful error messages when substrate.yaml is missing.
This is migration-specific behavior that can be removed once all configs have substrate.yaml.
"""

from pathlib import Path

import pytest
import torch

from townlet.environment.vectorized_env import VectorizedHamletEnv


def test_missing_substrate_yaml_raises_helpful_error(tmp_path):
    """TASK-002A: Should fail fast with migration instructions when substrate.yaml missing."""
    import shutil

    # Create config pack directory without substrate.yaml
    config_pack = tmp_path / "test_config"
    config_pack.mkdir()

    # Copy complete config files from test config (but NO substrate.yaml)
    test_config = Path("configs/test")
    shutil.copy(test_config / "bars.yaml", config_pack / "bars.yaml")
    shutil.copy(test_config / "affordances.yaml", config_pack / "affordances.yaml")
    shutil.copy(test_config / "cascades.yaml", config_pack / "cascades.yaml")

    # Attempt to create environment without substrate.yaml
    with pytest.raises(FileNotFoundError) as exc_info:
        VectorizedHamletEnv(
            config_pack_path=config_pack,
            num_agents=1,
            grid_size=8,
            partial_observability=False,
            vision_range=2,
            enable_temporal_mechanics=False,
            move_energy_cost=0.5,
            wait_energy_cost=0.1,
            interact_energy_cost=0.3,
            agent_lifespan=1000,
            device=torch.device("cpu"),
        )

    # Verify error message contains migration instructions
    error_msg = str(exc_info.value)
    assert "substrate.yaml is required" in error_msg
    assert "Quick fix:" in error_msg
    assert "configs/templates/substrate.yaml" in error_msg
    assert "TASK-002A" in error_msg
