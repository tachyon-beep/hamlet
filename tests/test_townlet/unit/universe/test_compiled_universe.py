"""Tests for CompiledUniverse Stage 7 artifact."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import torch

from townlet.universe.compiled import CompiledUniverse
from townlet.universe.compiler import UniverseCompiler


def test_compiler_returns_compiled_universe() -> None:
    compiler = UniverseCompiler()
    compiled = compiler.compile(Path("configs/L0_0_minimal"))

    assert isinstance(compiled, CompiledUniverse)
    assert compiled.metadata.universe_name == "L0_0_minimal"
    assert compiled.observation_spec.total_dims == compiled.metadata.observation_dim


def test_compiled_universe_is_frozen() -> None:
    compiler = UniverseCompiler()
    compiled = compiler.compile(Path("configs/L0_0_minimal"))

    try:
        compiled.metadata = None  # type: ignore[attr-defined]
    except FrozenInstanceError:
        pass
    else:
        raise AssertionError("CompiledUniverse should be frozen")


def test_compiled_universe_checkpoint_check() -> None:
    compiler = UniverseCompiler()
    compiled = compiler.compile(Path("configs/L0_0_minimal"))

    # Empty checkpoint is incompatible (missing drive_hash)
    compatible, _ = compiled.check_checkpoint_compatibility({})
    assert not compatible  # Changed: now expects incompatibility due to missing drive_hash

    checkpoint = {
        "config_hash": compiled.metadata.config_hash,
        "observation_dim": compiled.metadata.observation_dim,
        "action_dim": compiled.metadata.action_count,
        "observation_field_uuids": [field.uuid for field in compiled.observation_spec.fields],
        "drive_hash": compiled.drive_hash,  # Now required
    }
    compatible, reason = compiled.check_checkpoint_compatibility(checkpoint)
    assert compatible, reason

    checkpoint["observation_field_uuids"][0] = "deadbeefdeadbeef"
    compatible, reason = compiled.check_checkpoint_compatibility(checkpoint)
    assert not compatible


def test_compiled_universe_create_environment() -> None:
    compiler = UniverseCompiler()
    compiled = compiler.compile(Path("configs/L0_0_minimal"))

    env = compiled.create_environment(num_agents=1, device="cpu")
    from townlet.environment.vectorized_env import VectorizedHamletEnv

    assert isinstance(env, VectorizedHamletEnv)


def test_compiled_universe_environment_rollout(cpu_device: torch.device) -> None:
    compiler = UniverseCompiler()
    compiled = compiler.compile(Path("configs/L0_0_minimal"))

    env = compiled.create_environment(num_agents=2, device=str(cpu_device))

    observations = env.reset()
    assert observations.shape == (2, compiled.metadata.observation_dim)

    wait_idx = env.wait_action_idx
    actions = torch.full((2,), wait_idx, dtype=torch.long, device=env.device)

    next_obs, rewards, dones, info = env.step(actions)

    assert next_obs.shape == observations.shape
    assert rewards.shape == (2,)
    assert dones.shape == (2,)
    assert "positions" in info


def test_vectorized_env_from_universe_factory(cpu_device: torch.device) -> None:
    compiler = UniverseCompiler()
    compiled = compiler.compile(Path("configs/L0_0_minimal"))

    from townlet.environment.vectorized_env import VectorizedHamletEnv

    env = VectorizedHamletEnv.from_universe(compiled, num_agents=1, device=cpu_device)

    assert env.config_pack_path == compiled.config_dir
    assert env.observation_dim == compiled.metadata.observation_dim
    assert env.num_agents == 1


def test_check_checkpoint_compatibility_drive_hash_match() -> None:
    """Checkpoint with matching drive_hash is compatible."""
    import shutil
    import tempfile

    import yaml

    # Create temp config with DAC
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "test_config"
        shutil.copytree(Path("configs/L0_0_minimal"), tmp_path)

        # Add drive_as_code.yaml
        dac_config = {
            "drive_as_code": {
                "version": "1.0",
                "modifiers": {},
                "extrinsic": {"type": "multiplicative", "base": 1.0, "bars": ["energy"]},
                "intrinsic": {"strategy": "rnd", "base_weight": 0.1},
            }
        }
        (tmp_path / "drive_as_code.yaml").write_text(yaml.dump(dac_config))

        compiler = UniverseCompiler()
        compiled = compiler.compile(tmp_path, use_cache=False)

        # Create checkpoint with matching drive_hash
        checkpoint = {
            "config_hash": compiled.metadata.config_hash,
            "observation_dim": compiled.metadata.observation_dim,
            "action_dim": compiled.metadata.action_count,
            "observation_field_uuids": [field.uuid for field in compiled.observation_spec.fields],
            "drive_hash": compiled.drive_hash,  # Matching!
        }

        compatible, message = compiled.check_checkpoint_compatibility(checkpoint)
        assert compatible is True, f"Expected compatible, got: {message}"
        assert "compatible" in message.lower()


def test_check_checkpoint_compatibility_drive_hash_mismatch() -> None:
    """Checkpoint with mismatched drive_hash is incompatible."""
    import shutil
    import tempfile

    import yaml

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "test_config"
        shutil.copytree(Path("configs/L0_0_minimal"), tmp_path)

        # Add drive_as_code.yaml
        dac_config = {
            "drive_as_code": {
                "version": "1.0",
                "modifiers": {},
                "extrinsic": {"type": "multiplicative", "base": 1.0, "bars": ["energy"]},
                "intrinsic": {"strategy": "rnd", "base_weight": 0.1},
            }
        }
        (tmp_path / "drive_as_code.yaml").write_text(yaml.dump(dac_config))

        compiler = UniverseCompiler()
        compiled = compiler.compile(tmp_path, use_cache=False)

        # Create checkpoint with DIFFERENT drive_hash
        checkpoint = {
            "config_hash": compiled.metadata.config_hash,
            "observation_dim": compiled.metadata.observation_dim,
            "action_dim": compiled.metadata.action_count,
            "observation_field_uuids": [field.uuid for field in compiled.observation_spec.fields],
            "drive_hash": "xyz789_different_hash",  # Different!
        }

        compatible, message = compiled.check_checkpoint_compatibility(checkpoint)
        assert compatible is False
        assert "drive hash mismatch" in message.lower()
        assert "reward function has changed" in message.lower()


def test_check_checkpoint_compatibility_missing_drive_hash_in_checkpoint() -> None:
    """Checkpoint without drive_hash but universe has DAC is incompatible."""
    import shutil
    import tempfile

    import yaml

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "test_config"
        shutil.copytree(Path("configs/L0_0_minimal"), tmp_path)

        # Add drive_as_code.yaml
        dac_config = {
            "drive_as_code": {
                "version": "1.0",
                "modifiers": {},
                "extrinsic": {"type": "multiplicative", "base": 1.0, "bars": ["energy"]},
                "intrinsic": {"strategy": "rnd", "base_weight": 0.1},
            }
        }
        (tmp_path / "drive_as_code.yaml").write_text(yaml.dump(dac_config))

        compiler = UniverseCompiler()
        compiled = compiler.compile(tmp_path, use_cache=False)

        # Create checkpoint WITHOUT drive_hash (old checkpoint)
        checkpoint = {
            "config_hash": compiled.metadata.config_hash,
            "observation_dim": compiled.metadata.observation_dim,
            "action_dim": compiled.metadata.action_count,
            "observation_field_uuids": [field.uuid for field in compiled.observation_spec.fields],
            # drive_hash missing!
        }

        compatible, message = compiled.check_checkpoint_compatibility(checkpoint)
        assert compatible is False
        assert "missing drive_hash" in message.lower()
        assert "predates dac" in message.lower()


def test_check_checkpoint_compatibility_missing_drive_hash_in_universe() -> None:
    """Test removed: drive_hash is now always present in universe (DAC is required)."""
    # This test is no longer valid since drive_as_code.yaml is now required
    # All universes will always have a drive_hash
    pass
