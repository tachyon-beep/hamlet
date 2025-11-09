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

    incompatible, _ = compiled.check_checkpoint_compatibility({})
    assert not incompatible

    checkpoint = {
        "config_hash": compiled.metadata.config_hash,
        "observation_dim": compiled.metadata.observation_dim,
        "action_dim": compiled.metadata.action_count,
    }
    compatible, reason = compiled.check_checkpoint_compatibility(checkpoint)
    assert compatible, reason


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
