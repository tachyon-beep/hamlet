"""Runtime-specific tests for VectorizedHamletEnv."""

from __future__ import annotations

from pathlib import Path

import pytest

from townlet.universe.compiler import UniverseCompiler


@pytest.mark.parametrize("config_name", ["configs/L0_0_minimal"])
def test_vectorized_env_avoids_runtime_yaml_reads(monkeypatch, config_name: str) -> None:
    """Ensure compiled environments no longer reopen bars/variables/action label YAML files."""

    compiler = UniverseCompiler()
    compiled = compiler.compile(Path(config_name))

    blocked = {"bars.yaml", "variables_reference.yaml", "action_labels.yaml"}
    original_open = Path.open

    def guarded_open(self: Path, *args, **kwargs):  # type: ignore[override]
        if self.name in blocked:
            raise AssertionError(f"Unexpected runtime read of {self}")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open, raising=False)

    env = compiled.create_environment(num_agents=1)
    assert env.observation_dim == compiled.metadata.observation_dim
