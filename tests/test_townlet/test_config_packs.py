"""Tests for configuration pack loading (affordances/bars/cascades per pack)."""

import shutil
from pathlib import Path

import pytest

from townlet.environment.vectorized_env import VectorizedHamletEnv


@pytest.fixture
def test_pack_dir(tmp_path: Path) -> Path:
    """Return a writable copy of the default test config pack."""
    source_pack = Path("configs/test")
    target_pack = tmp_path / "custom_pack"
    shutil.copytree(source_pack, target_pack)
    return target_pack


def test_vectorized_env_uses_pack_specific_bars(test_pack_dir: Path):
    """Ensure VectorizedHamletEnv reads bars.yaml from the selected pack."""
    bars_path = test_pack_dir / "bars.yaml"
    original = bars_path.read_text()

    if "base_depletion: 0.005" not in original:
        pytest.fail("Unexpected bars.yaml fixture content: missing base_depletion 0.005")

    modified = original.replace("base_depletion: 0.005", "base_depletion: 0.010", 1)
    bars_path.write_text(modified)

    env = VectorizedHamletEnv(num_agents=1, config_pack_path=test_pack_dir)

    energy_base = env.meter_dynamics.cascade_engine._base_depletions[0].item()
    assert energy_base == pytest.approx(0.010, rel=1e-6)
