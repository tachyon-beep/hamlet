"""Tests for the UniverseCompiler Stage 1 loader."""

from pathlib import Path

import pytest

from townlet.universe.compiler import UniverseCompiler


@pytest.mark.parametrize("pack_name", ["L0_0_minimal", "L1_full_observability"])
def test_stage1_parses_config_pack(pack_name: str):
    compiler = UniverseCompiler()
    config_dir = Path("configs") / pack_name

    raw_configs = compiler._stage_1_parse_individual_files(config_dir)

    assert raw_configs.training.max_episodes > 0
    assert len(raw_configs.variables_reference) > 0
    assert len(raw_configs.global_actions.actions) > 0


def test_compile_not_implemented():
    compiler = UniverseCompiler()
    config_dir = Path("configs/L0_0_minimal")
    with pytest.raises(NotImplementedError):
        compiler.compile(config_dir)
