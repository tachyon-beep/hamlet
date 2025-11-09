"""Tests for Stage 5 metadata computation."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from townlet.universe.compiler import UniverseCompiler
from townlet.universe.compiler_inputs import RawConfigs


@pytest.fixture(scope="module")
def base_config_dir() -> Path:
    return Path("configs/L0_0_minimal")


@pytest.fixture(scope="module")
def base_raw_configs(base_config_dir: Path) -> RawConfigs:
    return RawConfigs.from_config_dir(base_config_dir)


def test_stage5_computes_metadata_and_observation_spec(base_config_dir: Path, base_raw_configs: RawConfigs) -> None:
    compiler = UniverseCompiler()
    symbol_table = compiler._stage_2_build_symbol_tables(base_raw_configs)

    metadata, observation_spec = compiler._stage_5_compute_metadata(base_config_dir, base_raw_configs, symbol_table)

    assert metadata.universe_name == base_config_dir.name
    assert metadata.meter_count == len(base_raw_configs.bars)
    assert metadata.affordance_count == len(base_raw_configs.affordances)
    assert metadata.action_count == len(base_raw_configs.global_actions.actions)
    assert metadata.observation_dim == observation_spec.total_dims
    assert metadata.config_hash
    assert metadata.provenance_id

    grid_field = observation_spec.get_field_by_name("obs_grid")
    assert grid_field.dims == 9
    affordance_field = observation_spec.get_field_by_name("obs_affordance")
    assert affordance_field.dims == 15


def test_stage5_config_hash_changes_when_config_changes(tmp_path: Path, base_config_dir: Path) -> None:
    temp_pack = tmp_path / "pack"
    shutil.copytree(base_config_dir, temp_pack)

    original = RawConfigs.from_config_dir(temp_pack)
    compiler = UniverseCompiler()
    symbol_table = compiler._stage_2_build_symbol_tables(original)
    metadata_before, _ = compiler._stage_5_compute_metadata(temp_pack, original, symbol_table)

    training_path = temp_pack / "training.yaml"
    training_path.write_text(training_path.read_text() + "\n# stage5-test\n")

    mutated = RawConfigs.from_config_dir(temp_pack)
    symbol_table_mut = compiler._stage_2_build_symbol_tables(mutated)
    metadata_after, _ = compiler._stage_5_compute_metadata(temp_pack, mutated, symbol_table_mut)

    assert metadata_before.config_hash != metadata_after.config_hash


def test_stage5_builds_rich_metadata(base_raw_configs: RawConfigs) -> None:
    compiler = UniverseCompiler()

    action_meta, meter_meta, affordance_meta = compiler._stage_5_build_rich_metadata(base_raw_configs)

    assert action_meta.total_actions == len(base_raw_configs.global_actions.actions)
    assert meter_meta.get_meter_by_name("energy").index == 0
    bed_info = affordance_meta.get_affordance_by_name("Bed")
    assert bed_info.enabled
    assert bed_info.effects, "Bed affordance should expose summarized effects"
