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
    assert grid_field.scope == "agent", "Grid encoding should stay agent scoped under full observability"
    affordance_field = observation_spec.get_field_by_name("obs_affordance")
    assert affordance_field.dims == 15
    assert affordance_field.scope == "agent"


@pytest.mark.parametrize(
    ("pack_name", "expected_fields", "extra_asserts"),
    [
        (
            "L0_0_minimal",
            [
                ("obs_grid", "agent"),
                ("obs_pos", "agent"),
                ("obs_energy", "agent"),
                ("obs_health", "agent"),
            ],
            (("obs_time_sin", "global"),),
        ),
        (
            "L2_partial_observability",
            [
                ("obs_local_window", "agent"),
                ("obs_pos", "agent"),
                ("obs_energy", "agent"),
                ("obs_health", "agent"),
            ],
            (("obs_time_sin", "global"), ("obs_time_cos", "global")),
        ),
    ],
)
def test_observation_spec_fields_order_and_scope(
    pack_name: str,
    expected_fields: list[tuple[str, str]],
    extra_asserts: tuple[str, ...],
) -> None:
    """Regress ObservationSpec ordering + scope for representative configs."""

    config_dir = Path("configs") / pack_name
    raw_configs = RawConfigs.from_config_dir(config_dir)
    compiler = UniverseCompiler()
    symbol_table = compiler._stage_2_build_symbol_tables(raw_configs)

    _, observation_spec = compiler._stage_5_compute_metadata(config_dir, raw_configs, symbol_table)

    actual_pairs = [(field.name, field.scope) for field in observation_spec.fields[: len(expected_fields)]]
    assert actual_pairs == expected_fields, f"{pack_name}: expected field ordering {expected_fields}, got {actual_pairs}"

    for field_name, expected_scope in extra_asserts:
        field = observation_spec.get_field_by_name(field_name)
        assert field.scope == expected_scope, f"{pack_name}: expected {field_name} scope {expected_scope}, got {field.scope}"


def test_stage5_config_hash_changes_when_config_changes(tmp_path: Path, base_config_dir: Path) -> None:
    temp_pack = tmp_path / "pack"
    shutil.copytree(base_config_dir, temp_pack)

    original = RawConfigs.from_config_dir(temp_pack)
    compiler = UniverseCompiler()
    symbol_table = compiler._stage_2_build_symbol_tables(original)
    metadata_before, _ = compiler._stage_5_compute_metadata(temp_pack, original, symbol_table)

    training_path = temp_pack / "training.yaml"
    text = training_path.read_text()
    text = text.replace("max_episodes: 500", "max_episodes: 501")
    training_path.write_text(text)

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
