"""Regression tests for the config_pack_factory fixture."""

from __future__ import annotations

import yaml


def test_config_pack_factory_returns_unique_directories(config_pack_factory):
    """Each invocation should return a distinct, existing directory."""

    pack_a = config_pack_factory()
    pack_b = config_pack_factory()

    assert pack_a != pack_b, "Fixture must create unique directories"
    assert pack_a.exists()
    assert pack_b.exists()


def test_config_pack_factory_applies_modifier(config_pack_factory):
    """Modifier callback should be applied to training.yaml content."""

    def _modifier(config: dict):
        config.setdefault("environment", {})["grid_size"] = 5

    pack = config_pack_factory(modifier=_modifier)

    with open(pack / "training.yaml") as handle:
        training = yaml.safe_load(handle)

    assert training["environment"]["grid_size"] == 5
