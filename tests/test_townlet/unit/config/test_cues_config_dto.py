"""Tests for CuesConfig DTO."""

from pathlib import Path

import pytest

from tests.test_townlet.unit.config.fixtures import VALID_CUES_CONFIG
from townlet.config.cues import CuesConfig, load_cues_config


class TestCuesConfigValidation:
    """Unit tests for cues schema."""

    def test_total_cues_property(self):
        """total_cues returns simple + compound."""
        config = CuesConfig(**VALID_CUES_CONFIG)
        assert config.total_cues == len(VALID_CUES_CONFIG["simple_cues"])

    def test_compound_cue_requires_condition(self):
        """Compound cues must specify at least one condition."""
        bad_config = {
            **VALID_CUES_CONFIG,
            "compound_cues": [
                {
                    "cue_id": "invalid",
                    "name": "Invalid",
                    "category": "test",
                    "visibility": "public",
                    "logic": "all_of",
                    "conditions": [],
                }
            ],
        }
        with pytest.raises(Exception):
            CuesConfig(**bad_config)


class TestCuesConfigLoading:
    """Tests for load_cues_config helper."""

    def test_load_cues_config(self, tmp_path: Path):
        """Valid cues.yaml loads successfully."""
        cues_path = tmp_path / "cues.yaml"
        cues_path.write_text("version: '1.0'\nstatus: TEMPLATE\nsimple_cues: []\ncompound_cues: []\n")

        config = load_cues_config(cues_path)
        assert config.version == "1.0"
        assert config.total_cues == 0

    def test_missing_file_error(self, tmp_path: Path):
        """Missing cues.yaml raises helpful error."""
        with pytest.raises(FileNotFoundError):
            load_cues_config(tmp_path / "missing.yaml")
