"""Comprehensive tests for CuesCompiler validation logic.

These tests address the critical 13% coverage gap in cues_compiler.py:
- Lines 37-68: Basic cue validation (threshold, meter references)
- Lines 83-113: Visual cue validation (coverage, overlap)
- Lines 116-143: Range validation helpers
"""

from __future__ import annotations

import pytest

from townlet.config.bar import BarConfig
from townlet.config.cues import CompoundCueConfig, CueCondition, CuesConfig, SimpleCueConfig, VisualCueConfig
from townlet.universe.cues_compiler import CuesCompiler
from townlet.universe.errors import CompilationErrorCollector, CompilationMessage
from townlet.universe.symbol_table import UniverseSymbolTable


@pytest.fixture
def symbol_table() -> UniverseSymbolTable:
    """Create symbol table with test meters."""
    table = UniverseSymbolTable()
    table.register_meter(
        BarConfig(
            name="energy",
            index=0,
            tier="pivotal",
            range=[0.0, 1.0],
            initial=1.0,
            base_depletion=0.1,
            base_move_depletion=0.01,
            base_interaction_cost=0.01,
        )
    )
    table.register_meter(
        BarConfig(
            name="health",
            index=1,
            tier="pivotal",
            range=[0.0, 1.0],
            initial=1.0,
            base_depletion=0.05,
            base_move_depletion=0.005,
            base_interaction_cost=0.005,
        )
    )
    table.register_meter(
        BarConfig(
            name="mood",
            index=2,
            tier="secondary",
            range=[0.0, 1.0],
            initial=1.0,
            base_depletion=0.02,
            base_move_depletion=0.002,
            base_interaction_cost=0.002,
        )
    )
    return table


@pytest.fixture
def compiler() -> CuesCompiler:
    """Create CuesCompiler instance."""
    return CuesCompiler()


@pytest.fixture
def formatter():
    """Create simple formatter for tests."""
    return lambda code, msg, loc: CompilationMessage(code=code, message=msg, location=loc)


class TestBasicCueValidation:
    """Test basic cue threshold and meter reference validation."""

    def test_simple_cue_with_valid_meter_passes(self, compiler, symbol_table, formatter):
        """Verify simple cue with valid meter passes validation."""
        cues = CuesConfig(
            version="1.0",
            simple_cues=[
                SimpleCueConfig(
                    cue_id="low_energy",
                    name="Test Cue",
                    category="test",
                    visibility="public",
                    condition=CueCondition(meter="energy", threshold=0.3, operator="<"),
                )
            ],
            compound_cues=[],
            visual_cues={},
        )
        errors = CompilationErrorCollector()

        compiler.validate(cues, symbol_table, errors, formatter)

        assert len(errors.errors) == 0

    def test_simple_cue_with_unknown_meter_fails(self, compiler, symbol_table, formatter):
        """Verify simple cue with unknown meter is rejected."""
        cues = CuesConfig(
            version="1.0",
            simple_cues=[
                SimpleCueConfig(
                    cue_id="low_stamina",
                    name="Test Cue",
                    category="test",
                    visibility="public",
                    condition=CueCondition(meter="stamina", threshold=0.3, operator="<"),
                )
            ],
            compound_cues=[],
            visual_cues={},
        )
        errors = CompilationErrorCollector()

        compiler.validate(cues, symbol_table, errors, formatter)

        assert len(errors.errors) == 1
        assert "UAC-VAL-005" in errors.errors[0]
        assert "unknown meter 'stamina'" in errors.errors[0]

    @pytest.mark.parametrize("invalid_threshold", [-0.1, 1.1, -10.0, 5.0, float("inf")])
    def test_simple_cue_threshold_out_of_range_fails(self, compiler, symbol_table, formatter, invalid_threshold):
        """Verify simple cue threshold outside [0.0, 1.0] is rejected."""
        cues = CuesConfig(
            version="1.0",
            simple_cues=[
                SimpleCueConfig(
                    cue_id="test_cue",
                    name="Test Cue",
                    category="test",
                    visibility="public",
                    condition=CueCondition(meter="energy", threshold=invalid_threshold, operator="<"),
                )
            ],
            compound_cues=[],
            visual_cues={},
        )
        errors = CompilationErrorCollector()

        compiler.validate(cues, symbol_table, errors, formatter)

        assert len(errors.errors) >= 1
        assert any("threshold must be within [0.0, 1.0]" in err for err in errors.errors)

    @pytest.mark.parametrize("valid_threshold", [0.0, 0.5, 1.0, 0.001, 0.999])
    def test_simple_cue_valid_threshold_passes(self, compiler, symbol_table, formatter, valid_threshold):
        """Verify simple cue with valid threshold passes."""
        cues = CuesConfig(
            version="1.0",
            simple_cues=[
                SimpleCueConfig(
                    cue_id="test_cue",
                    name="Test Cue",
                    category="test",
                    visibility="public",
                    condition=CueCondition(meter="energy", threshold=valid_threshold, operator="<"),
                )
            ],
            compound_cues=[],
            visual_cues={},
        )
        errors = CompilationErrorCollector()

        compiler.validate(cues, symbol_table, errors, formatter)

        assert len(errors.errors) == 0

    def test_compound_cue_with_valid_conditions_passes(self, compiler, symbol_table, formatter):
        """Verify compound cue with all valid conditions passes."""
        cues = CuesConfig(
            version="1.0",
            simple_cues=[],
            compound_cues=[
                CompoundCueConfig(
                    cue_id="critical_state",
                    name="Test Compound",
                    category="test",
                    visibility="public",
                    logic="all_of",
                    conditions=[
                        CueCondition(meter="energy", threshold=0.2, operator="<"),
                        CueCondition(meter="health", threshold=0.3, operator="<"),
                    ],
                )
            ],
            visual_cues={},
        )
        errors = CompilationErrorCollector()

        compiler.validate(cues, symbol_table, errors, formatter)

        assert len(errors.errors) == 0

    def test_compound_cue_with_unknown_meter_fails(self, compiler, symbol_table, formatter):
        """Verify compound cue condition with unknown meter is rejected."""
        cues = CuesConfig(
            version="1.0",
            simple_cues=[],
            compound_cues=[
                CompoundCueConfig(
                    cue_id="complex_state",
                    name="Test Compound",
                    category="test",
                    visibility="public",
                    logic="all_of",
                    conditions=[
                        CueCondition(meter="energy", threshold=0.2, operator="<"),
                        CueCondition(meter="stamina", threshold=0.3, operator="<"),  # Unknown!
                    ],
                )
            ],
            visual_cues={},
        )
        errors = CompilationErrorCollector()

        compiler.validate(cues, symbol_table, errors, formatter)

        assert len(errors.errors) == 1
        assert "unknown meter 'stamina'" in errors.errors[0]

    def test_compound_cue_invalid_threshold_fails(self, compiler, symbol_table, formatter):
        """Verify compound cue condition with invalid threshold is rejected."""
        cues = CuesConfig(
            version="1.0",
            simple_cues=[],
            compound_cues=[
                CompoundCueConfig(
                    cue_id="bad_compound",
                    name="Test Compound",
                    category="test",
                    visibility="public",
                    logic="all_of",
                    conditions=[
                        CueCondition(meter="energy", threshold=1.5, operator="<"),  # Invalid!
                    ],
                )
            ],
            visual_cues={},
        )
        errors = CompilationErrorCollector()

        compiler.validate(cues, symbol_table, errors, formatter)

        assert len(errors.errors) == 1
        assert "threshold must be within [0.0, 1.0]" in errors.errors[0]

    def test_multiple_invalid_cues_accumulate_errors(self, compiler, symbol_table, formatter):
        """Verify multiple validation errors are accumulated."""
        cues = CuesConfig(
            version="1.0",
            simple_cues=[
                SimpleCueConfig(
                    cue_id="bad_meter",
                    name="Test Cue",
                    category="test",
                    visibility="public",
                    condition=CueCondition(meter="nonexistent", threshold=0.5, operator="<"),
                ),
                SimpleCueConfig(
                    cue_id="bad_threshold",
                    name="Test Cue",
                    category="test",
                    visibility="public",
                    condition=CueCondition(meter="energy", threshold=2.0, operator="<"),
                ),
            ],
            compound_cues=[],
            visual_cues={},
        )
        errors = CompilationErrorCollector()

        compiler.validate(cues, symbol_table, errors, formatter)

        assert len(errors.errors) == 2


class TestVisualCueValidation:
    """Test visual cue range coverage and overlap validation."""

    def test_visual_cue_with_unknown_meter_fails(self, compiler, symbol_table, formatter):
        """Verify visual cue for unknown meter is rejected."""
        cues = CuesConfig(
            version="1.0",
            simple_cues=[],
            compound_cues=[],
            visual_cues={
                "stamina": [  # Unknown meter!
                    VisualCueConfig(range=(0.0, 1.0), label="green"),
                ]
            },
        )
        errors = CompilationErrorCollector()

        compiler.validate(cues, symbol_table, errors, formatter)

        assert len(errors.errors) == 1
        assert "UAC-VAL-009" in errors.errors[0]
        assert "unknown meter 'stamina'" in errors.errors[0]

    def test_visual_cue_full_coverage_passes(self, compiler, symbol_table, formatter):
        """Verify visual cue with full [0.0, 1.0] coverage passes."""
        cues = CuesConfig(
            version="1.0",
            simple_cues=[],
            compound_cues=[],
            visual_cues={
                "energy": [
                    VisualCueConfig(range=(0.0, 0.3), label="red"),
                    VisualCueConfig(range=(0.3, 0.7), label="yellow"),
                    VisualCueConfig(range=(0.7, 1.0), label="green"),
                ]
            },
        )
        errors = CompilationErrorCollector()

        compiler.validate(cues, symbol_table, errors, formatter)

        assert len(errors.errors) == 0

    def test_visual_cue_gap_in_coverage_fails(self, compiler, symbol_table, formatter):
        """Verify visual cue with gap in coverage is rejected."""
        cues = CuesConfig(
            version="1.0",
            simple_cues=[],
            compound_cues=[],
            visual_cues={
                "energy": [
                    VisualCueConfig(range=(0.0, 0.3), label="red"),
                    VisualCueConfig(range=(0.5, 1.0), label="green"),  # Gap: 0.3-0.5 missing!
                ]
            },
        )
        errors = CompilationErrorCollector()

        compiler.validate(cues, symbol_table, errors, formatter)

        assert len(errors.errors) == 1
        assert "do not cover [0.0, 1.0] without gaps" in errors.errors[0]

    def test_visual_cue_missing_start_fails(self, compiler, symbol_table, formatter):
        """Verify visual cue not starting at 0.0 is rejected."""
        cues = CuesConfig(
            version="1.0",
            simple_cues=[],
            compound_cues=[],
            visual_cues={
                "energy": [
                    VisualCueConfig(range=(0.1, 1.0), label="green"),  # Doesn't start at 0.0!
                ]
            },
        )
        errors = CompilationErrorCollector()

        compiler.validate(cues, symbol_table, errors, formatter)

        assert len(errors.errors) == 1
        assert "do not cover [0.0, 1.0]" in errors.errors[0]

    def test_visual_cue_missing_end_fails(self, compiler, symbol_table, formatter):
        """Verify visual cue not ending at 1.0 is rejected."""
        cues = CuesConfig(
            version="1.0",
            simple_cues=[],
            compound_cues=[],
            visual_cues={
                "energy": [
                    VisualCueConfig(range=(0.0, 0.9), label="green"),  # Doesn't end at 1.0!
                ]
            },
        )
        errors = CompilationErrorCollector()

        compiler.validate(cues, symbol_table, errors, formatter)

        assert len(errors.errors) == 1
        assert "do not cover [0.0, 1.0]" in errors.errors[0]

    def test_visual_cue_overlap_fails(self, compiler, symbol_table, formatter):
        """Verify overlapping visual cue ranges are rejected."""
        cues = CuesConfig(
            version="1.0",
            simple_cues=[],
            compound_cues=[],
            visual_cues={
                "energy": [
                    VisualCueConfig(range=(0.0, 0.6), label="red"),  # Overlaps with next!
                    VisualCueConfig(range=(0.5, 1.0), label="green"),
                ]
            },
        )
        errors = CompilationErrorCollector()

        compiler.validate(cues, symbol_table, errors, formatter)

        # Validator correctly reports both overlap AND coverage gap
        assert len(errors.errors) >= 1
        assert any("overlap" in err.lower() for err in errors.errors)

    def test_empty_visual_cues_passes(self, compiler, symbol_table, formatter):
        """Verify empty visual_cues dict passes validation."""
        cues = CuesConfig(version="1.0", simple_cues=[], compound_cues=[], visual_cues={})
        errors = CompilationErrorCollector()

        compiler.validate(cues, symbol_table, errors, formatter)

        assert len(errors.errors) == 0

    def test_visual_cue_multiple_meters_validated_independently(self, compiler, symbol_table, formatter):
        """Verify multiple visual cue meters are validated independently."""
        cues = CuesConfig(
            version="1.0",
            simple_cues=[],
            compound_cues=[],
            visual_cues={
                "energy": [  # Valid
                    VisualCueConfig(range=(0.0, 1.0), label="green"),
                ],
                "health": [  # Invalid - has gap
                    VisualCueConfig(range=(0.0, 0.5), label="red"),
                    VisualCueConfig(range=(0.7, 1.0), label="green"),
                ],
            },
        )
        errors = CompilationErrorCollector()

        compiler.validate(cues, symbol_table, errors, formatter)

        # Should only error on health, not energy
        assert len(errors.errors) == 1
        assert "health" in errors.errors[0]


class TestRangeHelpers:
    """Test range validation helper methods directly."""

    def test_ranges_cover_domain_with_full_coverage(self, compiler):
        """Verify _ranges_cover_domain returns True for full coverage."""
        ranges = [(0.0, 0.5), (0.5, 1.0)]
        assert compiler._ranges_cover_domain(ranges, 0.0, 1.0) is True

    def test_ranges_cover_domain_with_gap(self, compiler):
        """Verify _ranges_cover_domain returns False with gap."""
        ranges = [(0.0, 0.4), (0.6, 1.0)]
        assert compiler._ranges_cover_domain(ranges, 0.0, 1.0) is False

    def test_ranges_cover_domain_missing_start(self, compiler):
        """Verify _ranges_cover_domain returns False when not starting at domain_min."""
        ranges = [(0.1, 1.0)]
        assert compiler._ranges_cover_domain(ranges, 0.0, 1.0) is False

    def test_ranges_cover_domain_missing_end(self, compiler):
        """Verify _ranges_cover_domain returns False when not ending at domain_max."""
        ranges = [(0.0, 0.9)]
        assert compiler._ranges_cover_domain(ranges, 0.0, 1.0) is False

    def test_ranges_cover_domain_empty_ranges(self, compiler):
        """Verify _ranges_cover_domain returns False for empty list."""
        assert compiler._ranges_cover_domain([], 0.0, 1.0) is False

    def test_ranges_cover_domain_unsorted_ranges(self, compiler):
        """Verify _ranges_cover_domain handles unsorted ranges correctly."""
        ranges = [(0.5, 1.0), (0.0, 0.5)]  # Out of order
        assert compiler._ranges_cover_domain(ranges, 0.0, 1.0) is True

    def test_ranges_overlap_with_no_overlap(self, compiler):
        """Verify _ranges_overlap returns False when ranges don't overlap."""
        ranges = [(0.0, 0.3), (0.3, 0.7), (0.7, 1.0)]
        assert compiler._ranges_overlap(ranges) is False

    def test_ranges_overlap_with_overlap(self, compiler):
        """Verify _ranges_overlap returns True when ranges overlap."""
        ranges = [(0.0, 0.5), (0.4, 1.0)]  # 0.4-0.5 overlap
        assert compiler._ranges_overlap(ranges) is True

    def test_ranges_overlap_empty_ranges(self, compiler):
        """Verify _ranges_overlap returns False for empty list."""
        assert compiler._ranges_overlap([]) is False

    def test_ranges_overlap_single_range(self, compiler):
        """Verify _ranges_overlap returns False for single range."""
        assert compiler._ranges_overlap([(0.0, 1.0)]) is False

    def test_ranges_overlap_exact_boundary_touch(self, compiler):
        """Verify _ranges_overlap returns False when ranges touch at exact boundary."""
        ranges = [(0.0, 0.5), (0.5, 1.0)]  # Touch at 0.5, no overlap
        assert compiler._ranges_overlap(ranges) is False

    def test_ranges_cover_domain_handles_floating_point_precision(self, compiler):
        """Verify _ranges_cover_domain handles floating point precision correctly."""
        # Ranges with floating point precision issues
        ranges = [(0.0, 0.33333333), (0.33333333, 0.66666666), (0.66666666, 1.0)]
        # Should pass within epsilon tolerance
        assert compiler._ranges_cover_domain(ranges, 0.0, 1.0) is True
