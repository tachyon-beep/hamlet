"""Comprehensive tests for UniverseSymbolTable to achieve 70%+ coverage.

These tests address gaps in symbol_table.py:
- Lines 30-60: Registration and duplicate detection for all entity types
- Lines 63-97: Getter methods, properties, and lookups
"""

from __future__ import annotations

from pathlib import Path

import pytest

from townlet.config.affordance import AffordanceConfig
from townlet.config.bar import BarConfig
from townlet.universe.compiler_inputs import RawConfigs
from townlet.universe.errors import CompilationError
from townlet.universe.symbol_table import UniverseSymbolTable


@pytest.fixture(scope="module")
def raw_configs() -> RawConfigs:
    """Load real configs for testing."""
    return RawConfigs.from_config_dir(Path("configs/L0_0_minimal"))


@pytest.fixture
def table() -> UniverseSymbolTable:
    """Create fresh symbol table for each test."""
    return UniverseSymbolTable()


@pytest.fixture
def sample_meter(raw_configs) -> BarConfig:
    """Get sample meter from real config."""
    return raw_configs.bars[0]


@pytest.fixture
def sample_variable(raw_configs):
    """Get sample variable from real config."""
    # variables_reference is a list of VariableDef
    return raw_configs.variables_reference[0]


@pytest.fixture
def sample_action(raw_configs):
    """Get sample action from real config."""
    return raw_configs.global_actions.actions[0]


@pytest.fixture
def sample_cascade(raw_configs):
    """Get sample cascade from real config."""
    # L0_0_minimal may not have cascades, return None if empty
    return raw_configs.cascades[0] if raw_configs.cascades else None


@pytest.fixture
def sample_affordance(raw_configs) -> AffordanceConfig:
    """Get sample affordance from real config."""
    return raw_configs.affordances[0]


@pytest.fixture
def sample_simple_cue(raw_configs):
    """Get sample simple cue from real config."""
    return raw_configs.cues[0] if raw_configs.cues else None


@pytest.fixture
def sample_compound_cue(raw_configs):
    """Get sample compound cue from real config - may be None."""
    # Find a compound cue if exists
    for cue in raw_configs.cues:
        if hasattr(cue, "logic"):
            return cue
    return None


class TestMeterRegistration:
    """Test meter registration and retrieval."""

    def test_register_meter_stores_config(self, table, sample_meter):
        """Verify meter registration stores config."""
        table.register_meter(sample_meter)
        assert "energy" in table.meters
        assert table.meters["energy"] is sample_meter

    def test_register_duplicate_meter_raises_error(self, table, sample_meter):
        """Verify duplicate meter registration raises CompilationError."""
        table.register_meter(sample_meter)

        with pytest.raises(CompilationError, match="Duplicate meter 'energy'"):
            table.register_meter(sample_meter)

    def test_get_meter_returns_config(self, table, sample_meter):
        """Verify get_meter returns registered config."""
        table.register_meter(sample_meter)
        retrieved = table.get_meter("energy")
        assert retrieved is sample_meter

    def test_get_meter_raises_keyerror_for_unknown(self, table):
        """Verify get_meter raises KeyError for unknown meter."""
        with pytest.raises(KeyError):
            table.get_meter("nonexistent")

    def test_get_meter_count_returns_correct_count(self, table):
        """Verify get_meter_count returns number of registered meters."""
        assert table.get_meter_count() == 0

        table.register_meter(BarConfig(name="energy", index=0, tier="pivotal", range=[0, 1], initial=1.0, base_depletion=0.1))
        assert table.get_meter_count() == 1

        table.register_meter(BarConfig(name="health", index=1, tier="pivotal", range=[0, 1], initial=1.0, base_depletion=0.05))
        assert table.get_meter_count() == 2

    def test_get_meter_names_sorts_by_index(self, table):
        """Verify get_meter_names returns names sorted by meter index."""
        # Register out of order
        table.register_meter(BarConfig(name="health", index=2, tier="pivotal", range=[0, 1], initial=1.0, base_depletion=0.05))
        table.register_meter(BarConfig(name="energy", index=0, tier="pivotal", range=[0, 1], initial=1.0, base_depletion=0.1))
        table.register_meter(BarConfig(name="mood", index=1, tier="secondary", range=[0, 1], initial=1.0, base_depletion=0.02))

        names = table.get_meter_names()
        assert names == ["energy", "mood", "health"]

    def test_resolve_meter_reference_returns_config(self, table, sample_meter):
        """Verify resolve_meter_reference returns config for valid meter."""
        table.register_meter(sample_meter)
        resolved = table.resolve_meter_reference("energy", location="test.yaml:line1")
        assert resolved is sample_meter

    def test_resolve_meter_reference_raises_for_unknown(self, table, sample_meter):
        """Verify resolve_meter_reference raises ReferenceError for unknown meter."""
        table.register_meter(sample_meter)

        with pytest.raises(ReferenceError, match="non-existent meter 'stamina'"):
            table.resolve_meter_reference("stamina", location="test.yaml:line1")

    def test_resolve_meter_reference_includes_valid_meters_in_error(self, table, sample_meter):
        """Verify resolve_meter_reference error lists valid meters."""
        table.register_meter(sample_meter)

        with pytest.raises(ReferenceError) as exc_info:
            table.resolve_meter_reference("stamina")

        assert "['energy']" in str(exc_info.value)


class TestVariableRegistration:
    """Test variable registration and retrieval."""

    def test_register_variable_stores_config(self, table, sample_variable):
        """Verify variable registration stores config."""
        table.register_variable(sample_variable)
        assert sample_variable.id in table.variables
        assert table.variables[sample_variable.id] is sample_variable

    def test_register_duplicate_variable_raises_error(self, table, sample_variable):
        """Verify duplicate variable registration raises CompilationError."""
        table.register_variable(sample_variable)

        with pytest.raises(CompilationError, match=f"Duplicate variable '{sample_variable.id}'"):
            table.register_variable(sample_variable)

    def test_get_variable_returns_config(self, table, sample_variable):
        """Verify get_variable returns registered config."""
        table.register_variable(sample_variable)
        retrieved = table.get_variable(sample_variable.id)
        assert retrieved is sample_variable

    def test_get_variable_raises_keyerror_for_unknown(self, table):
        """Verify get_variable raises KeyError for unknown variable."""
        with pytest.raises(KeyError):
            table.get_variable("nonexistent")


class TestActionRegistration:
    """Test action registration and retrieval."""

    def test_register_action_stores_config(self, table, sample_action):
        """Verify action registration stores config."""
        table.register_action(sample_action)
        assert 0 in table.actions
        assert table.actions[0] is sample_action

    def test_register_duplicate_action_raises_error(self, table, sample_action):
        """Verify duplicate action registration raises CompilationError."""
        table.register_action(sample_action)

        with pytest.raises(CompilationError, match="Duplicate action id '0'"):
            table.register_action(sample_action)

    def test_get_action_returns_config(self, table, sample_action):
        """Verify get_action returns registered config."""
        table.register_action(sample_action)
        retrieved = table.get_action(0)
        assert retrieved is sample_action

    def test_get_action_raises_keyerror_for_unknown(self, table):
        """Verify get_action raises KeyError for unknown action."""
        with pytest.raises(KeyError):
            table.get_action(999)


class TestCascadeRegistration:
    """Test cascade registration."""

    def test_register_cascade_stores_config(self, table, sample_cascade):
        """Verify cascade registration stores config."""
        if sample_cascade is None:
            pytest.skip("No cascades in L0_0_minimal config")

        table.register_cascade(sample_cascade)
        assert sample_cascade.name in table.cascades
        assert table.cascades[sample_cascade.name] is sample_cascade

    def test_register_duplicate_cascade_raises_error(self, table, sample_cascade):
        """Verify duplicate cascade registration raises CompilationError."""
        if sample_cascade is None:
            pytest.skip("No cascades in L0_0_minimal config")

        table.register_cascade(sample_cascade)

        with pytest.raises(CompilationError, match=f"Duplicate cascade '{sample_cascade.name}'"):
            table.register_cascade(sample_cascade)


class TestAffordanceRegistration:
    """Test affordance registration with dual-key tracking."""

    def test_register_affordance_stores_by_id(self, table, sample_affordance):
        """Verify affordance registration stores by ID."""
        table.register_affordance(sample_affordance)
        assert "0" in table.affordances
        assert table.affordances["0"] is sample_affordance

    def test_register_affordance_stores_by_name(self, table, sample_affordance):
        """Verify affordance registration stores by name."""
        table.register_affordance(sample_affordance)
        assert "Bed" in table.affordances_by_name
        assert table.affordances_by_name["Bed"] is sample_affordance

    def test_register_duplicate_affordance_id_raises_error(self, table, sample_affordance):
        """Verify duplicate affordance ID raises CompilationError."""
        table.register_affordance(sample_affordance)

        with pytest.raises(CompilationError, match="Duplicate affordance '0'"):
            table.register_affordance(sample_affordance)

    def test_register_duplicate_affordance_name_raises_error(self, table, raw_configs):
        """Verify duplicate affordance name raises CompilationError."""
        aff1 = raw_configs.affordances[0]
        # Create a duplicate with different ID but same name
        aff2 = AffordanceConfig(**{**aff1.model_dump(), "id": "999"})

        table.register_affordance(aff1)

        with pytest.raises(CompilationError, match=f"Duplicate affordance name '{aff1.name}'"):
            table.register_affordance(aff2)

    def test_get_affordance_returns_config_by_id(self, table, sample_affordance):
        """Verify get_affordance returns config by ID."""
        table.register_affordance(sample_affordance)
        retrieved = table.get_affordance("0")
        assert retrieved is sample_affordance

    def test_get_affordance_raises_keyerror_for_unknown_id(self, table):
        """Verify get_affordance raises KeyError for unknown ID."""
        with pytest.raises(KeyError):
            table.get_affordance("999")

    def test_get_affordance_by_name_returns_config(self, table, sample_affordance):
        """Verify get_affordance_by_name returns config by name."""
        table.register_affordance(sample_affordance)
        retrieved = table.get_affordance_by_name("Bed")
        assert retrieved is sample_affordance

    def test_get_affordance_by_name_raises_keyerror_for_unknown(self, table):
        """Verify get_affordance_by_name raises KeyError for unknown name."""
        with pytest.raises(KeyError):
            table.get_affordance_by_name("Hospital")

    def test_get_affordance_count_returns_correct_count(self, table, raw_configs):
        """Verify get_affordance_count returns number of registered affordances."""
        assert table.get_affordance_count() == 0

        table.register_affordance(raw_configs.affordances[0])
        assert table.get_affordance_count() == 1

        if len(raw_configs.affordances) > 1:
            table.register_affordance(raw_configs.affordances[1])
            assert table.get_affordance_count() == 2

    def test_affordance_ids_property_returns_sorted_ids(self, table, raw_configs):
        """Verify affordance_ids property returns sorted list."""
        # Register affordances (they're already sorted in the config)
        for aff in raw_configs.affordances:
            table.register_affordance(aff)

        ids = table.affordance_ids
        assert ids == sorted(ids)  # Verify they're sorted

    def test_affordance_names_property_returns_sorted_names(self, table, raw_configs):
        """Verify affordance_names property returns sorted list."""
        # Register affordances
        for aff in raw_configs.affordances:
            table.register_affordance(aff)

        names = table.affordance_names
        assert names == sorted(names)  # Verify they're sorted


class TestCueRegistration:
    """Test cue registration for both simple and compound cues."""

    def test_register_simple_cue_stores_config(self, table, sample_simple_cue):
        """Verify simple cue registration stores config."""
        if sample_simple_cue is None:
            pytest.skip("No cues in L0_0_minimal config")

        table.register_cue(sample_simple_cue)
        assert sample_simple_cue.cue_id in table.cues
        assert table.cues[sample_simple_cue.cue_id] is sample_simple_cue

    def test_register_compound_cue_stores_config(self, table, sample_compound_cue):
        """Verify compound cue registration stores config."""
        if sample_compound_cue is None:
            pytest.skip("No compound cues in L0_0_minimal config")

        table.register_cue(sample_compound_cue)
        assert sample_compound_cue.cue_id in table.cues
        assert table.cues[sample_compound_cue.cue_id] is sample_compound_cue

    def test_register_duplicate_simple_cue_raises_error(self, table, sample_simple_cue):
        """Verify duplicate simple cue registration raises CompilationError."""
        if sample_simple_cue is None:
            pytest.skip("No cues in L0_0_minimal config")

        table.register_cue(sample_simple_cue)

        with pytest.raises(CompilationError, match=f"Duplicate cue '{sample_simple_cue.cue_id}'"):
            table.register_cue(sample_simple_cue)

    def test_register_duplicate_compound_cue_raises_error(self, table, sample_compound_cue):
        """Verify duplicate compound cue registration raises CompilationError."""
        if sample_compound_cue is None:
            pytest.skip("No compound cues in L0_0_minimal config")

        table.register_cue(sample_compound_cue)

        with pytest.raises(CompilationError, match=f"Duplicate cue '{sample_compound_cue.cue_id}'"):
            table.register_cue(sample_compound_cue)

    def test_register_simple_and_compound_cues_together(self, table, raw_configs):
        """Verify both simple and compound cues can be registered together."""
        if not raw_configs.cues or len(raw_configs.cues) < 2:
            pytest.skip("Not enough cues in config")

        table.register_cue(raw_configs.cues[0])
        table.register_cue(raw_configs.cues[1])

        assert len(table.cues) >= 2


class TestEmptyTable:
    """Test symbol table behavior when empty."""

    def test_empty_table_has_zero_meters(self, table):
        """Verify empty table reports 0 meters."""
        assert table.get_meter_count() == 0
        assert table.get_meter_names() == []

    def test_empty_table_has_zero_affordances(self, table):
        """Verify empty table reports 0 affordances."""
        assert table.get_affordance_count() == 0
        assert table.affordance_ids == []
        assert table.affordance_names == []

    def test_empty_table_has_empty_dicts(self, table):
        """Verify empty table has empty storage dicts."""
        assert len(table.meters) == 0
        assert len(table.cascades) == 0
        assert len(table.affordances) == 0
        assert len(table.affordances_by_name) == 0
        assert len(table.variables) == 0
        assert len(table.actions) == 0
        assert len(table.cues) == 0
