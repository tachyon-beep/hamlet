"""Test VFS schema definitions (Cycle 1 - TDD RED phase).

This module tests Pydantic schemas for the Variable & Feature System (VFS).
All tests should FAIL initially (RED), then pass after implementation (GREEN).
"""

import pytest
from pydantic import ValidationError


class TestVariableDef:
    """Test VariableDef schema validation."""

    def test_scalar_variable_valid(self):
        """Scalar variable with all required fields."""
        from townlet.vfs.schema import VariableDef

        var = VariableDef(
            id="energy",
            scope="agent",
            type="scalar",
            lifetime="episode",
            readable_by=["agent", "engine"],
            writable_by=["engine"],
            default=1.0,
        )

        assert var.id == "energy"
        assert var.scope == "agent"
        assert var.type == "scalar"
        assert var.lifetime == "episode"
        assert var.readable_by == ["agent", "engine"]
        assert var.writable_by == ["engine"]
        assert var.default == 1.0

    def test_vecNf_variable_valid(self):
        """N-dimensional float vector with dims specified."""
        from townlet.vfs.schema import VariableDef

        var = VariableDef(
            id="position",
            scope="agent",
            type="vecNf",
            dims=2,
            lifetime="episode",
            readable_by=["agent"],
            writable_by=["engine"],
            default=[0.0, 0.0],
        )

        assert var.type == "vecNf"
        assert var.dims == 2
        assert var.default == [0.0, 0.0]

    def test_vecNi_variable_valid(self):
        """N-dimensional int vector with dims specified."""
        from townlet.vfs.schema import VariableDef

        var = VariableDef(
            id="grid_pos",
            scope="agent",
            type="vecNi",
            dims=2,
            lifetime="episode",
            readable_by=["agent"],
            writable_by=["engine"],
            default=[0, 0],
        )

        assert var.type == "vecNi"
        assert var.dims == 2

    def test_bool_variable_valid(self):
        """Boolean variable."""
        from townlet.vfs.schema import VariableDef

        var = VariableDef(
            id="is_tired",
            scope="agent",
            type="bool",
            lifetime="tick",
            readable_by=["agent"],
            writable_by=["engine"],
            default=False,
        )

        assert var.type == "bool"
        assert var.default is False

    def test_global_scope_valid(self):
        """Global scope variable (single value for all agents)."""
        from townlet.vfs.schema import VariableDef

        var = VariableDef(
            id="time_sin",
            scope="global",
            type="scalar",
            lifetime="tick",
            readable_by=["agent"],
            writable_by=["engine"],
            default=0.0,
        )

        assert var.scope == "global"

    def test_agent_private_scope_valid(self):
        """Agent-private scope (not observable by other agents)."""
        from townlet.vfs.schema import VariableDef

        var = VariableDef(
            id="home_position",
            scope="agent_private",
            type="vecNf",
            dims=2,
            lifetime="episode",
            readable_by=["agent"],  # Owner agent only
            writable_by=["engine"],
            default=[0.0, 0.0],
        )

        assert var.scope == "agent_private"

    def test_invalid_scope_rejected(self):
        """Invalid scope should raise ValidationError."""
        from townlet.vfs.schema import VariableDef

        with pytest.raises(ValidationError):
            VariableDef(
                id="test",
                scope="invalid_scope",
                type="scalar",
                lifetime="episode",
                readable_by=["agent"],
                writable_by=["engine"],
                default=0.0,
            )

    def test_invalid_type_rejected(self):
        """Invalid type should raise ValidationError."""
        from townlet.vfs.schema import VariableDef

        with pytest.raises(ValidationError):
            VariableDef(
                id="test",
                scope="agent",
                type="invalid_type",
                lifetime="episode",
                readable_by=["agent"],
                writable_by=["engine"],
                default=0.0,
            )

    def test_vecNf_without_dims_rejected(self):
        """vecNf type requires dims field."""
        from townlet.vfs.schema import VariableDef

        with pytest.raises(ValidationError, match="dims.*required"):
            VariableDef(
                id="test",
                scope="agent",
                type="vecNf",
                # Missing dims!
                lifetime="episode",
                readable_by=["agent"],
                writable_by=["engine"],
                default=[0.0],
            )

    def test_scalar_with_dims_rejected(self):
        """scalar type should not have dims field."""
        from townlet.vfs.schema import VariableDef

        with pytest.raises(ValidationError, match="scalar.*dims"):
            VariableDef(
                id="test",
                scope="agent",
                type="scalar",
                dims=1,  # Should not be present!
                lifetime="episode",
                readable_by=["agent"],
                writable_by=["engine"],
                default=0.0,
            )


class TestNormalizationSpec:
    """Test NormalizationSpec schema validation."""

    def test_minmax_scalar_valid(self):
        """MinMax normalization with scalar bounds."""
        from townlet.vfs.schema import NormalizationSpec

        norm = NormalizationSpec(
            kind="minmax",
            min=0.0,
            max=1.0,
        )

        assert norm.kind == "minmax"
        assert norm.min == 0.0
        assert norm.max == 1.0

    def test_minmax_vector_valid(self):
        """MinMax normalization with vector bounds."""
        from townlet.vfs.schema import NormalizationSpec

        norm = NormalizationSpec(
            kind="minmax",
            min=[0.0, 0.0],
            max=[7.0, 7.0],
        )

        assert norm.kind == "minmax"
        assert norm.min == [0.0, 0.0]
        assert norm.max == [7.0, 7.0]

    def test_zscore_scalar_valid(self):
        """Z-score normalization with scalar parameters."""
        from townlet.vfs.schema import NormalizationSpec

        norm = NormalizationSpec(
            kind="zscore",
            mean=0.5,
            std=0.2,
        )

        assert norm.kind == "zscore"
        assert norm.mean == 0.5
        assert norm.std == 0.2

    def test_zscore_vector_valid(self):
        """Z-score normalization with vector parameters."""
        from townlet.vfs.schema import NormalizationSpec

        norm = NormalizationSpec(
            kind="zscore",
            mean=[0.5, 0.5],
            std=[0.2, 0.2],
        )

        assert norm.kind == "zscore"
        assert norm.mean == [0.5, 0.5]
        assert norm.std == [0.2, 0.2]

    def test_minmax_without_min_rejected(self):
        """MinMax normalization requires min field."""
        from townlet.vfs.schema import NormalizationSpec

        with pytest.raises(ValidationError, match="min.*required"):
            NormalizationSpec(
                kind="minmax",
                max=1.0,
                # Missing min!
            )

    def test_zscore_without_mean_rejected(self):
        """Z-score normalization requires mean field."""
        from townlet.vfs.schema import NormalizationSpec

        with pytest.raises(ValidationError, match="mean.*required"):
            NormalizationSpec(
                kind="zscore",
                std=0.2,
                # Missing mean!
            )


class TestObservationField:
    """Test ObservationField schema validation."""

    def test_scalar_observation_valid(self):
        """Scalar observation field (shape=[])."""
        from townlet.vfs.schema import ObservationField

        obs = ObservationField(
            id="obs_energy",
            source_variable="energy",
            exposed_to=["agent"],
            shape=[],
        )

        assert obs.id == "obs_energy"
        assert obs.source_variable == "energy"
        assert obs.exposed_to == ["agent"]
        assert obs.shape == []

    def test_vector_observation_valid(self):
        """Vector observation field (shape=[N])."""
        from townlet.vfs.schema import ObservationField

        obs = ObservationField(
            id="obs_position",
            source_variable="position",
            exposed_to=["agent"],
            shape=[2],
        )

        assert obs.shape == [2]

    def test_observation_with_normalization_valid(self):
        """Observation field with normalization spec."""
        from townlet.vfs.schema import ObservationField, NormalizationSpec

        obs = ObservationField(
            id="obs_energy",
            source_variable="energy",
            exposed_to=["agent"],
            shape=[],
            normalization=NormalizationSpec(
                kind="minmax",
                min=0.0,
                max=1.0,
            ),
        )

        assert obs.normalization is not None
        assert obs.normalization.kind == "minmax"

    def test_observation_without_normalization_valid(self):
        """Observation field without normalization (None)."""
        from townlet.vfs.schema import ObservationField

        obs = ObservationField(
            id="obs_money",
            source_variable="money",
            exposed_to=["agent"],
            shape=[],
            normalization=None,
        )

        assert obs.normalization is None


class TestWriteSpec:
    """Test WriteSpec schema validation."""

    def test_write_spec_valid(self):
        """Write specification with variable_id and expression."""
        from townlet.vfs.schema import WriteSpec

        write = WriteSpec(
            variable_id="energy",
            expression="-0.005",
        )

        assert write.variable_id == "energy"
        assert write.expression == "-0.005"

    def test_write_spec_complex_expression(self):
        """Write spec with complex expression (Phase 2 feature, Phase 1 stores as string)."""
        from townlet.vfs.schema import WriteSpec

        write = WriteSpec(
            variable_id="money",
            expression="money + 10.0",
        )

        assert write.expression == "money + 10.0"
        # Phase 1: No parsing, just store as string

    def test_write_spec_empty_variable_id_rejected(self):
        """Empty variable_id should be rejected."""
        from townlet.vfs.schema import WriteSpec

        with pytest.raises(ValidationError):
            WriteSpec(
                variable_id="",  # Empty!
                expression="-0.005",
            )

    def test_write_spec_empty_expression_rejected(self):
        """Empty expression should be rejected."""
        from townlet.vfs.schema import WriteSpec

        with pytest.raises(ValidationError):
            WriteSpec(
                variable_id="energy",
                expression="",  # Empty!
            )
