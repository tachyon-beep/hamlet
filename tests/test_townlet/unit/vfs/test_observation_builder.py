"""Test VFSObservationSpecBuilder for generating observation specs.

The VFSObservationSpecBuilder generates observation specifications (schemas)
from variable definitions and exposure configurations. This is used by the
BAC (Behavioral Action Compiler) for dynamic network input head generation.

NOTE: This is NOT the same as environment.observation_builder.ObservationBuilder!
- environment.ObservationBuilder: Runtime observation construction (tensors)
- vfs.VFSObservationSpecBuilder: Compile-time spec generation (schemas for BAC)
"""

from typing import Any

import pytest

from townlet.vfs.observation_builder import VFSObservationSpecBuilder
from townlet.vfs.schema import VariableDef


@pytest.fixture
def sample_variables():
    """Sample variables for testing."""
    return [
        VariableDef(
            id="energy",
            scope="agent",
            type="scalar",
            lifetime="episode",
            readable_by=["agent"],
            writable_by=["actions"],
            default=1.0,
        ),
        VariableDef(
            id="position",
            scope="agent",
            type="vec2i",
            lifetime="episode",
            readable_by=["agent"],
            writable_by=["actions"],
            default=[0, 0],
        ),
        VariableDef(
            id="velocity",
            scope="agent",
            type="vec3i",
            lifetime="tick",
            readable_by=["agent"],
            writable_by=["engine"],
            default=[0, 0, 0],
        ),
        VariableDef(
            id="position_7d",
            scope="agent",
            type="vecNi",
            dims=7,
            lifetime="episode",
            readable_by=["agent"],
            writable_by=["actions"],
            default=[0] * 7,
        ),
        VariableDef(
            id="grid_encoding",
            scope="agent",
            type="vecNf",
            dims=64,
            lifetime="tick",
            readable_by=["agent"],
            writable_by=["engine"],
            default=[0.0] * 64,
        ),
        VariableDef(
            id="is_alive",
            scope="agent",
            type="bool",
            lifetime="episode",
            readable_by=["agent", "engine"],
            writable_by=["engine"],
            default=True,
        ),
    ]


def exposures_from_dict(mapping: dict[str, dict[str, Any] | None]) -> list[dict[str, Any]]:
    """Convert legacy mapping-style exposure definitions into list form."""
    exposures: list[dict[str, Any]] = []
    for var_id, config in mapping.items():
        entry: dict[str, Any] = {"source_variable": var_id}
        if config:
            entry.update(config)
        exposures.append(entry)
    return exposures


class TestObservationSpecBuilderScalarTypes:
    """Test observation spec generation for scalar types."""

    def test_build_spec_for_scalar_variable(self, sample_variables):
        """Build observation spec for scalar variable."""
        builder = VFSObservationSpecBuilder()

        exposures = {"energy": {"normalization": {"kind": "minmax", "min": 0.0, "max": 1.0}}}
        exposures = exposures_from_dict(exposures)

        spec = builder.build_observation_spec(sample_variables, exposures)

        # Should have one field for energy
        assert len(spec) == 1
        assert spec[0].source_variable == "energy"
        assert spec[0].shape == []  # Scalar has empty shape
        assert spec[0].normalization is not None
        assert spec[0].normalization.kind == "minmax"

    def test_build_spec_for_bool_variable(self, sample_variables):
        """Build observation spec for bool variable."""
        builder = VFSObservationSpecBuilder()

        exposures = {"is_alive": {"normalization": None}}
        exposures = exposures_from_dict(exposures)

        spec = builder.build_observation_spec(sample_variables, exposures)

        assert len(spec) == 1
        assert spec[0].source_variable == "is_alive"
        assert spec[0].shape == []  # Bool has empty shape (scalar-like)


class TestObservationSpecBuilderVectorTypes:
    """Test observation spec generation for vector types."""

    def test_build_spec_for_vec2i_variable(self, sample_variables):
        """Build observation spec for vec2i variable."""
        builder = VFSObservationSpecBuilder()

        exposures = {"position": {"normalization": {"kind": "minmax", "min": [0, 0], "max": [7, 7]}}}
        exposures = exposures_from_dict(exposures)

        spec = builder.build_observation_spec(sample_variables, exposures)

        assert len(spec) == 1
        assert spec[0].source_variable == "position"
        assert spec[0].shape == [2]  # vec2i has 2 dimensions

    def test_build_spec_for_vec3i_variable(self, sample_variables):
        """Build observation spec for vec3i variable."""
        builder = VFSObservationSpecBuilder()

        exposures = {"velocity": {"normalization": None}}
        exposures = exposures_from_dict(exposures)

        spec = builder.build_observation_spec(sample_variables, exposures)

        assert len(spec) == 1
        assert spec[0].source_variable == "velocity"
        assert spec[0].shape == [3]  # vec3i has 3 dimensions

    def test_build_spec_for_vecNi_variable(self, sample_variables):  # noqa: N802
        """Build observation spec for vecNi variable (N-dimensional integer vector)."""
        builder = VFSObservationSpecBuilder()

        exposures = {"position_7d": {"normalization": None}}
        exposures = exposures_from_dict(exposures)

        spec = builder.build_observation_spec(sample_variables, exposures)

        assert len(spec) == 1
        assert spec[0].source_variable == "position_7d"
        assert spec[0].shape == [7]  # vecNi with dims=7

    def test_build_spec_for_vecNf_variable(self, sample_variables):  # noqa: N802
        """Build observation spec for vecNf variable (N-dimensional float vector)."""
        builder = VFSObservationSpecBuilder()

        exposures = {
            "grid_encoding": {
                "normalization": {
                    "kind": "zscore",
                    "mean": [0.0] * 64,
                    "std": [1.0] * 64,
                }
            }
        }
        exposures = exposures_from_dict(exposures)

        spec = builder.build_observation_spec(sample_variables, exposures)

        assert len(spec) == 1
        assert spec[0].source_variable == "grid_encoding"
        assert spec[0].shape == [64]  # vecNf with dims=64
        assert spec[0].normalization.kind == "zscore"


class TestObservationSpecBuilderMultipleVariables:
    """Test observation spec generation for multiple variables."""

    def test_build_spec_multiple_variables(self, sample_variables):
        """Build observation spec for multiple variables."""
        builder = VFSObservationSpecBuilder()

        exposures = {
            "energy": {"normalization": None},
            "position": {"normalization": None},
        }
        exposures = exposures_from_dict(exposures)

        spec = builder.build_observation_spec(sample_variables, exposures)

        assert len(spec) == 2
        source_vars = {field.source_variable for field in spec}
        assert source_vars == {"energy", "position"}

    def test_build_spec_all_variable_types(self, sample_variables):
        """Build observation spec including all variable types."""
        builder = VFSObservationSpecBuilder()

        exposures = {
            "energy": {"normalization": None},
            "position": {"normalization": None},
            "velocity": {"normalization": None},
            "position_7d": {"normalization": None},
            "grid_encoding": {"normalization": None},
            "is_alive": {"normalization": None},
        }
        exposures = exposures_from_dict(exposures)

        spec = builder.build_observation_spec(sample_variables, exposures)

        assert len(spec) == 6
        source_vars = {field.source_variable for field in spec}
        assert source_vars == {"energy", "position", "velocity", "position_7d", "grid_encoding", "is_alive"}


class TestObservationSpecBuilderValidation:
    """Validation-focused tests for observation spec builder."""

    def test_dict_exposures_are_rejected(self, sample_variables):
        """Legacy dict exposure format is no longer accepted."""
        builder = VFSObservationSpecBuilder()

        exposures = {"energy": {"normalization": None}}

        with pytest.raises(TypeError, match="list"):
            builder.build_observation_spec(sample_variables, exposures)  # type: ignore[arg-type]

    def test_vector_normalization_length_mismatch_raises(self, sample_variables):
        """Normalization list length must match flattened observation shape."""
        builder = VFSObservationSpecBuilder()

        exposures = {
            "position": {
                "normalization": {
                    "kind": "minmax",
                    "min": [0, 0, 0],  # Wrong length for vec2i
                    "max": [7, 7, 7],
                }
            }
        }
        exposures = exposures_from_dict(exposures)

        with pytest.raises(ValueError, match="must provide 2 values"):
            builder.build_observation_spec(sample_variables, exposures)

    def test_vector_normalization_scalar_values_rejected(self, sample_variables):
        """Vector observations require per-dimension normalization arrays."""
        builder = VFSObservationSpecBuilder()

        exposures = {
            "position": {
                "normalization": {
                    "kind": "minmax",
                    "min": 0.0,
                    "max": 1.0,
                }
            }
        }
        exposures = exposures_from_dict(exposures)

        with pytest.raises(ValueError, match="must be a list of length 2"):
            builder.build_observation_spec(sample_variables, exposures)

    def test_scalar_normalization_allows_scalar_values(self, sample_variables):
        """Scalar observations may use scalar normalization params."""
        builder = VFSObservationSpecBuilder()

        exposures = {
            "energy": {
                "normalization": {
                    "kind": "minmax",
                    "min": 0.0,
                    "max": 1.0,
                }
            }
        }
        exposures = exposures_from_dict(exposures)

        spec = builder.build_observation_spec(sample_variables, exposures)
        assert len(spec) == 1
        assert spec[0].source_variable == "energy"


class TestObservationDimensionCalculation:
    """Test total observation dimension calculation."""

    def test_total_observation_dim_scalar_only(self, sample_variables):
        """Calculate total observation dimension for scalar only."""
        builder = VFSObservationSpecBuilder()

        exposures = {"energy": {"normalization": None}}
        exposures = exposures_from_dict(exposures)

        spec = builder.build_observation_spec(sample_variables, exposures)

        # Scalar contributes 1 dimension
        total_dims = sum(len(field.shape) if field.shape else 1 for field in spec)
        assert total_dims == 1

    def test_total_observation_dim_vector_only(self, sample_variables):
        """Calculate total observation dimension for vector only."""
        builder = VFSObservationSpecBuilder()

        exposures = {"position": {"normalization": None}}
        exposures = exposures_from_dict(exposures)

        spec = builder.build_observation_spec(sample_variables, exposures)

        # vec2i contributes 2 dimensions
        total_dims = sum(field.shape[0] if field.shape else 1 for field in spec)
        assert total_dims == 2

    def test_total_observation_dim_mixed_types(self, sample_variables):
        """Calculate total observation dimension for mixed types."""
        builder = VFSObservationSpecBuilder()

        exposures = {
            "energy": {"normalization": None},  # 1 dim (scalar)
            "position": {"normalization": None},  # 2 dims (vec2i)
            "velocity": {"normalization": None},  # 3 dims (vec3i)
            "position_7d": {"normalization": None},  # 7 dims (vecNi)
        }
        exposures = exposures_from_dict(exposures)

        spec = builder.build_observation_spec(sample_variables, exposures)

        # energy (1) + position (2) + velocity (3) + position_7d (7) = 13
        total_dims = sum(field.shape[0] if field.shape else 1 for field in spec)
        assert total_dims == 13

    def test_total_observation_dim_large_vector(self, sample_variables):
        """Calculate total observation dimension with large vector."""
        builder = VFSObservationSpecBuilder()

        exposures = {
            "energy": {"normalization": None},  # 1 dim
            "grid_encoding": {"normalization": None},  # 64 dims
        }
        exposures = exposures_from_dict(exposures)

        spec = builder.build_observation_spec(sample_variables, exposures)

        # energy (1) + grid_encoding (64) = 65
        total_dims = sum(field.shape[0] if field.shape else 1 for field in spec)
        assert total_dims == 65


class TestObservationSpecBuilderNormalization:
    """Test normalization spec handling."""

    def test_build_spec_with_minmax_normalization(self, sample_variables):
        """Build observation spec with minmax normalization."""
        builder = VFSObservationSpecBuilder()

        exposures = {"energy": {"normalization": {"kind": "minmax", "min": 0.0, "max": 1.0}}}
        exposures = exposures_from_dict(exposures)

        spec = builder.build_observation_spec(sample_variables, exposures)

        assert spec[0].normalization is not None
        assert spec[0].normalization.kind == "minmax"
        assert spec[0].normalization.min == 0.0
        assert spec[0].normalization.max == 1.0

    def test_build_spec_with_zscore_normalization(self, sample_variables):
        """Build observation spec with zscore normalization."""
        builder = VFSObservationSpecBuilder()

        exposures = {
            "grid_encoding": {
                "normalization": {
                    "kind": "zscore",
                    "mean": [0.0] * 64,
                    "std": [1.0] * 64,
                }
            }
        }
        exposures = exposures_from_dict(exposures)

        spec = builder.build_observation_spec(sample_variables, exposures)

        assert spec[0].normalization is not None
        assert spec[0].normalization.kind == "zscore"
        assert spec[0].normalization.mean == [0.0] * 64
        assert spec[0].normalization.std == [1.0] * 64

    def test_build_spec_without_normalization(self, sample_variables):
        """Build observation spec without normalization."""
        builder = VFSObservationSpecBuilder()

        exposures = {"energy": {"normalization": None}}
        exposures = exposures_from_dict(exposures)

        spec = builder.build_observation_spec(sample_variables, exposures)

        assert spec[0].normalization is None

    def test_build_spec_vector_normalization_list_params(self, sample_variables):
        """Build observation spec with vector using list normalization params."""
        builder = VFSObservationSpecBuilder()

        # Vec2i with per-dimension normalization
        exposures = {"position": {"normalization": {"kind": "minmax", "min": [0, 0], "max": [7, 10]}}}
        exposures = exposures_from_dict(exposures)

        spec = builder.build_observation_spec(sample_variables, exposures)

        assert spec[0].normalization is not None
        assert spec[0].normalization.min == [0, 0]
        assert spec[0].normalization.max == [7, 10]


class TestObservationSpecBuilderErrorHandling:
    """Test error handling in observation spec builder."""

    def test_build_spec_missing_variable_raises_error(self, sample_variables):
        """Build observation spec with non-existent variable raises error."""
        builder = VFSObservationSpecBuilder()

        exposures = {"nonexistent_var": {"normalization": None}}
        exposures = exposures_from_dict(exposures)

        with pytest.raises(ValueError, match="Variable nonexistent_var not found"):
            builder.build_observation_spec(sample_variables, exposures)

    def test_build_spec_empty_exposures(self, sample_variables):
        """Build observation spec with empty exposures returns empty list."""
        builder = VFSObservationSpecBuilder()

        exposures: list[dict[str, Any]] = []

        spec = builder.build_observation_spec(sample_variables, exposures)

        assert len(spec) == 0

    def test_missing_source_variable_metadata_raises(self, sample_variables):
        """Exposure entries without source_variable should raise."""
        builder = VFSObservationSpecBuilder()

        exposures = [{"id": "broken_obs"}]

        with pytest.raises(ValueError, match="missing 'source_variable'"):
            builder.build_observation_spec(sample_variables, exposures)


class TestObservationFieldProperties:
    """Test generated ObservationField properties."""

    def test_observation_field_has_unique_id(self, sample_variables):
        """Generated observation fields have unique IDs."""
        builder = VFSObservationSpecBuilder()

        exposures = {
            "energy": {"normalization": None},
            "position": {"normalization": None},
        }
        exposures = exposures_from_dict(exposures)

        spec = builder.build_observation_spec(sample_variables, exposures)

        ids = [field.id for field in spec]
        assert len(ids) == len(set(ids))  # All IDs unique

    def test_observation_field_exposed_to_agent(self, sample_variables):
        """Generated observation fields are exposed to agent."""
        builder = VFSObservationSpecBuilder()

        exposures = {"energy": {"normalization": None}}
        exposures = exposures_from_dict(exposures)

        spec = builder.build_observation_spec(sample_variables, exposures)

        assert "agent" in spec[0].exposed_to

    def test_observation_field_source_variable_matches(self, sample_variables):
        """Generated observation fields have correct source variable."""
        builder = VFSObservationSpecBuilder()

        exposures = {"position": {"normalization": None}}
        exposures = exposures_from_dict(exposures)

        spec = builder.build_observation_spec(sample_variables, exposures)

        assert spec[0].source_variable == "position"

    def test_preserves_configured_metadata(self, sample_variables):
        """Builder should honor id, exposed_to, and shape from config."""
        builder = VFSObservationSpecBuilder()

        exposures = [
            {
                "id": "custom_position_obs",
                "source_variable": "position",
                "exposed_to": ["agent", "acs"],
                "shape": [4],  # Should override inferred shape
                "normalization": None,
            }
        ]

        spec = builder.build_observation_spec(sample_variables, exposures)
        assert spec[0].id == "custom_position_obs"
        assert spec[0].exposed_to == ["agent", "acs"]
        assert spec[0].shape == [4]

    def test_allows_duplicate_source_variables(self, sample_variables):
        """Builder should not deduplicate repeated source_variable exposures."""
        builder = VFSObservationSpecBuilder()

        exposures = [
            {
                "id": "obs_energy_agent",
                "source_variable": "energy",
                "exposed_to": ["agent"],
                "shape": [],
            },
            {
                "id": "obs_energy_acs",
                "source_variable": "energy",
                "exposed_to": ["acs"],
                "shape": [],
            },
        ]

        spec = builder.build_observation_spec(sample_variables, exposures)
        energy_fields = [field for field in spec if field.source_variable == "energy"]
        assert len(energy_fields) == 2
        assert {field.id for field in energy_fields} == {"obs_energy_agent", "obs_energy_acs"}
