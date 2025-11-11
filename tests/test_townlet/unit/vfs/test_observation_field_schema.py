"""Test ObservationField schema extensions for semantic grouping."""

import pytest
from pydantic import ValidationError

from townlet.vfs.schema import NormalizationSpec, ObservationField


class TestSemanticTypeField:
    def test_semantic_type_defaults_to_custom(self):
        """semantic_type should default to 'custom' if not specified."""
        field = ObservationField(
            id="test_field",
            source_variable="test_var",
            exposed_to=["agent"],
            shape=[1],
            normalization=None,
        )
        assert field.semantic_type == "custom"

    def test_semantic_type_accepts_valid_literals(self):
        """semantic_type should accept bars, spatial, affordance, temporal, custom."""
        valid_types = ["bars", "spatial", "affordance", "temporal", "custom"]

        for semantic_type in valid_types:
            field = ObservationField(
                id=f"test_{semantic_type}",
                source_variable="test_var",
                exposed_to=["agent"],
                shape=[1],
                normalization=None,
                semantic_type=semantic_type,
            )
            assert field.semantic_type == semantic_type

    def test_semantic_type_rejects_invalid_values(self):
        """semantic_type should reject values not in Literal."""
        with pytest.raises(ValidationError) as exc_info:
            ObservationField(
                id="test_field",
                source_variable="test_var",
                exposed_to=["agent"],
                shape=[1],
                normalization=None,
                semantic_type="invalid_type",  # Not in Literal
            )

        error = str(exc_info.value)
        assert "semantic_type" in error.lower()


class TestCurriculumActiveField:
    def test_curriculum_active_defaults_to_true(self):
        """curriculum_active should default to True if not specified."""
        field = ObservationField(
            id="test_field",
            source_variable="test_var",
            exposed_to=["agent"],
            shape=[1],
            normalization=None,
        )
        assert field.curriculum_active is True

    def test_curriculum_active_accepts_false(self):
        """curriculum_active should accept False (for padding dims)."""
        field = ObservationField(
            id="test_field",
            source_variable="test_var",
            exposed_to=["agent"],
            shape=[1],
            normalization=None,
            curriculum_active=False,
        )
        assert field.curriculum_active is False

    def test_curriculum_active_accepts_bool_like_values(self):
        """curriculum_active should coerce bool-like values (Pydantic behavior)."""
        # Pydantic coerces strings like "yes", "true", "1" to bool
        field = ObservationField(
            id="test_field",
            source_variable="test_var",
            exposed_to=["agent"],
            shape=[1],
            normalization=None,
            curriculum_active="yes",  # Coerced to True
        )
        assert field.curriculum_active is True


class TestBackwardCompatibility:
    def test_existing_fields_without_new_metadata_still_work(self):
        """Fields created without semantic_type/curriculum_active should use defaults."""
        # This simulates loading old configs that don't have the new fields
        field = ObservationField(
            id="legacy_field",
            source_variable="energy",
            exposed_to=["agent"],
            shape=[1],
            normalization=NormalizationSpec(kind="minmax", min=0.0, max=1.0),
        )

        assert field.semantic_type == "custom"  # Default
        assert field.curriculum_active is True  # Default
