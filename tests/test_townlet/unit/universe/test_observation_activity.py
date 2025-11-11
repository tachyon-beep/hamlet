"""Test ObservationActivity DTO for observation masking."""

import pytest

from townlet.universe.dto.observation_activity import ObservationActivity


class TestObservationActivityConstruction:
    def test_construct_with_all_active_dimensions(self):
        """All dimensions active (no padding)."""
        activity = ObservationActivity(
            active_mask=(True, True, True, True),
            group_slices={"bars": slice(0, 2), "spatial": slice(2, 4)},
            active_field_uuids=("uuid1", "uuid2", "uuid3", "uuid4"),
        )

        assert activity.active_mask == (True, True, True, True)
        assert activity.group_slices == {"bars": slice(0, 2), "spatial": slice(2, 4)}
        assert activity.active_field_uuids == ("uuid1", "uuid2", "uuid3", "uuid4")

    def test_construct_with_mixed_active_inactive(self):
        """Some dimensions inactive (padding present)."""
        activity = ObservationActivity(
            active_mask=(True, False, True, False, False, True),
            group_slices={
                "bars": slice(0, 3),  # Contains active + inactive
                "affordance": slice(3, 6),  # All inactive in this example
            },
            active_field_uuids=("uuid1", "uuid3", "uuid6"),  # Only active dims
        )

        assert activity.active_mask == (True, False, True, False, False, True)
        assert len(activity.active_field_uuids) == 3
        assert "uuid1" in activity.active_field_uuids
        assert "uuid2" not in activity.active_field_uuids  # Inactive

    def test_frozen_immutable_dataclass(self):
        """ObservationActivity should be frozen (immutable)."""
        activity = ObservationActivity(
            active_mask=(True, False),
            group_slices={"bars": slice(0, 2)},
            active_field_uuids=("uuid1",),
        )

        with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
            activity.active_mask = (False, True)


class TestObservationActivityHelperMethods:
    def test_total_dims_property(self):
        """total_dims should return length of active_mask."""
        activity = ObservationActivity(
            active_mask=(True, False, True, True, False),
            group_slices={},
            active_field_uuids=(),
        )

        assert activity.total_dims == 5

    def test_active_dim_count_property(self):
        """active_dim_count should count True values in active_mask."""
        activity = ObservationActivity(
            active_mask=(True, False, True, True, False),
            group_slices={},
            active_field_uuids=(),
        )

        assert activity.active_dim_count == 3

    def test_get_group_slice_existing_key(self):
        """get_group_slice should return slice for existing group."""
        activity = ObservationActivity(
            active_mask=(True, True, True, True),
            group_slices={"bars": slice(0, 2), "spatial": slice(2, 4)},
            active_field_uuids=(),
        )

        assert activity.get_group_slice("bars") == slice(0, 2)
        assert activity.get_group_slice("spatial") == slice(2, 4)

    def test_get_group_slice_missing_key_returns_none(self):
        """get_group_slice should return None for missing group."""
        activity = ObservationActivity(
            active_mask=(True, True),
            group_slices={"bars": slice(0, 2)},
            active_field_uuids=(),
        )

        assert activity.get_group_slice("nonexistent") is None


class TestObservationActivityValidation:
    def test_mask_length_matches_field_uuid_consistency(self):
        """active_field_uuids should have one UUID per True in mask."""
        # This is a logical consistency check, not enforced by DTO
        # But we document expected usage
        activity = ObservationActivity(
            active_mask=(True, False, True, False),  # 2 active
            group_slices={},
            active_field_uuids=("uuid1", "uuid3"),  # 2 UUIDs (matches)
        )

        active_count = sum(activity.active_mask)
        assert len(activity.active_field_uuids) == active_count
