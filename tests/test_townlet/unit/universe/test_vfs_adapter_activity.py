"""Test VFS adapter builds ObservationActivity from observation spec."""

from townlet.universe.adapters.vfs_adapter import VFSAdapter
from townlet.vfs.schema import ObservationField


class TestBuildObservationActivity:
    def test_all_active_dimensions_no_padding(self):
        """All fields active → mask all True."""
        observation_spec = [
            ObservationField(
                id="health",
                source_variable="health",
                exposed_to=["agent"],
                shape=[1],
                normalization=None,
                semantic_type="bars",
                curriculum_active=True,
            ),
            ObservationField(
                id="position",
                source_variable="position",
                exposed_to=["agent"],
                shape=[2],
                normalization=None,
                semantic_type="spatial",
                curriculum_active=True,
            ),
        ]

        field_uuids = {"health": "uuid_health", "position": "uuid_position"}

        activity = VFSAdapter.build_observation_activity(observation_spec, field_uuids)

        # 1 health + 2 position = 3 total dims
        assert activity.total_dims == 3
        assert activity.active_mask == (True, True, True)
        assert activity.active_dim_count == 3
        assert activity.active_field_uuids == ("uuid_health", "uuid_position", "uuid_position")

        # Group slices
        assert activity.get_group_slice("bars") == slice(0, 1)
        assert activity.get_group_slice("spatial") == slice(1, 3)

    def test_mixed_active_inactive_creates_padding_mask(self):
        """Some fields inactive → mask has False values."""
        observation_spec = [
            ObservationField(
                id="health",
                source_variable="health",
                exposed_to=["agent"],
                shape=[1],
                normalization=None,
                semantic_type="bars",
                curriculum_active=True,  # ACTIVE
            ),
            ObservationField(
                id="mood",
                source_variable="mood",
                exposed_to=["agent"],
                shape=[1],
                normalization=None,
                semantic_type="bars",
                curriculum_active=False,  # INACTIVE (padding)
            ),
            ObservationField(
                id="position",
                source_variable="position",
                exposed_to=["agent"],
                shape=[2],
                normalization=None,
                semantic_type="spatial",
                curriculum_active=True,  # ACTIVE
            ),
        ]

        field_uuids = {
            "health": "uuid_health",
            "mood": "uuid_mood",
            "position": "uuid_position",
        }

        activity = VFSAdapter.build_observation_activity(observation_spec, field_uuids)

        # 1 health + 1 mood + 2 position = 4 total dims
        assert activity.total_dims == 4
        assert activity.active_mask == (True, False, True, True)
        assert activity.active_dim_count == 3  # Only health + position

        # Only active field UUIDs
        assert len(activity.active_field_uuids) == 3
        assert "uuid_health" in activity.active_field_uuids
        assert "uuid_mood" not in activity.active_field_uuids  # Inactive
        assert activity.active_field_uuids.count("uuid_position") == 2  # 2 dims

    def test_group_slices_cover_all_semantic_types(self):
        """Group slices should be created for each semantic_type present."""
        observation_spec = [
            ObservationField(
                id="health",
                source_variable="health",
                exposed_to=["agent"],
                shape=[1],
                normalization=None,
                semantic_type="bars",
                curriculum_active=True,
            ),
            ObservationField(
                id="energy",
                source_variable="energy",
                exposed_to=["agent"],
                shape=[1],
                normalization=None,
                semantic_type="bars",
                curriculum_active=True,
            ),
            ObservationField(
                id="position",
                source_variable="position",
                exposed_to=["agent"],
                shape=[2],
                normalization=None,
                semantic_type="spatial",
                curriculum_active=True,
            ),
            ObservationField(
                id="affordance_state",
                source_variable="affordance_state",
                exposed_to=["agent"],
                shape=[3],
                normalization=None,
                semantic_type="affordance",
                curriculum_active=True,
            ),
            ObservationField(
                id="time_sin",
                source_variable="time_sin",
                exposed_to=["agent"],
                shape=[1],
                normalization=None,
                semantic_type="temporal",
                curriculum_active=False,
            ),
        ]

        field_uuids = {
            "health": "uuid1",
            "energy": "uuid2",
            "position": "uuid3",
            "affordance_state": "uuid4",
            "time_sin": "uuid5",
        }

        activity = VFSAdapter.build_observation_activity(observation_spec, field_uuids)

        # Verify all groups present
        assert activity.get_group_slice("bars") == slice(0, 2)  # health, energy
        assert activity.get_group_slice("spatial") == slice(2, 4)  # position (2 dims)
        assert activity.get_group_slice("affordance") == slice(4, 7)  # affordance_state (3 dims)
        assert activity.get_group_slice("temporal") == slice(7, 8)  # time_sin (1 dim)

    def test_empty_observation_spec_creates_empty_activity(self):
        """Empty observation spec → empty mask and slices."""
        activity = VFSAdapter.build_observation_activity([], {})

        assert activity.total_dims == 0
        assert activity.active_mask == ()
        assert activity.active_dim_count == 0
        assert activity.active_field_uuids == ()
        assert activity.group_slices == {}
