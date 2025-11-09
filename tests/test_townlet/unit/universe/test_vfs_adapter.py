"""Unit tests for vfs_to_observation_spec adapter."""

from townlet.universe.adapters.vfs_adapter import vfs_to_observation_spec
from townlet.vfs.schema import ObservationField as VFSObservationField


def make_field(**kwargs):
    base = {
        "id": "obs_energy",
        "source_variable": "energy",
        "exposed_to": ["agent"],
        "shape": [],
        "normalization": None,
    }
    base.update(kwargs)
    return VFSObservationField(**base)


def test_adapter_scalar_and_vector():
    fields = [
        make_field(id="obs_energy", shape=[]),
        make_field(id="obs_position", shape=[2]),
    ]

    spec = vfs_to_observation_spec(fields)

    assert spec.total_dims == 3
    assert spec.fields[0].name == "obs_energy"
    assert spec.fields[0].dims == 1
    assert spec.fields[1].name == "obs_position"
    assert spec.fields[1].dims == 2
    assert spec.fields[0].uuid != spec.fields[1].uuid


def test_adapter_grid_field():
    fields = [make_field(id="local_grid", shape=[5, 5], source_variable="grid_encoding")]

    spec = vfs_to_observation_spec(fields)

    assert spec.total_dims == 25
    assert spec.fields[0].start_index == 0
    assert spec.fields[0].end_index == 25
    assert spec.fields[0].uuid is not None
