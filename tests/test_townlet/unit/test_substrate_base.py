"""Test spatial substrate abstract interface."""

import pytest

from townlet.substrate.base import SpatialSubstrate


def test_substrate_module_exists():
    """Substrate module should be importable."""
    assert SpatialSubstrate is not None


def test_substrate_is_abstract():
    """SpatialSubstrate should not be instantiable directly."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        SpatialSubstrate()
