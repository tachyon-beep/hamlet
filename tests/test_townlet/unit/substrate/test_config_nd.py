"""Test N-dimensional substrate configuration schema (GridND and ContinuousND)."""

import pytest

from townlet.substrate.config import (
    ContinuousConfig,
    GridNDConfig,
    SubstrateConfig,
)

# ============================================================================
# GridNDConfig Tests
# ============================================================================


def test_gridnd_config_valid_4d():
    """Valid 4D GridND config should parse successfully."""
    config_data = {
        "dimension_sizes": [8, 8, 8, 8],
        "boundary": "clamp",
        "distance_metric": "manhattan",
        "observation_encoding": "relative",
        "topology": "hypercube",
    }

    config = GridNDConfig(**config_data)

    assert config.dimension_sizes == [8, 8, 8, 8]
    assert config.boundary == "clamp"
    assert config.distance_metric == "manhattan"
    assert config.observation_encoding == "relative"


def test_gridnd_config_valid_asymmetric():
    """GridND config with asymmetric dimensions should parse successfully."""
    config_data = {
        "dimension_sizes": [10, 5, 3, 8],
        "boundary": "wrap",
        "distance_metric": "euclidean",
        "observation_encoding": "scaled",
        "topology": "hypercube",
    }

    config = GridNDConfig(**config_data)

    assert config.dimension_sizes == [10, 5, 3, 8]
    assert config.boundary == "wrap"
    assert config.observation_encoding == "scaled"


def test_gridnd_config_valid_high_dimensional():
    """GridND config with 10 dimensions should parse successfully."""
    config_data = {
        "dimension_sizes": [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],  # 10D
        "boundary": "bounce",
        "distance_metric": "chebyshev",
        "observation_encoding": "relative",
        "topology": "hypercube",
    }

    config = GridNDConfig(**config_data)

    assert len(config.dimension_sizes) == 10
    assert all(size == 3 for size in config.dimension_sizes)


def test_gridnd_config_invalid_too_few_dimensions():
    """GridND config with <4 dimensions should fail."""
    config_data = {
        "dimension_sizes": [8, 8, 8],  # Only 3D!
        "boundary": "clamp",
        "distance_metric": "manhattan",
        "observation_encoding": "relative",
        "topology": "hypercube",
    }

    with pytest.raises(ValueError, match="at least 4 dimensions"):
        GridNDConfig(**config_data)


def test_gridnd_config_invalid_zero_dimension():
    """GridND config with zero dimension size should fail."""
    config_data = {
        "dimension_sizes": [8, 8, 0, 8],  # Invalid!
        "boundary": "clamp",
        "distance_metric": "manhattan",
        "observation_encoding": "relative",
        "topology": "hypercube",
    }

    with pytest.raises(ValueError, match="positive"):
        GridNDConfig(**config_data)


def test_gridnd_config_invalid_negative_dimension():
    """GridND config with negative dimension size should fail."""
    config_data = {
        "dimension_sizes": [8, 8, -5, 8],  # Invalid!
        "boundary": "clamp",
        "distance_metric": "manhattan",
        "observation_encoding": "relative",
        "topology": "hypercube",
    }

    with pytest.raises(ValueError, match="positive"):
        GridNDConfig(**config_data)


def test_gridnd_config_invalid_too_many_dimensions():
    """GridND config with >100 dimensions should fail."""
    config_data = {
        "dimension_sizes": [3] * 101,  # 101D exceeds limit!
        "boundary": "clamp",
        "distance_metric": "manhattan",
        "observation_encoding": "relative",
        "topology": "hypercube",
    }

    with pytest.raises(ValueError, match="exceeds limit"):
        GridNDConfig(**config_data)


def test_gridnd_config_observation_encoding_all_modes():
    """GridND config should accept all observation encoding modes."""
    for encoding in ["relative", "scaled", "absolute"]:
        config = GridNDConfig(
            dimension_sizes=[4, 4, 4, 4],
            boundary="clamp",
            distance_metric="manhattan",
            observation_encoding=encoding,
            topology="hypercube",
        )
        assert config.observation_encoding == encoding


# ============================================================================
# ContinuousConfig Extended Tests (N≥4 support)
# ============================================================================


def test_continuous_config_valid_1d():
    """ContinuousConfig with 1D should still work (backward compatibility)."""
    config = ContinuousConfig(
        dimensions=1,
        bounds=[(0.0, 10.0)],
        boundary="clamp",
        movement_delta=0.5,
        interaction_radius=1.0,
        distance_metric="euclidean",
        observation_encoding="relative",
    )

    assert config.dimensions == 1
    assert len(config.bounds) == 1


def test_continuous_config_valid_2d():
    """ContinuousConfig with 2D should work (backward compatibility)."""
    config = ContinuousConfig(
        dimensions=2,
        bounds=[(0.0, 10.0), (0.0, 10.0)],
        boundary="clamp",
        movement_delta=0.5,
        interaction_radius=1.0,
        distance_metric="euclidean",
        observation_encoding="relative",
    )

    assert config.dimensions == 2
    assert len(config.bounds) == 2


def test_continuous_config_valid_3d():
    """ContinuousConfig with 3D should work (backward compatibility)."""
    config = ContinuousConfig(
        dimensions=3,
        bounds=[(0.0, 10.0), (0.0, 10.0), (0.0, 10.0)],
        boundary="clamp",
        movement_delta=0.5,
        interaction_radius=1.0,
        distance_metric="euclidean",
        observation_encoding="relative",
    )

    assert config.dimensions == 3
    assert len(config.bounds) == 3


def test_continuous_config_valid_4d():
    """ContinuousConfig with 4D should now work (new feature)."""
    config = ContinuousConfig(
        dimensions=4,
        bounds=[(0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0)],
        boundary="clamp",
        movement_delta=0.5,
        interaction_radius=1.0,
        distance_metric="euclidean",
        observation_encoding="relative",
    )

    assert config.dimensions == 4
    assert len(config.bounds) == 4


def test_continuous_config_valid_high_dimensional():
    """ContinuousConfig with 10D should work."""
    bounds = [(0.0, 10.0) for _ in range(10)]
    config = ContinuousConfig(
        dimensions=10,
        bounds=bounds,
        boundary="wrap",
        movement_delta=0.5,
        interaction_radius=1.0,
        distance_metric="manhattan",
        observation_encoding="relative",
    )

    assert config.dimensions == 10
    assert len(config.bounds) == 10


def test_continuous_config_valid_asymmetric_bounds():
    """ContinuousConfig with asymmetric bounds should work."""
    config = ContinuousConfig(
        dimensions=4,
        bounds=[(-10.0, 10.0), (0.0, 5.0), (-100.0, 100.0), (0.0, 1.0)],
        boundary="clamp",
        movement_delta=0.5,
        interaction_radius=1.0,
        distance_metric="euclidean",
        observation_encoding="relative",
    )

    assert config.bounds[0] == (-10.0, 10.0)
    assert config.bounds[1] == (0.0, 5.0)
    assert config.bounds[2] == (-100.0, 100.0)
    assert config.bounds[3] == (0.0, 1.0)


def test_continuous_config_invalid_bounds_mismatch():
    """ContinuousConfig with mismatched bounds should fail."""
    config_data = {
        "dimensions": 4,
        "bounds": [(0.0, 10.0), (0.0, 10.0)],  # Only 2 bounds!
        "boundary": "clamp",
        "movement_delta": 0.5,
        "interaction_radius": 1.0,
        "distance_metric": "euclidean",
        "observation_encoding": "relative",
    }

    with pytest.raises(ValueError, match="must match dimensions"):
        ContinuousConfig(**config_data)


def test_continuous_config_invalid_bound_order():
    """ContinuousConfig with min >= max should fail."""
    config_data = {
        "dimensions": 4,
        "bounds": [(0.0, 10.0), (5.0, 5.0), (0.0, 10.0), (0.0, 10.0)],  # min == max!
        "boundary": "clamp",
        "movement_delta": 0.5,
        "interaction_radius": 1.0,
        "distance_metric": "euclidean",
        "observation_encoding": "relative",
    }

    with pytest.raises(ValueError, match="must be < max"):
        ContinuousConfig(**config_data)


def test_continuous_config_invalid_too_small_range():
    """ContinuousConfig with range < interaction_radius should fail."""
    config_data = {
        "dimensions": 4,
        "bounds": [(0.0, 10.0), (0.0, 0.5), (0.0, 10.0), (0.0, 10.0)],  # Range 0.5 < interaction 1.0!
        "boundary": "clamp",
        "movement_delta": 0.5,
        "interaction_radius": 1.0,
        "distance_metric": "euclidean",
        "observation_encoding": "relative",
    }

    with pytest.raises(ValueError, match="Space too small"):
        ContinuousConfig(**config_data)


def test_continuous_config_invalid_too_many_dimensions():
    """ContinuousConfig with >100 dimensions should fail."""
    bounds = [(0.0, 10.0) for _ in range(101)]
    config_data = {
        "dimensions": 101,
        "bounds": bounds,
        "boundary": "clamp",
        "movement_delta": 0.5,
        "interaction_radius": 1.0,
        "distance_metric": "euclidean",
    }

    with pytest.raises(Exception, match="less than or equal to 100"):
        ContinuousConfig(**config_data)


def test_continuous_config_chebyshev_metric():
    """ContinuousConfig should support chebyshev metric (new for ND)."""
    config = ContinuousConfig(
        dimensions=4,
        bounds=[(0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0)],
        boundary="clamp",
        movement_delta=0.5,
        interaction_radius=1.0,
        distance_metric="chebyshev",
        observation_encoding="relative",
    )

    assert config.distance_metric == "chebyshev"


# ============================================================================
# SubstrateConfig Integration Tests
# ============================================================================


def test_substrate_config_gridnd():
    """SubstrateConfig with type='gridnd' should require gridnd config."""
    config_data = {
        "version": "1.0",
        "description": "Test GridND substrate",
        "type": "gridnd",
        "gridnd": {
            "dimension_sizes": [8, 8, 8, 8],
            "boundary": "clamp",
            "distance_metric": "manhattan",
            "observation_encoding": "relative",
            "topology": "hypercube",
        },
    }

    config = SubstrateConfig(**config_data)

    assert config.type == "gridnd"
    assert config.gridnd is not None
    assert config.gridnd.dimension_sizes == [8, 8, 8, 8]


def test_substrate_config_gridnd_missing_config():
    """SubstrateConfig with type='gridnd' but missing gridnd config should fail."""
    config_data = {
        "version": "1.0",
        "description": "Test",
        "type": "gridnd",
        # Missing gridnd config!
    }

    with pytest.raises(ValueError, match="gridnd configuration"):
        SubstrateConfig(**config_data)


def test_substrate_config_continuousnd():
    """SubstrateConfig with type='continuousnd' should use continuous config with 4+ dims."""
    config_data = {
        "version": "1.0",
        "description": "Test ContinuousND substrate",
        "type": "continuousnd",
        "continuous": {
            "dimensions": 5,
            "bounds": [(0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0)],
            "boundary": "clamp",
            "movement_delta": 0.5,
            "interaction_radius": 1.0,
            "distance_metric": "euclidean",
            "observation_encoding": "relative",
        },
    }

    config = SubstrateConfig(**config_data)

    assert config.type == "continuousnd"
    assert config.continuous is not None
    assert config.continuous.dimensions == 5


def test_substrate_config_continuousnd_missing_config():
    """SubstrateConfig with type='continuousnd' but missing continuous config should fail."""
    config_data = {
        "version": "1.0",
        "description": "Test",
        "type": "continuousnd",
        # Missing continuous config!
    }

    with pytest.raises(ValueError, match="continuous configuration"):
        SubstrateConfig(**config_data)


def test_substrate_config_continuous_wrong_dimensions():
    """SubstrateConfig type='continuous' with 4+ dimensions should fail."""
    config_data = {
        "version": "1.0",
        "description": "Test",
        "type": "continuous",
        "continuous": {
            "dimensions": 4,  # Should use continuousnd!
            "bounds": [(0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0)],
            "boundary": "clamp",
            "movement_delta": 0.5,
            "interaction_radius": 1.0,
            "distance_metric": "euclidean",
            "observation_encoding": "relative",
        },
    }

    with pytest.raises(ValueError, match="expects 1-3 dimensions.*use type='continuousnd'"):
        SubstrateConfig(**config_data)


def test_substrate_config_continuousnd_wrong_dimensions():
    """SubstrateConfig type='continuousnd' with <4 dimensions should fail."""
    config_data = {
        "version": "1.0",
        "description": "Test",
        "type": "continuousnd",
        "continuous": {
            "dimensions": 3,  # Should use continuous!
            "bounds": [(0.0, 10.0), (0.0, 10.0), (0.0, 10.0)],
            "boundary": "clamp",
            "movement_delta": 0.5,
            "interaction_radius": 1.0,
            "distance_metric": "euclidean",
            "observation_encoding": "relative",
        },
    }

    with pytest.raises(ValueError, match="expects 4\\+ dimensions.*use type='continuous'"):
        SubstrateConfig(**config_data)


def test_substrate_config_gridnd_wrong_config_type():
    """SubstrateConfig type='gridnd' with grid config should fail."""
    config_data = {
        "version": "1.0",
        "description": "Test",
        "type": "gridnd",
        "grid": {  # Wrong config type!
            "topology": "square",
            "width": 8,
            "height": 8,
            "boundary": "clamp",
            "distance_metric": "manhattan",
            "observation_encoding": "relative",
        },
    }

    with pytest.raises(ValueError, match="gridnd configuration"):
        SubstrateConfig(**config_data)


def test_substrate_config_multiple_configs_provided():
    """SubstrateConfig with multiple configs should fail."""
    config_data = {
        "version": "1.0",
        "description": "Test",
        "type": "gridnd",
        "gridnd": {
            "dimension_sizes": [8, 8, 8, 8],
            "boundary": "clamp",
            "distance_metric": "manhattan",
            "observation_encoding": "relative",
            "topology": "hypercube",
        },
        "continuous": {  # Extra config!
            "dimensions": 2,
            "bounds": [(0.0, 10.0), (0.0, 10.0)],
            "boundary": "clamp",
            "movement_delta": 0.5,
            "interaction_radius": 1.0,
            "distance_metric": "euclidean",
            "observation_encoding": "relative",
        },
    }

    with pytest.raises(ValueError, match="Only one substrate configuration"):
        SubstrateConfig(**config_data)


# ============================================================================
# YAML Round-Trip Tests
# ============================================================================


def test_gridnd_yaml_round_trip(tmp_path):
    """GridND config should round-trip through YAML."""
    import yaml

    from townlet.substrate.config import load_substrate_config

    config_data = {
        "version": "1.0",
        "description": "Test GridND YAML",
        "type": "gridnd",
        "gridnd": {
            "dimension_sizes": [5, 5, 5, 5],
            "boundary": "wrap",
            "distance_metric": "euclidean",
            "observation_encoding": "scaled",
            "topology": "hypercube",
        },
    }

    # Write to YAML
    config_path = tmp_path / "substrate.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(config_data, f)

    # Load and validate
    config = load_substrate_config(config_path)

    assert config.type == "gridnd"
    assert config.gridnd.dimension_sizes == [5, 5, 5, 5]
    assert config.gridnd.boundary == "wrap"
    assert config.gridnd.observation_encoding == "scaled"


def test_continuousnd_yaml_round_trip(tmp_path):
    """ContinuousND config should round-trip through YAML."""
    import yaml

    from townlet.substrate.config import load_substrate_config

    config_data = {
        "version": "1.0",
        "description": "Test ContinuousND YAML",
        "type": "continuousnd",
        "continuous": {
            "dimensions": 4,
            "bounds": [(0.0, 10.0), (0.0, 5.0), (-10.0, 10.0), (0.0, 1.0)],
            "boundary": "bounce",
            "movement_delta": 0.5,
            "interaction_radius": 1.0,
            "distance_metric": "manhattan",
            "observation_encoding": "absolute",
        },
    }

    # Write to YAML
    config_path = tmp_path / "substrate.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(config_data, f)

    # Load and validate
    config = load_substrate_config(config_path)

    assert config.type == "continuousnd"
    assert config.continuous.dimensions == 4
    assert config.continuous.bounds == [(0.0, 10.0), (0.0, 5.0), (-10.0, 10.0), (0.0, 1.0)]
    assert config.continuous.boundary == "bounce"
    assert config.continuous.observation_encoding == "absolute"
