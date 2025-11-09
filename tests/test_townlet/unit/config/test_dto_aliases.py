"""Regression tests for DTO collection aliases."""


def test_bars_config_alias():
    from townlet.config.bar import BarsConfig as ConfigBars
    from townlet.environment.cascade_config import BarsConfig as EnvBars

    assert ConfigBars is EnvBars


def test_cascades_config_alias():
    from townlet.config.cascade import CascadesConfig as ConfigCascades
    from townlet.environment.cascade_config import CascadesConfig as EnvCascades

    assert ConfigCascades is EnvCascades


def test_affordance_collection_alias():
    from townlet.config.affordance import (
        AffordanceConfigCollection as ConfigCollection,
    )
    from townlet.environment.affordance_config import (
        AffordanceConfigCollection as EnvCollection,
    )

    assert ConfigCollection is EnvCollection
