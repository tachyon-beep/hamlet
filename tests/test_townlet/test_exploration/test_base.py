"""Tests for ExplorationStrategy abstract interface."""

import pytest


def test_exploration_strategy_cannot_instantiate():
    """ExplorationStrategy ABC should not be instantiable."""
    from townlet.exploration.base import ExplorationStrategy

    with pytest.raises(TypeError) as exc_info:
        ExplorationStrategy()

    assert "abstract" in str(exc_info.value).lower()


def test_exploration_strategy_requires_all_methods():
    """Subclass must implement all abstract methods."""
    from townlet.exploration.base import ExplorationStrategy

    class IncompleteExploration(ExplorationStrategy):
        pass

    with pytest.raises(TypeError):
        IncompleteExploration()


def test_exploration_strategy_interface_signature():
    """ExplorationStrategy should have expected method signatures."""
    from townlet.exploration.base import ExplorationStrategy
    import inspect

    abstract_methods = {
        name for name, method in inspect.getmembers(ExplorationStrategy)
        if getattr(method, '__isabstractmethod__', False)
    }

    assert 'select_actions' in abstract_methods
    assert 'compute_intrinsic_rewards' in abstract_methods
    assert 'update' in abstract_methods
    assert 'checkpoint_state' in abstract_methods
    assert 'load_state' in abstract_methods
