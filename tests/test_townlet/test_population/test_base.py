"""Tests for PopulationManager abstract interface."""

import pytest


def test_population_manager_cannot_instantiate():
    """PopulationManager ABC should not be instantiable."""
    from townlet.population.base import PopulationManager

    with pytest.raises(TypeError) as exc_info:
        PopulationManager()

    assert "abstract" in str(exc_info.value).lower()


def test_population_manager_requires_all_methods():
    """Subclass must implement all abstract methods."""
    from townlet.population.base import PopulationManager

    class IncompletePopulation(PopulationManager):
        pass

    with pytest.raises(TypeError):
        IncompletePopulation()


def test_population_manager_interface_signature():
    """PopulationManager should have expected method signatures."""
    from townlet.population.base import PopulationManager
    import inspect

    abstract_methods = {
        name for name, method in inspect.getmembers(PopulationManager)
        if getattr(method, '__isabstractmethod__', False)
    }

    assert 'step_population' in abstract_methods
    assert 'get_checkpoint' in abstract_methods
