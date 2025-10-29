"""Tests for CurriculumManager abstract interface."""

import pytest


def test_curriculum_manager_cannot_instantiate():
    """CurriculumManager ABC should not be instantiable."""
    from townlet.curriculum.base import CurriculumManager

    with pytest.raises(TypeError) as exc_info:
        CurriculumManager()

    assert "abstract" in str(exc_info.value).lower()


def test_curriculum_manager_requires_all_methods():
    """Subclass must implement all abstract methods."""
    from townlet.curriculum.base import CurriculumManager

    class IncompleteCurriculum(CurriculumManager):
        # Missing implementations
        pass

    with pytest.raises(TypeError):
        IncompleteCurriculum()


def test_curriculum_manager_interface_signature():
    """CurriculumManager should have expected method signatures."""
    from townlet.curriculum.base import CurriculumManager
    import inspect

    # Check that abstract methods exist
    abstract_methods = {
        name for name, method in inspect.getmembers(CurriculumManager)
        if getattr(method, '__isabstractmethod__', False)
    }

    assert 'get_batch_decisions' in abstract_methods
    assert 'checkpoint_state' in abstract_methods
    assert 'load_state' in abstract_methods
