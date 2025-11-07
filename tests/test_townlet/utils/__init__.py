"""Test utilities for Townlet test suite.

This module provides reusable builder factories, assertion helpers, and polling
utilities to reduce boilerplate in test files.

Usage:
    from tests.test_townlet.utils.builders import make_grid2d_substrate
    from tests.test_townlet.utils.assertions import assert_valid_observation
    from tests.test_townlet.utils.polling import wait_for_condition

    def test_something():
        substrate = make_grid2d_substrate(width=3, height=3)
        obs = env.reset()
        assert_valid_observation(env, obs)

    def test_threading():
        # Wait for background thread to process item
        assert wait_for_condition(lambda: queue.empty(), timeout=2.0)
"""

from tests.test_townlet.utils.polling import wait_for_condition

__all__ = ["wait_for_condition"]
