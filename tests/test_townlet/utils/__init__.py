"""Test utilities for Townlet test suite.

This module provides reusable builder factories and assertion helpers to reduce
boilerplate in test files.

Usage:
    from tests.test_townlet.utils.builders import make_grid2d_substrate
    from tests.test_townlet.utils.assertions import assert_valid_observation

    def test_something():
        substrate = make_grid2d_substrate(width=3, height=3)
        obs = env.reset()
        assert_valid_observation(env, obs)
"""
