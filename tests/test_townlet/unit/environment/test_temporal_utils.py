"""Tests for temporal_utils.py (JANK-09 fix)."""

from townlet.environment.temporal_utils import is_affordance_open


class TestIsAffordanceOpen:
    """Test canonical is_affordance_open() implementation."""

    def test_normal_hours_inside(self):
        """Test normal hours (open < close), time inside range."""
        # 9am-5pm, check at noon
        assert is_affordance_open(12, (9, 17)) is True
        # 8am-6pm, check at 8am (inclusive)
        assert is_affordance_open(8, (8, 18)) is True
        # 0-12, check at 11am
        assert is_affordance_open(11, (0, 12)) is True

    def test_normal_hours_outside(self):
        """Test normal hours (open < close), time outside range."""
        # 9am-5pm, check at 8am
        assert is_affordance_open(8, (9, 17)) is False
        # 9am-5pm, check at 5pm (exclusive)
        assert is_affordance_open(17, (9, 17)) is False
        # 9am-5pm, check at 6pm
        assert is_affordance_open(18, (9, 17)) is False

    def test_wraparound_modulo_notation(self):
        """Test wraparound with >24 notation (e.g., Bar: 18-28 = 6pm-4am)."""
        # 6pm-4am (18-28 notation)
        assert is_affordance_open(20, (18, 28)) is True  # 8pm - open
        assert is_affordance_open(2, (18, 28)) is True  # 2am - open
        assert is_affordance_open(18, (18, 28)) is True  # 6pm - open (inclusive)
        assert is_affordance_open(4, (18, 28)) is False  # 4am - closed (exclusive)
        assert is_affordance_open(5, (18, 28)) is False  # 5am - closed
        assert is_affordance_open(12, (18, 28)) is False  # Noon - closed

    def test_wraparound_negative_notation(self):
        """Test wraparound with open>close notation (e.g., 22-6 = 10pm-6am)."""
        # 10pm-6am (22-6 notation)
        assert is_affordance_open(23, (22, 6)) is True  # 11pm - open
        assert is_affordance_open(1, (22, 6)) is True  # 1am - open
        assert is_affordance_open(22, (22, 6)) is True  # 10pm - open (inclusive)
        assert is_affordance_open(6, (22, 6)) is False  # 6am - closed (exclusive)
        assert is_affordance_open(12, (22, 6)) is False  # Noon - closed

    def test_24_7_operation(self):
        """Test 24/7 operation (interval spans full day)."""
        # Standard 24/7: 0-24
        assert is_affordance_open(0, (0, 24)) is True
        assert is_affordance_open(12, (0, 24)) is True
        assert is_affordance_open(23, (0, 24)) is True

        # Wraparound 24/7: 8-32 (covers full day)
        assert is_affordance_open(0, (8, 32)) is True
        assert is_affordance_open(12, (8, 32)) is True
        assert is_affordance_open(23, (8, 32)) is True

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Midnight crossing (23-1 = 11pm-1am)
        assert is_affordance_open(23, (23, 1)) is True
        assert is_affordance_open(0, (23, 1)) is True
        assert is_affordance_open(1, (23, 1)) is False

        # Single hour window
        assert is_affordance_open(12, (12, 13)) is True
        assert is_affordance_open(13, (12, 13)) is False

    def test_time_normalization(self):
        """Test that time_of_day is normalized to [0-23]."""
        # 25 should normalize to 1
        assert is_affordance_open(25, (0, 12)) is True
        assert is_affordance_open(25, (12, 18)) is False

        # 48 should normalize to 0
        assert is_affordance_open(48, (0, 12)) is True
        assert is_affordance_open(48, (12, 18)) is False

    def test_consistency_all_three_notations(self):
        """Regression test: All notation styles must agree."""
        # Bar hours: 6pm-4am
        # Can be expressed as [18, 28] or [18, 4] (after modulo)
        # Both should give identical results
        for hour in range(24):
            result_modulo = is_affordance_open(hour, (18, 28))
            # After modulo: 28 % 24 = 4, so [18, 28] becomes [18, 4]
            # The function should handle this internally
            expected_result = hour >= 18 or hour < 4
            assert result_modulo == expected_result, f"Hour {hour} failed"

    def test_jank_09_regression(self):
        """Regression test for JANK-09 bugs.

        Previously three implementations disagreed on these cases:
        - affordance_config.is_affordance_open() failed on [18, 28] notation
        - AffordanceEngine.is_affordance_open() failed on [22, 6] notation
        - UniverseCompiler._is_open() was correct (became this function)
        """
        # Case 1: [18, 28] notation at 2am (previously failed in V1)
        assert is_affordance_open(2, (18, 28)) is True

        # Case 2: [22, 6] notation at 23:00 (previously failed in V2)
        assert is_affordance_open(23, (22, 6)) is True

        # Case 3: [22, 6] notation at 1am (previously failed in V2)
        assert is_affordance_open(1, (22, 6)) is True
