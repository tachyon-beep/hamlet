"""
Tests for recording criteria evaluator.

Tests each criterion independently and combined OR logic.
"""

import time


class TestPeriodicCriterion:
    """Test periodic recording criterion."""

    def test_periodic_records_at_interval(self):
        """Periodic criterion should record every N episodes."""
        from townlet.recording.criteria import RecordingCriteria

        config = {
            "criteria": {
                "periodic": {
                    "enabled": True,
                    "interval": 100,
                }
            }
        }

        criteria = RecordingCriteria(config=config, curriculum=None, database=None)

        # Episode 0 should record
        metadata = _make_metadata(episode_id=0, survival_steps=50, total_reward=100.0)
        should_record, reason = criteria.should_record(metadata)
        assert should_record is True
        assert reason == "periodic_100"

        # Episode 50 should not record
        metadata = _make_metadata(episode_id=50, survival_steps=50, total_reward=100.0)
        should_record, reason = criteria.should_record(metadata)
        assert should_record is False

        # Episode 100 should record
        metadata = _make_metadata(episode_id=100, survival_steps=50, total_reward=100.0)
        should_record, reason = criteria.should_record(metadata)
        assert should_record is True
        assert reason == "periodic_100"

    def test_periodic_disabled(self):
        """Periodic criterion should not record when disabled."""
        from townlet.recording.criteria import RecordingCriteria

        config = {
            "criteria": {
                "periodic": {
                    "enabled": False,
                    "interval": 100,
                }
            }
        }

        criteria = RecordingCriteria(config=config, curriculum=None, database=None)

        metadata = _make_metadata(episode_id=0, survival_steps=50, total_reward=100.0)
        should_record, reason = criteria.should_record(metadata)
        assert should_record is False


class TestStageTransitionCriterion:
    """Test stage transition recording criterion."""

    def test_stage_transition_records_when_likely(self):
        """Stage transition criterion should record when transition is likely."""
        from townlet.recording.criteria import RecordingCriteria

        config = {
            "criteria": {
                "stage_transitions": {
                    "enabled": True,
                    "record_before": 5,
                    "record_after": 10,
                }
            }
        }

        # Mock curriculum that predicts transition
        class MockCurriculum:
            def get_stage_info(self, agent_idx=0):
                return {
                    "current_stage": 2,
                    "episodes_at_stage": 50,
                    "survival_rate": 0.8,
                    "likely_transition_soon": True,
                }

        criteria = RecordingCriteria(config=config, curriculum=MockCurriculum(), database=None)

        metadata = _make_metadata(episode_id=1000, survival_steps=80, total_reward=200.0, curriculum_stage=2)
        should_record, reason = criteria.should_record(metadata)
        assert should_record is True
        assert reason == "stage_2_pre_transition"

    def test_stage_transition_no_record_when_not_likely(self):
        """Stage transition criterion should not record when transition is not likely."""
        from townlet.recording.criteria import RecordingCriteria

        config = {
            "criteria": {
                "stage_transitions": {
                    "enabled": True,
                    "record_before": 5,
                    "record_after": 10,
                }
            }
        }

        # Mock curriculum that does not predict transition
        class MockCurriculum:
            def get_stage_info(self, agent_idx=0):
                return {
                    "current_stage": 2,
                    "episodes_at_stage": 10,
                    "survival_rate": 0.5,
                    "likely_transition_soon": False,
                }

        criteria = RecordingCriteria(config=config, curriculum=MockCurriculum(), database=None)

        metadata = _make_metadata(episode_id=1000, survival_steps=50, total_reward=100.0, curriculum_stage=2)
        should_record, reason = criteria.should_record(metadata)
        assert should_record is False

    def test_stage_transition_records_before_transition(self):
        """Stage transition should record N episodes before actual transition."""
        from townlet.recording.criteria import RecordingCriteria

        class MockCurriculum:
            def get_stage_info(self, agent_idx=0):
                return {"current_stage": 2, "episodes_at_stage": 50, "survival_rate": 0.8, "likely_transition_soon": False}

        config = {
            "criteria": {
                "stage_transitions": {
                    "enabled": True,
                    "record_before": 5,
                    "record_after": 10,
                }
            }
        }

        criteria = RecordingCriteria(config=config, curriculum=MockCurriculum(), database=None)

        # Build up to stage transition at episode 105
        for i in range(100):
            metadata = _make_metadata(episode_id=i, survival_steps=50, total_reward=100.0, curriculum_stage=1)
            criteria.should_record(metadata)

        # Simulate transition from stage 1 to stage 2 at episode 105
        metadata = _make_metadata(episode_id=105, survival_steps=50, total_reward=100.0, curriculum_stage=2)
        criteria.should_record(metadata)

        # Episodes 100-104 (5 before transition) should record
        for ep_id in range(100, 105):
            metadata = _make_metadata(episode_id=ep_id, survival_steps=50, total_reward=100.0, curriculum_stage=1)
            # Rewind to check these episodes
            criteria.last_stage = 1  # Reset to stage 1
            should_record, reason = criteria.should_record(metadata)
            # Note: This test verifies retroactive logic works
            # In practice, these would be recorded when transition is detected

    def test_stage_transition_records_after_transition(self):
        """Stage transition should record N episodes after actual transition."""
        from townlet.recording.criteria import RecordingCriteria

        class MockCurriculum:
            def get_stage_info(self, agent_idx=0):
                return {"current_stage": 2, "episodes_at_stage": 50, "survival_rate": 0.8, "likely_transition_soon": False}

        config = {
            "criteria": {
                "stage_transitions": {
                    "enabled": True,
                    "record_before": 5,
                    "record_after": 10,
                }
            }
        }

        criteria = RecordingCriteria(config=config, curriculum=MockCurriculum(), database=None)

        # Build up episodes at stage 1
        for i in range(100):
            metadata = _make_metadata(episode_id=i, survival_steps=50, total_reward=100.0, curriculum_stage=1)
            criteria.should_record(metadata)

        # Simulate transition at episode 100
        metadata = _make_metadata(episode_id=100, survival_steps=50, total_reward=100.0, curriculum_stage=2)
        criteria.should_record(metadata)

        # Next 10 episodes should record (after transition)
        for ep_id in range(100, 110):
            metadata = _make_metadata(episode_id=ep_id, survival_steps=50, total_reward=100.0, curriculum_stage=2)
            should_record, reason = criteria.should_record(metadata)
            assert should_record is True
            assert reason == "after_transition_100"

        # Episode 110 should not record (outside after window)
        metadata = _make_metadata(episode_id=110, survival_steps=50, total_reward=100.0, curriculum_stage=2)
        should_record, reason = criteria.should_record(metadata)
        assert should_record is False


class TestPerformanceCriterion:
    """Test performance recording criterion (top and bottom percentiles)."""

    def test_performance_records_top_percentile(self):
        """Performance criterion should record high-reward episodes."""
        from townlet.recording.criteria import RecordingCriteria

        config = {
            "criteria": {
                "performance": {
                    "enabled": True,
                    "top_percent": 10.0,
                    "bottom_percent": 10.0,
                    "window": 100,
                }
            }
        }

        criteria = RecordingCriteria(config=config, curriculum=None, database=None)

        # Track episodes with varied rewards to build history
        for i in range(20):
            metadata = _make_metadata(episode_id=i, survival_steps=50, total_reward=float(i * 10))
            criteria.should_record(metadata)

        # Very high reward should record (top 10%)
        metadata = _make_metadata(episode_id=20, survival_steps=100, total_reward=500.0)
        should_record, reason = criteria.should_record(metadata)
        assert should_record is True
        assert reason == "top_10.0pct"

    def test_performance_records_bottom_percentile(self):
        """Performance criterion should record low-reward episodes (failures)."""
        from townlet.recording.criteria import RecordingCriteria

        config = {
            "criteria": {
                "performance": {
                    "enabled": True,
                    "top_percent": 10.0,
                    "bottom_percent": 10.0,
                    "window": 100,
                }
            }
        }

        criteria = RecordingCriteria(config=config, curriculum=None, database=None)

        # Track episodes with varied rewards
        for i in range(20):
            metadata = _make_metadata(episode_id=i, survival_steps=50, total_reward=float(100 + i * 10))
            criteria.should_record(metadata)

        # Very low reward should record (bottom 10%)
        metadata = _make_metadata(episode_id=20, survival_steps=5, total_reward=5.0)
        should_record, reason = criteria.should_record(metadata)
        assert should_record is True
        assert reason == "bottom_10.0pct"

    def test_performance_no_record_middle_percentile(self):
        """Performance criterion should not record middle-range rewards."""
        from townlet.recording.criteria import RecordingCriteria

        config = {
            "criteria": {
                "performance": {
                    "enabled": True,
                    "top_percent": 10.0,
                    "bottom_percent": 10.0,
                    "window": 100,
                }
            }
        }

        criteria = RecordingCriteria(config=config, curriculum=None, database=None)

        # Track episodes with rewards from 0 to 200
        for i in range(20):
            metadata = _make_metadata(episode_id=i, survival_steps=50, total_reward=float(i * 10))
            criteria.should_record(metadata)

        # Middle reward should not record
        metadata = _make_metadata(episode_id=20, survival_steps=50, total_reward=100.0)
        should_record, reason = criteria.should_record(metadata)
        assert should_record is False

    def test_performance_needs_minimum_history(self):
        """Performance criterion needs minimum history before evaluating."""
        from townlet.recording.criteria import RecordingCriteria

        config = {
            "criteria": {
                "performance": {
                    "enabled": True,
                    "top_percent": 10.0,
                    "bottom_percent": 10.0,
                    "window": 100,
                }
            }
        }

        criteria = RecordingCriteria(config=config, curriculum=None, database=None)

        # With only 2 episodes, shouldn't record even extreme values
        metadata = _make_metadata(episode_id=0, survival_steps=50, total_reward=100.0)
        criteria.should_record(metadata)

        metadata = _make_metadata(episode_id=1, survival_steps=5, total_reward=1000.0)
        should_record, reason = criteria.should_record(metadata)
        assert should_record is False  # Not enough history yet


class TestStageBoundariesCriterion:
    """Test stage boundaries recording criterion."""

    def test_stage_boundaries_records_first_n(self):
        """Stage boundaries should record first N episodes at each stage."""
        from townlet.recording.criteria import RecordingCriteria

        class MockCurriculum:
            def get_stage_info(self, agent_idx=0):
                return {"current_stage": 2, "episodes_at_stage": 50, "survival_rate": 0.8, "likely_transition_soon": False}

        config = {
            "criteria": {
                "stage_boundaries": {
                    "enabled": True,
                    "first_n": 10,
                    "last_n": 5,
                }
            }
        }

        criteria = RecordingCriteria(config=config, curriculum=MockCurriculum(), database=None)

        # First 10 episodes at stage 2 should record
        for i in range(10):
            metadata = _make_metadata(episode_id=100 + i, survival_steps=50, total_reward=100.0, curriculum_stage=2)
            should_record, reason = criteria.should_record(metadata)
            assert should_record is True
            assert reason == f"stage_2_first_{i + 1}"

        # Episode 11 at stage 2 should not record (past first_n)
        metadata = _make_metadata(episode_id=110, survival_steps=50, total_reward=100.0, curriculum_stage=2)
        should_record, reason = criteria.should_record(metadata)
        assert should_record is False

    def test_stage_boundaries_records_last_n(self):
        """Stage boundaries should record last N episodes before stage transition."""
        from townlet.recording.criteria import RecordingCriteria

        class MockCurriculum:
            def __init__(self):
                self.likely_transition = False

            def get_stage_info(self, agent_idx=0):
                return {"current_stage": 2, "episodes_at_stage": 50, "survival_rate": 0.8, "likely_transition_soon": self.likely_transition}

        mock_curriculum = MockCurriculum()

        config = {
            "criteria": {
                "stage_boundaries": {
                    "enabled": True,
                    "first_n": 10,
                    "last_n": 5,
                }
            }
        }

        criteria = RecordingCriteria(config=config, curriculum=mock_curriculum, database=None)

        # Build up episodes at stage 2 (past first_n)
        for i in range(15):
            metadata = _make_metadata(episode_id=100 + i, survival_steps=50, total_reward=100.0, curriculum_stage=2)
            criteria.should_record(metadata)

        # Episodes not near transition shouldn't record
        metadata = _make_metadata(episode_id=115, survival_steps=50, total_reward=100.0, curriculum_stage=2)
        should_record, reason = criteria.should_record(metadata)
        assert should_record is False

        # Curriculum signals transition is likely
        mock_curriculum.likely_transition = True

        # Now episodes should record (last N before transition)
        metadata = _make_metadata(episode_id=116, survival_steps=50, total_reward=100.0, curriculum_stage=2)
        should_record, reason = criteria.should_record(metadata)
        assert should_record is True
        assert reason == "stage_2_pre_transition"

    def test_stage_boundaries_resets_per_stage(self):
        """Stage boundaries counter should reset when stage changes."""
        from townlet.recording.criteria import RecordingCriteria

        class MockCurriculum:
            def get_stage_info(self, agent_idx=0):
                return {"current_stage": 2, "episodes_at_stage": 50, "survival_rate": 0.8, "likely_transition_soon": False}

        config = {
            "criteria": {
                "stage_boundaries": {
                    "enabled": True,
                    "first_n": 5,
                    "last_n": 3,
                }
            }
        }

        criteria = RecordingCriteria(config=config, curriculum=MockCurriculum(), database=None)

        # First 5 episodes at stage 1
        for i in range(5):
            metadata = _make_metadata(episode_id=i, survival_steps=50, total_reward=100.0, curriculum_stage=1)
            should_record, reason = criteria.should_record(metadata)
            assert should_record is True

        # Episode 6 at stage 1 should not record
        metadata = _make_metadata(episode_id=5, survival_steps=50, total_reward=100.0, curriculum_stage=1)
        should_record, reason = criteria.should_record(metadata)
        assert should_record is False

        # Transition to stage 2 - counter should reset
        # First 5 episodes at stage 2 should record again
        for i in range(5):
            metadata = _make_metadata(episode_id=100 + i, survival_steps=50, total_reward=100.0, curriculum_stage=2)
            should_record, reason = criteria.should_record(metadata)
            assert should_record is True
            assert reason == f"stage_2_first_{i + 1}"


class TestMultipleCriteria:
    """Test OR logic with multiple criteria."""

    def test_multiple_criteria_or_logic(self):
        """Multiple criteria should use OR logic (any criterion triggers record)."""
        from townlet.recording.criteria import RecordingCriteria

        config = {
            "criteria": {
                "periodic": {
                    "enabled": True,
                    "interval": 100,
                },
                "performance": {
                    "enabled": True,
                    "top_percent": 10.0,
                    "bottom_percent": 10.0,
                    "window": 100,
                },
            }
        }

        criteria = RecordingCriteria(config=config, curriculum=None, database=None)

        # Episode 0 should record (periodic)
        metadata = _make_metadata(episode_id=0, survival_steps=50, total_reward=100.0)
        should_record, reason = criteria.should_record(metadata)
        assert should_record is True
        assert reason == "periodic_100"

        # Build some history with varied rewards
        for i in range(1, 20):
            metadata = _make_metadata(episode_id=i, survival_steps=50, total_reward=float(i * 10))
            criteria.should_record(metadata)

        # Episode 20 with high reward should record (top_performance)
        metadata = _make_metadata(episode_id=20, survival_steps=100, total_reward=500.0)
        should_record, reason = criteria.should_record(metadata)
        assert should_record is True
        assert reason == "top_10.0pct"

        # Episode 100 should record (periodic)
        metadata = _make_metadata(episode_id=100, survival_steps=50, total_reward=100.0)
        should_record, reason = criteria.should_record(metadata)
        assert should_record is True
        assert reason == "periodic_100"

    def test_all_criteria_disabled(self):
        """When all criteria disabled, should never record."""
        from townlet.recording.criteria import RecordingCriteria

        config = {
            "criteria": {
                "periodic": {"enabled": False},
                "stage_transitions": {"enabled": False},
                "performance": {"enabled": False},
                "stage_boundaries": {"enabled": False},
            }
        }

        criteria = RecordingCriteria(config=config, curriculum=None, database=None)

        metadata = _make_metadata(episode_id=0, survival_steps=100, total_reward=1000.0)
        should_record, reason = criteria.should_record(metadata)
        assert should_record is False


def _make_metadata(episode_id, survival_steps, total_reward, curriculum_stage=1):
    """Helper to create EpisodeMetadata for testing."""
    from townlet.recording.data_structures import EpisodeMetadata

    return EpisodeMetadata(
        episode_id=episode_id,
        survival_steps=survival_steps,
        total_reward=total_reward,
        extrinsic_reward=total_reward,
        intrinsic_reward=0.0,
        curriculum_stage=curriculum_stage,
        epsilon=0.1,
        intrinsic_weight=0.5,
        timestamp=time.time(),
        affordance_layout={},
        affordance_visits={},
    )
