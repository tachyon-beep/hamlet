"""
Tests for database recording integration.

Tests the episode_recordings table and query methods.
"""

import tempfile
import time
from pathlib import Path


class TestDatabaseRecording:
    """Test database methods for recording metadata."""

    def test_create_recordings_table(self):
        """Database should create episode_recordings table."""
        from townlet.demo.database import DemoDatabase

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = DemoDatabase(db_path)

            # Verify table exists
            cursor = db.conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='episode_recordings'")
            result = cursor.fetchone()
            assert result is not None
            assert result[0] == "episode_recordings"

            db.close()

    def test_insert_recording(self):
        """insert_recording should add recording metadata to database."""
        from townlet.demo.database import DemoDatabase
        from townlet.recording.data_structures import EpisodeMetadata

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = DemoDatabase(db_path)

            metadata = EpisodeMetadata(
                episode_id=100,
                survival_steps=487,
                total_reward=423.7,
                extrinsic_reward=410.2,
                intrinsic_reward=13.5,
                curriculum_stage=3,
                epsilon=0.15,
                intrinsic_weight=0.3,
                timestamp=time.time(),
                affordance_layout={"Bed": (2, 3)},
                affordance_visits={"Bed": 15},
            )

            db.insert_recording(
                episode_id=100,
                file_path="recordings/episode_000100.msgpack.lz4",
                metadata=metadata,
                reason="periodic_100",
                file_size=50000,
                compressed_size=15000,
            )

            # Verify inserted
            cursor = db.conn.execute("SELECT * FROM episode_recordings WHERE episode_id = ?", (100,))
            row = cursor.fetchone()

            assert row is not None
            assert row["episode_id"] == 100
            assert row["file_path"] == "recordings/episode_000100.msgpack.lz4"
            assert row["survival_steps"] == 487
            assert row["total_reward"] == 423.7
            assert row["curriculum_stage"] == 3
            assert row["recording_reason"] == "periodic_100"
            assert row["file_size_bytes"] == 50000
            assert row["compressed_size_bytes"] == 15000

            db.close()

    def test_get_recording(self):
        """get_recording should retrieve recording by episode_id."""
        from townlet.demo.database import DemoDatabase
        from townlet.recording.data_structures import EpisodeMetadata

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = DemoDatabase(db_path)

            metadata = EpisodeMetadata(
                episode_id=200,
                survival_steps=300,
                total_reward=250.0,
                extrinsic_reward=240.0,
                intrinsic_reward=10.0,
                curriculum_stage=2,
                epsilon=0.5,
                intrinsic_weight=0.5,
                timestamp=time.time(),
                affordance_layout={},
                affordance_visits={},
            )

            db.insert_recording(
                episode_id=200,
                file_path="recordings/episode_000200.msgpack.lz4",
                metadata=metadata,
                reason="stage_transition",
                file_size=40000,
                compressed_size=12000,
            )

            # Get recording
            recording = db.get_recording(200)

            assert recording is not None
            assert recording["episode_id"] == 200
            assert recording["file_path"] == "recordings/episode_000200.msgpack.lz4"
            assert recording["recording_reason"] == "stage_transition"

            db.close()

    def test_get_recording_not_found(self):
        """get_recording should return None if episode not found."""
        from townlet.demo.database import DemoDatabase

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = DemoDatabase(db_path)

            recording = db.get_recording(999)
            assert recording is None

            db.close()

    def test_list_recordings_all(self):
        """list_recordings should return all recordings."""
        from townlet.demo.database import DemoDatabase
        from townlet.recording.data_structures import EpisodeMetadata

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = DemoDatabase(db_path)

            # Insert multiple recordings
            for i in range(5):
                metadata = EpisodeMetadata(
                    episode_id=i * 100,
                    survival_steps=100 + i * 50,
                    total_reward=100.0 + i * 50,
                    extrinsic_reward=100.0 + i * 50,
                    intrinsic_reward=0.0,
                    curriculum_stage=1,
                    epsilon=1.0,
                    intrinsic_weight=1.0,
                    timestamp=time.time(),
                    affordance_layout={},
                    affordance_visits={},
                )
                db.insert_recording(
                    episode_id=i * 100,
                    file_path=f"recordings/episode_{i*100:06d}.msgpack.lz4",
                    metadata=metadata,
                    reason="periodic",
                    file_size=10000,
                    compressed_size=3000,
                )

            # List all
            recordings = db.list_recordings()

            assert len(recordings) == 5
            # Should be in descending order by episode_id
            assert recordings[0]["episode_id"] == 400
            assert recordings[4]["episode_id"] == 0

            db.close()

    def test_list_recordings_filter_by_stage(self):
        """list_recordings should filter by curriculum stage."""
        from townlet.demo.database import DemoDatabase
        from townlet.recording.data_structures import EpisodeMetadata

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = DemoDatabase(db_path)

            # Insert recordings at different stages
            for stage in [1, 2, 3]:
                for i in range(2):
                    episode_id = stage * 100 + i
                    metadata = EpisodeMetadata(
                        episode_id=episode_id,
                        survival_steps=100,
                        total_reward=100.0,
                        extrinsic_reward=100.0,
                        intrinsic_reward=0.0,
                        curriculum_stage=stage,
                        epsilon=1.0,
                        intrinsic_weight=1.0,
                        timestamp=time.time(),
                        affordance_layout={},
                        affordance_visits={},
                    )
                    db.insert_recording(
                        episode_id=episode_id,
                        file_path=f"recordings/episode_{episode_id:06d}.msgpack.lz4",
                        metadata=metadata,
                        reason="periodic",
                        file_size=10000,
                        compressed_size=3000,
                    )

            # Filter by stage 2
            recordings = db.list_recordings(stage=2)

            assert len(recordings) == 2
            assert all(r["curriculum_stage"] == 2 for r in recordings)

            db.close()

    def test_list_recordings_filter_by_reason(self):
        """list_recordings should filter by recording reason."""
        from townlet.demo.database import DemoDatabase
        from townlet.recording.data_structures import EpisodeMetadata

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = DemoDatabase(db_path)

            # Insert recordings with different reasons
            reasons = ["periodic", "stage_transition", "top_performance"]
            for i, reason in enumerate(reasons):
                metadata = EpisodeMetadata(
                    episode_id=i * 100,
                    survival_steps=100,
                    total_reward=100.0,
                    extrinsic_reward=100.0,
                    intrinsic_reward=0.0,
                    curriculum_stage=1,
                    epsilon=1.0,
                    intrinsic_weight=1.0,
                    timestamp=time.time(),
                    affordance_layout={},
                    affordance_visits={},
                )
                db.insert_recording(
                    episode_id=i * 100,
                    file_path=f"recordings/episode_{i*100:06d}.msgpack.lz4",
                    metadata=metadata,
                    reason=reason,
                    file_size=10000,
                    compressed_size=3000,
                )

            # Filter by reason
            recordings = db.list_recordings(reason="stage_transition")

            assert len(recordings) == 1
            assert recordings[0]["recording_reason"] == "stage_transition"

            db.close()

    def test_list_recordings_filter_by_reward_range(self):
        """list_recordings should filter by reward range."""
        from townlet.demo.database import DemoDatabase
        from townlet.recording.data_structures import EpisodeMetadata

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = DemoDatabase(db_path)

            # Insert recordings with different rewards
            for i in range(5):
                reward = 100.0 + i * 100.0  # 100, 200, 300, 400, 500
                metadata = EpisodeMetadata(
                    episode_id=i,
                    survival_steps=100,
                    total_reward=reward,
                    extrinsic_reward=reward,
                    intrinsic_reward=0.0,
                    curriculum_stage=1,
                    epsilon=1.0,
                    intrinsic_weight=1.0,
                    timestamp=time.time(),
                    affordance_layout={},
                    affordance_visits={},
                )
                db.insert_recording(
                    episode_id=i,
                    file_path=f"recordings/episode_{i:06d}.msgpack.lz4",
                    metadata=metadata,
                    reason="periodic",
                    file_size=10000,
                    compressed_size=3000,
                )

            # Filter by reward range [250, 450]
            recordings = db.list_recordings(min_reward=250.0, max_reward=450.0)

            # 300, 400 satisfy 250 <= reward <= 450 (not 500)
            assert len(recordings) == 2
            rewards = [r["total_reward"] for r in recordings]
            assert 300.0 in rewards
            assert 400.0 in rewards

            db.close()

    def test_list_recordings_limit(self):
        """list_recordings should respect limit parameter."""
        from townlet.demo.database import DemoDatabase
        from townlet.recording.data_structures import EpisodeMetadata

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = DemoDatabase(db_path)

            # Insert many recordings
            for i in range(20):
                metadata = EpisodeMetadata(
                    episode_id=i,
                    survival_steps=100,
                    total_reward=100.0,
                    extrinsic_reward=100.0,
                    intrinsic_reward=0.0,
                    curriculum_stage=1,
                    epsilon=1.0,
                    intrinsic_weight=1.0,
                    timestamp=time.time(),
                    affordance_layout={},
                    affordance_visits={},
                )
                db.insert_recording(
                    episode_id=i,
                    file_path=f"recordings/episode_{i:06d}.msgpack.lz4",
                    metadata=metadata,
                    reason="periodic",
                    file_size=10000,
                    compressed_size=3000,
                )

            # Limit to 10
            recordings = db.list_recordings(limit=10)

            assert len(recordings) == 10
            # Should be most recent (highest episode_id)
            assert recordings[0]["episode_id"] == 19
            assert recordings[9]["episode_id"] == 10

            db.close()


class TestAffordanceTransitions:
    """Test affordance transition tracking in database."""

    def test_insert_affordance_visits(self):
        """insert_affordance_visits should insert transition counts."""
        from townlet.demo.database import DemoDatabase

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = DemoDatabase(db_path)

            transitions = {
                "Bed": {"Hospital": 3, "Job": 1},
                "Hospital": {"Bed": 2},
            }

            db.insert_affordance_visits(episode_id=100, transitions=transitions)

            # Query back
            cursor = db.conn.execute("SELECT * FROM affordance_visits WHERE episode_id = ? ORDER BY from_affordance, to_affordance", (100,))
            rows = cursor.fetchall()

            assert len(rows) == 3
            assert rows[0]["episode_id"] == 100
            assert rows[0]["from_affordance"] == "Bed"
            assert rows[0]["to_affordance"] == "Hospital"
            assert rows[0]["visit_count"] == 3

            assert rows[1]["episode_id"] == 100
            assert rows[1]["from_affordance"] == "Bed"
            assert rows[1]["to_affordance"] == "Job"
            assert rows[1]["visit_count"] == 1

            assert rows[2]["episode_id"] == 100
            assert rows[2]["from_affordance"] == "Hospital"
            assert rows[2]["to_affordance"] == "Bed"
            assert rows[2]["visit_count"] == 2

            db.close()

    def test_insert_affordance_visits_empty(self):
        """Empty transitions should not crash."""
        from townlet.demo.database import DemoDatabase

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = DemoDatabase(db_path)

            # Should complete without error
            db.insert_affordance_visits(episode_id=100, transitions={})

            # Verify no rows inserted
            cursor = db.conn.execute("SELECT COUNT(*) as count FROM affordance_visits WHERE episode_id = ?", (100,))
            result = cursor.fetchone()
            assert result["count"] == 0

            db.close()

    def test_insert_affordance_visits_self_loop(self):
        """Self-loops (Bed→Bed) should be recorded."""
        from townlet.demo.database import DemoDatabase

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = DemoDatabase(db_path)

            transitions = {"Bed": {"Bed": 5}}  # Agent used Bed 5 times consecutively

            db.insert_affordance_visits(episode_id=100, transitions=transitions)

            cursor = db.conn.execute("SELECT * FROM affordance_visits WHERE episode_id = ?", (100,))
            rows = cursor.fetchall()

            assert len(rows) == 1
            assert rows[0]["episode_id"] == 100
            assert rows[0]["from_affordance"] == "Bed"
            assert rows[0]["to_affordance"] == "Bed"
            assert rows[0]["visit_count"] == 5

            db.close()
