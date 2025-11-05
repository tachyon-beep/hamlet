"""
Tests for ReplayManager.

Tests loading and controlling episode replay.
"""

import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import lz4.frame
import msgpack


class TestReplayManager:
    """Test ReplayManager functionality."""

    def test_replay_manager_initialization(self):
        """ReplayManager should initialize with database and directory."""
        from townlet.demo.database import DemoDatabase
        from townlet.recording.replay import ReplayManager

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            db_path = tmpdir_path / "test.db"
            recordings_dir = tmpdir_path / "recordings"
            recordings_dir.mkdir()

            db = DemoDatabase(db_path)
            replay = ReplayManager(db, recordings_dir)

            assert replay.database is db
            assert replay.recordings_base_dir == recordings_dir
            assert replay.is_loaded() is False

    def test_load_episode_from_file(self):
        """ReplayManager should load and decompress episode."""
        from townlet.demo.database import DemoDatabase
        from townlet.recording.data_structures import EpisodeMetadata, RecordedStep
        from townlet.recording.replay import ReplayManager

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            db_path = tmpdir_path / "test.db"
            recordings_dir = tmpdir_path / "recordings"
            recordings_dir.mkdir()

            # Create database and recording
            db = DemoDatabase(db_path)

            # Create sample episode
            metadata = EpisodeMetadata(
                episode_id=100,
                survival_steps=10,
                total_reward=10.0,
                extrinsic_reward=9.5,
                intrinsic_reward=0.5,
                curriculum_stage=1,
                epsilon=0.1,
                intrinsic_weight=0.5,
                timestamp=time.time(),
                affordance_layout={"Bed": (2, 3)},
                affordance_visits={"Bed": 1},
            )

            steps = [
                RecordedStep(
                    step=i,
                    position=(3, 4),
                    meters=(0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1),
                    action=i % 6,
                    reward=1.0,
                    intrinsic_reward=0.05,
                    done=(i == 9),
                    q_values=(0.1, 0.2, 0.3, 0.4, 0.5),
                )
                for i in range(10)
            ]

            # Serialize and write
            episode_data = {
                "version": 1,
                "metadata": asdict(metadata),
                "steps": [asdict(step) for step in steps],
                "affordances": metadata.affordance_layout,
            }
            serialized = msgpack.packb(episode_data, use_bin_type=True)
            compressed = lz4.frame.compress(serialized, compression_level=0)

            file_path = recordings_dir / "episode_000100.msgpack.lz4"
            file_path.write_bytes(compressed)

            # Insert into database
            db.insert_recording(
                episode_id=100,
                file_path=str(file_path.relative_to(tmpdir_path)),
                metadata=metadata,
                reason="periodic_100",
                file_size=len(serialized),
                compressed_size=len(compressed),
            )

            # Load with ReplayManager
            replay = ReplayManager(db, tmpdir_path)
            success = replay.load_episode(100)

            assert success is True
            assert replay.is_loaded() is True
            assert replay.episode_id == 100
            assert replay.get_total_steps() == 10
            assert replay.get_metadata()["episode_id"] == 100
            assert replay.get_affordances() == {"Bed": [2, 3]}  # msgpack converts tuples to lists

    def test_load_nonexistent_episode(self):
        """ReplayManager should return False for nonexistent episode."""
        from townlet.demo.database import DemoDatabase
        from townlet.recording.replay import ReplayManager

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            db_path = tmpdir_path / "test.db"
            recordings_dir = tmpdir_path / "recordings"
            recordings_dir.mkdir()

            db = DemoDatabase(db_path)
            replay = ReplayManager(db, recordings_dir)

            success = replay.load_episode(999)
            assert success is False
            assert replay.is_loaded() is False

    def test_replay_step_progression(self):
        """ReplayManager should advance through steps."""
        from townlet.demo.database import DemoDatabase
        from townlet.recording.data_structures import EpisodeMetadata, RecordedStep
        from townlet.recording.replay import ReplayManager

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            db_path = tmpdir_path / "test.db"
            recordings_dir = tmpdir_path / "recordings"
            recordings_dir.mkdir()

            db = DemoDatabase(db_path)

            # Create minimal episode
            metadata = EpisodeMetadata(
                episode_id=200,
                survival_steps=5,
                total_reward=5.0,
                extrinsic_reward=5.0,
                intrinsic_reward=0.0,
                curriculum_stage=1,
                epsilon=0.1,
                intrinsic_weight=0.5,
                timestamp=time.time(),
                affordance_layout={},
                affordance_visits={},
            )

            steps = [
                RecordedStep(
                    step=i,
                    position=(i, i),
                    meters=(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
                    action=0,
                    reward=1.0,
                    intrinsic_reward=0.0,
                    done=(i == 4),
                    q_values=None,
                )
                for i in range(5)
            ]

            episode_data = {
                "version": 1,
                "metadata": asdict(metadata),
                "steps": [asdict(step) for step in steps],
                "affordances": {},
            }
            serialized = msgpack.packb(episode_data, use_bin_type=True)
            compressed = lz4.frame.compress(serialized)

            file_path = recordings_dir / "episode_000200.msgpack.lz4"
            file_path.write_bytes(compressed)

            db.insert_recording(
                episode_id=200,
                file_path=str(file_path.relative_to(tmpdir_path)),
                metadata=metadata,
                reason="test",
                file_size=len(serialized),
                compressed_size=len(compressed),
            )

            # Load and step through
            replay = ReplayManager(db, tmpdir_path)
            replay.load_episode(200)

            # Initially at step 0
            assert replay.get_current_step_index() == 0
            step = replay.get_current_step()
            assert step["step"] == 0
            assert step["position"] == [0, 0]  # msgpack converts tuples to lists

            # Advance to step 1
            next_step = replay.next_step()
            assert replay.get_current_step_index() == 1
            assert next_step["step"] == 1
            assert next_step["position"] == [1, 1]

            # Advance to step 2
            next_step = replay.next_step()
            assert replay.get_current_step_index() == 2

    def test_replay_at_end(self):
        """ReplayManager should detect end of episode."""
        from townlet.demo.database import DemoDatabase
        from townlet.recording.data_structures import EpisodeMetadata, RecordedStep
        from townlet.recording.replay import ReplayManager

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            db_path = tmpdir_path / "test.db"
            recordings_dir = tmpdir_path / "recordings"
            recordings_dir.mkdir()

            db = DemoDatabase(db_path)

            # Create 3-step episode
            metadata = EpisodeMetadata(
                episode_id=300,
                survival_steps=3,
                total_reward=3.0,
                extrinsic_reward=3.0,
                intrinsic_reward=0.0,
                curriculum_stage=1,
                epsilon=0.1,
                intrinsic_weight=0.5,
                timestamp=time.time(),
                affordance_layout={},
                affordance_visits={},
            )

            steps = [
                RecordedStep(
                    step=i,
                    position=(0, 0),
                    meters=(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
                    action=0,
                    reward=1.0,
                    intrinsic_reward=0.0,
                    done=(i == 2),
                    q_values=None,
                )
                for i in range(3)
            ]

            episode_data = {
                "version": 1,
                "metadata": asdict(metadata),
                "steps": [asdict(step) for step in steps],
                "affordances": {},
            }
            serialized = msgpack.packb(episode_data, use_bin_type=True)
            compressed = lz4.frame.compress(serialized)

            file_path = recordings_dir / "episode_000300.msgpack.lz4"
            file_path.write_bytes(compressed)

            db.insert_recording(
                episode_id=300,
                file_path=str(file_path.relative_to(tmpdir_path)),
                metadata=metadata,
                reason="test",
                file_size=len(serialized),
                compressed_size=len(compressed),
            )

            # Load and advance to end
            replay = ReplayManager(db, tmpdir_path)
            replay.load_episode(300)

            assert replay.is_at_end() is False

            replay.next_step()  # step 1
            replay.next_step()  # step 2
            replay.next_step()  # past end

            assert replay.is_at_end() is True
            assert replay.get_current_step() is None

    def test_replay_seek(self):
        """ReplayManager should support seeking."""
        from townlet.demo.database import DemoDatabase
        from townlet.recording.data_structures import EpisodeMetadata, RecordedStep
        from townlet.recording.replay import ReplayManager

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            db_path = tmpdir_path / "test.db"
            recordings_dir = tmpdir_path / "recordings"
            recordings_dir.mkdir()

            db = DemoDatabase(db_path)

            # Create 10-step episode
            metadata = EpisodeMetadata(
                episode_id=400,
                survival_steps=10,
                total_reward=10.0,
                extrinsic_reward=10.0,
                intrinsic_reward=0.0,
                curriculum_stage=1,
                epsilon=0.1,
                intrinsic_weight=0.5,
                timestamp=time.time(),
                affordance_layout={},
                affordance_visits={},
            )

            steps = [
                RecordedStep(
                    step=i,
                    position=(0, 0),
                    meters=(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
                    action=0,
                    reward=1.0,
                    intrinsic_reward=0.0,
                    done=(i == 9),
                    q_values=None,
                )
                for i in range(10)
            ]

            episode_data = {
                "version": 1,
                "metadata": asdict(metadata),
                "steps": [asdict(step) for step in steps],
                "affordances": {},
            }
            serialized = msgpack.packb(episode_data, use_bin_type=True)
            compressed = lz4.frame.compress(serialized)

            file_path = recordings_dir / "episode_000400.msgpack.lz4"
            file_path.write_bytes(compressed)

            db.insert_recording(
                episode_id=400,
                file_path=str(file_path.relative_to(tmpdir_path)),
                metadata=metadata,
                reason="test",
                file_size=len(serialized),
                compressed_size=len(compressed),
            )

            # Load and seek
            replay = ReplayManager(db, tmpdir_path)
            replay.load_episode(400)

            # Seek to step 5
            success = replay.seek(5)
            assert success is True
            assert replay.get_current_step_index() == 5
            assert replay.get_current_step()["step"] == 5

            # Seek to beginning
            success = replay.seek(0)
            assert success is True
            assert replay.get_current_step_index() == 0

            # Seek out of bounds
            success = replay.seek(100)
            assert success is False
            assert replay.get_current_step_index() == 0  # Unchanged

    def test_replay_reset(self):
        """ReplayManager should reset to beginning."""
        from townlet.demo.database import DemoDatabase
        from townlet.recording.data_structures import EpisodeMetadata, RecordedStep
        from townlet.recording.replay import ReplayManager

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            db_path = tmpdir_path / "test.db"
            recordings_dir = tmpdir_path / "recordings"
            recordings_dir.mkdir()

            db = DemoDatabase(db_path)

            # Create episode
            metadata = EpisodeMetadata(
                episode_id=500,
                survival_steps=5,
                total_reward=5.0,
                extrinsic_reward=5.0,
                intrinsic_reward=0.0,
                curriculum_stage=1,
                epsilon=0.1,
                intrinsic_weight=0.5,
                timestamp=time.time(),
                affordance_layout={},
                affordance_visits={},
            )

            steps = [
                RecordedStep(
                    step=i,
                    position=(0, 0),
                    meters=(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
                    action=0,
                    reward=1.0,
                    intrinsic_reward=0.0,
                    done=(i == 4),
                    q_values=None,
                )
                for i in range(5)
            ]

            episode_data = {
                "version": 1,
                "metadata": asdict(metadata),
                "steps": [asdict(step) for step in steps],
                "affordances": {},
            }
            serialized = msgpack.packb(episode_data, use_bin_type=True)
            compressed = lz4.frame.compress(serialized)

            file_path = recordings_dir / "episode_000500.msgpack.lz4"
            file_path.write_bytes(compressed)

            db.insert_recording(
                episode_id=500,
                file_path=str(file_path.relative_to(tmpdir_path)),
                metadata=metadata,
                reason="test",
                file_size=len(serialized),
                compressed_size=len(compressed),
            )

            # Load, advance, then reset
            replay = ReplayManager(db, tmpdir_path)
            replay.load_episode(500)

            replay.next_step()
            replay.next_step()
            assert replay.get_current_step_index() == 2

            replay.reset()
            assert replay.get_current_step_index() == 0
            assert replay.playing is False

    def test_replay_unload(self):
        """ReplayManager should unload episode."""
        from townlet.demo.database import DemoDatabase
        from townlet.recording.data_structures import EpisodeMetadata, RecordedStep
        from townlet.recording.replay import ReplayManager

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            db_path = tmpdir_path / "test.db"
            recordings_dir = tmpdir_path / "recordings"
            recordings_dir.mkdir()

            db = DemoDatabase(db_path)

            # Create episode
            metadata = EpisodeMetadata(
                episode_id=600,
                survival_steps=1,
                total_reward=1.0,
                extrinsic_reward=1.0,
                intrinsic_reward=0.0,
                curriculum_stage=1,
                epsilon=0.1,
                intrinsic_weight=0.5,
                timestamp=time.time(),
                affordance_layout={},
                affordance_visits={},
            )

            steps = [
                RecordedStep(
                    step=0,
                    position=(0, 0),
                    meters=(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
                    action=0,
                    reward=1.0,
                    intrinsic_reward=0.0,
                    done=True,
                    q_values=None,
                )
            ]

            episode_data = {
                "version": 1,
                "metadata": asdict(metadata),
                "steps": [asdict(step) for step in steps],
                "affordances": {},
            }
            serialized = msgpack.packb(episode_data, use_bin_type=True)
            compressed = lz4.frame.compress(serialized)

            file_path = recordings_dir / "episode_000600.msgpack.lz4"
            file_path.write_bytes(compressed)

            db.insert_recording(
                episode_id=600,
                file_path=str(file_path.relative_to(tmpdir_path)),
                metadata=metadata,
                reason="test",
                file_size=len(serialized),
                compressed_size=len(compressed),
            )

            # Load and unload
            replay = ReplayManager(db, tmpdir_path)
            replay.load_episode(600)
            assert replay.is_loaded() is True

            replay.unload()
            assert replay.is_loaded() is False
            assert replay.episode_id is None
            assert replay.get_total_steps() == 0
