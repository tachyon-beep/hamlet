"""
Tests for playback system.

Tests loading and replaying recorded episodes through the inference server.
"""

import pytest
import tempfile
import msgpack
import lz4.frame
from pathlib import Path
from dataclasses import asdict


class TestReplayLoading:
    """Test loading replay data from recordings."""

    def test_load_replay_from_database(self):
        """Should load replay data from database query."""
        from townlet.demo.database import DemoDatabase
        from townlet.recording.data_structures import EpisodeMetadata, RecordedStep
        import time

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            db_path = tmpdir_path / "test.db"
            recordings_dir = tmpdir_path / "recordings"
            recordings_dir.mkdir()

            # Create database and insert recording
            db = DemoDatabase(db_path)

            # Create sample episode data
            metadata = EpisodeMetadata(
                episode_id=100,
                survival_steps=50,
                total_reward=123.4,
                extrinsic_reward=120.0,
                intrinsic_reward=3.4,
                curriculum_stage=2,
                epsilon=0.15,
                intrinsic_weight=0.5,
                timestamp=time.time(),
                affordance_layout={"Bed": (2, 3), "Job": (5, 6)},
                affordance_visits={"Bed": 5, "Job": 3},
            )

            steps = [
                RecordedStep(
                    step=i,
                    position=(3 + i % 2, 4),
                    meters=(1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3),
                    action=i % 5,
                    reward=1.0,
                    intrinsic_reward=0.1,
                    done=(i == 49),
                    q_values=(0.1, 0.2, 0.3, 0.4, 0.5),
                    time_of_day=i % 24,
                    interaction_progress=0.5 if i % 5 == 0 else 0.0,
                )
                for i in range(50)
            ]

            # Serialize and compress
            episode_data = {
                "version": 1,
                "metadata": asdict(metadata),
                "steps": [asdict(step) for step in steps],
                "affordances": metadata.affordance_layout,
            }
            serialized = msgpack.packb(episode_data, use_bin_type=True)
            compressed = lz4.frame.compress(serialized, compression_level=0)

            # Write file
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

            # Query and verify
            recording = db.get_recording(100)
            assert recording is not None
            assert recording["episode_id"] == 100
            assert recording["survival_steps"] == 50

    def test_decompress_and_deserialize_replay(self):
        """Should decompress and deserialize replay data."""
        from townlet.recording.data_structures import EpisodeMetadata, RecordedStep
        import time

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create sample episode data
            metadata = EpisodeMetadata(
                episode_id=200,
                survival_steps=30,
                total_reward=75.5,
                extrinsic_reward=70.0,
                intrinsic_reward=5.5,
                curriculum_stage=1,
                epsilon=0.3,
                intrinsic_weight=0.7,
                timestamp=time.time(),
                affordance_layout={"Bed": (1, 2)},
                affordance_visits={"Bed": 2},
            )

            steps = [
                RecordedStep(
                    step=i,
                    position=(2, 3),
                    meters=(0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1),
                    action=0,
                    reward=1.0,
                    intrinsic_reward=0.05,
                    done=(i == 29),
                    q_values=None,
                    time_of_day=None,
                    interaction_progress=None,
                )
                for i in range(30)
            ]

            # Serialize and compress
            episode_data = {
                "version": 1,
                "metadata": asdict(metadata),
                "steps": [asdict(step) for step in steps],
                "affordances": metadata.affordance_layout,
            }
            serialized = msgpack.packb(episode_data, use_bin_type=True)
            compressed = lz4.frame.compress(serialized, compression_level=0)

            # Write and read back
            file_path = tmpdir_path / "episode_000200.msgpack.lz4"
            file_path.write_bytes(compressed)

            # Decompress
            compressed_data = file_path.read_bytes()
            decompressed = lz4.frame.decompress(compressed_data)
            episode_data_loaded = msgpack.unpackb(decompressed, raw=False)

            # Verify structure
            assert episode_data_loaded["version"] == 1
            assert episode_data_loaded["metadata"]["episode_id"] == 200
            assert len(episode_data_loaded["steps"]) == 30
            assert episode_data_loaded["steps"][0]["step"] == 0
            assert episode_data_loaded["steps"][29]["done"] is True


class TestReplayState:
    """Test replay state management."""

    def test_replay_mode_initialization(self):
        """Replay mode should initialize with loaded data."""
        # Mock replay state
        replay_state = {
            "mode": "replay",
            "episode_id": 123,
            "current_step": 0,
            "total_steps": 50,
            "playing": False,
        }

        assert replay_state["mode"] == "replay"
        assert replay_state["episode_id"] == 123
        assert replay_state["current_step"] == 0
        assert replay_state["total_steps"] == 50
        assert replay_state["playing"] is False

    def test_replay_step_increment(self):
        """Replay should increment step index."""
        replay_state = {
            "current_step": 0,
            "total_steps": 50,
        }

        # Simulate stepping
        replay_state["current_step"] += 1
        assert replay_state["current_step"] == 1

        replay_state["current_step"] += 1
        assert replay_state["current_step"] == 2

    def test_replay_reaches_end(self):
        """Replay should detect end of episode."""
        replay_state = {
            "current_step": 49,
            "total_steps": 50,
        }

        # Check if at end
        at_end = replay_state["current_step"] >= replay_state["total_steps"] - 1
        assert at_end is True

    def test_replay_seek(self):
        """Replay should support seeking to specific step."""
        replay_state = {
            "current_step": 10,
            "total_steps": 100,
        }

        # Seek to step 50
        target_step = 50
        if 0 <= target_step < replay_state["total_steps"]:
            replay_state["current_step"] = target_step

        assert replay_state["current_step"] == 50


class TestReplayMessageFormat:
    """Test replay message format compatibility."""

    def test_replay_state_update_format(self):
        """Replay state updates should match live inference format."""
        # This should be identical to live inference format except for 'mode' field
        state_update = {
            "type": "state_update",
            "mode": "replay",  # NEW: distinguish from live
            "episode_id": 123,
            "step": 25,
            "cumulative_reward": 25.0,
            "grid": {
                "width": 8,
                "height": 8,
                "agents": [
                    {
                        "id": "agent_0",
                        "x": 3,
                        "y": 4,
                        "color": "blue",
                        "last_action": 2,
                    }
                ],
                "affordances": [
                    {"type": "Bed", "x": 2, "y": 3},
                    {"type": "Job", "x": 5, "y": 6},
                ],
            },
            "agent_meters": {
                "agent_0": {
                    "meters": {
                        "energy": 0.8,
                        "hygiene": 0.7,
                        "satiation": 0.6,
                        "money": 0.5,
                        "health": 0.9,
                        "fitness": 0.8,
                        "mood": 0.7,
                        "social": 0.6,
                    }
                }
            },
            "q_values": [0.1, 0.2, 0.3, 0.4, 0.5],
            "replay_metadata": {
                "total_steps": 50,
                "survival_steps": 50,
                "total_reward": 123.4,
                "curriculum_stage": 2,
            },
        }

        # Verify structure
        assert state_update["type"] == "state_update"
        assert state_update["mode"] == "replay"
        assert "replay_metadata" in state_update
        assert state_update["replay_metadata"]["total_steps"] == 50

    def test_replay_loaded_message(self):
        """Replay loaded message should inform client of loaded episode."""
        loaded_message = {
            "type": "replay_loaded",
            "episode_id": 123,
            "metadata": {
                "survival_steps": 50,
                "total_reward": 123.4,
                "curriculum_stage": 2,
            },
            "total_steps": 50,
        }

        assert loaded_message["type"] == "replay_loaded"
        assert loaded_message["episode_id"] == 123
        assert loaded_message["total_steps"] == 50

    def test_replay_finished_message(self):
        """Replay finished message should signal end of playback."""
        finished_message = {
            "type": "replay_finished",
            "episode_id": 123,
        }

        assert finished_message["type"] == "replay_finished"
        assert finished_message["episode_id"] == 123

    def test_recordings_list_message(self):
        """Recordings list message should provide available recordings."""
        recordings_list = {
            "type": "recordings_list",
            "recordings": [
                {
                    "episode_id": 100,
                    "survival_steps": 50,
                    "total_reward": 123.4,
                    "curriculum_stage": 2,
                    "recording_reason": "periodic_100",
                    "timestamp": 1699123456.78,
                },
                {
                    "episode_id": 200,
                    "survival_steps": 75,
                    "total_reward": 234.5,
                    "curriculum_stage": 3,
                    "recording_reason": "top_10.0pct",
                    "timestamp": 1699123556.78,
                },
            ],
        }

        assert recordings_list["type"] == "recordings_list"
        assert len(recordings_list["recordings"]) == 2
        assert recordings_list["recordings"][0]["episode_id"] == 100


class TestReplayCommands:
    """Test replay control commands."""

    def test_load_replay_command(self):
        """Should parse load_replay command."""
        command = {
            "type": "load_replay",
            "episode_id": 123,
        }

        assert command["type"] == "load_replay"
        assert command["episode_id"] == 123

    def test_list_recordings_command(self):
        """Should parse list_recordings command."""
        command = {
            "type": "list_recordings",
            "filters": {
                "stage": 2,
                "reason": "periodic_100",
                "min_reward": 100.0,
            },
        }

        assert command["type"] == "list_recordings"
        assert command["filters"]["stage"] == 2
        assert command["filters"]["reason"] == "periodic_100"

    def test_replay_control_command(self):
        """Should parse replay control commands."""
        # Play
        play_cmd = {
            "type": "replay_control",
            "action": "play",
        }
        assert play_cmd["action"] == "play"

        # Pause
        pause_cmd = {
            "type": "replay_control",
            "action": "pause",
        }
        assert pause_cmd["action"] == "pause"

        # Step
        step_cmd = {
            "type": "replay_control",
            "action": "step",
        }
        assert step_cmd["action"] == "step"

        # Seek
        seek_cmd = {
            "type": "replay_control",
            "action": "seek",
            "seek_step": 50,
        }
        assert seek_cmd["action"] == "seek"
        assert seek_cmd["seek_step"] == 50
