"""Comprehensive unit tests for recorder.py module.

These unit tests complement the integration tests by providing:
- Full mocking for deterministic test execution
- Coverage of edge cases and error paths
- Testing without thread complexity
- Focus on achieving 75%+ coverage

Current integration test coverage: 57%
Target with unit tests: 75%+

Uncovered lines to target:
- Lines 107, 113-116: action_masks tuple conversion (non-tensor path)
- Lines 154-155: Queue full error in finish_episode
- Lines 159-160: shutdown method
- Lines 215-220: EpisodeEndMarker processing
- Lines 229-240: _process_episode_end logging
- Lines 254-263: _should_record_episode criteria
- Lines 272-296: _write_episode implementation
"""

import queue
from unittest.mock import Mock, patch

import torch

from tests.test_townlet.utils.builders import make_test_episode_metadata, make_test_recorded_step
from townlet.recording.data_structures import EpisodeEndMarker, EpisodeMetadata, RecordedStep


@patch("townlet.recording.recorder.RecordingWriter")
@patch("townlet.recording.recorder.threading.Thread")
def test_record_step_uses_recording_output_dir_fixture(mock_thread, mock_writer_class, recording_output_dir):
    """EpisodeRecorder should respect the shared recording_output_dir fixture."""

    from townlet.recording.recorder import EpisodeRecorder

    recorder = EpisodeRecorder(
        config={"max_queue_size": 10},
        output_dir=recording_output_dir,
        database=None,
        curriculum=None,
    )

    positions = torch.tensor([0, 1])
    meters = torch.ones(8)

    recorder.record_step(
        step=0,
        positions=positions,
        meters=meters,
        action=1,
        reward=0.5,
        intrinsic_reward=0.0,
        done=False,
    )

    assert recording_output_dir.exists()
    assert recorder.queue.qsize() == 1


# =============================================================================
# EPISODE RECORDER TESTS
# =============================================================================


class TestEpisodeRecorderEdgeCases:
    """Test EpisodeRecorder edge cases and error paths."""

    @patch("townlet.recording.recorder.RecordingWriter")
    @patch("townlet.recording.recorder.threading.Thread")
    def test_record_step_with_list_q_values(self, mock_thread, mock_writer_class, recording_output_dir):
        """Should handle q_values as list (line 107 else branch)."""
        from townlet.recording.recorder import EpisodeRecorder

        config = {"max_queue_size": 100}

        recorder = EpisodeRecorder(
            config=config,
            output_dir=recording_output_dir,
            database=None,
            curriculum=None,
        )

        positions = torch.tensor([3, 5])
        meters = torch.tensor([1.0, 0.9, 0.8, 0.5, 0.7, 0.6, 0.95, 0.85])

        # Provide q_values as plain list (not tensor)
        q_values_list = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

        recorder.record_step(
            step=0,
            positions=positions,
            meters=meters,
            action=2,
            reward=1.0,
            intrinsic_reward=0.1,
            done=False,
            q_values=q_values_list,  # List, not tensor
        )

        # Should convert list to tuple without error
        item = recorder.queue.get_nowait()
        assert item.q_values == tuple(q_values_list)

    @patch("townlet.recording.recorder.RecordingWriter")
    @patch("townlet.recording.recorder.threading.Thread")
    def test_record_step_with_list_action_masks(self, mock_thread, mock_writer_class, recording_output_dir):
        """Should handle action_masks as list (lines 113-116 else branch)."""
        from townlet.recording.recorder import EpisodeRecorder

        config = {"max_queue_size": 100}

        recorder = EpisodeRecorder(
            config=config,
            output_dir=recording_output_dir,
            database=None,
            curriculum=None,
        )

        positions = torch.tensor([3, 5])
        meters = torch.tensor([1.0, 0.9, 0.8, 0.5, 0.7, 0.6, 0.95, 0.85])

        # Provide action_masks as plain list (not tensor)
        action_masks_list = [True, True, False, True, True, False]

        recorder.record_step(
            step=0,
            positions=positions,
            meters=meters,
            action=2,
            reward=1.0,
            intrinsic_reward=0.1,
            done=False,
            action_masks=action_masks_list,  # List, not tensor
        )

        # Should convert list to tuple without error
        item = recorder.queue.get_nowait()
        assert item.action_masks == tuple(action_masks_list)

    @patch("townlet.recording.recorder.RecordingWriter")
    @patch("townlet.recording.recorder.threading.Thread")
    def test_finish_episode_queue_full_error(self, mock_thread, mock_writer_class, recording_output_dir):
        """Should log error when queue is full on finish_episode (lines 154-155)."""
        from townlet.recording.recorder import EpisodeRecorder

        config = {"max_queue_size": 1}  # Very small queue

        recorder = EpisodeRecorder(
            config=config,
            output_dir=recording_output_dir,
            database=None,
            curriculum=None,
        )

        # Fill the queue first
        positions = torch.tensor([3, 5])
        meters = torch.tensor([1.0, 0.9, 0.8, 0.5, 0.7, 0.6, 0.95, 0.85])
        recorder.record_step(
            step=0,
            positions=positions,
            meters=meters,
            action=0,
            reward=1.0,
            intrinsic_reward=0.0,
            done=False,
        )

        # Queue is now full
        assert recorder.queue.full()

        # Try to finish episode - should log error but not crash
        metadata = make_test_episode_metadata()

        with patch("townlet.recording.recorder.logger") as mock_logger:
            recorder.finish_episode(metadata)
            # Should log error
            mock_logger.error.assert_called_once()

    @patch("townlet.recording.recorder.RecordingWriter")
    def test_shutdown_method(self, mock_writer_class, recording_output_dir):
        """Should call stop() and join thread (lines 159-160)."""
        from townlet.recording.recorder import EpisodeRecorder

        config = {"max_queue_size": 100}

        mock_writer = Mock()
        mock_writer_class.return_value = mock_writer

        mock_thread = Mock()

        with patch("townlet.recording.recorder.threading.Thread", return_value=mock_thread):
            recorder = EpisodeRecorder(
                config=config,
                output_dir=recording_output_dir,
                database=None,
                curriculum=None,
            )

            recorder.shutdown()

            # Should call writer.stop()
            mock_writer.stop.assert_called_once()

            # Should join thread with timeout
            mock_thread.join.assert_called_once_with(timeout=10.0)


# =============================================================================
# RECORDING WRITER TESTS
# =============================================================================


class TestRecordingWriterProcessing:
    """Test RecordingWriter episode processing logic."""

    def test_writer_loop_processes_recorded_step(self, recording_output_dir):
        from townlet.recording.recorder import RecordingWriter

        test_queue = queue.Queue()
        writer = RecordingWriter(
            queue=test_queue,
            config={"criteria": {}},
            output_dir=recording_output_dir,
            database=None,
            curriculum=None,
        )

        step = make_test_recorded_step()
        test_queue.put(step)
        writer.running = False
        item = writer.queue.get(timeout=0.1)
        if isinstance(item, RecordedStep):
            writer.episode_buffer.append(item)

        assert len(writer.episode_buffer) == 1
        assert writer.episode_buffer[0] == step

    def test_writer_loop_processes_episode_end_marker(self, recording_output_dir):
        from townlet.recording.recorder import RecordingWriter

        writer = RecordingWriter(
            queue=queue.Queue(),
            config={"criteria": {}},
            output_dir=recording_output_dir,
            database=None,
            curriculum=None,
        )

        writer.episode_buffer.append(make_test_recorded_step())
        metadata = make_test_episode_metadata()
        marker = EpisodeEndMarker(metadata=metadata)

        with patch.object(writer, "_process_episode_end") as mock_process:
            if isinstance(marker, EpisodeEndMarker):
                writer._process_episode_end(marker.metadata)
                writer.episode_buffer.clear()

            mock_process.assert_called_once_with(metadata)
            assert len(writer.episode_buffer) == 0

    def test_process_episode_end_should_record_true(self, recording_output_dir):
        from townlet.recording.recorder import RecordingWriter

        writer = RecordingWriter(
            queue=queue.Queue(),
            config={"criteria": {"periodic": {"enabled": True, "interval": 100}}},
            output_dir=recording_output_dir,
            database=None,
            curriculum=None,
        )

        writer.episode_buffer.append(make_test_recorded_step())
        metadata = EpisodeMetadata(
            episode_id=100,
            survival_steps=10,
            total_reward=10.0,
            extrinsic_reward=10.0,
            intrinsic_reward=0.0,
            curriculum_stage=1,
            epsilon=0.5,
            intrinsic_weight=0.0,
            timestamp=1234567890.0,
            affordance_layout={"Bed": (2, 3)},
            affordance_visits={"Bed": 1},
            custom_action_uses={},
        )

        with patch.object(writer, "_write_episode") as mock_write, patch("townlet.recording.recorder.logger") as mock_logger:
            writer._process_episode_end(metadata)
            mock_write.assert_called_once_with(metadata)
            mock_logger.info.assert_called_once()

    def test_process_episode_end_should_record_false(self, recording_output_dir):
        from townlet.recording.recorder import RecordingWriter

        writer = RecordingWriter(
            queue=queue.Queue(),
            config={"criteria": {"periodic": {"enabled": True, "interval": 100}}},
            output_dir=recording_output_dir,
            database=None,
            curriculum=None,
        )

        metadata = EpisodeMetadata(
            episode_id=42,
            survival_steps=10,
            total_reward=10.0,
            extrinsic_reward=10.0,
            intrinsic_reward=0.0,
            curriculum_stage=1,
            epsilon=0.5,
            intrinsic_weight=0.0,
            timestamp=1234567890.0,
            affordance_layout={"Bed": (2, 3)},
            affordance_visits={"Bed": 1},
            custom_action_uses={},
        )

        with patch.object(writer, "_write_episode") as mock_write, patch("townlet.recording.recorder.logger") as mock_logger:
            writer._process_episode_end(metadata)
            mock_write.assert_not_called()
            mock_logger.debug.assert_called_once()

    def test_should_record_episode_periodic_enabled(self, recording_output_dir):
        from townlet.recording.recorder import RecordingWriter

        writer = RecordingWriter(
            queue=queue.Queue(),
            config={"criteria": {"periodic": {"enabled": True, "interval": 50}}},
            output_dir=recording_output_dir,
            database=None,
            curriculum=None,
        )

        metadata = make_test_episode_metadata(episode_id=100)
        assert writer._should_record_episode(metadata) is True

    def test_should_record_episode_periodic_disabled(self, recording_output_dir):
        from townlet.recording.recorder import RecordingWriter

        writer = RecordingWriter(
            queue=queue.Queue(),
            config={"criteria": {"periodic": {"enabled": False, "interval": 50}}},
            output_dir=recording_output_dir,
            database=None,
            curriculum=None,
        )

        metadata = make_test_episode_metadata()
        assert writer._should_record_episode(metadata) is False

    def test_should_record_episode_no_criteria(self, recording_output_dir):
        from townlet.recording.recorder import RecordingWriter

        writer = RecordingWriter(
            queue=queue.Queue(),
            config={"criteria": {}},
            output_dir=recording_output_dir,
            database=None,
            curriculum=None,
        )

        metadata = make_test_episode_metadata()
        assert writer._should_record_episode(metadata) is False

    def test_write_episode_with_lz4_compression(self, recording_output_dir):
        from townlet.recording.recorder import RecordingWriter

        writer = RecordingWriter(
            queue=queue.Queue(),
            config={"compression": "lz4"},
            output_dir=recording_output_dir,
            database=None,
            curriculum=None,
        )

        writer.episode_buffer.append(make_test_recorded_step())
        writer._write_episode(make_test_episode_metadata(episode_id=42))

        expected_path = recording_output_dir / "episode_000042.msgpack.lz4"
        assert expected_path.exists()
        assert expected_path.stat().st_size > 0

    def test_write_episode_no_compression(self, recording_output_dir):
        from townlet.recording.recorder import RecordingWriter

        writer = RecordingWriter(
            queue=queue.Queue(),
            config={"compression": "none"},
            output_dir=recording_output_dir,
            database=None,
            curriculum=None,
        )

        writer.episode_buffer.append(make_test_recorded_step())
        writer._write_episode(make_test_episode_metadata(episode_id=99))

        expected_path = recording_output_dir / "episode_000099.msgpack.lz4"
        assert expected_path.exists()

    def test_write_episode_with_database_insertion(self, recording_output_dir):
        from townlet.recording.recorder import RecordingWriter

        mock_database = Mock()
        extra_dir = recording_output_dir / "nested"
        extra_dir.mkdir(exist_ok=True)

        writer = RecordingWriter(
            queue=queue.Queue(),
            config={"compression": "lz4"},
            output_dir=extra_dir,
            database=mock_database,
            curriculum=None,
        )

        writer.episode_buffer.append(make_test_recorded_step())
        writer._write_episode(make_test_episode_metadata(episode_id=55))

        mock_database.insert_recording.assert_called_once()
        call_args = mock_database.insert_recording.call_args
        assert call_args[1]["episode_id"] == 55
        assert call_args[1]["reason"] == "periodic"

    def test_stop_method(self, recording_output_dir):
        from townlet.recording.recorder import RecordingWriter

        writer = RecordingWriter(
            queue=queue.Queue(),
            config={},
            output_dir=recording_output_dir,
            database=None,
            curriculum=None,
        )

        assert writer.running is True
        writer.stop()
        assert writer.running is False
