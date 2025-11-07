"""Comprehensive unit tests for video_export.py module.

This test suite focuses on testing video export logic with mocked dependencies
to achieve 70%+ coverage without requiring FFmpeg installation.

Test Coverage:
- export_episode_video() success and error paths
- _encode_video_ffmpeg() with subprocess mocking
- batch_export_videos() filtering and iteration
- Grid size auto-detection logic
- Error handling for missing dependencies
"""

import subprocess
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

# =============================================================================
# EXPORT_EPISODE_VIDEO TESTS
# =============================================================================


class TestExportEpisodeVideo:
    """Test export_episode_video() function with mocked dependencies."""

    @patch("townlet.recording.video_export.DemoDatabase")
    @patch("townlet.recording.video_export.ReplayManager")
    @patch("townlet.recording.video_export.EpisodeVideoRenderer")
    @patch("townlet.recording.video_export._encode_video_ffmpeg")
    @patch("PIL.Image.fromarray")
    def test_export_success_with_explicit_grid_size(
        self, mock_fromarray, mock_encode, mock_renderer_class, mock_replay_class, mock_db_class
    ):
        """Should successfully export episode with explicit grid size."""
        from townlet.recording.video_export import export_episode_video

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            db_path = tmpdir_path / "test.db"
            recordings_dir = tmpdir_path / "recordings"
            output_path = tmpdir_path / "output.mp4"
            output_path.touch()  # Create file for stat() call

            # Mock database
            mock_db = Mock()
            mock_db_class.return_value = mock_db

            # Mock replay manager
            mock_replay = Mock()
            mock_replay_class.return_value = mock_replay
            mock_replay.load_episode.return_value = True
            mock_replay.get_metadata.return_value = {"curriculum_stage": 2}
            mock_replay.get_affordances.return_value = {"Bed": [2, 3]}
            mock_replay.get_total_steps.return_value = 5

            # Mock step data
            mock_replay.get_current_step.return_value = {
                "step": 0,
                "position": [3, 4],
                "meters": [0.8, 0.7, 0.6, 0.5, 0.9, 0.8, 0.7, 0.6],
                "action": 2,
                "reward": 1.0,
            }

            # Mock renderer
            mock_renderer = Mock()
            mock_renderer_class.return_value = mock_renderer
            mock_renderer.render_frame.return_value = np.zeros((900, 1600, 3), dtype=np.uint8)

            # Mock Image.fromarray
            mock_img = Mock()
            mock_fromarray.return_value = mock_img

            # Mock ffmpeg encoding
            mock_encode.return_value = True

            # Execute
            result = export_episode_video(
                episode_id=100,
                database_path=db_path,
                recordings_base_dir=recordings_dir,
                output_path=output_path,
                grid_size=8,  # Explicit grid size
                fps=30,
                dpi=100,
            )

            # Verify success
            assert result is True

            # Verify replay loaded
            mock_replay.load_episode.assert_called_once_with(100)

            # Verify renderer initialized with correct grid size
            mock_renderer_class.assert_called_once_with(grid_size=8, dpi=100, style="dark")

            # Verify all frames rendered
            assert mock_renderer.render_frame.call_count == 5
            assert mock_replay.seek.call_count == 5

            # Verify encoding called
            mock_encode.assert_called_once()

    @patch("townlet.recording.video_export.DemoDatabase")
    @patch("townlet.recording.video_export.ReplayManager")
    def test_export_failure_episode_load_failed(self, mock_replay_class, mock_db_class):
        """Should return False when episode load fails."""
        from townlet.recording.video_export import export_episode_video

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Mock replay load failure
            mock_replay = Mock()
            mock_replay_class.return_value = mock_replay
            mock_replay.load_episode.return_value = False

            result = export_episode_video(
                episode_id=999,
                database_path=tmpdir_path / "test.db",
                recordings_base_dir=tmpdir_path / "recordings",
                output_path=tmpdir_path / "output.mp4",
            )

            assert result is False

    @patch("townlet.recording.video_export.DemoDatabase")
    @patch("townlet.recording.video_export.ReplayManager")
    def test_export_failure_metadata_missing(self, mock_replay_class, mock_db_class):
        """Should return False when metadata is None."""
        from townlet.recording.video_export import export_episode_video

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            mock_replay = Mock()
            mock_replay_class.return_value = mock_replay
            mock_replay.load_episode.return_value = True
            mock_replay.get_metadata.return_value = None  # Metadata missing

            result = export_episode_video(
                episode_id=100,
                database_path=tmpdir_path / "test.db",
                recordings_base_dir=tmpdir_path / "recordings",
                output_path=tmpdir_path / "output.mp4",
            )

            assert result is False

    @patch("townlet.recording.video_export.DemoDatabase")
    @patch("townlet.recording.video_export.ReplayManager")
    @patch("townlet.recording.video_export.EpisodeVideoRenderer")
    @patch("PIL.Image.fromarray")
    def test_export_grid_size_auto_detection(self, mock_fromarray, mock_renderer_class, mock_replay_class, mock_db_class):
        """Should auto-detect grid size from affordance positions."""
        from townlet.recording.video_export import export_episode_video

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            output_path = tmpdir_path / "output.mp4"

            mock_replay = Mock()
            mock_replay_class.return_value = mock_replay
            mock_replay.load_episode.return_value = True
            mock_replay.get_metadata.return_value = {"curriculum_stage": 1}
            mock_replay.get_affordances.return_value = {
                "Bed": [2, 3],
                "Job": [7, 6],  # Max coord is 7
                "Hospital": [5, 7],  # Max coord is 7
            }
            mock_replay.get_total_steps.return_value = 0  # No frames to render

            # Don't provide grid_size (should auto-detect)
            with patch("townlet.recording.video_export._encode_video_ffmpeg", return_value=True):
                # Create output file for stat() call
                output_path.touch()

                export_episode_video(
                    episode_id=100,
                    database_path=tmpdir_path / "test.db",
                    recordings_base_dir=tmpdir_path / "recordings",
                    output_path=output_path,
                    grid_size=None,  # Trigger auto-detection
                )

            # Should auto-detect grid_size = 8 (7 + 1)
            mock_renderer_class.assert_called_once()
            call_args = mock_renderer_class.call_args
            assert call_args[1]["grid_size"] == 8

    @patch("townlet.recording.video_export.DemoDatabase")
    @patch("townlet.recording.video_export.ReplayManager")
    @patch("townlet.recording.video_export.EpisodeVideoRenderer")
    @patch("townlet.recording.video_export._encode_video_ffmpeg")
    @patch("PIL.Image.fromarray")
    def test_export_failure_step_data_missing(
        self, mock_fromarray, mock_encode, mock_renderer_class, mock_replay_class, mock_db_class
    ):
        """Should return False when step data retrieval fails."""
        from townlet.recording.video_export import export_episode_video

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            mock_replay = Mock()
            mock_replay_class.return_value = mock_replay
            mock_replay.load_episode.return_value = True
            mock_replay.get_metadata.return_value = {"curriculum_stage": 2}
            mock_replay.get_affordances.return_value = {"Bed": [2, 3]}
            mock_replay.get_total_steps.return_value = 3
            mock_replay.get_current_step.return_value = None  # Step data missing

            result = export_episode_video(
                episode_id=100,
                database_path=tmpdir_path / "test.db",
                recordings_base_dir=tmpdir_path / "recordings",
                output_path=tmpdir_path / "output.mp4",
                grid_size=8,
            )

            assert result is False

    @patch("townlet.recording.video_export.DemoDatabase")
    @patch("townlet.recording.video_export.ReplayManager")
    @patch("townlet.recording.video_export.EpisodeVideoRenderer")
    @patch("townlet.recording.video_export._encode_video_ffmpeg")
    @patch("PIL.Image.fromarray")
    def test_export_failure_encoding_failed(
        self, mock_fromarray, mock_encode, mock_renderer_class, mock_replay_class, mock_db_class
    ):
        """Should return False when ffmpeg encoding fails."""
        from townlet.recording.video_export import export_episode_video

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            mock_replay = Mock()
            mock_replay_class.return_value = mock_replay
            mock_replay.load_episode.return_value = True
            mock_replay.get_metadata.return_value = {"curriculum_stage": 2}
            mock_replay.get_affordances.return_value = {"Bed": [2, 3]}
            mock_replay.get_total_steps.return_value = 2
            mock_replay.get_current_step.return_value = {
                "step": 0,
                "position": [3, 4],
                "meters": [0.5] * 8,
                "action": 0,
                "reward": 1.0,
            }

            mock_renderer = Mock()
            mock_renderer_class.return_value = mock_renderer
            mock_renderer.render_frame.return_value = np.zeros((900, 1600, 3), dtype=np.uint8)

            # Encoding fails
            mock_encode.return_value = False

            result = export_episode_video(
                episode_id=100,
                database_path=tmpdir_path / "test.db",
                recordings_base_dir=tmpdir_path / "recordings",
                output_path=tmpdir_path / "output.mp4",
                grid_size=8,
            )

            assert result is False

    @patch("townlet.recording.video_export.DemoDatabase")
    @patch("townlet.recording.video_export.ReplayManager")
    @patch("townlet.recording.video_export.EpisodeVideoRenderer")
    @patch("townlet.recording.video_export._encode_video_ffmpeg")
    @patch("PIL.Image.fromarray")
    def test_export_custom_style_and_dpi(
        self, mock_fromarray, mock_encode, mock_renderer_class, mock_replay_class, mock_db_class
    ):
        """Should pass custom style and DPI to renderer."""
        from townlet.recording.video_export import export_episode_video

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            output_path = tmpdir_path / "output.mp4"
            output_path.touch()

            mock_replay = Mock()
            mock_replay_class.return_value = mock_replay
            mock_replay.load_episode.return_value = True
            mock_replay.get_metadata.return_value = {"curriculum_stage": 1}
            mock_replay.get_affordances.return_value = {}
            mock_replay.get_total_steps.return_value = 0

            mock_encode.return_value = True

            export_episode_video(
                episode_id=100,
                database_path=tmpdir_path / "test.db",
                recordings_base_dir=tmpdir_path / "recordings",
                output_path=output_path,
                grid_size=8,
                style="light",  # Custom style
                dpi=150,  # Custom DPI
            )

            # Verify renderer called with custom parameters
            mock_renderer_class.assert_called_once_with(grid_size=8, dpi=150, style="light")


# =============================================================================
# _ENCODE_VIDEO_FFMPEG TESTS
# =============================================================================


class TestEncodeVideoFFmpeg:
    """Test _encode_video_ffmpeg() function with subprocess mocking."""

    @patch("townlet.recording.video_export.subprocess.run")
    def test_encode_success(self, mock_run):
        """Should successfully encode video when ffmpeg succeeds."""
        from townlet.recording.video_export import _encode_video_ffmpeg

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            frames_dir = tmpdir_path / "frames"
            frames_dir.mkdir()
            output_path = tmpdir_path / "output.mp4"

            # Mock ffmpeg availability check (success)
            # Mock ffmpeg encoding (success)
            mock_run.side_effect = [
                Mock(returncode=0),  # Version check
                Mock(returncode=0),  # Encoding
            ]

            result = _encode_video_ffmpeg(frames_dir, output_path, fps=30, speed=1.0)

            assert result is True
            assert mock_run.call_count == 2

            # Verify ffmpeg command structure
            encoding_call = mock_run.call_args_list[1]
            cmd = encoding_call[0][0]
            assert "ffmpeg" in cmd
            assert "-framerate" in cmd
            assert "30" in cmd
            assert "-c:v" in cmd
            assert "libx264" in cmd

    @patch("townlet.recording.video_export.subprocess.run")
    def test_encode_ffmpeg_not_found(self, mock_run):
        """Should return False when ffmpeg is not installed."""
        from townlet.recording.video_export import _encode_video_ffmpeg

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Mock ffmpeg not found
            mock_run.side_effect = FileNotFoundError("ffmpeg not found")

            result = _encode_video_ffmpeg(
                tmpdir_path / "frames", tmpdir_path / "output.mp4"
            )

            assert result is False

    @patch("townlet.recording.video_export.subprocess.run")
    def test_encode_ffmpeg_version_check_failed(self, mock_run):
        """Should return False when ffmpeg version check fails."""
        from townlet.recording.video_export import _encode_video_ffmpeg

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Mock ffmpeg version check failure
            mock_run.side_effect = subprocess.CalledProcessError(1, "ffmpeg")

            result = _encode_video_ffmpeg(
                tmpdir_path / "frames", tmpdir_path / "output.mp4"
            )

            assert result is False

    @patch("townlet.recording.video_export.subprocess.run")
    def test_encode_ffmpeg_encoding_failed(self, mock_run):
        """Should return False when ffmpeg encoding fails."""
        from townlet.recording.video_export import _encode_video_ffmpeg

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Mock version check success, encoding failure
            mock_run.side_effect = [
                Mock(returncode=0),  # Version check
                subprocess.CalledProcessError(1, "ffmpeg", stderr="Encoding error"),
            ]

            result = _encode_video_ffmpeg(
                tmpdir_path / "frames", tmpdir_path / "output.mp4"
            )

            assert result is False

    @patch("townlet.recording.video_export.subprocess.run")
    def test_encode_speed_adjustment(self, mock_run):
        """Should adjust FPS based on speed multiplier."""
        from townlet.recording.video_export import _encode_video_ffmpeg

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            mock_run.side_effect = [
                Mock(returncode=0),  # Version check
                Mock(returncode=0),  # Encoding
            ]

            _encode_video_ffmpeg(
                tmpdir_path / "frames",
                tmpdir_path / "output.mp4",
                fps=30,
                speed=2.0,  # 2x speed
            )

            # Verify effective FPS = 30 * 2.0 = 60
            encoding_call = mock_run.call_args_list[1]
            cmd = encoding_call[0][0]
            framerate_idx = cmd.index("-framerate")
            assert cmd[framerate_idx + 1] == "60"

    @patch("townlet.recording.video_export.subprocess.run")
    def test_encode_custom_fps(self, mock_run):
        """Should use custom FPS value."""
        from townlet.recording.video_export import _encode_video_ffmpeg

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            mock_run.side_effect = [
                Mock(returncode=0),
                Mock(returncode=0),
            ]

            _encode_video_ffmpeg(
                tmpdir_path / "frames",
                tmpdir_path / "output.mp4",
                fps=60,  # Custom FPS
                speed=1.0,
            )

            encoding_call = mock_run.call_args_list[1]
            cmd = encoding_call[0][0]
            framerate_idx = cmd.index("-framerate")
            assert cmd[framerate_idx + 1] == "60"


# =============================================================================
# BATCH_EXPORT_VIDEOS TESTS
# =============================================================================


class TestBatchExportVideos:
    """Test batch_export_videos() function."""

    @patch("townlet.recording.video_export.DemoDatabase")
    @patch("townlet.recording.video_export.export_episode_video")
    def test_batch_export_success(self, mock_export, mock_db_class):
        """Should export multiple episodes and return success count."""
        from townlet.recording.video_export import batch_export_videos

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Mock database
            mock_db = Mock()
            mock_db_class.return_value = mock_db
            mock_db.list_recordings.return_value = [
                {"episode_id": 100},
                {"episode_id": 200},
                {"episode_id": 300},
            ]

            # Mock export success
            mock_export.return_value = True

            result = batch_export_videos(
                database_path=tmpdir_path / "test.db",
                recordings_base_dir=tmpdir_path / "recordings",
                output_dir=tmpdir_path / "videos",
            )

            # Should export 3 videos
            assert result == 3
            assert mock_export.call_count == 3

            # Verify output directory created
            assert (tmpdir_path / "videos").exists()

    @patch("townlet.recording.video_export.DemoDatabase")
    @patch("townlet.recording.video_export.export_episode_video")
    def test_batch_export_partial_failures(self, mock_export, mock_db_class):
        """Should count only successful exports."""
        from townlet.recording.video_export import batch_export_videos

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            mock_db = Mock()
            mock_db_class.return_value = mock_db
            mock_db.list_recordings.return_value = [
                {"episode_id": 100},
                {"episode_id": 200},
                {"episode_id": 300},
            ]

            # Mock mixed success/failure
            mock_export.side_effect = [True, False, True]  # 2 successes, 1 failure

            result = batch_export_videos(
                database_path=tmpdir_path / "test.db",
                recordings_base_dir=tmpdir_path / "recordings",
                output_dir=tmpdir_path / "videos",
            )

            # Should return 2 (only successful exports)
            assert result == 2

    @patch("townlet.recording.video_export.DemoDatabase")
    @patch("townlet.recording.video_export.export_episode_video")
    def test_batch_export_with_filters(self, mock_export, mock_db_class):
        """Should pass filters to database query."""
        from townlet.recording.video_export import batch_export_videos

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            mock_db = Mock()
            mock_db_class.return_value = mock_db
            mock_db.list_recordings.return_value = []

            batch_export_videos(
                database_path=tmpdir_path / "test.db",
                recordings_base_dir=tmpdir_path / "recordings",
                output_dir=tmpdir_path / "videos",
                stage=2,
                reason="high_reward",
                min_reward=50.0,
                max_reward=100.0,
                limit=10,
            )

            # Verify filters passed to database
            mock_db.list_recordings.assert_called_once_with(
                stage=2,
                reason="high_reward",
                min_reward=50.0,
                max_reward=100.0,
                limit=10,
            )

    @patch("townlet.recording.video_export.DemoDatabase")
    @patch("townlet.recording.video_export.export_episode_video")
    def test_batch_export_custom_rendering_params(self, mock_export, mock_db_class):
        """Should pass custom rendering parameters to export function."""
        from townlet.recording.video_export import batch_export_videos

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            mock_db = Mock()
            mock_db_class.return_value = mock_db
            mock_db.list_recordings.return_value = [{"episode_id": 100}]
            mock_export.return_value = True

            batch_export_videos(
                database_path=tmpdir_path / "test.db",
                recordings_base_dir=tmpdir_path / "recordings",
                output_dir=tmpdir_path / "videos",
                fps=60,  # Custom FPS
                speed=2.0,  # Custom speed
                dpi=150,  # Custom DPI
                style="light",  # Custom style
            )

            # Verify custom parameters passed to export
            call_args = mock_export.call_args
            assert call_args[1]["fps"] == 60
            assert call_args[1]["speed"] == 2.0
            assert call_args[1]["dpi"] == 150
            assert call_args[1]["style"] == "light"

    @patch("townlet.recording.video_export.DemoDatabase")
    @patch("townlet.recording.video_export.export_episode_video")
    def test_batch_export_output_filename_format(self, mock_export, mock_db_class):
        """Should generate correct output filenames."""
        from townlet.recording.video_export import batch_export_videos

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            mock_db = Mock()
            mock_db_class.return_value = mock_db
            mock_db.list_recordings.return_value = [
                {"episode_id": 42},
                {"episode_id": 1000},
            ]
            mock_export.return_value = True

            batch_export_videos(
                database_path=tmpdir_path / "test.db",
                recordings_base_dir=tmpdir_path / "recordings",
                output_dir=tmpdir_path / "videos",
            )

            # Verify output paths have correct format
            calls = mock_export.call_args_list
            assert calls[0][1]["output_path"] == tmpdir_path / "videos" / "episode_000042.mp4"
            assert calls[1][1]["output_path"] == tmpdir_path / "videos" / "episode_001000.mp4"

    @patch("townlet.recording.video_export.DemoDatabase")
    def test_batch_export_no_recordings_found(self, mock_db_class):
        """Should return 0 when no recordings match filters."""
        from townlet.recording.video_export import batch_export_videos

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            mock_db = Mock()
            mock_db_class.return_value = mock_db
            mock_db.list_recordings.return_value = []  # No recordings

            result = batch_export_videos(
                database_path=tmpdir_path / "test.db",
                recordings_base_dir=tmpdir_path / "recordings",
                output_dir=tmpdir_path / "videos",
            )

            assert result == 0
