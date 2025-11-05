"""
Tests for video export functionality.

Tests rendering frames and exporting to MP4.
"""

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest


class TestVideoRenderer:
    """Test video frame rendering."""

    def test_renderer_initialization(self):
        """Renderer should initialize with grid size and style."""
        from townlet.recording.video_renderer import EpisodeVideoRenderer

        renderer = EpisodeVideoRenderer(grid_size=8, dpi=100, style="dark")

        assert renderer.grid_size == 8
        assert renderer.dpi == 100
        assert renderer.style == "dark"

    def test_render_simple_frame(self):
        """Renderer should produce numpy array frame."""
        from townlet.recording.video_renderer import EpisodeVideoRenderer

        renderer = EpisodeVideoRenderer(grid_size=8, dpi=100, style="dark")

        # Simple step data
        step_data = {
            "step": 0,
            "position": [3, 4],
            "meters": [0.8, 0.7, 0.6, 0.5, 0.9, 0.8, 0.7, 0.6],
            "action": 2,
            "reward": 1.0,
            "intrinsic_reward": 0.1,
            "done": False,
            "q_values": None,
        }

        metadata = {
            "episode_id": 100,
            "survival_steps": 50,
            "total_reward": 50.0,
            "curriculum_stage": 2,
        }

        affordances = {
            "Bed": [2, 3],
            "Job": [5, 6],
        }

        frame = renderer.render_frame(step_data, metadata, affordances)

        # Should return numpy array (RGB image)
        assert isinstance(frame, np.ndarray)
        assert frame.ndim == 3
        assert frame.shape[2] == 3  # RGB channels
        assert frame.dtype == np.uint8

    def test_render_frame_with_q_values(self):
        """Renderer should display Q-values if provided."""
        from townlet.recording.video_renderer import EpisodeVideoRenderer

        renderer = EpisodeVideoRenderer(grid_size=8, dpi=100, style="dark")

        step_data = {
            "step": 10,
            "position": [4, 4],
            "meters": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            "action": 0,
            "reward": 1.0,
            "intrinsic_reward": 0.05,
            "done": False,
            "q_values": [0.1, 0.2, 0.3, 0.4, 0.5],
        }

        metadata = {
            "episode_id": 200,
            "survival_steps": 30,
            "total_reward": 30.0,
            "curriculum_stage": 1,
        }

        affordances = {}

        frame = renderer.render_frame(step_data, metadata, affordances)

        assert isinstance(frame, np.ndarray)
        assert frame.shape[2] == 3

    def test_render_frame_with_temporal_mechanics(self):
        """Renderer should display temporal mechanics info."""
        from townlet.recording.video_renderer import EpisodeVideoRenderer

        renderer = EpisodeVideoRenderer(grid_size=8, dpi=100, style="dark")

        step_data = {
            "step": 20,
            "position": [3, 3],
            "meters": [0.7, 0.6, 0.5, 0.4, 0.8, 0.7, 0.6, 0.5],
            "action": 4,
            "reward": 1.0,
            "intrinsic_reward": 0.0,
            "done": False,
            "q_values": None,
            "time_of_day": 12,  # Noon
            "interaction_progress": 0.5,
        }

        metadata = {
            "episode_id": 300,
            "survival_steps": 40,
            "total_reward": 40.0,
            "curriculum_stage": 3,
        }

        affordances = {"CoffeeShop": [1, 1]}

        frame = renderer.render_frame(step_data, metadata, affordances)

        assert isinstance(frame, np.ndarray)

    def test_renderer_consistent_dimensions(self):
        """Renderer should produce consistent frame dimensions."""
        from townlet.recording.video_renderer import EpisodeVideoRenderer

        renderer = EpisodeVideoRenderer(grid_size=8, dpi=100, style="dark")

        # Render multiple frames
        frames = []
        for i in range(5):
            step_data = {
                "step": i,
                "position": [i % 8, i % 8],
                "meters": [0.5] * 8,
                "action": i % 6,
                "reward": 1.0,
                "intrinsic_reward": 0.0,
                "done": False,
                "q_values": None,
            }

            metadata = {
                "episode_id": 400,
                "survival_steps": 10,
                "total_reward": 10.0,
                "curriculum_stage": 1,
            }

            frame = renderer.render_frame(step_data, metadata, {})
            frames.append(frame)

        # All frames should have same dimensions
        first_shape = frames[0].shape
        for frame in frames[1:]:
            assert frame.shape == first_shape


def _ffmpeg_available() -> bool:
    """Check if ffmpeg is available."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


class TestVideoExport:
    """Test video export to MP4."""

    @pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg not installed")
    def test_export_episode_to_mp4(self):
        """Should export episode to MP4 file."""
        import time
        from dataclasses import asdict

        import lz4.frame
        import msgpack

        from townlet.demo.database import DemoDatabase
        from townlet.recording.data_structures import EpisodeMetadata, RecordedStep
        from townlet.recording.video_export import export_episode_video

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            db_path = tmpdir_path / "test.db"
            recordings_dir = tmpdir_path / "recordings"
            recordings_dir.mkdir()

            # Create database and recording
            db = DemoDatabase(db_path)

            metadata = EpisodeMetadata(
                episode_id=500,
                survival_steps=10,
                total_reward=10.0,
                extrinsic_reward=10.0,
                intrinsic_reward=0.0,
                curriculum_stage=1,
                epsilon=0.1,
                intrinsic_weight=0.5,
                timestamp=time.time(),
                affordance_layout={"Bed": (2, 3)},
                affordance_visits={"Bed": 2},
            )

            steps = [
                RecordedStep(
                    step=i,
                    position=(3 + i % 2, 4),
                    meters=(0.8, 0.7, 0.6, 0.5, 0.9, 0.8, 0.7, 0.6),
                    action=i % 6,
                    reward=1.0,
                    intrinsic_reward=0.0,
                    done=(i == 9),
                    q_values=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),  # 6 actions: UP, DOWN, LEFT, RIGHT, INTERACT, WAIT
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

            # Close database to prevent resource warnings
            db.close()

            # Export video
            output_path = tmpdir_path / "episode_500.mp4"
            export_episode_video(
                episode_id=500,
                database_path=db_path,
                recordings_base_dir=tmpdir_path,
                output_path=output_path,
                fps=10,
                dpi=80,
            )

            # Verify file exists
            assert output_path.exists()
            assert output_path.stat().st_size > 0


class TestBatchExport:
    """Test batch video export."""

    def test_batch_export_filters(self):
        """Should export multiple episodes with filters."""
        # This is more of an integration test
        # Just verify the API exists
        from townlet.recording import video_export

        assert hasattr(video_export, "batch_export_videos")
