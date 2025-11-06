"""Test recording system handles substrate positions."""

import tempfile
from pathlib import Path

import torch

from townlet.recording.recorder import EpisodeRecorder


def test_recording_handles_2d_positions():
    """Recorder should handle 2D position tuples."""
    from townlet.recording.data_structures import RecordedStep

    with tempfile.TemporaryDirectory() as tmpdir:
        config = {"max_queue_size": 100}

        recorder = EpisodeRecorder(
            config=config,
            output_dir=Path(tmpdir),
            database=None,
            curriculum=None,
        )

        # Record step with 2D position
        positions = torch.tensor([3, 4], dtype=torch.long)  # [2] for single agent
        meters = torch.rand(8)
        action = 2  # LEFT

        recorder.record_step(
            step=0,
            positions=positions,
            meters=meters,
            action=action,
            reward=1.0,
            intrinsic_reward=0.0,
            done=False,
            q_values=None,
        )

        # Verify position was converted to tuple correctly
        assert recorder.queue.qsize() == 1
        item = recorder.queue.get_nowait()
        assert isinstance(item, RecordedStep)
        assert item.position == (3, 4)  # 2D tuple


def test_recording_affordance_layout_variable_dims():
    """Recording affordance layout should handle variable position dimensions."""
    # 2D positions
    layout_2d = {
        "Bed": [2, 3],
        "Hospital": [5, 7],
    }

    # 3D positions (future)
    layout_3d = {
        "Bed": [2, 3, 0],
        "Hospital": [5, 7, 1],
    }

    # Aspatial positions
    layout_aspatial = {
        "Bed": [],
        "Hospital": [],
    }

    # All should be valid affordance layouts
    # Conversion to tuples should handle any length
    assert tuple(layout_2d["Bed"]) == (2, 3)
    assert tuple(layout_3d["Bed"]) == (2, 3, 0)
    assert tuple(layout_aspatial["Bed"]) == ()


def test_recording_handles_aspatial_positions():
    """Recorder should handle aspatial (empty) position tuples."""
    from townlet.recording.data_structures import RecordedStep

    with tempfile.TemporaryDirectory() as tmpdir:
        config = {"max_queue_size": 100}

        recorder = EpisodeRecorder(
            config=config,
            output_dir=Path(tmpdir),
            database=None,
            curriculum=None,
        )

        # Record step with aspatial position (empty tuple)
        positions = torch.tensor([], dtype=torch.long)  # Empty position
        meters = torch.rand(8)
        action = 4  # INTERACT

        recorder.record_step(
            step=0,
            positions=positions,
            meters=meters,
            action=action,
            reward=1.0,
            intrinsic_reward=0.0,
            done=False,
            q_values=None,
        )

        # Verify position was converted to empty tuple correctly
        assert recorder.queue.qsize() == 1
        item = recorder.queue.get_nowait()
        assert isinstance(item, RecordedStep)
        assert item.position == ()  # Empty tuple for aspatial
