"""
Tests for episode recorder.

Tests the core recording infrastructure: EpisodeRecorder and RecordingWriter.
"""

import queue
import tempfile
import threading
import time
from pathlib import Path

import torch


class TestEpisodeRecorder:
    """Test EpisodeRecorder queue and threading."""

    def test_recorder_creates_queue(self):
        """Recorder should create a bounded queue on initialization."""
        from townlet.recording.recorder import EpisodeRecorder

        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock minimal dependencies
            config = {"max_queue_size": 100}

            recorder = EpisodeRecorder(
                config=config,
                output_dir=Path(tmpdir),
                database=None,  # Mock
                curriculum=None,  # Mock
            )

            # Verify queue exists and has correct size
            assert recorder.queue is not None
            assert recorder.queue.maxsize == 100

    def test_record_step_adds_to_queue(self):
        """record_step should add RecordedStep to queue."""
        from townlet.recording.data_structures import RecordedStep
        from townlet.recording.recorder import EpisodeRecorder

        with tempfile.TemporaryDirectory() as tmpdir:
            config = {"max_queue_size": 100}

            recorder = EpisodeRecorder(
                config=config,
                output_dir=Path(tmpdir),
                database=None,
                curriculum=None,
            )

            # Record a step
            positions = torch.tensor([3, 5])
            meters = torch.tensor([1.0, 0.9, 0.8, 0.5, 0.7, 0.6, 0.95, 0.85])

            recorder.record_step(
                step=0,
                positions=positions,
                meters=meters,
                action=4,
                reward=1.0,
                intrinsic_reward=0.15,
                done=False,
                q_values=None,
            )

            # Verify item added to queue
            assert recorder.queue.qsize() == 1

            # Get item and verify it's a RecordedStep
            item = recorder.queue.get_nowait()
            assert isinstance(item, RecordedStep)
            assert item.step == 0
            assert item.position == (3, 5)
            assert item.action == 4

    def test_record_step_clones_tensors(self, cpu_device):
        """record_step should clone tensors to prevent training loop blocking."""
        from townlet.recording.recorder import EpisodeRecorder

        with tempfile.TemporaryDirectory() as tmpdir:
            config = {"max_queue_size": 100}

            recorder = EpisodeRecorder(
                config=config,
                output_dir=Path(tmpdir),
                database=None,
                curriculum=None,
            )

            # Create tensors on CPU device for determinism
            positions = torch.tensor([3, 5], device=cpu_device)
            meters = torch.tensor([1.0, 0.9, 0.8, 0.5, 0.7, 0.6, 0.95, 0.85], device=cpu_device)

            recorder.record_step(
                step=0,
                positions=positions,
                meters=meters,
                action=4,
                reward=1.0,
                intrinsic_reward=0.15,
                done=False,
            )

            # Get recorded step
            item = recorder.queue.get_nowait()

            # Verify position is a tuple (converted from tensor)
            assert isinstance(item.position, tuple)
            assert item.position == (3, 5)

    def test_record_step_with_q_values(self):
        """record_step should handle optional Q-values."""
        import numpy as np

        from townlet.recording.recorder import EpisodeRecorder

        with tempfile.TemporaryDirectory() as tmpdir:
            config = {"max_queue_size": 100}

            recorder = EpisodeRecorder(
                config=config,
                output_dir=Path(tmpdir),
                database=None,
                curriculum=None,
            )

            positions = torch.tensor([0, 0])
            meters = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
            q_values = torch.tensor([0.8, 0.7, 0.9, 0.6, 1.2, 0.5])

            recorder.record_step(
                step=10,
                positions=positions,
                meters=meters,
                action=2,
                reward=0.8,
                intrinsic_reward=0.05,
                done=False,
                q_values=q_values,
            )

            item = recorder.queue.get_nowait()
            # Use approximate equality due to float32 precision
            assert np.allclose(item.q_values, (0.8, 0.7, 0.9, 0.6, 1.2, 0.5), rtol=1e-5)

    def test_record_step_with_temporal_mechanics(self):
        """record_step should handle temporal mechanics fields."""
        from townlet.recording.recorder import EpisodeRecorder

        with tempfile.TemporaryDirectory() as tmpdir:
            config = {"max_queue_size": 100}

            recorder = EpisodeRecorder(
                config=config,
                output_dir=Path(tmpdir),
                database=None,
                curriculum=None,
            )

            positions = torch.tensor([4, 4])
            meters = torch.tensor([0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3])

            recorder.record_step(
                step=100,
                positions=positions,
                meters=meters,
                action=4,
                reward=0.5,
                intrinsic_reward=0.1,
                done=False,
                time_of_day=12,
                interaction_progress=0.33,
            )

            item = recorder.queue.get_nowait()
            assert item.time_of_day == 12
            assert item.interaction_progress == 0.33

    def test_record_temporal_mechanics_from_environment(self, cpu_device):
        """record_step should capture temporal state from VectorizedHamletEnv.

        Integration test: Verify that temporal mechanics state (time_of_day,
        interaction_progress) from a live environment gets correctly recorded.
        """
        from townlet.environment.vectorized_env import VectorizedHamletEnv
        from townlet.recording.recorder import EpisodeRecorder

        # Create environment with temporal mechanics enabled
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            partial_observability=False,
            device=cpu_device,
            enable_temporal_mechanics=True,
            vision_range=8,
            move_energy_cost=0.005,
            wait_energy_cost=0.001,
            interact_energy_cost=0.0,
            agent_lifespan=1000,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = {"max_queue_size": 100}

            recorder = EpisodeRecorder(
                config=config,
                output_dir=Path(tmpdir),
                database=None,
                curriculum=None,
            )

            # Stop writer thread to prevent it from consuming queue items
            recorder.writer.stop()
            recorder.writer_thread.join(timeout=1.0)

            # Reset environment and set agent on Bed
            env.reset()
            assert "Bed" in env.affordances, "Bed affordance not deployed"
            env.positions[0] = env.affordances["Bed"]
            env.meters[0, 0] = 0.3  # Low energy

            # Record 3 interaction steps
            for i in range(3):
                # Take INTERACT action
                obs, reward, done, info = env.step(torch.tensor([4], device=cpu_device))

                # Record step with temporal state
                recorder.record_step(
                    step=i,
                    positions=env.positions[0],
                    meters=env.meters[0],
                    action=4,
                    reward=reward[0].item(),
                    intrinsic_reward=0.0,
                    done=done[0].item(),
                    time_of_day=env.time_of_day,
                    interaction_progress=env.interaction_progress[0].item() / 10.0,  # Normalized
                )

            # Verify 3 steps recorded with temporal state
            assert recorder.queue.qsize() == 3

            # Check each recorded step
            for i in range(3):
                item = recorder.queue.get_nowait()

                # Time advances with each step (24-tick cycle)
                assert item.time_of_day is not None
                assert 0 <= item.time_of_day < 24

                # Interaction progress increases (Bed requires 5 ticks)
                assert item.interaction_progress is not None
                assert item.interaction_progress >= 0.0

                # Progress should increase with each interaction
                expected_progress = (i + 1) / 10.0  # Raw progress / 10.0
                assert item.interaction_progress == expected_progress

    def test_finish_episode_adds_marker(self):
        """finish_episode should add EpisodeEndMarker to queue."""
        from townlet.recording.data_structures import EpisodeEndMarker, EpisodeMetadata
        from townlet.recording.recorder import EpisodeRecorder

        with tempfile.TemporaryDirectory() as tmpdir:
            config = {"max_queue_size": 100}

            recorder = EpisodeRecorder(
                config=config,
                output_dir=Path(tmpdir),
                database=None,
                curriculum=None,
            )

            # Create metadata
            metadata = EpisodeMetadata(
                episode_id=100,
                survival_steps=200,
                total_reward=150.0,
                extrinsic_reward=145.0,
                intrinsic_reward=5.0,
                curriculum_stage=2,
                epsilon=0.5,
                intrinsic_weight=0.5,
                timestamp=1699123456.78,
                affordance_layout={"Bed": (2, 3)},
                affordance_visits={"Bed": 5},
            )

            recorder.finish_episode(metadata)

            # Verify marker added
            assert recorder.queue.qsize() == 1
            item = recorder.queue.get_nowait()
            assert isinstance(item, EpisodeEndMarker)
            assert item.metadata.episode_id == 100

    def test_queue_overflow_graceful_degradation(self):
        """Queue overflow should drop frames without crashing."""
        from townlet.recording.recorder import EpisodeRecorder

        with tempfile.TemporaryDirectory() as tmpdir:
            # Very small queue
            config = {"max_queue_size": 2}

            recorder = EpisodeRecorder(
                config=config,
                output_dir=Path(tmpdir),
                database=None,
                curriculum=None,
            )

            # Stop the writer thread to prevent it from consuming items
            recorder.writer.stop()
            recorder.writer_thread.join(timeout=1.0)

            positions = torch.tensor([0, 0])
            meters = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

            # Fill queue
            recorder.record_step(step=0, positions=positions, meters=meters, action=0, reward=1.0, intrinsic_reward=0.0, done=False)
            recorder.record_step(step=1, positions=positions, meters=meters, action=0, reward=1.0, intrinsic_reward=0.0, done=False)

            # Queue is full (size=2)
            assert recorder.queue.full()

            # Try to add one more (should drop gracefully)
            recorder.record_step(step=2, positions=positions, meters=meters, action=0, reward=1.0, intrinsic_reward=0.0, done=False)

            # Should still have 2 items (3rd was dropped)
            assert recorder.queue.qsize() == 2


class TestRecordingWriter:
    """Test RecordingWriter background thread."""

    def test_writer_starts_thread(self):
        """Writer should start background thread on initialization."""
        from townlet.recording.recorder import RecordingWriter

        with tempfile.TemporaryDirectory() as tmpdir:
            test_queue = queue.Queue(maxsize=100)
            config = {"compression": "lz4"}

            writer = RecordingWriter(
                queue=test_queue,
                config=config,
                output_dir=Path(tmpdir),
                database=None,
                curriculum=None,
            )

            # Verify writer is ready
            assert writer.running is True
            assert writer.episode_buffer == []

    def test_writer_buffers_steps(self):
        """Writer should buffer RecordedStep items."""
        from townlet.recording.data_structures import RecordedStep
        from townlet.recording.recorder import RecordingWriter

        with tempfile.TemporaryDirectory() as tmpdir:
            test_queue = queue.Queue(maxsize=100)
            config = {"compression": "lz4"}

            writer = RecordingWriter(
                queue=test_queue,
                config=config,
                output_dir=Path(tmpdir),
                database=None,
                curriculum=None,
            )

            # Start writer thread
            writer_thread = threading.Thread(target=writer.writer_loop, daemon=True)
            writer_thread.start()

            # Add step to queue
            step = RecordedStep(
                step=0,
                position=(0, 0),
                meters=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
                action=0,
                reward=1.0,
                intrinsic_reward=0.0,
                done=False,
                q_values=None,
            )
            test_queue.put(step)

            # Give writer time to process
            time.sleep(0.2)

            # Verify step was buffered
            assert len(writer.episode_buffer) == 1
            assert writer.episode_buffer[0] == step

            # Stop writer
            writer.stop()
            writer_thread.join(timeout=1.0)

    def test_writer_clears_buffer_on_episode_end(self):
        """Writer should clear buffer after processing episode end."""
        from townlet.recording.data_structures import EpisodeEndMarker, EpisodeMetadata, RecordedStep
        from townlet.recording.recorder import RecordingWriter

        with tempfile.TemporaryDirectory() as tmpdir:
            test_queue = queue.Queue(maxsize=100)
            config = {"compression": "lz4", "criteria": {"periodic": {"enabled": True, "interval": 1}}}  # Record everything

            # Mock database
            class MockDatabase:
                def insert_recording(self, **kwargs):
                    pass

            writer = RecordingWriter(
                queue=test_queue,
                config=config,
                output_dir=Path(tmpdir),
                database=MockDatabase(),
                curriculum=None,
            )

            # Add step
            step = RecordedStep(
                step=0,
                position=(0, 0),
                meters=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
                action=0,
                reward=1.0,
                intrinsic_reward=0.0,
                done=False,
                q_values=None,
            )
            test_queue.put(step)

            # Add episode end marker
            metadata = EpisodeMetadata(
                episode_id=1,
                survival_steps=1,
                total_reward=1.0,
                extrinsic_reward=1.0,
                intrinsic_reward=0.0,
                curriculum_stage=1,
                epsilon=1.0,
                intrinsic_weight=1.0,
                timestamp=time.time(),
                affordance_layout={},
                affordance_visits={},
            )
            marker = EpisodeEndMarker(metadata=metadata)
            test_queue.put(marker)

            # Give writer time to process
            time.sleep(0.5)

            # Buffer should be cleared
            assert len(writer.episode_buffer) == 0

    def test_writer_stops_gracefully(self):
        """Writer should stop when stop() is called."""
        from townlet.recording.recorder import RecordingWriter

        with tempfile.TemporaryDirectory() as tmpdir:
            test_queue = queue.Queue(maxsize=100)
            config = {"compression": "lz4"}

            writer = RecordingWriter(
                queue=test_queue,
                config=config,
                output_dir=Path(tmpdir),
                database=None,
                curriculum=None,
            )

            assert writer.running is True

            # Stop writer
            writer.stop()

            assert writer.running is False
