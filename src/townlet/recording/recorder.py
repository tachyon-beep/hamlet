"""
Episode recorder implementation.

Provides non-blocking recording of episode data via async queue + background writer thread.
"""

import logging
import queue
import threading
from dataclasses import asdict
from pathlib import Path

import lz4.frame  # type: ignore[import-untyped]
import msgpack  # type: ignore[import-untyped]
import torch

from townlet.recording.data_structures import EpisodeEndMarker, EpisodeMetadata, RecordedStep

logger = logging.getLogger(__name__)


class EpisodeRecorder:
    """Main interface for episode recording.

    Thread-safe, non-blocking capture of episode data to a bounded queue.
    Background writer thread handles expensive I/O operations.
    """

    def __init__(
        self,
        config: dict,
        output_dir: Path,
        database,
        curriculum,
    ):
        """Initialize episode recorder.

        Args:
            config: Recording configuration dict
            output_dir: Directory for saving episode files
            database: Database instance for metadata
            curriculum: Curriculum instance for stage info
        """
        self.config = config
        self.output_dir = output_dir
        self.database = database
        self.curriculum = curriculum

        # Create bounded queue
        max_queue_size = config.get("max_queue_size", 1000)
        self.queue: queue.Queue = queue.Queue(maxsize=max_queue_size)

        # Start writer thread
        self.writer = RecordingWriter(
            queue=self.queue,
            config=config,
            output_dir=output_dir,
            database=database,
            curriculum=curriculum,
        )
        self.writer_thread = threading.Thread(
            target=self.writer.writer_loop,
            daemon=True,
            name="RecordingWriter",
        )
        self.writer_thread.start()

    def record_step(
        self,
        step: int,
        positions: torch.Tensor,
        meters: torch.Tensor,
        action: int,
        reward: float,
        intrinsic_reward: float,
        done: bool,
        q_values: torch.Tensor | None = None,
        epsilon: float | None = None,
        action_masks: torch.Tensor | None = None,
        time_of_day: int | None = None,
        interaction_progress: float | None = None,
    ):
        """Record a single step (non-blocking).

        Clones tensors to prevent training loop from blocking on I/O.

        Args:
            step: Step number within episode
            positions: [2] Agent (x, y) position
            meters: [8] All meters, normalized [0,1]
            action: Action taken (0-5)
            reward: Extrinsic reward
            intrinsic_reward: RND novelty reward
            done: Terminal state flag
            q_values: Optional [6] Q-values for all actions
            epsilon: Optional exploration rate (epsilon-greedy)
            action_masks: Optional [6] valid action flags (True=valid)
            time_of_day: Optional time of day (temporal mechanics)
            interaction_progress: Optional interaction progress (temporal mechanics)
        """
        # Convert tensors to Python types (cheap, avoids GPU blocking)
        # Handle q_values: could be tensor or list (if already converted by population)
        if q_values is not None:
            if hasattr(q_values, "tolist"):
                q_values_tuple = tuple(q_values.tolist())
            else:
                q_values_tuple = tuple(q_values)
        else:
            q_values_tuple = None

        # Handle action_masks: convert tensor to tuple of bools
        if action_masks is not None:
            if hasattr(action_masks, "tolist"):
                action_masks_tuple = tuple(action_masks.tolist())
            else:
                action_masks_tuple = tuple(action_masks)
        else:
            action_masks_tuple = None

        recorded_step = RecordedStep(
            step=step,
            # Convert position to tuple (handles any dimensionality)
            # - 2D: (x, y)
            # - 3D: (x, y, z)
            # - Aspatial: ()
            position=tuple(int(positions[i].item()) for i in range(positions.shape[0])),
            meters=tuple(meters.tolist()),
            action=action,
            reward=reward,
            intrinsic_reward=intrinsic_reward,
            done=done,
            q_values=q_values_tuple,
            epsilon=epsilon,
            action_masks=action_masks_tuple,
            time_of_day=time_of_day,
            interaction_progress=interaction_progress,
        )

        try:
            self.queue.put_nowait(recorded_step)
        except queue.Full:
            # Graceful degradation: drop frame if queue is full
            logger.warning(f"Recording queue full (step {step}), dropping frame")

    def finish_episode(self, metadata: EpisodeMetadata):
        """Mark episode boundary (non-blocking).

        Args:
            metadata: Episode metadata for recording decisions
        """
        marker = EpisodeEndMarker(metadata=metadata)
        try:
            self.queue.put_nowait(marker)
        except queue.Full:
            logger.error(f"Recording queue full, episode {metadata.episode_id} metadata lost")

    def shutdown(self):
        """Graceful shutdown: drain queue and stop writer thread."""
        self.writer.stop()
        self.writer_thread.join(timeout=10.0)


class RecordingWriter:
    """Background thread for writing episode recordings.

    Pulls items from queue, buffers steps until episode end, evaluates
    recording criteria, and writes selected episodes to disk.
    """

    def __init__(
        self,
        queue: queue.Queue,
        config: dict,
        output_dir: Path,
        database,
        curriculum,
    ):
        """Initialize recording writer.

        Args:
            queue: Queue to pull items from
            config: Recording configuration dict
            output_dir: Directory for saving episode files
            database: Database instance for metadata
            curriculum: Curriculum instance for stage info
        """
        self.queue = queue
        self.config = config
        self.output_dir = output_dir
        self.database = database
        self.curriculum = curriculum

        self.episode_buffer: list[RecordedStep] = []
        self.running = True

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def writer_loop(self):
        """Main writer thread loop.

        Pulls items from queue, buffers steps, and writes episodes when
        episode boundary is reached.
        """
        while self.running:
            try:
                item = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if isinstance(item, RecordedStep):
                # Buffer step
                self.episode_buffer.append(item)

            elif isinstance(item, EpisodeEndMarker):
                # Episode ended - evaluate criteria and write if needed
                self._process_episode_end(item.metadata)

                # Clear buffer regardless of whether we recorded
                self.episode_buffer.clear()

    def _process_episode_end(self, metadata: EpisodeMetadata):
        """Process episode end: evaluate criteria and write if needed.

        Args:
            metadata: Episode metadata
        """
        # Check if we should record this episode
        should_record = self._should_record_episode(metadata)

        if should_record:
            self._write_episode(metadata)
            logger.info(
                f"Recorded episode {metadata.episode_id}: "
                f"{len(self.episode_buffer)} steps, "
                f"survival={metadata.survival_steps}, "
                f"reward={metadata.total_reward:.1f}"
            )
        else:
            logger.debug(f"Skipped episode {metadata.episode_id} (no criteria matched)")

    def _should_record_episode(self, metadata: EpisodeMetadata) -> bool:
        """Evaluate recording criteria for episode.

        For now, just check periodic criterion. Full criteria evaluator
        will be implemented in Phase 2.

        Args:
            metadata: Episode metadata

        Returns:
            True if episode should be recorded
        """
        criteria = self.config.get("criteria", {})

        # Periodic criterion (simple implementation)
        periodic = criteria.get("periodic", {})
        if periodic.get("enabled", False):
            interval = periodic.get("interval", 100)
            if metadata.episode_id % interval == 0:
                return True

        return False

    def _write_episode(self, metadata: EpisodeMetadata):
        """Serialize, compress, and write episode to disk.

        Args:
            metadata: Episode metadata
        """
        # Build episode data structure
        episode_data = {
            "version": 1,
            "metadata": asdict(metadata),
            "steps": [asdict(step) for step in self.episode_buffer],
            "affordances": metadata.affordance_layout,
        }

        # Serialize with msgpack
        serialized = msgpack.packb(episode_data, use_bin_type=True)

        # Compress with LZ4
        compression = self.config.get("compression", "lz4")
        if compression == "lz4":
            compressed = lz4.frame.compress(serialized, compression_level=0)
        else:
            compressed = serialized  # No compression

        # Write to file
        episode_id = metadata.episode_id
        file_path = self.output_dir / f"episode_{episode_id:06d}.msgpack.lz4"
        file_path.write_bytes(compressed)

        # Index in database (if database provided)
        if self.database is not None:
            self.database.insert_recording(
                episode_id=episode_id,
                file_path=str(file_path.relative_to(self.output_dir.parent)),
                metadata=metadata,
                reason="periodic",  # For now, all recordings are periodic
                file_size=len(serialized),
                compressed_size=len(compressed),
            )

    def stop(self):
        """Signal writer thread to stop."""
        self.running = False
