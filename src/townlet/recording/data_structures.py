"""
Data structures for episode recording.

Defines frozen dataclasses for recording episode data with minimal memory footprint.
All structures are designed for serialization with msgpack.
"""

from dataclasses import dataclass


def deserialize_step(data: dict) -> "RecordedStep":
    """Deserialize RecordedStep from msgpack dict.

    Handles conversion of lists back to tuples (msgpack deserializes tuples as lists).

    Args:
        data: Dictionary from msgpack.unpackb()

    Returns:
        RecordedStep instance
    """
    # Convert lists back to tuples
    data["position"] = tuple(data["position"])
    data["meters"] = tuple(data["meters"])
    if data["q_values"] is not None:
        data["q_values"] = tuple(data["q_values"])

    return RecordedStep(**data)


def deserialize_metadata(data: dict) -> "EpisodeMetadata":
    """Deserialize EpisodeMetadata from msgpack dict.

    Handles conversion of lists back to tuples for affordance positions.

    Args:
        data: Dictionary from msgpack.unpackb()

    Returns:
        EpisodeMetadata instance
    """
    # Convert affordance positions from lists to tuples
    data["affordance_layout"] = {
        name: tuple(pos) for name, pos in data["affordance_layout"].items()
    }

    return EpisodeMetadata(**data)


@dataclass(frozen=True, slots=True)
class RecordedStep:
    """Single step of episode recording.

    Optimized for:
    - Fast clone from GPU tensors
    - Minimal memory footprint (~100-150 bytes/step)
    - Lock-free queue compatibility
    - msgpack serialization
    """

    step: int  # Step number within episode
    position: tuple[int, int]  # Agent (x, y)
    meters: tuple[float, ...]  # 8 meters, normalized [0,1]
    action: int  # Action taken (0-5)
    reward: float  # Extrinsic reward
    intrinsic_reward: float  # RND novelty reward
    done: bool  # Terminal state
    q_values: tuple[float, ...] | None  # Optional Q-values for all actions

    # Optional temporal mechanics (Level 2.5+)
    time_of_day: int | None = None
    interaction_progress: float | None = None


@dataclass(frozen=True, slots=True)
class EpisodeMetadata:
    """Episode-level metadata for recording decisions.

    Contains summary statistics and context needed for:
    - Recording criteria evaluation
    - Database indexing
    - Replay playback
    """

    episode_id: int
    survival_steps: int
    total_reward: float
    extrinsic_reward: float
    intrinsic_reward: float
    curriculum_stage: int
    epsilon: float
    intrinsic_weight: float
    timestamp: float

    # Affordance context
    affordance_layout: dict[str, tuple[int, int]]  # name → (x, y)
    affordance_visits: dict[str, int]  # name → count


@dataclass
class EpisodeEndMarker:
    """Sentinel value marking episode boundary in queue.

    Not frozen to allow writer thread to process and discard.
    """

    metadata: EpisodeMetadata
