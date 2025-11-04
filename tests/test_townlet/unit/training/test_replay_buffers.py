"""Consolidated tests for replay buffer implementations.

This file consolidates tests from:
- test_replay_buffer.py (standard replay buffer for DQN)
- test_sequential_replay_buffer.py (sequential buffer for LSTM)
- test_sequential_replay_buffer_masking.py (post-terminal masking)

Coverage:
- ReplayBuffer: Circular buffer, random sampling, dual rewards
- SequentialReplayBuffer: Episode storage, sequence sampling, temporal continuity
- Post-terminal masking: Validity masks for recurrent training
"""

import pytest
import torch

from townlet.training.replay_buffer import ReplayBuffer
from townlet.training.sequential_replay_buffer import SequentialReplayBuffer


# =============================================================================
# STANDARD REPLAY BUFFER (Feed-forward DQN)
# =============================================================================


class TestReplayBufferInitialization:
    """Test replay buffer initialization and lazy storage allocation."""

    def test_initialization(self):
        """Buffer should initialize with correct capacity and empty storage."""
        buffer = ReplayBuffer(capacity=1000)

        assert buffer.capacity == 1000
        assert buffer.position == 0
        assert buffer.size == 0
        assert len(buffer) == 0

        # Storage should be None until first push (lazy initialization)
        assert buffer.observations is None
        assert buffer.actions is None
        assert buffer.rewards_extrinsic is None
        assert buffer.rewards_intrinsic is None
        assert buffer.next_observations is None
        assert buffer.dones is None

    def test_custom_capacity(self):
        """Buffer should accept custom capacity."""
        buffer = ReplayBuffer(capacity=500)
        assert buffer.capacity == 500

    def test_device_specification(self):
        """Buffer should accept device specification."""
        cpu_buffer = ReplayBuffer(capacity=100, device=torch.device("cpu"))
        assert cpu_buffer.device == torch.device("cpu")

        # CUDA buffer (only test if CUDA available)
        if torch.cuda.is_available():
            cuda_buffer = ReplayBuffer(capacity=100, device=torch.device("cuda"))
            assert cuda_buffer.device.type == "cuda"


class TestReplayBufferStorage:
    """Test storing transitions in the replay buffer."""

    @pytest.fixture
    def buffer(self):
        """Create small buffer for testing."""
        return ReplayBuffer(capacity=10)

    def test_single_transition_push(self, buffer):
        """Buffer should store single transition correctly."""
        obs = torch.randn(1, 5)  # batch=1, obs_dim=5
        actions = torch.tensor([2])
        rewards_ext = torch.tensor([1.0])
        rewards_int = torch.tensor([0.5])
        next_obs = torch.randn(1, 5)
        dones = torch.tensor([False])

        buffer.push(obs, actions, rewards_ext, rewards_int, next_obs, dones)

        assert len(buffer) == 1
        assert buffer.position == 1
        assert buffer.size == 1

        # Storage should now be initialized
        assert buffer.observations is not None
        assert buffer.observations.shape == (10, 5)  # capacity × obs_dim

    def test_batch_push(self, buffer):
        """Buffer should store batch of transitions correctly."""
        batch_size = 4
        obs = torch.randn(batch_size, 5)
        actions = torch.randint(0, 5, (batch_size,))
        rewards_ext = torch.randn(batch_size)
        rewards_int = torch.randn(batch_size)
        next_obs = torch.randn(batch_size, 5)
        dones = torch.rand(batch_size) > 0.5

        buffer.push(obs, actions, rewards_ext, rewards_int, next_obs, dones)

        assert len(buffer) == 4
        assert buffer.position == 4
        assert buffer.size == 4

    def test_lazy_storage_initialization(self, buffer):
        """Storage tensors should be initialized on first push, not construction."""
        # Before first push
        assert buffer.observations is None

        obs = torch.randn(2, 8)  # batch=2, obs_dim=8
        actions = torch.tensor([0, 1])
        rewards_ext = torch.tensor([1.0, 2.0])
        rewards_int = torch.tensor([0.1, 0.2])
        next_obs = torch.randn(2, 8)
        dones = torch.tensor([False, True])

        buffer.push(obs, actions, rewards_ext, rewards_int, next_obs, dones)

        # After first push
        assert buffer.observations is not None
        assert buffer.observations.shape == (10, 8)  # capacity × obs_dim (inferred)
        assert buffer.actions.shape == (10,)
        assert buffer.rewards_extrinsic.shape == (10,)
        assert buffer.rewards_intrinsic.shape == (10,)
        assert buffer.next_observations.shape == (10, 8)
        assert buffer.dones.shape == (10,)

    def test_stored_data_integrity(self, buffer):
        """Stored data should match input data."""
        obs = torch.tensor([[1.0, 2.0, 3.0]])
        actions = torch.tensor([2])
        rewards_ext = torch.tensor([10.0])
        rewards_int = torch.tensor([0.5])
        next_obs = torch.tensor([[4.0, 5.0, 6.0]])
        dones = torch.tensor([True])

        buffer.push(obs, actions, rewards_ext, rewards_int, next_obs, dones)

        # Verify stored data
        assert torch.allclose(buffer.observations[0], obs[0])
        assert buffer.actions[0] == actions[0]
        assert buffer.rewards_extrinsic[0] == rewards_ext[0]
        assert buffer.rewards_intrinsic[0] == rewards_int[0]
        assert torch.allclose(buffer.next_observations[0], next_obs[0])
        assert buffer.dones[0] == dones[0]


class TestReplayBufferCircularLogic:
    """Test circular buffer (FIFO) behavior when buffer fills up."""

    @pytest.fixture
    def small_buffer(self):
        """Create tiny buffer to test wraparound quickly."""
        return ReplayBuffer(capacity=5)

    def test_fill_to_capacity(self, small_buffer):
        """Buffer should accept transitions up to capacity."""
        for i in range(5):
            obs = torch.tensor([[float(i)]])
            actions = torch.tensor([i])
            rewards_ext = torch.tensor([float(i)])
            rewards_int = torch.tensor([float(i)])
            next_obs = torch.tensor([[float(i + 1)]])
            dones = torch.tensor([False])

            small_buffer.push(obs, actions, rewards_ext, rewards_int, next_obs, dones)

        assert len(small_buffer) == 5
        assert small_buffer.position == 5
        assert small_buffer.size == 5

    def test_fifo_eviction_when_full(self, small_buffer):
        """Old transitions should be evicted (FIFO) when buffer is full."""
        # Fill buffer with 5 transitions (values 0-4)
        for i in range(5):
            obs = torch.tensor([[float(i)]])
            actions = torch.tensor([i])
            rewards_ext = torch.tensor([float(i)])
            rewards_int = torch.tensor([float(i)])
            next_obs = torch.tensor([[float(i + 1)]])
            dones = torch.tensor([False])
            small_buffer.push(obs, actions, rewards_ext, rewards_int, next_obs, dones)

        # Verify first transition (value 0) is at position 0
        assert small_buffer.observations[0, 0] == 0.0
        assert small_buffer.actions[0] == 0

        # Add one more transition (value 5) - should overwrite position 0
        obs = torch.tensor([[5.0]])
        actions = torch.tensor([5])
        rewards_ext = torch.tensor([5.0])
        rewards_int = torch.tensor([5.0])
        next_obs = torch.tensor([[6.0]])
        dones = torch.tensor([True])
        small_buffer.push(obs, actions, rewards_ext, rewards_int, next_obs, dones)

        # Size should stay at capacity
        assert len(small_buffer) == 5
        assert small_buffer.position == 6  # Position keeps incrementing
        assert small_buffer.size == 5

        # Position 0 should now have value 5 (overwritten)
        assert small_buffer.observations[0, 0] == 5.0
        assert small_buffer.actions[0] == 5
        assert small_buffer.dones[0]

    def test_wraparound_multiple_times(self, small_buffer):
        """Buffer should handle multiple wraparounds correctly."""
        # Add 15 transitions (3× capacity)
        for i in range(15):
            obs = torch.tensor([[float(i)]])
            actions = torch.tensor([i % 5])  # Actions wrap
            rewards_ext = torch.tensor([float(i)])
            rewards_int = torch.tensor([float(i) * 0.1])
            next_obs = torch.tensor([[float(i + 1)]])
            dones = torch.tensor([i % 7 == 0])  # Some dones
            small_buffer.push(obs, actions, rewards_ext, rewards_int, next_obs, dones)

        # Size should be capped at capacity
        assert len(small_buffer) == 5
        assert small_buffer.size == 5

        # Position should be 15 (keeps incrementing)
        assert small_buffer.position == 15

        # Buffer should contain last 5 transitions (10-14)
        expected_values = [10.0, 11.0, 12.0, 13.0, 14.0]
        for i, expected in enumerate(expected_values):
            assert small_buffer.observations[i, 0] == expected


class TestReplayBufferSampling:
    """Test random sampling from replay buffer."""

    @pytest.fixture
    def filled_buffer(self):
        """Create buffer with 20 transitions."""
        buffer = ReplayBuffer(capacity=100)

        for i in range(20):
            obs = torch.tensor([[float(i), float(i * 2)]])
            actions = torch.tensor([i % 5])
            rewards_ext = torch.tensor([float(i)])
            rewards_int = torch.tensor([float(i) * 0.1])
            next_obs = torch.tensor([[float(i + 1), float((i + 1) * 2)]])
            dones = torch.tensor([i == 19])  # Last one is done
            buffer.push(obs, actions, rewards_ext, rewards_int, next_obs, dones)

        return buffer

    def test_sample_basic(self, filled_buffer):
        """Should sample correct batch size with correct shapes."""
        batch = filled_buffer.sample(batch_size=8, intrinsic_weight=0.5)

        assert "observations" in batch
        assert "actions" in batch
        assert "rewards" in batch
        assert "next_observations" in batch
        assert "dones" in batch
        assert "mask" in batch

        assert batch["observations"].shape == (8, 2)  # batch × obs_dim
        assert batch["actions"].shape == (8,)
        assert batch["rewards"].shape == (8,)
        assert batch["next_observations"].shape == (8, 2)
        assert batch["dones"].shape == (8,)
        assert batch["mask"].shape == (8,)
        assert batch["mask"].dtype == torch.bool
        assert torch.all(batch["mask"]), "Feed-forward mask should be all True"

    def test_sample_insufficient_data_raises_error(self):
        """Should raise error if buffer has fewer transitions than batch_size."""
        buffer = ReplayBuffer(capacity=100)

        # Add only 3 transitions
        for i in range(3):
            obs = torch.tensor([[float(i)]])
            actions = torch.tensor([i])
            rewards_ext = torch.tensor([1.0])
            rewards_int = torch.tensor([0.1])
            next_obs = torch.tensor([[float(i + 1)]])
            dones = torch.tensor([False])
            buffer.push(obs, actions, rewards_ext, rewards_int, next_obs, dones)

        # Try to sample batch of 5
        with pytest.raises(ValueError, match="Buffer size.*<.*batch_size"):
            buffer.sample(batch_size=5, intrinsic_weight=1.0)

    def test_combined_rewards_calculation(self, filled_buffer):
        """Rewards should combine: extrinsic + intrinsic * weight."""
        # Sample with weight 0.5
        batch = filled_buffer.sample(batch_size=5, intrinsic_weight=0.5)

        # Verify combined rewards are in valid range
        for reward in batch["rewards"]:
            # Extrinsic: 0-19, Intrinsic: 0-1.9
            # Combined: extrinsic + intrinsic * 0.5 = [0, 19.95]
            assert 0.0 <= reward <= 20.0

    def test_zero_intrinsic_weight(self, filled_buffer):
        """With intrinsic_weight=0, rewards should equal extrinsic only."""
        batch = filled_buffer.sample(batch_size=10, intrinsic_weight=0.0)

        # Sampled rewards should be in extrinsic range (0-19)
        for reward in batch["rewards"]:
            assert 0.0 <= reward <= 19.0

    def test_full_intrinsic_weight(self, filled_buffer):
        """With intrinsic_weight=1.0, rewards should include full intrinsic."""
        batch = filled_buffer.sample(batch_size=10, intrinsic_weight=1.0)

        # Combined rewards in range [0, 20.9] (extrinsic 0-19 + intrinsic 0-1.9)
        for reward in batch["rewards"]:
            assert 0.0 <= reward <= 21.0

    def test_sample_returns_different_batches(self, filled_buffer):
        """Multiple samples should return different batches (randomness)."""
        batch1 = filled_buffer.sample(batch_size=5, intrinsic_weight=0.5)
        batch2 = filled_buffer.sample(batch_size=5, intrinsic_weight=0.5)

        # With 20 transitions sampling 5, unlikely to get identical batches
        assert not torch.equal(batch1["observations"], batch2["observations"])
        assert torch.all(batch1["mask"])
        assert torch.all(batch2["mask"])

    def test_sample_full_buffer(self, filled_buffer):
        """Should be able to sample entire buffer (uses permutation)."""
        batch = filled_buffer.sample(batch_size=20, intrinsic_weight=0.5)

        assert batch["observations"].shape == (20, 2)
        assert batch["actions"].shape == (20,)
        assert batch["mask"].shape == (20,)
        assert torch.all(batch["mask"])

        # When sampling full buffer, should get permutation (no duplicates)
        obs_first_column = batch["observations"][:, 0].sort()[0]
        expected = torch.arange(20, dtype=torch.float32)
        assert torch.allclose(obs_first_column, expected)


class TestReplayBufferDeviceHandling:
    """Test device placement for CPU/CUDA."""

    def test_cpu_buffer_storage(self):
        """CPU buffer should store tensors on CPU."""
        buffer = ReplayBuffer(capacity=10, device=torch.device("cpu"))

        obs = torch.randn(2, 3)
        actions = torch.tensor([0, 1])
        rewards_ext = torch.tensor([1.0, 2.0])
        rewards_int = torch.tensor([0.1, 0.2])
        next_obs = torch.randn(2, 3)
        dones = torch.tensor([False, True])

        buffer.push(obs, actions, rewards_ext, rewards_int, next_obs, dones)

        assert buffer.observations.device.type == "cpu"
        assert buffer.actions.device.type == "cpu"

    def test_moves_tensors_to_device(self):
        """Buffer should move input tensors to its device."""
        buffer = ReplayBuffer(capacity=10, device=torch.device("cpu"))

        # Create tensors on CPU
        obs = torch.randn(1, 3)
        actions = torch.tensor([0])
        rewards_ext = torch.tensor([1.0])
        rewards_int = torch.tensor([0.1])
        next_obs = torch.randn(1, 3)
        dones = torch.tensor([False])

        buffer.push(obs, actions, rewards_ext, rewards_int, next_obs, dones)

        # Verify stored on correct device
        assert buffer.observations.device == buffer.device

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_buffer_storage(self):
        """CUDA buffer should store tensors on GPU."""
        buffer = ReplayBuffer(capacity=10, device=torch.device("cuda"))

        obs = torch.randn(2, 3)
        actions = torch.tensor([0, 1])
        rewards_ext = torch.tensor([1.0, 2.0])
        rewards_int = torch.tensor([0.1, 0.2])
        next_obs = torch.randn(2, 3)
        dones = torch.tensor([False, True])

        buffer.push(obs, actions, rewards_ext, rewards_int, next_obs, dones)

        assert buffer.observations.device.type == "cuda"
        assert buffer.actions.device.type == "cuda"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_sampling(self):
        """Sampling from CUDA buffer should return CUDA tensors."""
        buffer = ReplayBuffer(capacity=10, device=torch.device("cuda"))

        # Add transitions
        for i in range(5):
            obs = torch.tensor([[float(i)]])
            actions = torch.tensor([i])
            rewards_ext = torch.tensor([float(i)])
            rewards_int = torch.tensor([float(i) * 0.1])
            next_obs = torch.tensor([[float(i + 1)]])
            dones = torch.tensor([False])
            buffer.push(obs, actions, rewards_ext, rewards_int, next_obs, dones)

        batch = buffer.sample(batch_size=3, intrinsic_weight=0.5)

        assert batch["observations"].device.type == "cuda"
        assert batch["actions"].device.type == "cuda"
        assert batch["rewards"].device.type == "cuda"


class TestReplayBufferEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_capacity_buffer(self):
        """Buffer with capacity=1 should work correctly."""
        buffer = ReplayBuffer(capacity=1)

        # Add first transition
        obs1 = torch.tensor([[1.0]])
        buffer.push(
            obs1,
            torch.tensor([0]),
            torch.tensor([1.0]),
            torch.tensor([0.1]),
            torch.tensor([[2.0]]),
            torch.tensor([False]),
        )

        assert len(buffer) == 1

        # Add second transition (should overwrite first)
        obs2 = torch.tensor([[3.0]])
        buffer.push(
            obs2,
            torch.tensor([1]),
            torch.tensor([2.0]),
            torch.tensor([0.2]),
            torch.tensor([[4.0]]),
            torch.tensor([True]),
        )

        assert len(buffer) == 1
        assert buffer.observations[0, 0] == 3.0

    def test_empty_buffer_length(self):
        """Empty buffer should have length 0."""
        buffer = ReplayBuffer(capacity=10)
        assert len(buffer) == 0

    def test_large_batch_push(self):
        """Buffer should handle large batches correctly."""
        buffer = ReplayBuffer(capacity=1000)

        batch_size = 256
        obs = torch.randn(batch_size, 10)
        actions = torch.randint(0, 5, (batch_size,))
        rewards_ext = torch.randn(batch_size)
        rewards_int = torch.randn(batch_size)
        next_obs = torch.randn(batch_size, 10)
        dones = torch.rand(batch_size) > 0.9

        buffer.push(obs, actions, rewards_ext, rewards_int, next_obs, dones)

        assert len(buffer) == 256
        assert buffer.position == 256

    def test_different_observation_dimensions(self):
        """Buffer should infer observation dimension from first push."""
        buffer = ReplayBuffer(capacity=10)

        # First push with obs_dim=7
        obs = torch.randn(2, 7)
        actions = torch.tensor([0, 1])
        rewards_ext = torch.tensor([1.0, 2.0])
        rewards_int = torch.tensor([0.1, 0.2])
        next_obs = torch.randn(2, 7)
        dones = torch.tensor([False, True])

        buffer.push(obs, actions, rewards_ext, rewards_int, next_obs, dones)

        assert buffer.observations.shape == (10, 7)
        assert buffer.next_observations.shape == (10, 7)


# =============================================================================
# SEQUENTIAL REPLAY BUFFER (LSTM training)
# =============================================================================


class TestSequentialReplayBufferInitialization:
    """Test sequential buffer initialization and configuration."""

    def test_initialization(self):
        """Buffer should initialize with correct capacity and device."""
        buffer = SequentialReplayBuffer(capacity=1000, device=torch.device("cpu"))

        assert buffer.capacity == 1000
        assert buffer.device.type == "cpu"
        assert len(buffer) == 0

    def test_custom_capacity(self):
        """Buffer should support custom capacity."""
        buffer = SequentialReplayBuffer(capacity=500, device=torch.device("cpu"))
        assert buffer.capacity == 500

    def test_device_specification(self):
        """Buffer should support both CPU and CUDA devices."""
        cpu_buffer = SequentialReplayBuffer(capacity=100, device=torch.device("cpu"))
        assert cpu_buffer.device.type == "cpu"

        if torch.cuda.is_available():
            cuda_buffer = SequentialReplayBuffer(capacity=100, device=torch.device("cuda"))
            assert cuda_buffer.device.type == "cuda"


class TestEpisodeStorage:
    """Test storing complete episodes."""

    def test_store_single_episode(self):
        """Buffer should store a complete episode."""
        buffer = SequentialReplayBuffer(capacity=1000, device=torch.device("cpu"))

        # Create a short episode (5 steps)
        episode = {
            "observations": torch.randn(5, 10),  # [seq_len, obs_dim]
            "actions": torch.randint(0, 4, (5,)),  # [seq_len]
            "rewards": torch.randn(5),  # [seq_len]
            "dones": torch.tensor([False, False, False, False, True]),  # [seq_len]
        }

        buffer.store_episode(episode)

        assert len(buffer) == 1  # One episode stored
        assert buffer.num_transitions == 5  # 5 transitions total

    def test_store_multiple_episodes(self):
        """Buffer should store multiple episodes."""
        buffer = SequentialReplayBuffer(capacity=1000, device=torch.device("cpu"))

        # Store 3 episodes of different lengths
        for length in [5, 10, 7]:
            episode = {
                "observations": torch.randn(length, 10),
                "actions": torch.randint(0, 4, (length,)),
                "rewards": torch.randn(length),
                "dones": torch.zeros(length, dtype=torch.bool),
            }
            episode["dones"][-1] = True  # Mark last as done
            buffer.store_episode(episode)

        assert len(buffer) == 3
        assert buffer.num_transitions == 5 + 10 + 7

    def test_episode_validation(self):
        """Buffer should validate episode structure."""
        buffer = SequentialReplayBuffer(capacity=1000, device=torch.device("cpu"))

        # Missing required keys should raise error
        with pytest.raises(ValueError, match="Missing required keys"):
            buffer.store_episode({"observations": torch.randn(5, 10)})

        # Mismatched lengths should raise error
        with pytest.raises(ValueError, match="length mismatch"):
            buffer.store_episode(
                {
                    "observations": torch.randn(5, 10),
                    "actions": torch.randint(0, 4, (3,)),  # Wrong length
                    "rewards": torch.randn(5),
                    "dones": torch.zeros(5, dtype=torch.bool),
                }
            )


class TestSequentialCircularBufferLogic:
    """Test FIFO eviction when sequential buffer is full."""

    def test_capacity_enforcement(self):
        """Buffer should not exceed capacity."""
        buffer = SequentialReplayBuffer(capacity=100, device=torch.device("cpu"))

        # Store episodes until we exceed capacity
        for _ in range(30):  # 30 episodes * 5 transitions = 150 transitions
            episode = {
                "observations": torch.randn(5, 10),
                "actions": torch.randint(0, 4, (5,)),
                "rewards": torch.randn(5),
                "dones": torch.zeros(5, dtype=torch.bool),
            }
            episode["dones"][-1] = True
            buffer.store_episode(episode)

        # Buffer should evict oldest episodes to stay at capacity
        assert buffer.num_transitions <= 100

    def test_fifo_eviction(self):
        """Oldest episodes should be evicted first."""
        buffer = SequentialReplayBuffer(capacity=20, device=torch.device("cpu"))

        # Store episodes with unique reward values to track them
        for reward_val in [1.0, 2.0, 3.0, 4.0, 5.0]:
            episode = {
                "observations": torch.randn(5, 10),
                "actions": torch.randint(0, 4, (5,)),
                "rewards": torch.full((5,), reward_val),  # Unique marker
                "dones": torch.zeros(5, dtype=torch.bool),
            }
            episode["dones"][-1] = True
            buffer.store_episode(episode)

        # Buffer capacity is 20, stored 25 (5 episodes * 5 steps)
        # First episode (reward=1.0) should be evicted
        assert len(buffer) == 4  # Only 4 episodes remain
        assert buffer.num_transitions == 20


class TestSequenceSampling:
    """Test sampling sequences for LSTM training."""

    def test_sample_single_sequence(self):
        """Buffer should return a single sequence of specified length."""
        buffer = SequentialReplayBuffer(capacity=1000, device=torch.device("cpu"))

        # Store one long episode
        episode = {
            "observations": torch.randn(20, 10),
            "actions": torch.randint(0, 4, (20,)),
            "rewards": torch.randn(20),
            "dones": torch.zeros(20, dtype=torch.bool),
        }
        episode["dones"][-1] = True
        buffer.store_episode(episode)

        # Sample a sequence of length 5
        batch = buffer.sample_sequences(batch_size=1, seq_len=5)

        assert "observations" in batch
        assert batch["observations"].shape == (1, 5, 10)  # [batch, seq_len, obs_dim]
        assert batch["actions"].shape == (1, 5)  # [batch, seq_len]
        assert batch["rewards"].shape == (1, 5)  # [batch, seq_len]
        assert batch["dones"].shape == (1, 5)  # [batch, seq_len]

    def test_sample_batch_of_sequences(self):
        """Buffer should return multiple sequences."""
        buffer = SequentialReplayBuffer(capacity=1000, device=torch.device("cpu"))

        # Store multiple episodes
        for _ in range(5):
            episode = {
                "observations": torch.randn(20, 10),
                "actions": torch.randint(0, 4, (20,)),
                "rewards": torch.randn(20),
                "dones": torch.zeros(20, dtype=torch.bool),
            }
            episode["dones"][-1] = True
            buffer.store_episode(episode)

        # Sample batch of 4 sequences, each length 8
        batch = buffer.sample_sequences(batch_size=4, seq_len=8)

        assert batch["observations"].shape == (4, 8, 10)
        assert batch["actions"].shape == (4, 8)
        assert batch["rewards"].shape == (4, 8)
        assert batch["dones"].shape == (4, 8)

    def test_temporal_continuity(self):
        """Sampled sequences should be temporally continuous."""
        buffer = SequentialReplayBuffer(capacity=1000, device=torch.device("cpu"))

        # Store episode with sequential observations
        obs_seq = torch.arange(20).unsqueeze(-1).float()  # [20, 1] with values 0-19
        episode = {
            "observations": obs_seq,
            "actions": torch.zeros(20, dtype=torch.long),
            "rewards": torch.zeros(20),
            "dones": torch.zeros(20, dtype=torch.bool),
        }
        episode["dones"][-1] = True
        buffer.store_episode(episode)

        # Sample a sequence
        torch.manual_seed(42)  # For reproducibility
        batch = buffer.sample_sequences(batch_size=1, seq_len=5)

        # Observations should be consecutive (e.g., [3, 4, 5, 6, 7])
        obs_values = batch["observations"][0, :, 0]  # Get the sequence
        for i in range(len(obs_values) - 1):
            assert obs_values[i + 1] == obs_values[i] + 1, "Sequence not continuous!"

    def test_insufficient_data_raises_error(self):
        """Sampling should fail if not enough data."""
        buffer = SequentialReplayBuffer(capacity=1000, device=torch.device("cpu"))

        # Try to sample before storing any episodes
        with pytest.raises(ValueError, match="not enough data"):
            buffer.sample_sequences(batch_size=1, seq_len=5)

        # Store short episode
        episode = {
            "observations": torch.randn(3, 10),
            "actions": torch.randint(0, 4, (3,)),
            "rewards": torch.randn(3),
            "dones": torch.zeros(3, dtype=torch.bool),
        }
        episode["dones"][-1] = True
        buffer.store_episode(episode)

        # Try to sample longer sequence than available
        with pytest.raises(ValueError, match="not enough data"):
            buffer.sample_sequences(batch_size=1, seq_len=5)


class TestEpisodeBoundaryHandling:
    """Test that sequences don't cross episode boundaries."""

    def test_sequences_dont_cross_episodes(self):
        """Sampled sequences should not span multiple episodes."""
        buffer = SequentialReplayBuffer(capacity=1000, device=torch.device("cpu"))

        # Store two episodes with distinct patterns
        episode1 = {
            "observations": torch.ones(10, 10),  # All ones
            "actions": torch.zeros(10, dtype=torch.long),
            "rewards": torch.zeros(10),
            "dones": torch.zeros(10, dtype=torch.bool),
        }
        episode1["dones"][-1] = True

        episode2 = {
            "observations": torch.full((10, 10), 2.0),  # All twos
            "actions": torch.zeros(10, dtype=torch.long),
            "rewards": torch.zeros(10),
            "dones": torch.zeros(10, dtype=torch.bool),
        }
        episode2["dones"][-1] = True

        buffer.store_episode(episode1)
        buffer.store_episode(episode2)

        # Sample many sequences to test boundary handling
        for _ in range(20):
            batch = buffer.sample_sequences(batch_size=1, seq_len=5)
            obs = batch["observations"][0, :, 0]  # [seq_len]

            # All observations should be same value (all 1s or all 2s, never mixed)
            unique_values = obs.unique()
            assert len(unique_values) == 1, f"Sequence crossed episode boundary! Values: {obs}"


class TestSequentialDeviceHandling:
    """Test device management for stored tensors."""

    def test_cpu_storage(self):
        """Buffer should store and return CPU tensors."""
        buffer = SequentialReplayBuffer(capacity=100, device=torch.device("cpu"))

        episode = {
            "observations": torch.randn(5, 10),
            "actions": torch.randint(0, 4, (5,)),
            "rewards": torch.randn(5),
            "dones": torch.zeros(5, dtype=torch.bool),
        }
        episode["dones"][-1] = True
        buffer.store_episode(episode)

        batch = buffer.sample_sequences(batch_size=1, seq_len=3)
        assert batch["observations"].device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_storage(self):
        """Buffer should support CUDA tensors."""
        buffer = SequentialReplayBuffer(capacity=100, device=torch.device("cuda"))

        episode = {
            "observations": torch.randn(5, 10, device="cuda"),
            "actions": torch.randint(0, 4, (5,), device="cuda"),
            "rewards": torch.randn(5, device="cuda"),
            "dones": torch.zeros(5, dtype=torch.bool, device="cuda"),
        }
        episode["dones"][-1] = True
        buffer.store_episode(episode)

        batch = buffer.sample_sequences(batch_size=1, seq_len=3)
        assert batch["observations"].device.type == "cuda"


class TestIntrinsicRewardSupport:
    """Test support for dual reward system (extrinsic + intrinsic)."""

    def test_store_intrinsic_rewards(self):
        """Buffer should support intrinsic rewards."""
        buffer = SequentialReplayBuffer(capacity=1000, device=torch.device("cpu"))

        episode = {
            "observations": torch.randn(5, 10),
            "actions": torch.randint(0, 4, (5,)),
            "rewards_extrinsic": torch.randn(5),
            "rewards_intrinsic": torch.randn(5),
            "dones": torch.zeros(5, dtype=torch.bool),
        }
        episode["dones"][-1] = True
        buffer.store_episode(episode)

        assert len(buffer) == 1

    def test_sample_with_combined_rewards(self):
        """Buffer should combine extrinsic + intrinsic rewards when sampling."""
        buffer = SequentialReplayBuffer(capacity=1000, device=torch.device("cpu"))

        episode = {
            "observations": torch.randn(10, 10),
            "actions": torch.randint(0, 4, (10,)),
            "rewards_extrinsic": torch.ones(10),  # All 1.0
            "rewards_intrinsic": torch.full((10,), 0.5),  # All 0.5
            "dones": torch.zeros(10, dtype=torch.bool),
        }
        episode["dones"][-1] = True
        buffer.store_episode(episode)

        # Sample with intrinsic weight
        batch = buffer.sample_sequences(batch_size=1, seq_len=5, intrinsic_weight=1.0)

        # Combined reward should be 1.0 + 0.5*1.0 = 1.5
        assert "rewards" in batch
        assert torch.allclose(batch["rewards"], torch.tensor([1.5] * 5).unsqueeze(0))

    def test_intrinsic_weight_zero(self):
        """With zero intrinsic weight, only extrinsic rewards should be used."""
        buffer = SequentialReplayBuffer(capacity=1000, device=torch.device("cpu"))

        episode = {
            "observations": torch.randn(10, 10),
            "actions": torch.randint(0, 4, (10,)),
            "rewards_extrinsic": torch.ones(10),
            "rewards_intrinsic": torch.full((10,), 100.0),  # Large intrinsic
            "dones": torch.zeros(10, dtype=torch.bool),
        }
        episode["dones"][-1] = True
        buffer.store_episode(episode)

        batch = buffer.sample_sequences(batch_size=1, seq_len=5, intrinsic_weight=0.0)

        # Should only use extrinsic (1.0)
        assert torch.allclose(batch["rewards"], torch.ones(1, 5))


# =============================================================================
# POST-TERMINAL MASKING (Recurrent networks)
# =============================================================================


class TestPostTerminalMasking:
    """Test that sampled sequences include validity masks."""

    @pytest.fixture
    def buffer(self, cpu_device):
        """Empty sequential replay buffer on CPU for deterministic tests."""
        return SequentialReplayBuffer(capacity=10000, device=cpu_device)

    def test_sample_returns_mask(self, buffer, cpu_device):
        """Sample should return a mask field."""
        # Store episode with terminal in middle
        episode = {
            "observations": torch.randn(10, 72, device=cpu_device),
            "actions": torch.randint(0, 6, (10,), device=cpu_device),
            "rewards_extrinsic": torch.randn(10, device=cpu_device),
            "rewards_intrinsic": torch.randn(10, device=cpu_device),
            "dones": torch.tensor([False] * 5 + [True] + [False] * 4, device=cpu_device),
        }
        buffer.store_episode(episode)

        batch = buffer.sample_sequences(batch_size=2, seq_len=5)

        assert "mask" in batch, "Sample should return mask field"
        assert batch["mask"].shape == (2, 5), f"Mask shape should be [batch, seq_len], got {batch['mask'].shape}"
        assert batch["mask"].dtype == torch.bool, f"Mask should be bool, got {batch['mask'].dtype}"

    def test_mask_all_true_when_no_terminal(self, buffer, cpu_device):
        """Mask should be all True when sequence has no terminal."""
        # Episode with no terminal
        episode = {
            "observations": torch.randn(10, 72, device=cpu_device),
            "actions": torch.randint(0, 6, (10,), device=cpu_device),
            "rewards_extrinsic": torch.randn(10, device=cpu_device),
            "rewards_intrinsic": torch.randn(10, device=cpu_device),
            "dones": torch.zeros(10, dtype=torch.bool, device=cpu_device),
        }
        buffer.store_episode(episode)

        batch = buffer.sample_sequences(batch_size=4, seq_len=5)

        # All masks should be True (no terminals)
        assert torch.all(batch["mask"]), "Mask should be all True when no terminal in sequence"

    def test_mask_false_after_terminal(self, buffer, cpu_device):
        """Mask should be False after terminal state."""
        # Episode: terminal at index 3
        episode = {
            "observations": torch.randn(10, 72, device=cpu_device),
            "actions": torch.randint(0, 6, (10,), device=cpu_device),
            "rewards_extrinsic": torch.randn(10, device=cpu_device),
            "rewards_intrinsic": torch.randn(10, device=cpu_device),
            "dones": torch.tensor(
                [False, False, False, True, False, False, False, False, False, False],
                device=cpu_device
            ),
        }
        buffer.store_episode(episode)

        # Sample sequence starting at index 0 (length 7)
        # Should have terminal at index 3
        batch = buffer.sample_sequences(batch_size=1, seq_len=7)

        mask = batch["mask"][0]  # First (only) sequence
        dones = batch["dones"][0]

        # Find terminal in sampled sequence
        if torch.any(dones):
            terminal_idx = torch.where(dones)[0][0].item()

            # Mask should be True up to and including terminal
            assert torch.all(mask[: terminal_idx + 1]), f"Mask should be True before terminal at {terminal_idx}"

            # Mask should be False after terminal
            if terminal_idx < len(mask) - 1:
                assert torch.all(~mask[terminal_idx + 1 :]), f"Mask should be False after terminal at {terminal_idx}"

    def test_mask_includes_terminal_timestep(self, buffer, cpu_device):
        """Mask should include the terminal timestep itself (True at terminal)."""
        # Episode with terminal at index 2
        episode = {
            "observations": torch.randn(5, 72, device=cpu_device),
            "actions": torch.randint(0, 6, (5,), device=cpu_device),
            "rewards_extrinsic": torch.randn(5, device=cpu_device),
            "rewards_intrinsic": torch.randn(5, device=cpu_device),
            "dones": torch.tensor([False, False, True, False, False], device=cpu_device),
        }
        buffer.store_episode(episode)

        # Sample full episode
        batch = buffer.sample_sequences(batch_size=1, seq_len=5)

        mask = batch["mask"][0]
        dones = batch["dones"][0]

        terminal_idx = torch.where(dones)[0][0].item()

        # Terminal timestep itself should be valid (True)
        assert mask[terminal_idx], f"Mask should be True at terminal timestep {terminal_idx}"

        # Only timesteps AFTER terminal should be False
        assert torch.all(~mask[terminal_idx + 1 :]), "Only timesteps after terminal should be False"

    def test_mask_with_multiple_sequences(self, buffer, cpu_device):
        """Each sequence in batch should have its own mask based on its terminal."""
        # Store multiple episodes with different terminal positions
        episodes = [
            {  # Terminal at index 2
                "observations": torch.randn(10, 72, device=cpu_device),
                "actions": torch.randint(0, 6, (10,), device=cpu_device),
                "rewards_extrinsic": torch.randn(10, device=cpu_device),
                "rewards_intrinsic": torch.randn(10, device=cpu_device),
                "dones": torch.tensor([False, False, True] + [False] * 7, device=cpu_device),
            },
            {  # Terminal at index 5
                "observations": torch.randn(10, 72, device=cpu_device),
                "actions": torch.randint(0, 6, (10,), device=cpu_device),
                "rewards_extrinsic": torch.randn(10, device=cpu_device),
                "rewards_intrinsic": torch.randn(10, device=cpu_device),
                "dones": torch.tensor([False] * 5 + [True] + [False] * 4, device=cpu_device),
            },
            {  # No terminal
                "observations": torch.randn(10, 72, device=cpu_device),
                "actions": torch.randint(0, 6, (10,), device=cpu_device),
                "rewards_extrinsic": torch.randn(10, device=cpu_device),
                "rewards_intrinsic": torch.randn(10, device=cpu_device),
                "dones": torch.zeros(10, dtype=torch.bool, device=cpu_device),
            },
        ]

        for ep in episodes:
            buffer.store_episode(ep)

        # Sample batch
        batch = buffer.sample_sequences(batch_size=8, seq_len=6)

        # Check each sequence has valid mask
        for i in range(8):
            mask = batch["mask"][i]
            dones = batch["dones"][i]

            if torch.any(dones):
                terminal_idx = torch.where(dones)[0][0].item()
                # Should be True up to terminal, False after
                assert torch.all(mask[: terminal_idx + 1]), f"Sequence {i}: mask wrong before terminal"
                if terminal_idx < len(mask) - 1:
                    assert torch.all(~mask[terminal_idx + 1 :]), f"Sequence {i}: mask wrong after terminal"
            else:
                # No terminal -> all True
                assert torch.all(mask), f"Sequence {i}: should be all True (no terminal)"

    def test_mask_shape_matches_batch(self, buffer, cpu_device):
        """Mask shape should match [batch_size, seq_len]."""
        episode = {
            "observations": torch.randn(20, 72, device=cpu_device),
            "actions": torch.randint(0, 6, (20,), device=cpu_device),
            "rewards_extrinsic": torch.randn(20, device=cpu_device),
            "rewards_intrinsic": torch.randn(20, device=cpu_device),
            "dones": torch.zeros(20, dtype=torch.bool, device=cpu_device),
        }
        buffer.store_episode(episode)

        batch = buffer.sample_sequences(batch_size=16, seq_len=10)

        assert batch["mask"].shape == (16, 10), f"Mask shape should be [16, 10], got {batch['mask'].shape}"

    def test_mask_works_with_unified_rewards(self, buffer, cpu_device):
        """Mask should work when episode has unified 'rewards' instead of split."""
        episode = {
            "observations": torch.randn(10, 72, device=cpu_device),
            "actions": torch.randint(0, 6, (10,), device=cpu_device),
            "rewards": torch.randn(10, device=cpu_device),  # Unified rewards
            "dones": torch.tensor([False] * 4 + [True] + [False] * 5, device=cpu_device),
        }
        buffer.store_episode(episode)

        batch = buffer.sample_sequences(batch_size=2, seq_len=8)

        assert "mask" in batch, "Should have mask with unified rewards"
        assert batch["mask"].dtype == torch.bool, "Mask should be bool"


class TestMaskIntegrationWithLoss:
    """Test that mask can be used correctly in loss computation."""

    @pytest.fixture
    def buffer(self, cpu_device):
        """Empty sequential replay buffer on CPU for deterministic tests."""
        return SequentialReplayBuffer(capacity=10000, device=cpu_device)

    def test_mask_sum_counts_valid_timesteps(self, buffer, cpu_device):
        """Mask sum should equal number of valid timesteps."""
        # Episode with terminal at index 3 (4 valid timesteps: 0,1,2,3)
        episode = {
            "observations": torch.randn(10, 72, device=cpu_device),
            "actions": torch.randint(0, 6, (10,), device=cpu_device),
            "rewards_extrinsic": torch.randn(10, device=cpu_device),
            "rewards_intrinsic": torch.randn(10, device=cpu_device),
            "dones": torch.tensor([False, False, False, True] + [False] * 6, device=cpu_device),
        }
        buffer.store_episode(episode)

        # Sample sequence of length 10 starting at 0
        batch = buffer.sample_sequences(batch_size=1, seq_len=10)

        mask = batch["mask"][0]
        valid_count = mask.sum().item()

        # Should have 4 valid timesteps (0,1,2,3 - including terminal)
        assert valid_count == 4, f"Expected 4 valid timesteps, got {valid_count}"

    def test_masked_loss_example(self, buffer, cpu_device):
        """Demonstrate masked loss computation pattern."""
        episode = {
            "observations": torch.randn(10, 72, device=cpu_device),
            "actions": torch.randint(0, 6, (10,), device=cpu_device),
            "rewards_extrinsic": torch.randn(10, device=cpu_device),
            "rewards_intrinsic": torch.randn(10, device=cpu_device),
            "dones": torch.tensor([False] * 5 + [True] + [False] * 4, device=cpu_device),
        }
        buffer.store_episode(episode)

        batch = buffer.sample_sequences(batch_size=2, seq_len=8)

        # Simulate loss computation
        fake_losses = torch.randn(2, 8, device=cpu_device)  # [batch, seq_len]
        mask = batch["mask"]

        # Apply mask
        masked_losses = fake_losses * mask.float()
        total_loss = masked_losses.sum() / mask.sum().clamp_min(1)

        # Should be finite
        assert torch.isfinite(total_loss), "Masked loss should be finite"

        # Denominator should equal number of valid timesteps
        assert mask.sum() > 0, "Should have at least some valid timesteps"
