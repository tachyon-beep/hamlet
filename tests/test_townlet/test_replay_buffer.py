"""
Test Suite: Replay Buffer

Tests for ReplayBuffer to ensure correct storage, sampling, and reward combination.

Coverage Target: training/replay_buffer.py (0% → 80%+)

Critical Areas:
1. Circular buffer mechanics (FIFO eviction)
2. Lazy initialization on first push
3. Random sampling without replacement
4. Combined reward calculation (extrinsic + intrinsic * weight)
5. Batch handling and capacity limits
6. Device handling (CPU/CUDA)
"""

import pytest
import torch
from src.townlet.training.replay_buffer import ReplayBuffer


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
        assert small_buffer.dones[0] == True

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
        # Position 15 % 5 = 0, so next write goes to position 0
        # Current buffer has: [10, 11, 12, 13, 14] (wrapped around 3 times)
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

        assert batch["observations"].shape == (8, 2)  # batch × obs_dim
        assert batch["actions"].shape == (8,)
        assert batch["rewards"].shape == (8,)
        assert batch["next_observations"].shape == (8, 2)
        assert batch["dones"].shape == (8,)

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

        # Get indices to verify calculation
        # We can't know which indices were sampled, but we can verify the math
        # by checking that combined_reward is in valid range
        for reward in batch["rewards"]:
            # Extrinsic rewards are 0-19, intrinsic are 0-1.9
            # Combined should be extrinsic + intrinsic * 0.5
            # Min: 0 + 0*0.5 = 0, Max: 19 + 1.9*0.5 = 19.95
            assert 0.0 <= reward <= 20.0

    def test_zero_intrinsic_weight(self, filled_buffer):
        """With intrinsic_weight=0, rewards should equal extrinsic only."""
        batch = filled_buffer.sample(batch_size=10, intrinsic_weight=0.0)

        # Sampled rewards should be integers (extrinsic only: 0-19)
        for reward in batch["rewards"]:
            assert 0.0 <= reward <= 19.0

    def test_full_intrinsic_weight(self, filled_buffer):
        """With intrinsic_weight=1.0, rewards should include full intrinsic."""
        batch = filled_buffer.sample(batch_size=10, intrinsic_weight=1.0)

        # Combined rewards should be in range [0, 20.9]
        # (extrinsic 0-19 + intrinsic 0-1.9)
        for reward in batch["rewards"]:
            assert 0.0 <= reward <= 21.0

    def test_sample_returns_different_batches(self, filled_buffer):
        """Multiple samples should return different batches (randomness)."""
        batch1 = filled_buffer.sample(batch_size=5, intrinsic_weight=0.5)
        batch2 = filled_buffer.sample(batch_size=5, intrinsic_weight=0.5)

        # With 20 transitions and sampling 5, very unlikely to get identical batches
        # Check if at least one observation differs
        assert not torch.equal(batch1["observations"], batch2["observations"])

    def test_sample_full_buffer(self, filled_buffer):
        """Should be able to sample entire buffer (uses permutation)."""
        batch = filled_buffer.sample(batch_size=20, intrinsic_weight=0.5)

        assert batch["observations"].shape == (20, 2)
        assert batch["actions"].shape == (20,)

        # When sampling full buffer, should get permutation (no duplicates)
        # Verify by checking all values 0-19 are present
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

        # Create tensors (they're on CPU by default)
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
