"""
Tests for SequentialReplayBuffer - stores sequences for LSTM training.

This buffer is designed for recurrent networks that need temporal context.
Unlike the standard replay buffer which samples random individual transitions,
this buffer samples sequences of consecutive transitions to maintain temporal
structure for LSTM/GRU training.
"""

import pytest
import torch

from townlet.training.sequential_replay_buffer import SequentialReplayBuffer


class TestSequentialReplayBufferInitialization:
    """Test buffer initialization and configuration."""

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
                    "actions": torch.randint(0, 4, (3,)),  # Wrong length!
                    "rewards": torch.randn(5),
                    "dones": torch.zeros(5, dtype=torch.bool),
                }
            )


class TestCircularBufferLogic:
    """Test FIFO eviction when buffer is full."""

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

        # Buffer capacity is 20 transitions, we stored 25 (5 episodes * 5 steps)
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


class TestDeviceHandling:
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
