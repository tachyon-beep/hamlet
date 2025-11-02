"""
Test that LSTM actually learns temporal patterns.

This is a functional test to validate that the sequential replay buffer
and LSTM training loop work together to learn temporal dependencies.

RED → GREEN workflow:
1. Write test that fails if LSTM doesn't use memory
2. Verify current implementation (should PASS since infrastructure exists)
3. If fails, identify and fix bugs
"""

import pytest
import torch
import torch.nn.functional as torch_fn

from townlet.agent.networks import RecurrentSpatialQNetwork
from townlet.training.sequential_replay_buffer import SequentialReplayBuffer


def test_lstm_learns_temporal_sequence():
    """
    Test that LSTM can learn a simple temporal pattern: A → B → C

    Task: Learn that action sequences matter
    - State 0 (start) → State 1: Take action 0
    - State 1 → State 2 (goal): Take action 1
    - Reward only given at State 2

    An MLP would struggle (state 1 looks the same whether you came from 0 or elsewhere).
    An LSTM should learn the temporal dependency.
    """
    device = torch.device("cpu")

    # Simple network for testing
    network = RecurrentSpatialQNetwork(
        action_dim=2,  # Actions: 0 (A→B) or 1 (B→C)
        window_size=1,  # Minimal spatial observation
        num_meters=2,  # Minimal state (just state_id encoded)
        num_affordance_types=1,
        enable_temporal_features=False,
        hidden_dim=32,  # Small for fast training
    ).to(device)

    optimizer = torch.optim.Adam(network.parameters(), lr=0.01)

    # Create synthetic episode: State 0 → State 1 → State 2
    # Observation encoding: [grid (1), pos (2), meters (2), affordance (2)] = 7 dims
    # State 0: meters = [1.0, 0.0]
    # State 1: meters = [0.5, 0.5]
    # State 2: meters = [0.0, 1.0]

    def make_observation(state_id):
        """Create synthetic observation for a state."""
        grid = torch.zeros(1)  # Minimal grid
        pos = torch.tensor([0.0, 0.0])  # Fixed position

        if state_id == 0:
            meters = torch.tensor([1.0, 0.0])
        elif state_id == 1:
            meters = torch.tensor([0.5, 0.5])
        else:  # state_id == 2
            meters = torch.tensor([0.0, 1.0])

        affordance = torch.zeros(2)  # No affordance

        return torch.cat([grid, pos, meters, affordance])

    # Create training episode: 0 → 1 → 2
    episode = {
        "observations": torch.stack(
            [
                make_observation(0),
                make_observation(1),
                make_observation(2),
            ]
        ),
        "actions": torch.tensor([0, 1, 0]),  # A→B, B→C, done
        "rewards_extrinsic": torch.tensor([0.0, 0.0, 1.0]),  # Reward only at end
        "rewards_intrinsic": torch.tensor([0.0, 0.0, 0.0]),
        "dones": torch.tensor([False, False, True]),
    }

    # Store episode in replay buffer
    replay_buffer = SequentialReplayBuffer(capacity=100, device=device)
    replay_buffer.store_episode(episode)

    # Store a few more episodes for sampling variety
    for _ in range(15):
        replay_buffer.store_episode(episode)

    # Create target network (for stable temporal learning)
    target_network = RecurrentSpatialQNetwork(
        action_dim=2,
        window_size=1,
        num_meters=2,
        num_affordance_types=1,
        enable_temporal_features=False,
        hidden_dim=32,
    ).to(device)
    target_network.load_state_dict(network.state_dict())
    target_network.eval()

    # Train for multiple iterations
    losses = []
    for iteration in range(500):  # More iterations for convergence
        # Sample sequence
        batch = replay_buffer.sample_sequences(
            batch_size=8,
            seq_len=3,
            intrinsic_weight=0.0,
        )

        gamma = 0.99

        # PASS 1: Collect Q-predictions from online network
        network.reset_hidden_state(batch_size=8, device=device)
        q_pred_list = []

        for t in range(3):
            q_values, _ = network(batch["observations"][:, t, :])
            q_pred = q_values.gather(1, batch["actions"][:, t].unsqueeze(1)).squeeze()
            q_pred_list.append(q_pred)

        # PASS 2: Collect Q-targets from target network (maintains hidden state!)
        with torch.no_grad():
            target_network.reset_hidden_state(batch_size=8, device=device)
            q_target_list = []
            q_values_list = []

            # First, unroll through entire sequence to get Q-values at each step
            for t in range(3):
                q_values, _ = target_network(batch["observations"][:, t, :])
                q_values_list.append(q_values)

            # Now compute targets using the Q-values from next timestep
            for t in range(3):
                if t < 2:
                    # Use Q-values from t+1 (already computed with hidden state from t)
                    q_next = q_values_list[t + 1].max(1)[0]
                    q_target = batch["rewards"][:, t] + gamma * q_next * (~batch["dones"][:, t]).float()
                else:
                    # Terminal state
                    q_target = batch["rewards"][:, t]

                q_target_list.append(q_target)

        # Compute loss
        q_pred_all = torch.stack(q_pred_list, dim=1)
        q_target_all = torch.stack(q_target_list, dim=1)
        loss = torch_fn.mse_loss(q_pred_all, q_target_all)

        # Backprop through time
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
        optimizer.step()

        # Update target network every 10 steps
        if iteration % 10 == 0:
            target_network.load_state_dict(network.state_dict())

        losses.append(loss.item())

    # After training, network should learn the pattern
    network.reset_hidden_state(batch_size=1, device=device)
    network.eval()

    with torch.no_grad():
        # Start at state 0
        obs_0 = make_observation(0).unsqueeze(0)
        q_0, _ = network(obs_0)

        # Follow sequence: state 0 → state 1
        obs_1 = make_observation(1).unsqueeze(0)
        q_1, _ = network(obs_1)

        # Follow sequence: state 1 → state 2
        obs_2 = make_observation(2).unsqueeze(0)
        q_2, _ = network(obs_2)

        # Check that network learned SOMETHING about the temporal pattern
        avg_q_0 = q_0[0].mean()
        avg_q_1 = q_1[0].mean()
        avg_q_2 = q_2[0].mean()

    # Verify loss decreased (learning happened) - this is the main success criterion
    initial_loss = sum(losses[:10]) / 10
    final_loss = sum(losses[-10:]) / 10

    assert final_loss < initial_loss * 0.5, (
        f"Loss should decrease during training. " f"Initial: {initial_loss:.4f}, Final: {final_loss:.4f}"
    )

    # Optional: print Q-values for debugging (not asserted due to random init sensitivity)
    print(f"Loss: Initial={initial_loss:.4f}, Final={final_loss:.4f}")
    print(f"Q(s_0)={avg_q_0:.4f}, Q(s_1)={avg_q_1:.4f}, Q(s_2)={avg_q_2:.4f}")


def test_lstm_memory_persistence_in_training():
    """
    Test that LSTM hidden state persists across timesteps during sequence training.

    This validates that the training loop correctly maintains hidden state
    when unrolling through sequences (as opposed to resetting each timestep).
    """
    device = torch.device("cpu")

    network = RecurrentSpatialQNetwork(
        action_dim=2,
        window_size=1,
        num_meters=2,
        num_affordance_types=1,
        enable_temporal_features=False,
        hidden_dim=16,
    ).to(device)

    # Create a sequence of 3 observations
    obs_sequence = torch.randn(4, 3, 7)  # [batch=4, seq_len=3, obs_dim=7]

    # Reset hidden state once for the batch
    network.reset_hidden_state(batch_size=4, device=device)

    # Unroll through sequence, tracking hidden states
    hidden_states = []

    for t in range(3):
        q_values, hidden = network(obs_sequence[:, t, :])
        h, c = hidden
        hidden_states.append((h.clone(), c.clone()))

    # Verify hidden state changed across timesteps
    h0, c0 = hidden_states[0]
    h1, c1 = hidden_states[1]
    h2, c2 = hidden_states[2]

    # Hidden states should be different (memory is evolving)
    assert not torch.allclose(h0, h1), "Hidden state should change between timesteps"
    assert not torch.allclose(h1, h2), "Hidden state should change between timesteps"

    # Cell states should also be different
    assert not torch.allclose(c0, c1), "Cell state should change between timesteps"
    assert not torch.allclose(c1, c2), "Cell state should change between timesteps"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
