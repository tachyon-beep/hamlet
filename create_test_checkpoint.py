"""Create a minimal test checkpoint for UI development."""

import torch
from pathlib import Path

# Import the network architecture
from townlet.population.vectorized import VectorizedPopulation
from townlet.environment.vectorized_env import VectorizedHamletEnv

def create_test_checkpoint():
    """Create a minimal checkpoint for testing the UI."""

    # Create a minimal environment to get dimensions
    device = torch.device('cpu')
    env = VectorizedHamletEnv(num_agents=1, grid_size=8, device=device)

    obs_dim = env.observation_dim
    action_dim = env.action_dim

    print(f"Creating checkpoint with obs_dim={obs_dim}, action_dim={action_dim}")

    # Create a simple Q-network (we just need the structure, not training)
    from townlet.agent.networks import SimpleQNetwork

    q_network = SimpleQNetwork(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=128,
    ).to(device)

    # Create checkpoint
    checkpoint = {
        'episode': 1,
        'epsilon': 0.5,
        'timestamp': 0.0,
        'population_state': {
            'q_network': q_network.state_dict(),
        }
    }

    # Save checkpoint
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    checkpoint_path = checkpoint_dir / "checkpoint_ep00001.pt"
    torch.save(checkpoint, checkpoint_path)

    print(f"Test checkpoint created: {checkpoint_path}")
    print(f"  Episode: {checkpoint['episode']}")
    print(f"  Epsilon: {checkpoint['epsilon']}")
    print(f"  Q-network parameters: {sum(p.numel() for p in q_network.parameters())}")

if __name__ == '__main__':
    create_test_checkpoint()
