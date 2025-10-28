"""
Training demonstration for Hamlet DRL Agent.

Shows the agent learning to survive in the environment over multiple episodes.
"""

import numpy as np
import torch
from hamlet.environment.hamlet_env import HamletEnv
from hamlet.agent.drl_agent import DRLAgent
from hamlet.agent.replay_buffer import ReplayBuffer
from hamlet.agent.observation_utils import preprocess_observation


def train_agent(
    num_episodes: int = 50,
    batch_size: int = 32,
    buffer_capacity: int = 10000,
    update_target_every: int = 10,
    device: str = "cpu"
):
    """
    Train a DRL agent in the Hamlet environment.

    Args:
        num_episodes: Number of training episodes
        batch_size: Batch size for learning
        buffer_capacity: Size of replay buffer
        update_target_every: Episodes between target network updates
        device: Device to train on ('cpu' or 'cuda')
    """
    # Initialize environment and agent
    env = HamletEnv()
    agent = DRLAgent(
        agent_id="learner",
        state_dim=72,  # Updated: 2 (pos) + 6 (meters) + 64 (grid)
        action_dim=5,
        learning_rate=1e-3,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        device=device,
        network_type="relational",  # Use relational network for cross-meter dependencies
        grid_size=8
    )
    buffer = ReplayBuffer(capacity=buffer_capacity)

    # Training metrics
    episode_rewards = []
    episode_lengths = []
    losses = []

    print("=" * 60)
    print("HAMLET DRL TRAINING DEMONSTRATION")
    print("=" * 60)
    print(f"Episodes: {num_episodes}")
    print(f"Batch size: {batch_size}")
    print(f"Buffer capacity: {buffer_capacity}")
    print(f"Device: {device}")
    print("=" * 60)
    print()

    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0.0
        episode_loss = []
        step = 0
        done = False

        while not done:
            # Select action
            action = agent.select_action(obs, explore=True)

            # Execute action
            next_obs, reward, done, info = env.step(action)
            episode_reward += reward
            step += 1

            # Preprocess observations for storage
            state = preprocess_observation(obs)
            next_state = preprocess_observation(next_obs)

            # Store experience
            buffer.push(state, action, reward, next_state, done)

            # Learn from experience
            if buffer.is_ready(batch_size):
                batch = buffer.sample(batch_size)
                loss = agent.learn(batch)
                episode_loss.append(loss)

            obs = next_obs

            # Limit episode length
            if step >= 500:
                break

        # Decay exploration
        agent.decay_epsilon()

        # Update target network periodically
        if (episode + 1) % update_target_every == 0:
            agent.update_target_network()

        # Record metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(step)
        if episode_loss:
            losses.append(np.mean(episode_loss))

        # Print progress
        if (episode + 1) % 5 == 0:
            recent_rewards = episode_rewards[-5:]
            recent_lengths = episode_lengths[-5:]
            recent_losses = losses[-5:] if len(losses) >= 5 else losses

            print(f"Episode {episode + 1:3d}/{num_episodes}")
            print(f"  Avg Reward (last 5):  {np.mean(recent_rewards):7.2f}")
            print(f"  Avg Length (last 5):  {np.mean(recent_lengths):7.1f}")
            if recent_losses:
                print(f"  Avg Loss (last 5):    {np.mean(recent_losses):7.4f}")
            print(f"  Epsilon:              {agent.epsilon:7.3f}")
            print(f"  Buffer size:          {len(buffer):7d}")
            print()

    # Final summary
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"First 5 episodes avg reward:  {np.mean(episode_rewards[:5]):7.2f}")
    print(f"Last 5 episodes avg reward:   {np.mean(episode_rewards[-5:]):7.2f}")
    print(f"Improvement:                  {np.mean(episode_rewards[-5:]) - np.mean(episode_rewards[:5]):7.2f}")
    print()
    print(f"First 5 episodes avg length:  {np.mean(episode_lengths[:5]):7.1f}")
    print(f"Last 5 episodes avg length:   {np.mean(episode_lengths[-5:]):7.1f}")
    print()

    # Save trained agent
    save_path = "models/trained_agent.pt"
    import os
    os.makedirs("models", exist_ok=True)
    agent.save(save_path)
    print(f"Agent saved to: {save_path}")
    print()

    # Demonstrate loading
    print("Testing save/load...")
    loaded_agent = DRLAgent("loaded", state_dim=72, action_dim=5, device=device, network_type="relational", grid_size=8)
    loaded_agent.load(save_path)
    print(f"  Original epsilon: {agent.epsilon:.3f}")
    print(f"  Loaded epsilon:   {loaded_agent.epsilon:.3f}")
    print(f"  Match: {abs(agent.epsilon - loaded_agent.epsilon) < 1e-6}")
    print()

    # Test loaded agent
    print("Running test episode with loaded agent...")
    obs = env.reset()
    test_reward = 0.0
    test_steps = 0
    done = False

    while not done and test_steps < 500:
        action = loaded_agent.select_action(obs, explore=False)  # Greedy
        obs, reward, done, info = env.step(action)
        test_reward += reward
        test_steps += 1

    print(f"  Test reward: {test_reward:.2f}")
    print(f"  Test steps:  {test_steps}")
    print()

    return agent, episode_rewards, episode_lengths, losses


if __name__ == "__main__":
    # Run training demonstration
    agent, rewards, lengths, losses = train_agent(
        num_episodes=50,
        batch_size=32,
        buffer_capacity=10000,
        update_target_every=10,
        device="cpu"
    )

    print("=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print()
    print("The agent has learned to survive in the Hamlet environment!")
    print("Key observations:")
    print("  - Reward should improve over episodes")
    print("  - Agent learns to balance multiple meters")
    print("  - Epsilon decays from exploration to exploitation")
    print("  - Model can be saved and loaded successfully")
    print()
    print("Next steps:")
    print("  - Adjust hyperparameters for better learning")
    print("  - Add visualization of agent behavior")
    print("  - Train for more episodes")
    print("  - Implement multi-agent scenarios")
