#!/usr/bin/env python3
"""
Demo a trained agent in the Hamlet environment.

Shows the agent acting in real-time (no exploration).
"""

import time
import argparse
from pathlib import Path

from hamlet.environment.hamlet_env import HamletEnv
from hamlet.training.trainer import Trainer
from hamlet.training.config import EnvironmentConfig


def demo_agent(checkpoint_path: str, num_episodes: int = 3, delay: float = 0.1):
    """
    Demo a trained agent.

    Args:
        checkpoint_path: Path to checkpoint directory
        num_episodes: Number of episodes to run
        delay: Delay between steps (seconds) for visualization
    """
    print("=" * 70)
    print("Hamlet Trained Agent Demo")
    print("=" * 70)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Episodes: {num_episodes}")
    print()

    # Create environment
    env = HamletEnv(config=EnvironmentConfig())

    # Load trainer to get agent
    # For now, create a minimal config and load just the agent
    from hamlet.training.config import FullConfig, AgentConfig, TrainingConfig, MetricsConfig, ExperimentConfig

    config = FullConfig(
        experiment=ExperimentConfig(name="demo"),
        environment=EnvironmentConfig(),
        agents=[AgentConfig(agent_id="agent_0", algorithm="dqn")],
        training=TrainingConfig(),
        metrics=MetricsConfig(tensorboard=False, database=False, replay_storage=False),
    )

    trainer = Trainer(config)

    # Load checkpoint
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.exists():
        print(f"Loading checkpoint from {checkpoint_path}...")
        trainer.checkpoint_manager.load_checkpoint(
            checkpoint_path,
            {"agent_0": trainer.agent_manager.get_agent("agent_0")}
        )
        print("Checkpoint loaded!")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        print("Running with untrained agent...")

    print()

    agent = trainer.agent_manager.get_agent("agent_0")

    # Run episodes
    for episode in range(num_episodes):
        print(f"\n{'='*70}")
        print(f"Episode {episode + 1}/{num_episodes}")
        print(f"{'='*70}\n")

        obs = env.reset()

        from hamlet.agent.observation_utils import preprocess_observation
        state = preprocess_observation(obs)

        episode_reward = 0
        step = 0

        # Print initial state
        print(f"Step {step}:")
        print(f"  Position: ({obs['agent']['x']}, {obs['agent']['y']})")
        print(f"  Energy: {obs['agent']['energy']:.1f}")
        print(f"  Hygiene: {obs['agent']['hygiene']:.1f}")
        print(f"  Satiation: {obs['agent']['satiation']:.1f}")
        print(f"  Money: ${obs['agent']['money']:.1f}")

        while step < 100:  # Max 100 steps per episode
            # Select action (no exploration)
            action = agent.select_action(state, explore=False)

            # Action names for display
            action_names = ["UP", "DOWN", "LEFT", "RIGHT", "INTERACT"]

            # Execute action
            next_obs, reward, done, info = env.step(action)
            next_state = preprocess_observation(next_obs)

            episode_reward += reward
            step += 1

            # Print step info
            print(f"\nStep {step}:")
            print(f"  Action: {action_names[action]}")
            print(f"  Position: ({next_obs['agent']['x']}, {next_obs['agent']['y']})")
            print(f"  Reward: {reward:.2f}")
            print(f"  Energy: {next_obs['agent']['energy']:.1f}")
            print(f"  Hygiene: {next_obs['agent']['hygiene']:.1f}")
            print(f"  Satiation: {next_obs['agent']['satiation']:.1f}")
            print(f"  Money: ${next_obs['agent']['money']:.1f}")

            state = next_state

            # Delay for visualization
            time.sleep(delay)

            if done:
                print(f"\n{'='*40}")
                print(f"Episode ended at step {step}")
                print(f"Reason: Agent died (meter reached 0)")
                print(f"{'='*40}")
                break

        print(f"\nEpisode {episode + 1} Summary:")
        print(f"  Total Reward: {episode_reward:.2f}")
        print(f"  Steps: {step}")
        print(f"  Survival Time: {step}")


def main():
    parser = argparse.ArgumentParser(description="Demo a trained Hamlet agent")

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="test_checkpoints/checkpoint_ep10",
        help="Path to checkpoint directory",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes to run",
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=0.1,
        help="Delay between steps (seconds)",
    )

    args = parser.parse_args()

    demo_agent(args.checkpoint, args.episodes, args.delay)


if __name__ == "__main__":
    main()
