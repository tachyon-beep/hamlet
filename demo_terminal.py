#!/usr/bin/env python3
"""
Quick demo of terminal visualization during training.

Shows agent learning with simple ASCII rendering.
"""

from hamlet.training.trainer import Trainer
from hamlet.training.config import *


def main():
    # Minimal config for quick demo
    config = FullConfig(
        experiment=ExperimentConfig(
            name="terminal_demo",
            description="Quick terminal demo",
            tracking_uri="mlruns"
        ),
        environment=EnvironmentConfig(
            grid_width=8,
            grid_height=8,
        ),
        agents=[
            AgentConfig(
                agent_id="agent_0",
                algorithm="dqn",
                state_dim=72,
                action_dim=5,
            )
        ],
        training=TrainingConfig(
            num_episodes=3,  # Just 3 episodes for demo
            max_steps_per_episode=50,
            batch_size=32,
            learning_starts=10,
            target_update_frequency=1,
            replay_buffer_size=1000,
            save_frequency=10,
            checkpoint_dir="demo_checkpoints/",
            log_frequency=1,
        ),
        metrics=MetricsConfig(
            tensorboard=False,
            database=False,
            replay_storage=False,
            live_broadcast=False,
        ),
    )

    print("=" * 70)
    print("Hamlet Terminal Demo")
    print("=" * 70)
    print("Watch the agent (A) move around the grid")
    print("B=Bed, S=Shower, F=Fridge, J=Job")
    print("Meters will deplete, agent needs to manage survival")
    print()
    print("Press Ctrl+C to stop")
    print("=" * 70)
    print()

    # Create trainer with terminal visualization enabled
    trainer = Trainer(config, enable_terminal_viz=True)

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\nDemo stopped by user.")


if __name__ == "__main__":
    main()
