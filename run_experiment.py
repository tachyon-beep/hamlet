#!/usr/bin/env python3
"""
Run a Hamlet training experiment from a YAML configuration file.

Usage:
    python run_experiment.py configs/example_dqn.yaml
"""

import sys
import argparse
from pathlib import Path

from hamlet.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(
        description="Run Hamlet DRL training experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with default config
    python run_experiment.py configs/example_dqn.yaml

    # Create experiment directory first
    mkdir -p experiments/my_run
    cd experiments/my_run
    python ../../run_experiment.py ../../configs/example_dqn.yaml
        """,
    )

    parser.add_argument(
        "config",
        type=str,
        help="Path to YAML configuration file",
    )

    args = parser.parse_args()

    # Check config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    print("=" * 70)
    print("Hamlet DRL Training")
    print("=" * 70)
    print(f"Config: {config_path}")
    print()

    try:
        # Create trainer from YAML config
        trainer = Trainer.from_yaml(str(config_path))

        # Run training
        trainer.train()

        print()
        print("=" * 70)
        print("Training complete!")
        print("=" * 70)
        print()
        print("Results:")
        print(f"  - TensorBoard logs: {trainer.metrics_manager.config.tensorboard_dir}")
        print(f"  - Metrics database: {trainer.metrics_manager.config.database_path}")
        print(f"  - Checkpoints: {trainer.checkpoint_manager.checkpoint_dir}")
        print(f"  - MLflow tracking: {trainer.experiment_manager.config.tracking_uri}")
        print()
        print("To view results:")
        print(f"  tensorboard --logdir {trainer.metrics_manager.config.tensorboard_dir}")
        print(f"  mlflow ui --backend-store-uri {trainer.experiment_manager.config.tracking_uri}")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
