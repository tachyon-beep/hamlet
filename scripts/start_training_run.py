#!/usr/bin/env python3
"""Helper script to start organized training runs.

Creates a clean directory structure for each training run:
    runs/
    └── run_name/
        ├── config.yaml (copy of config)
        ├── checkpoints/
        └── metrics.db

Usage:
    python scripts/start_training_run.py <run_name> <config_path>

Examples:
    python scripts/start_training_run.py L2_pomdp configs/townlet_level_2_pomdp.yaml
    python scripts/start_training_run.py L2_5_temporal configs/townlet_level_2_5_temporal.yaml
"""

import shutil
import subprocess
import sys
from pathlib import Path

import yaml


def main():
    if len(sys.argv) < 3:
        print("Usage: python scripts/start_training_run.py <run_name> <config_path>")
        print()
        print("See docs/TRAINING_LEVELS.md for level specifications.")
        print()
        print("Examples:")
        print("  python scripts/start_training_run.py L1_baseline "
              "configs/level_1_full_observability.yaml")
        print("  python scripts/start_training_run.py L2_pomdp "
              "configs/level_2_pomdp.yaml")
        print("  python scripts/start_training_run.py L3_temporal "
              "configs/level_3_temporal.yaml")
        sys.exit(1)

    run_name = sys.argv[1]
    config_path = Path(sys.argv[2])

    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    # Load config to get max_episodes
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    num_episodes = config.get("training", {}).get("max_episodes")
    if num_episodes is None:
        print("Error: 'training.max_episodes' not found in config file")
        print("Please add to config:")
        print("  training:")
        print("    max_episodes: 10000")
        sys.exit(1)

    # Create run directory structure
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    # Copy config into run directory for reproducibility
    config_copy = run_dir / config_path.name
    shutil.copy2(config_path, config_copy)

    # Database path
    db_path = run_dir / "metrics.db"

    print(f"🚀 Starting training run: {run_name}")
    print(f"   Config: {config_path}")
    print(f"   Run directory: {run_dir}")
    print(f"   Episodes: {num_episodes}")
    print()

    # Build command
    cmd = [
        "python", "-m", "townlet.demo.runner",
        str(config_copy),  # Use the copied config
        str(db_path),
        str(checkpoint_dir),
        str(num_episodes),
    ]

    print(f"Command: {' '.join(cmd)}")
    print()

    # Run training
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        sys.exit(130)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with exit code {e.returncode}")
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
