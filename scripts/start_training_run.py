#!/usr/bin/env python3
"""Helper script to start organized training runs.

Creates a clean directory structure for each training run:
    runs/
    └── L1_full_observability/
        └── 2025-11-02_143022/
            ├── config.yaml (copy of config)
            ├── checkpoints/
            ├── tensorboard/
            └── metrics.db

Usage:
    python scripts/start_training_run.py <config_path>

Examples:
    python scripts/start_training_run.py configs/level_1_full_observability.yaml
    python scripts/start_training_run.py configs/level_2_pomdp.yaml
    python scripts/start_training_run.py configs/level_3_temporal.yaml
"""

import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml


def infer_level_name(config_path: Path) -> str:
    """Infer level name from config filename."""
    filename = config_path.stem.lower()

    # Map level prefixes to human-readable names
    level_map = {
        "level_1": "L1_full_observability",
        "level_2": "L2_partial_observability",
        "level_3": "L3_temporal_mechanics",
        "level_4": "L4_multi_agent",
    }

    for prefix, level_name in level_map.items():
        if prefix in filename:
            return level_name

    # Fallback: use filename as-is
    return config_path.stem


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/start_training_run.py <config_path>")
        print()
        print("See docs/TRAINING_LEVELS.md for level specifications.")
        print()
        print("Examples:")
        print("  python scripts/start_training_run.py configs/level_1_full_observability.yaml")
        print("  python scripts/start_training_run.py configs/level_2_pomdp.yaml")
        print("  python scripts/start_training_run.py configs/level_3_temporal.yaml")
        sys.exit(1)

    config_path = Path(sys.argv[1])

    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    # Infer level name from config filename
    level_name = infer_level_name(config_path)

    # Generate timestamp for this run
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

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

    # Create run directory structure: runs/L1_full_observability/2025-11-02_143022/
    run_dir = Path("runs") / level_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    # Create tensorboard directory
    tensorboard_dir = run_dir / "tensorboard"
    tensorboard_dir.mkdir(exist_ok=True)

    # Copy config into run directory for reproducibility
    config_copy = run_dir / config_path.name
    shutil.copy2(config_path, config_copy)

    # Database path
    db_path = run_dir / "metrics.db"

    print(f"🚀 Starting training run: {level_name}")
    print(f"   Config: {config_path}")
    print(f"   Run directory: {run_dir}")
    print(f"   Episodes: {num_episodes}")
    print(f"   TensorBoard: tensorboard --logdir {tensorboard_dir}")
    print()

    # Build command
    cmd = [
        "python",
        "-m",
        "townlet.demo.runner",
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
