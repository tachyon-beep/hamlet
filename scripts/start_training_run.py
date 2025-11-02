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
    python scripts/start_training_run.py <config_dir_or_training_yaml>

Examples:
    python scripts/start_training_run.py configs/L1_full_observability
    python scripts/start_training_run.py configs/L2_partial_observability/training.yaml
"""

import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml


def infer_level_name(config_dir: Path, config: dict) -> str:
    """Determine run folder name from config metadata."""
    metadata = config.get("run_metadata") or {}
    output_subdir = metadata.get("output_subdir")
    if output_subdir:
        return output_subdir
    return config_dir.name


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/start_training_run.py <config_dir_or_training_yaml>")
        print()
        print("See docs/TRAINING_LEVELS.md for level specifications.")
        print()
        print("Examples:")
        print("  python scripts/start_training_run.py configs/L1_full_observability")
        print("  python scripts/start_training_run.py configs/L2_partial_observability/training.yaml")
        sys.exit(1)

    config_input = Path(sys.argv[1])
    if config_input.is_dir():
        config_dir = config_input
        training_config = config_dir / "training.yaml"
    else:
        training_config = config_input
        config_dir = training_config.parent

    if not training_config.exists():
        print(f"Error: Training config not found: {training_config}")
        sys.exit(1)

    # Load config to get metadata and max_episodes
    with open(training_config) as f:
        config = yaml.safe_load(f)

    level_name = infer_level_name(config_dir, config)

    # Generate timestamp for this run
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

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

    # Copy entire config pack for reproducibility
    pack_copy_dir = run_dir / "config_pack"
    if pack_copy_dir.exists():
        shutil.rmtree(pack_copy_dir)
    shutil.copytree(config_dir, pack_copy_dir)

    training_copy = pack_copy_dir / training_config.name

    # Database path
    db_path = run_dir / "metrics.db"

    print(f"🚀 Starting training run: {level_name}")
    print(f"   Config pack: {config_dir}")
    print(f"   Run directory: {run_dir}")
    print(f"   Episodes: {num_episodes}")
    print(f"   TensorBoard: tensorboard --logdir {tensorboard_dir}")
    print()

    # Build command
    cmd = [
        "python",
        "-m",
        "townlet.demo.runner",
        str(pack_copy_dir),
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
