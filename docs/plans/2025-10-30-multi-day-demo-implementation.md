# Multi-Day Demo Implementation Plan (Phase 3.5)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a multi-day demonstration system that runs 10K episodes unattended, streams live to browsers, and captures teaching materials.

**Architecture:** Separate processes (training, visualization, snapshots) sharing state via SQLite. Training auto-resumes from checkpoints via systemd. Containerization-ready design ("for but not with").

**Tech Stack:** Python 3.11, SQLite, PyTorch, FastAPI, WebSocket, Vue 3, systemd, selenium/playwright

**Estimated Duration:** 2-3 days implementation

**Exit Criteria:**
- [ ] demo_runner.py runs 10K episodes with checkpointing
- [ ] SQLite database captures all metrics
- [ ] viz_server.py streams to browsers
- [ ] snapshot_daemon.py generates screenshots/GIFs/CSVs
- [ ] systemd auto-restart works
- [ ] Generalization test (affordance randomization at ep 5000)
- [ ] End-to-end test validates 100+ episodes

---

## Task 1: SQLite Schema and Database Helper

**Files:**
- Create: `src/hamlet/demo/database.py`
- Create: `tests/test_demo/test_database.py`

**Step 1: Write the failing test**

Create `tests/test_demo/__init__.py`:
```python
"""Tests for multi-day demo infrastructure."""
```

Create `tests/test_demo/test_database.py`:
```python
"""Tests for demo database operations."""

import pytest
import sqlite3
import tempfile
from pathlib import Path
from hamlet.demo.database import DemoDatabase


def test_demo_database_initialization():
    """Database should initialize schema on creation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        db = DemoDatabase(db_path)

        # Verify tables exist
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check episodes table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='episodes'")
        assert cursor.fetchone() is not None

        # Check system_state table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='system_state'")
        assert cursor.fetchone() is not None

        conn.close()


def test_insert_episode_metrics():
    """Should insert and retrieve episode metrics."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        db = DemoDatabase(db_path)

        # Insert episode
        db.insert_episode(
            episode_id=1,
            timestamp=1234567890.0,
            survival_time=150,
            total_reward=42.5,
            extrinsic_reward=30.0,
            intrinsic_reward=12.5,
            intrinsic_weight=0.8,
            curriculum_stage=3,
            epsilon=0.1
        )

        # Retrieve
        episodes = db.get_latest_episodes(limit=1)
        assert len(episodes) == 1
        assert episodes[0]['episode_id'] == 1
        assert episodes[0]['survival_time'] == 150
        assert episodes[0]['total_reward'] == 42.5


def test_system_state_get_set():
    """Should store and retrieve system state."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        db = DemoDatabase(db_path)

        # Set state
        db.set_system_state('current_episode', '42')
        db.set_system_state('training_status', 'running')

        # Get state
        assert db.get_system_state('current_episode') == '42'
        assert db.get_system_state('training_status') == 'running'
        assert db.get_system_state('nonexistent') is None
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_demo/test_database.py -xvs`

Expected: FAIL with "No module named 'hamlet.demo.database'"

**Step 3: Write minimal implementation**

Create `src/hamlet/demo/__init__.py`:
```python
"""Multi-day demonstration infrastructure."""
```

Create `src/hamlet/demo/database.py`:
```python
"""SQLite database for multi-day demo state management."""

import sqlite3
from pathlib import Path
from typing import Optional, List, Dict, Any


class DemoDatabase:
    """Manages SQLite database for demo metrics and state."""

    def __init__(self, db_path: Path | str):
        """Initialize database, creating schema if needed.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Enable WAL mode for better concurrent access
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.row_factory = sqlite3.Row

        self._create_schema()

    def _create_schema(self):
        """Create database schema if it doesn't exist."""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS episodes (
                episode_id INTEGER PRIMARY KEY,
                timestamp REAL NOT NULL,
                survival_time INTEGER NOT NULL,
                total_reward REAL NOT NULL,
                extrinsic_reward REAL NOT NULL,
                intrinsic_reward REAL NOT NULL,
                intrinsic_weight REAL NOT NULL,
                curriculum_stage INTEGER NOT NULL,
                epsilon REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_episodes_timestamp ON episodes(timestamp);

            CREATE TABLE IF NOT EXISTS affordance_visits (
                episode_id INTEGER NOT NULL,
                from_affordance TEXT NOT NULL,
                to_affordance TEXT NOT NULL,
                visit_count INTEGER NOT NULL,
                FOREIGN KEY (episode_id) REFERENCES episodes(episode_id)
            );
            CREATE INDEX IF NOT EXISTS idx_visits_episode ON affordance_visits(episode_id);

            CREATE TABLE IF NOT EXISTS position_heatmap (
                episode_id INTEGER NOT NULL,
                x INTEGER NOT NULL,
                y INTEGER NOT NULL,
                visit_count INTEGER NOT NULL,
                novelty_value REAL,
                FOREIGN KEY (episode_id) REFERENCES episodes(episode_id)
            );
            CREATE INDEX IF NOT EXISTS idx_heatmap_episode ON position_heatmap(episode_id);

            CREATE TABLE IF NOT EXISTS system_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
        """)
        self.conn.commit()

    def insert_episode(
        self,
        episode_id: int,
        timestamp: float,
        survival_time: int,
        total_reward: float,
        extrinsic_reward: float,
        intrinsic_reward: float,
        intrinsic_weight: float,
        curriculum_stage: int,
        epsilon: float,
    ):
        """Insert episode metrics into database.

        Args:
            episode_id: Episode number
            timestamp: Unix timestamp
            survival_time: Steps survived
            total_reward: Combined reward
            extrinsic_reward: Environment reward
            intrinsic_reward: RND novelty reward
            intrinsic_weight: Current intrinsic weight
            curriculum_stage: Current curriculum stage (1-5)
            epsilon: Current exploration epsilon
        """
        self.conn.execute(
            """INSERT INTO episodes
               (episode_id, timestamp, survival_time, total_reward, extrinsic_reward,
                intrinsic_reward, intrinsic_weight, curriculum_stage, epsilon)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (episode_id, timestamp, survival_time, total_reward, extrinsic_reward,
             intrinsic_reward, intrinsic_weight, curriculum_stage, epsilon)
        )
        self.conn.commit()

    def get_latest_episodes(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get most recent episodes.

        Args:
            limit: Maximum number of episodes to return

        Returns:
            List of episode dictionaries
        """
        cursor = self.conn.execute(
            "SELECT * FROM episodes ORDER BY episode_id DESC LIMIT ?",
            (limit,)
        )
        return [dict(row) for row in cursor.fetchall()]

    def set_system_state(self, key: str, value: str):
        """Set system state key-value pair.

        Args:
            key: State key
            value: State value (will be converted to string)
        """
        self.conn.execute(
            "INSERT OR REPLACE INTO system_state (key, value) VALUES (?, ?)",
            (key, str(value))
        )
        self.conn.commit()

    def get_system_state(self, key: str) -> Optional[str]:
        """Get system state value.

        Args:
            key: State key

        Returns:
            State value or None if not found
        """
        cursor = self.conn.execute(
            "SELECT value FROM system_state WHERE key = ?",
            (key,)
        )
        row = cursor.fetchone()
        return row['value'] if row else None

    def close(self):
        """Close database connection."""
        self.conn.close()
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_demo/test_database.py -xvs`

Expected: PASS (all 3 tests)

**Step 5: Commit**

```bash
git add src/hamlet/demo/ tests/test_demo/
git commit -m "feat(demo): add SQLite database schema and helper"
```

---

## Task 2: Demo Runner with Checkpointing

**Files:**
- Create: `src/hamlet/demo/runner.py`
- Create: `tests/test_demo/test_runner.py`
- Modify: `configs/townlet/sparse_adaptive.yaml` (add demo-specific params)

**Step 1: Write the failing test**

Create `tests/test_demo/test_runner.py`:
```python
"""Tests for demo runner."""

import pytest
import tempfile
import torch
from pathlib import Path
from hamlet.demo.runner import DemoRunner
from hamlet.demo.database import DemoDatabase


def test_demo_runner_initialization():
    """DemoRunner should initialize from config."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(__file__).parent.parent.parent / "configs" / "townlet" / "sparse_adaptive.yaml"
        runner = DemoRunner(
            config_path=config_path,
            db_path=Path(tmpdir) / "demo.db",
            checkpoint_dir=Path(tmpdir) / "checkpoints",
            max_episodes=100  # Override for testing
        )

        assert runner.max_episodes == 100
        assert runner.current_episode == 0
        assert runner.db is not None


def test_checkpoint_save_load():
    """Should save and load checkpoints."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(__file__).parent.parent.parent / "configs" / "townlet" / "sparse_adaptive.yaml"
        checkpoint_dir = Path(tmpdir) / "checkpoints"

        runner = DemoRunner(
            config_path=config_path,
            db_path=Path(tmpdir) / "demo.db",
            checkpoint_dir=checkpoint_dir,
            max_episodes=100
        )

        # Run a few episodes
        runner.current_episode = 42

        # Save checkpoint
        runner.save_checkpoint()

        # Verify file exists
        checkpoint_files = list(checkpoint_dir.glob("checkpoint_ep*.pt"))
        assert len(checkpoint_files) == 1
        assert "ep00042" in str(checkpoint_files[0])

        # Load checkpoint in new runner
        runner2 = DemoRunner(
            config_path=config_path,
            db_path=Path(tmpdir) / "demo.db",
            checkpoint_dir=checkpoint_dir,
            max_episodes=100
        )
        loaded_episode = runner2.load_checkpoint()

        assert loaded_episode == 42
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_demo/test_runner.py -xvs`

Expected: FAIL with "No module named 'hamlet.demo.runner'"

**Step 3: Write minimal implementation**

Create `src/hamlet/demo/runner.py`:
```python
"""Demo runner for multi-day training."""

import logging
import signal
import time
from pathlib import Path
from typing import Optional
import torch
import yaml

from hamlet.demo.database import DemoDatabase
from townlet.population.vectorized import VectorizedPopulation
from townlet.environment.vectorized_env import VectorizedHamletEnv
from townlet.curriculum.adversarial import AdversarialCurriculum
from townlet.exploration.adaptive_intrinsic import AdaptiveIntrinsicExploration

logger = logging.getLogger(__name__)


class DemoRunner:
    """Orchestrates multi-day demo training with checkpointing."""

    def __init__(
        self,
        config_path: Path | str,
        db_path: Path | str,
        checkpoint_dir: Path | str,
        max_episodes: int = 10000,
    ):
        """Initialize demo runner.

        Args:
            config_path: Path to YAML config file
            db_path: Path to SQLite database
            checkpoint_dir: Directory for checkpoint files
            max_episodes: Maximum number of episodes to run
        """
        self.config_path = Path(config_path)
        self.db_path = Path(db_path)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_episodes = max_episodes
        self.current_episode = 0

        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self.db = DemoDatabase(self.db_path)

        # Load config
        with open(self.config_path) as f:
            self.config = yaml.safe_load(f)

        # Initialize components (will be created in run())
        self.env = None
        self.population = None
        self.curriculum = None
        self.exploration = None

        # Shutdown flag
        self.should_shutdown = False
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)

    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signal gracefully."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.should_shutdown = True

    def save_checkpoint(self):
        """Save checkpoint at current episode."""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_ep{self.current_episode:05d}.pt"

        # For now, just save episode number (full checkpoint implementation later)
        checkpoint = {
            'episode': self.current_episode,
            'timestamp': time.time(),
        }

        # Add population state if initialized
        if self.population:
            checkpoint['population_state'] = {
                'q_network': self.population.q_network.state_dict(),
                'optimizer': self.population.optimizer.state_dict(),
            }

            # Add exploration state
            if hasattr(self.population.exploration, 'checkpoint_state'):
                checkpoint['exploration_state'] = self.population.exploration.checkpoint_state()

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

        # Update system state
        self.db.set_system_state('last_checkpoint', str(checkpoint_path))

    def load_checkpoint(self) -> Optional[int]:
        """Load latest checkpoint if exists.

        Returns:
            Episode number of loaded checkpoint, or None if no checkpoint
        """
        # Find latest checkpoint
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_ep*.pt"))
        if not checkpoints:
            logger.info("No checkpoints found, starting from scratch")
            return None

        latest_checkpoint = checkpoints[-1]
        logger.info(f"Loading checkpoint: {latest_checkpoint}")

        checkpoint = torch.load(latest_checkpoint, weights_only=False)
        self.current_episode = checkpoint['episode']

        # Load population state if present
        if 'population_state' in checkpoint and self.population:
            self.population.q_network.load_state_dict(checkpoint['population_state']['q_network'])
            self.population.optimizer.load_state_dict(checkpoint['population_state']['optimizer'])

        # Load exploration state if present
        if 'exploration_state' in checkpoint and self.population:
            if hasattr(self.population.exploration, 'load_state'):
                self.population.exploration.load_state(checkpoint['exploration_state'])

        logger.info(f"Resumed from episode {self.current_episode}")
        return self.current_episode

    def run(self):
        """Run demo training loop."""
        logger.info(f"Starting demo runner: {self.max_episodes} episodes")
        logger.info(f"Database: {self.db_path}")
        logger.info(f"Checkpoints: {self.checkpoint_dir}")

        # Initialize training components
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create curriculum
        self.curriculum = AdversarialCurriculum(
            max_steps_per_episode=500,
            survival_advance_threshold=0.7,
            survival_retreat_threshold=0.3,
            entropy_gate=0.5,
            min_steps_at_stage=1000,
            device=device,
        )

        # Create exploration
        self.exploration = AdaptiveIntrinsicExploration(
            obs_dim=70,
            embed_dim=128,
            initial_intrinsic_weight=1.0,
            variance_threshold=10.0,
            survival_window=100,
            device=device,
        )

        # Create population
        num_agents = 1  # Single agent for Phase 3.5
        self.population = VectorizedPopulation(
            num_agents=num_agents,
            state_dim=70,
            action_dim=5,
            grid_size=8,
            curriculum=self.curriculum,
            exploration=self.exploration,
            replay_buffer_capacity=10000,
            device=device,
        )

        self.curriculum.initialize_population(num_agents)

        # Create environment
        self.env = VectorizedHamletEnv(num_agents=num_agents, grid_size=8, device=device)

        # Try to resume from checkpoint
        loaded_episode = self.load_checkpoint()
        if loaded_episode is not None:
            self.current_episode = loaded_episode + 1

        # Mark training started
        self.db.set_system_state('training_status', 'running')
        self.db.set_system_state('start_time', str(time.time()))

        # Training loop
        try:
            while self.current_episode < self.max_episodes and not self.should_shutdown:
                episode_start = time.time()

                # Reset environment
                self.env.reset()
                self.population.reset(self.env)

                # Run episode
                survival_time = 0
                episode_reward = 0.0
                max_steps = 500

                for step in range(max_steps):
                    agent_state = self.population.step_population(self.env)
                    self.population.update_curriculum_tracker(agent_state.rewards, agent_state.dones)

                    survival_time += 1
                    episode_reward += agent_state.rewards[0].item()

                    if agent_state.dones[0]:
                        break

                # Log metrics to database
                self.db.insert_episode(
                    episode_id=self.current_episode,
                    timestamp=time.time(),
                    survival_time=survival_time,
                    total_reward=episode_reward,
                    extrinsic_reward=0.0,  # TODO: track separately
                    intrinsic_reward=0.0,  # TODO: track separately
                    intrinsic_weight=self.exploration.get_intrinsic_weight(),
                    curriculum_stage=self.curriculum.tracker.agent_stages[0].item(),
                    epsilon=self.exploration.rnd.epsilon,
                )

                # Heartbeat log every 10 episodes
                if self.current_episode % 10 == 0:
                    elapsed = time.time() - episode_start
                    logger.info(
                        f"Episode {self.current_episode}/{self.max_episodes} | "
                        f"Survival: {survival_time} steps | "
                        f"Reward: {episode_reward:.2f} | "
                        f"Intrinsic Weight: {self.exploration.get_intrinsic_weight():.3f} | "
                        f"Stage: {self.curriculum.tracker.agent_stages[0].item()}/5 | "
                        f"Time: {elapsed:.2f}s"
                    )

                # Checkpoint every 100 episodes
                if self.current_episode % 100 == 0:
                    self.save_checkpoint()

                self.current_episode += 1

        finally:
            # Save final checkpoint
            logger.info("Training complete, saving final checkpoint...")
            self.save_checkpoint()
            self.db.set_system_state('training_status', 'completed')
            self.db.close()


if __name__ == '__main__':
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
    )

    # Get paths from environment or args
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/townlet/sparse_adaptive.yaml"
    db_path = sys.argv[2] if len(sys.argv) > 2 else "demo_state.db"
    checkpoint_dir = sys.argv[3] if len(sys.argv) > 3 else "checkpoints"

    runner = DemoRunner(
        config_path=config_path,
        db_path=db_path,
        checkpoint_dir=checkpoint_dir,
    )
    runner.run()
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_demo/test_runner.py -xvs`

Expected: PASS (all 2 tests)

**Step 5: Commit**

```bash
git add src/hamlet/demo/runner.py tests/test_demo/test_runner.py
git commit -m "feat(demo): add demo runner with checkpointing"
```

---

## Task 3: Generalization Test (Affordance Randomization)

**Files:**
- Modify: `src/townlet/environment/vectorized_env.py` (add randomize_affordance_positions method)
- Modify: `src/hamlet/demo/runner.py` (add generalization test at episode 5000)
- Create: `tests/test_environment/test_affordance_randomization.py`

**Step 1: Write the failing test**

Create `tests/test_environment/test_affordance_randomization.py`:
```python
"""Tests for affordance randomization (generalization test)."""

import pytest
import torch
from townlet.environment.vectorized_env import VectorizedHamletEnv


def test_randomize_affordance_positions():
    """Should randomize affordance positions while maintaining validity."""
    env = VectorizedHamletEnv(num_agents=1, grid_size=8, device=torch.device('cpu'))

    # Get initial positions
    initial_positions = env.get_affordance_positions()

    # Randomize
    env.randomize_affordance_positions()

    # Get new positions
    new_positions = env.get_affordance_positions()

    # Verify positions changed
    assert initial_positions != new_positions

    # Verify all affordances still exist
    assert set(initial_positions.keys()) == set(new_positions.keys())

    # Verify positions are valid (within grid)
    for name, pos in new_positions.items():
        assert 0 <= pos[0] < 8
        assert 0 <= pos[1] < 8


def test_get_affordance_positions():
    """Should return dict of affordance positions."""
    env = VectorizedHamletEnv(num_agents=1, grid_size=8, device=torch.device('cpu'))

    positions = env.get_affordance_positions()

    # Should have 4 affordances
    assert len(positions) == 4
    assert 'Bed' in positions
    assert 'Shower' in positions
    assert 'Fridge' in positions
    assert 'Job' in positions

    # Each position should be (x, y) tuple
    for name, pos in positions.items():
        assert len(pos) == 2
        assert isinstance(pos[0], int)
        assert isinstance(pos[1], int)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_environment/test_affordance_randomization.py -xvs`

Expected: FAIL with "AttributeError: 'VectorizedHamletEnv' object has no attribute 'get_affordance_positions'"

**Step 3: Implement affordance randomization**

Modify `src/townlet/environment/vectorized_env.py`, add these methods to the VectorizedHamletEnv class:

```python
def get_affordance_positions(self) -> dict[str, tuple[int, int]]:
    """Get current affordance positions.

    Returns:
        Dictionary mapping affordance names to (x, y) positions
    """
    positions = {}
    for affordance in self.affordances:
        # Convert tensor position to tuple
        pos = affordance.position.cpu().tolist()
        positions[affordance.name] = (int(pos[0]), int(pos[1]))
    return positions

def randomize_affordance_positions(self):
    """Randomize affordance positions for generalization testing.

    Ensures no two affordances occupy the same position.
    """
    import random

    # Generate list of all grid positions
    all_positions = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size)]

    # Shuffle and assign to affordances
    random.shuffle(all_positions)

    for i, affordance in enumerate(self.affordances):
        new_pos = all_positions[i]
        affordance.position = torch.tensor(new_pos, dtype=torch.long, device=self.device)

    # Clear grid and rebuild
    self.grid.fill_(0)
    for affordance in self.affordances:
        x, y = affordance.position[0].item(), affordance.position[1].item()
        self.grid[:, y, x] = 1  # Mark affordance position
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_environment/test_affordance_randomization.py -xvs`

Expected: PASS (all 2 tests)

**Step 5: Add generalization test to demo runner**

Modify `src/hamlet/demo/runner.py`, in the training loop, add after the episode loop starts:

```python
# In run() method, after "while self.current_episode < self.max_episodes":

                # Generalization test at episode 5000
                if self.current_episode == 5000:
                    logger.info("=" * 60)
                    logger.info("GENERALIZATION TEST: Randomizing affordance positions")
                    logger.info("=" * 60)

                    # Store old positions
                    old_positions = self.env.get_affordance_positions()
                    logger.info(f"Old positions: {old_positions}")

                    # Randomize
                    self.env.randomize_affordance_positions()

                    # Store new positions
                    new_positions = self.env.get_affordance_positions()
                    logger.info(f"New positions: {new_positions}")

                    # Mark in database
                    import json
                    self.db.set_system_state('affordance_randomization_episode', '5000')
                    self.db.set_system_state('old_affordance_positions', json.dumps(old_positions))
                    self.db.set_system_state('new_affordance_positions', json.dumps(new_positions))
```

**Step 6: Commit**

```bash
git add src/townlet/environment/vectorized_env.py src/hamlet/demo/runner.py tests/test_environment/test_affordance_randomization.py
git commit -m "feat(demo): add affordance randomization for generalization test"
```

---

## Task 4: Systemd Service Configuration

**Files:**
- Create: `deploy/hamlet-demo.service`
- Create: `deploy/install-service.sh`
- Create: `docs/DEPLOYMENT.md`

**Step 1: Create systemd service file**

Create `deploy/hamlet-demo.service`:
```ini
[Unit]
Description=Hamlet Multi-Day Demo Training
After=network.target

[Service]
Type=simple
User=%USER%
WorkingDirectory=%WORKDIR%
ExecStart=%VENV_PYTHON% -m hamlet.demo.runner %CONFIG% %DB_PATH% %CHECKPOINT_DIR%
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

# Environment variables (will be substituted by install script)
Environment="HAMLET_CONFIG=%CONFIG%"
Environment="HAMLET_DB_PATH=%DB_PATH%"
Environment="HAMLET_CHECKPOINT_DIR=%CHECKPOINT_DIR%"

[Install]
WantedBy=multi-user.target
```

**Step 2: Create installation script**

Create `deploy/install-service.sh`:
```bash
#!/bin/bash
set -e

# Installation script for hamlet-demo systemd service

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Default values
USER="${USER:-$USER}"
WORKDIR="${WORKDIR:-$PROJECT_DIR}"
VENV_PYTHON="${VENV_PYTHON:-$PROJECT_DIR/.venv/bin/python}"
CONFIG="${CONFIG:-$PROJECT_DIR/configs/townlet/sparse_adaptive.yaml}"
DB_PATH="${DB_PATH:-$PROJECT_DIR/demo_state.db}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$PROJECT_DIR/checkpoints}"

echo "Installing hamlet-demo.service with:"
echo "  User: $USER"
echo "  WorkingDirectory: $WORKDIR"
echo "  Python: $VENV_PYTHON"
echo "  Config: $CONFIG"
echo "  Database: $DB_PATH"
echo "  Checkpoints: $CHECKPOINT_DIR"

# Create service file with substitutions
SERVICE_FILE="/tmp/hamlet-demo.service"
sed -e "s|%USER%|$USER|g" \
    -e "s|%WORKDIR%|$WORKDIR|g" \
    -e "s|%VENV_PYTHON%|$VENV_PYTHON|g" \
    -e "s|%CONFIG%|$CONFIG|g" \
    -e "s|%DB_PATH%|$DB_PATH|g" \
    -e "s|%CHECKPOINT_DIR%|$CHECKPOINT_DIR|g" \
    "$SCRIPT_DIR/hamlet-demo.service" > "$SERVICE_FILE"

# Install service
sudo cp "$SERVICE_FILE" /etc/systemd/system/hamlet-demo.service
sudo systemctl daemon-reload

echo ""
echo "Service installed! Commands:"
echo "  sudo systemctl start hamlet-demo    # Start training"
echo "  sudo systemctl stop hamlet-demo     # Stop training"
echo "  sudo systemctl status hamlet-demo   # Check status"
echo "  sudo journalctl -u hamlet-demo -f   # View logs"
echo ""
echo "To enable auto-start on boot:"
echo "  sudo systemctl enable hamlet-demo"
```

**Step 3: Create deployment documentation**

Create `docs/DEPLOYMENT.md`:
```markdown
# Multi-Day Demo Deployment Guide

## Prerequisites

- Ubuntu 24.04 (or compatible Linux with systemd)
- Python 3.11+
- uv package manager
- GPU (optional but recommended)

## Installation

### 1. Install Dependencies

\`\`\`bash
cd /path/to/hamlet
uv sync
\`\`\`

### 2. Install systemd Service

\`\`\`bash
chmod +x deploy/install-service.sh
./deploy/install-service.sh
\`\`\`

This installs the `hamlet-demo` service with auto-restart on failure.

### 3. Start Training

\`\`\`bash
sudo systemctl start hamlet-demo
\`\`\`

### 4. Monitor Progress

\`\`\`bash
# View live logs
sudo journalctl -u hamlet-demo -f

# Check status
sudo systemctl status hamlet-demo

# Query database
sqlite3 demo_state.db "SELECT COUNT(*) FROM episodes"
\`\`\`

## Starting Visualization

In a separate terminal:

\`\`\`bash
# Terminal 1: Visualization server (TODO: implement in Task 4)
python -m hamlet.demo.viz_server

# Terminal 2: Frontend
cd frontend && npm run dev
\`\`\`

Open browser to `http://localhost:5173`

## Stopping the Demo

\`\`\`bash
# Graceful shutdown (saves checkpoint)
sudo systemctl stop hamlet-demo

# Check final status
sqlite3 demo_state.db "SELECT episode_id, survival_time FROM episodes ORDER BY episode_id DESC LIMIT 10"
\`\`\`

## Troubleshooting

**Training not starting:**
\`\`\`bash
sudo journalctl -u hamlet-demo --since "5 minutes ago"
\`\`\`

**Database locked:**
- Ensure only one training process is running
- Check for stale lock files: `ls -la demo_state.db*`

**Out of disk space:**
\`\`\`bash
du -sh checkpoints/  # Check checkpoint size
df -h                # Check disk usage
\`\`\`
```

**Step 4: Make install script executable and test syntax**

Run: `chmod +x deploy/install-service.sh && bash -n deploy/install-service.sh`

Expected: No syntax errors

**Step 5: Commit**

```bash
git add deploy/ docs/DEPLOYMENT.md
git commit -m "feat(deploy): add systemd service and deployment docs"
```

---

## Task 5: Visualization Server (Streaming from SQLite)

**Files:**
- Create: `src/hamlet/demo/viz_server.py`
- Modify: `src/hamlet/web/websocket.py` (extend for demo)
- Create: `tests/test_demo/test_viz_server.py`

**Note:** This task is more integration-focused, so TDD is lighter.

**Step 1: Create viz server**

Create `src/hamlet/demo/viz_server.py`:
```python
"""Visualization server for multi-day demo."""

import asyncio
import logging
from pathlib import Path
from typing import Optional
import json

from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from hamlet.demo.database import DemoDatabase

logger = logging.getLogger(__name__)

app = FastAPI(title="Hamlet Demo Visualization")


class VizServer:
    """Streams demo state from SQLite to browsers via WebSocket."""

    def __init__(
        self,
        db_path: Path | str,
        frontend_dir: Path | str,
        port: int = 8765,
    ):
        """Initialize visualization server.

        Args:
            db_path: Path to demo SQLite database
            frontend_dir: Path to built frontend assets
            port: WebSocket port
        """
        self.db_path = Path(db_path)
        self.frontend_dir = Path(frontend_dir)
        self.port = port

        self.db = DemoDatabase(db_path)
        self.clients = set()

        # Mount static files
        if self.frontend_dir.exists():
            app.mount("/assets", StaticFiles(directory=self.frontend_dir / "assets"), name="assets")

    @app.get("/")
    async def serve_index():
        """Serve frontend index.html."""
        index_path = Path(self.frontend_dir) / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
        return {"error": "Frontend not built. Run: cd frontend && npm run build"}

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket endpoint for streaming updates."""
        await websocket.accept()
        self.clients.add(websocket)

        try:
            while True:
                # Send update every 1 second
                await self.broadcast_update()
                await asyncio.sleep(1)
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            self.clients.remove(websocket)

    async def broadcast_update(self):
        """Query database and broadcast to all clients."""
        try:
            # Get latest episode
            episodes = self.db.get_latest_episodes(limit=1)
            if not episodes:
                return

            latest = episodes[0]

            # Build update message
            update = {
                'type': 'state_update',
                'episode': latest['episode_id'],
                'timestamp': latest['timestamp'],
                'metrics': {
                    'survival_time': latest['survival_time'],
                    'total_reward': latest['total_reward'],
                    'extrinsic_reward': latest['extrinsic_reward'],
                    'intrinsic_reward': latest['intrinsic_reward'],
                    'intrinsic_weight': latest['intrinsic_weight'],
                    'curriculum_stage': latest['curriculum_stage'],
                    'epsilon': latest['epsilon'],
                },
                # TODO: Add position_heatmap and affordance_graph
            }

            # Broadcast to all clients
            for client in list(self.clients):
                try:
                    await client.send_json(update)
                except Exception as e:
                    logger.error(f"Failed to send to client: {e}")
                    self.clients.remove(client)

        except Exception as e:
            logger.error(f"Error broadcasting update: {e}")


def run_server(
    db_path: str = "demo_state.db",
    frontend_dir: str = "frontend/dist",
    port: int = 8765,
):
    """Run visualization server."""
    import uvicorn

    logging.basicConfig(level=logging.INFO)

    # Initialize server
    server = VizServer(db_path, frontend_dir, port)

    logger.info(f"Starting visualization server on port {port}")
    logger.info(f"Database: {db_path}")
    logger.info(f"Frontend: {frontend_dir}")

    # Run FastAPI
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == '__main__':
    import sys
    run_server(
        db_path=sys.argv[1] if len(sys.argv) > 1 else "demo_state.db",
        frontend_dir=sys.argv[2] if len(sys.argv) > 2 else "frontend/dist",
        port=int(sys.argv[3]) if len(sys.argv) > 3 else 8765,
    )
```

**Step 2: Test manually (no automated test for now)**

This requires running demo to generate data. Manual test after implementation complete.

**Step 3: Commit**

```bash
git add src/hamlet/demo/viz_server.py
git commit -m "feat(demo): add visualization server streaming from SQLite"
```

---

## Task 6: Snapshot Daemon (Screenshots, GIFs, CSVs)

**Files:**
- Create: `src/hamlet/demo/snapshot_daemon.py`

**Note:** This is primarily integration code, light on TDD.

**Step 1: Create snapshot daemon**

Create `src/hamlet/demo/snapshot_daemon.py`:
```python
"""Snapshot daemon for capturing demo visualizations."""

import asyncio
import logging
import time
from pathlib import Path
from datetime import datetime
import subprocess

from hamlet.demo.database import DemoDatabase

logger = logging.getLogger(__name__)


class SnapshotDaemon:
    """Periodically captures screenshots, GIFs, and CSV exports."""

    def __init__(
        self,
        db_path: Path | str,
        output_dir: Path | str,
        browser_url: str = "http://localhost:5173",
    ):
        """Initialize snapshot daemon.

        Args:
            db_path: Path to demo database
            output_dir: Output directory for snapshots
            browser_url: URL to screenshot
        """
        self.db_path = Path(db_path)
        self.output_dir = Path(output_dir)
        self.browser_url = browser_url

        # Create output directories
        (self.output_dir / "screenshots").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "gifs").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "exports").mkdir(parents=True, exist_ok=True)

        self.db = DemoDatabase(db_path)

        self.last_screenshot_time = 0
        self.last_gif_episode = 0
        self.last_csv_export_time = 0

    async def run(self):
        """Main daemon loop."""
        logger.info(f"Starting snapshot daemon")
        logger.info(f"Database: {self.db_path}")
        logger.info(f"Output: {self.output_dir}")
        logger.info(f"Browser: {self.browser_url}")

        while True:
            try:
                current_time = time.time()

                # Screenshot every 5 minutes
                if current_time - self.last_screenshot_time > 300:
                    await self.capture_screenshot()
                    self.last_screenshot_time = current_time

                # Check for GIF generation (every 50 episodes)
                if self.should_generate_gif():
                    await self.generate_heatmap_gif()

                # CSV export every hour
                if current_time - self.last_csv_export_time > 3600:
                    await self.export_metrics_csv()
                    self.last_csv_export_time = current_time

                # Check every minute
                await asyncio.sleep(60)

            except Exception as e:
                logger.error(f"Error in snapshot daemon: {e}")
                await asyncio.sleep(60)

    async def capture_screenshot(self):
        """Capture browser screenshot using headless Chrome."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / "screenshots" / f"screenshot_{timestamp}.png"

        try:
            # Use playwright for headless screenshots (requires: uv add playwright)
            # For now, use simple screencapture on macOS or import via selenium
            logger.info(f"Capturing screenshot: {output_path}")

            # TODO: Implement with playwright or selenium
            # For now, just log
            logger.warning("Screenshot capture not implemented yet (needs playwright/selenium)")

        except Exception as e:
            logger.error(f"Failed to capture screenshot: {e}")

    def should_generate_gif(self) -> bool:
        """Check if enough episodes have passed for GIF generation."""
        episodes = self.db.get_latest_episodes(limit=1)
        if not episodes:
            return False

        current_episode = episodes[0]['episode_id']

        # Generate GIF every 50 episodes
        if current_episode >= self.last_gif_episode + 50:
            return True

        return False

    async def generate_heatmap_gif(self):
        """Generate novelty heatmap GIF for last 50 episodes."""
        episodes = self.db.get_latest_episodes(limit=1)
        if not episodes:
            return

        current_episode = episodes[0]['episode_id']
        start_episode = self.last_gif_episode
        end_episode = current_episode

        output_path = self.output_dir / "gifs" / f"novelty_ep{start_episode:05d}-{end_episode:05d}.gif"

        try:
            logger.info(f"Generating novelty GIF: {output_path}")

            # TODO: Implement GIF generation from position_heatmap table
            # Query position_heatmap for episodes [start, end]
            # Generate frames
            # Compile to GIF using Pillow
            logger.warning("GIF generation not implemented yet")

            self.last_gif_episode = end_episode

        except Exception as e:
            logger.error(f"Failed to generate GIF: {e}")

    async def export_metrics_csv(self):
        """Export episodes table to CSV."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / "exports" / f"metrics_{timestamp}.csv"

        try:
            logger.info(f"Exporting metrics to CSV: {output_path}")

            # Use sqlite3 command line tool for export
            cmd = f'sqlite3 {self.db_path} ".mode csv" ".output {output_path}" "SELECT * FROM episodes"'
            subprocess.run(cmd, shell=True, check=True)

            logger.info(f"CSV export complete: {output_path}")

        except Exception as e:
            logger.error(f"Failed to export CSV: {e}")


def run_daemon(
    db_path: str = "demo_state.db",
    output_dir: str = "snapshots",
    browser_url: str = "http://localhost:5173",
):
    """Run snapshot daemon."""
    logging.basicConfig(level=logging.INFO)

    daemon = SnapshotDaemon(db_path, output_dir, browser_url)
    asyncio.run(daemon.run())


if __name__ == '__main__':
    import sys
    run_daemon(
        db_path=sys.argv[1] if len(sys.argv) > 1 else "demo_state.db",
        output_dir=sys.argv[2] if len(sys.argv) > 2 else "snapshots",
        browser_url=sys.argv[3] if len(sys.argv) > 3 else "http://localhost:5173",
    )
```

**Step 2: Commit**

```bash
git add src/hamlet/demo/snapshot_daemon.py
git commit -m "feat(demo): add snapshot daemon for screenshots/GIFs/CSVs"
```

---

## Task 7: Frontend AffordanceGraph Component

**Files:**
- Create: `frontend/src/components/AffordanceGraph.vue`
- Modify: `frontend/src/App.vue` (integrate component)

**Step 1: Create AffordanceGraph component**

Create `frontend/src/components/AffordanceGraph.vue`:
```vue
<template>
  <div class="affordance-graph">
    <h4>Learned Routines (Affordance Transitions)</h4>
    <svg :width="width" :height="height" v-if="hasData">
      <!-- Nodes (affordances) -->
      <g v-for="(node, name) in nodes" :key="name">
        <circle
          :cx="node.x"
          :cy="node.y"
          :r="30"
          :fill="getNodeColor(name)"
          stroke="#333"
          stroke-width="2"
        />
        <text
          :x="node.x"
          :y="node.y + 5"
          text-anchor="middle"
          font-size="12"
          fill="white"
          font-weight="bold"
        >
          {{ name }}
        </text>
      </g>

      <!-- Edges (transitions) -->
      <g v-for="(edge, idx) in edges" :key="idx">
        <line
          :x1="edge.x1"
          :y1="edge.y1"
          :x2="edge.x2"
          :y2="edge.y2"
          :stroke="edge.color"
          :stroke-width="edge.width"
          stroke-opacity="0.6"
          marker-end="url(#arrowhead)"
        />
      </g>

      <!-- Arrow marker definition -->
      <defs>
        <marker
          id="arrowhead"
          markerWidth="10"
          markerHeight="10"
          refX="8"
          refY="3"
          orient="auto"
        >
          <polygon points="0 0, 10 3, 0 6" fill="#666" />
        </marker>
      </defs>
    </svg>
    <div v-else class="no-data">
      No transition data yet...
    </div>
  </div>
</template>

<script>
export default {
  name: 'AffordanceGraph',
  props: {
    transitionData: {
      type: Object,
      default: () => ({}),
    },
    width: {
      type: Number,
      default: 400,
    },
    height: {
      type: Number,
      default: 300,
    },
  },
  computed: {
    hasData() {
      return Object.keys(this.transitionData).length > 0
    },
    nodes() {
      // Position nodes in a circle
      const names = ['Bed', 'Job', 'Shower', 'Fridge']
      const centerX = this.width / 2
      const centerY = this.height / 2
      const radius = Math.min(this.width, this.height) / 3

      const nodes = {}
      names.forEach((name, idx) => {
        const angle = (idx / names.length) * 2 * Math.PI - Math.PI / 2
        nodes[name] = {
          x: centerX + radius * Math.cos(angle),
          y: centerY + radius * Math.sin(angle),
        }
      })

      return nodes
    },
    edges() {
      const edges = []

      // Build edges from transition data
      for (const [from, toDict] of Object.entries(this.transitionData)) {
        for (const [to, count] of Object.entries(toDict)) {
          if (from === to) continue // Skip self-loops

          const fromNode = this.nodes[from]
          const toNode = this.nodes[to]

          if (!fromNode || !toNode) continue

          // Edge thickness based on count (log scale)
          const width = Math.log(count + 1) * 2 + 1

          // Edge color based on recency (for now, just use count)
          const color = count > 10 ? '#10b981' : '#3b82f6'

          edges.push({
            x1: fromNode.x,
            y1: fromNode.y,
            x2: toNode.x,
            y2: toNode.y,
            width,
            color,
          })
        }
      }

      return edges
    },
  },
  methods: {
    getNodeColor(name) {
      const colors = {
        'Bed': '#8b5cf6',
        'Job': '#f59e0b',
        'Shower': '#3b82f6',
        'Fridge': '#10b981',
      }
      return colors[name] || '#666'
    },
  },
}
</script>

<style scoped>
.affordance-graph {
  margin: 10px 0;
  padding: 10px;
  border: 1px solid #ccc;
  border-radius: 4px;
  background: #f9f9f9;
}

h4 {
  margin: 0 0 10px 0;
  font-size: 14px;
  color: #333;
}

.no-data {
  text-align: center;
  padding: 40px;
  color: #999;
  font-style: italic;
}
</style>
```

**Step 2: Modify App.vue to integrate**

Modify `frontend/src/App.vue`, add import and component:

```vue
<script>
import AffordanceGraph from './components/AffordanceGraph.vue'

export default {
  components: {
    // ... existing components
    AffordanceGraph,
  },
  // ... rest of component
}
</script>

<template>
  <div class="app">
    <!-- ... existing components -->

    <!-- Add AffordanceGraph in side panel or bottom -->
    <AffordanceGraph
      v-if="transitionData"
      :transition-data="transitionData"
      :width="400"
      :height="300"
    />
  </div>
</template>
```

**Step 3: Commit**

```bash
git add frontend/src/components/AffordanceGraph.vue frontend/src/App.vue
git commit -m "feat(viz): add affordance transition graph component"
```

---

## Task 8: End-to-End Integration Test

**Files:**
- Create: `tests/test_demo/test_integration.py`

**Step 1: Write integration test**

Create `tests/test_demo/test_integration.py`:
```python
"""End-to-end integration test for multi-day demo."""

import pytest
import tempfile
import time
from pathlib import Path

from hamlet.demo.runner import DemoRunner
from hamlet.demo.database import DemoDatabase


@pytest.mark.slow
def test_demo_integration_100_episodes():
    """Run 100 episodes end-to-end and verify system works."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        config_path = Path(__file__).parent.parent.parent / "configs" / "townlet" / "sparse_adaptive.yaml"
        db_path = tmpdir / "demo.db"
        checkpoint_dir = tmpdir / "checkpoints"

        # Create runner
        runner = DemoRunner(
            config_path=config_path,
            db_path=db_path,
            checkpoint_dir=checkpoint_dir,
            max_episodes=100,
        )

        # Run demo (will take several minutes)
        start_time = time.time()
        runner.run()
        elapsed = time.time() - start_time

        # Verify database has episodes
        db = DemoDatabase(db_path)
        episodes = db.get_latest_episodes(limit=100)

        assert len(episodes) == 100, f"Expected 100 episodes, got {len(episodes)}"

        # Verify metrics are reasonable
        for ep in episodes:
            assert ep['survival_time'] > 0
            assert 0.0 <= ep['intrinsic_weight'] <= 1.0
            assert 1 <= ep['curriculum_stage'] <= 5
            assert 0.0 <= ep['epsilon'] <= 1.0

        # Verify checkpoint was created
        checkpoints = list(checkpoint_dir.glob("checkpoint_ep*.pt"))
        assert len(checkpoints) >= 1, "At least one checkpoint should exist"

        # Verify system state
        status = db.get_system_state('training_status')
        assert status == 'completed'

        print(f"\n✅ Integration test passed!")
        print(f"   Episodes: {len(episodes)}")
        print(f"   Time: {elapsed:.1f}s ({elapsed/100:.2f}s per episode)")
        print(f"   Checkpoints: {len(checkpoints)}")
        print(f"   Final survival: {episodes[0]['survival_time']} steps")
        print(f"   Final intrinsic weight: {episodes[0]['intrinsic_weight']:.3f}")
```

**Step 2: Run integration test**

Run: `uv run pytest tests/test_demo/test_integration.py -xvs -m slow`

Expected: PASS (takes ~5-10 minutes for 100 episodes)

**Step 3: Commit**

```bash
git add tests/test_demo/test_integration.py
git commit -m "test(demo): add end-to-end integration test"
```

---

## Verification

After completing all 8 tasks, verify the system:

```bash
# 1. Run all demo tests
uv run pytest tests/test_demo/ -v

# 2. Run integration test
uv run pytest tests/test_demo/test_integration.py -m slow -xvs

# 3. Manually test demo runner (short run)
python -m hamlet.demo.runner configs/townlet/sparse_adaptive.yaml demo_test.db checkpoints_test

# Stop after a few episodes with Ctrl+C, verify it saved checkpoint

# 4. Test checkpoint resume
python -m hamlet.demo.runner configs/townlet/sparse_adaptive.yaml demo_test.db checkpoints_test

# Should resume from last episode

# 5. Inspect database
sqlite3 demo_test.db "SELECT * FROM episodes LIMIT 5"
sqlite3 demo_test.db "SELECT * FROM system_state"

# 6. Clean up test files
rm -rf demo_test.db checkpoints_test/
```

**Exit Criteria Checklist:**
- [ ] All unit tests pass (database, runner, affordance randomization)
- [ ] Integration test completes 100 episodes successfully
- [ ] Checkpoints save and load correctly
- [ ] Database captures all metrics
- [ ] Generalization test triggers at episode 5000
- [ ] systemd service installs without errors
- [ ] Visualization server starts (manual test)
- [ ] Snapshot daemon runs (manual test)

---

## Execution Options

Plan saved to: `docs/plans/2025-10-30-multi-day-demo-implementation.md`

**Two execution options:**

**1. Subagent-Driven (this session)**
- I dispatch fresh subagent per task
- Code review after each task
- Fast iteration, same session

**2. Parallel Session (separate)**
- Open new session with executing-plans
- Batch execution with checkpoints
- Good for long-running implementation

**Which approach would you prefer?**
