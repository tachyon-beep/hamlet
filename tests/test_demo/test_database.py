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
