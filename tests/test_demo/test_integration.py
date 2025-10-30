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
