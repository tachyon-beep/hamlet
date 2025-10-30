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
        device_str = self.config.get('training', {}).get('device', 'cuda')
        device = torch.device(device_str if torch.cuda.is_available() else 'cpu')

        # Extract config parameters
        curriculum_cfg = self.config.get('curriculum', {})
        exploration_cfg = self.config.get('exploration', {})
        population_cfg = self.config.get('population', {})
        environment_cfg = self.config.get('environment', {})

        # Get environment parameters from config
        num_agents = population_cfg.get('num_agents', 1)
        grid_size = environment_cfg.get('grid_size', 8)
        partial_observability = environment_cfg.get('partial_observability', False)
        vision_range = environment_cfg.get('vision_range', 2)

        # Create environment FIRST (need it to auto-detect dimensions)
        self.env = VectorizedHamletEnv(
            num_agents=num_agents,
            grid_size=grid_size,
            device=device,
            partial_observability=partial_observability,
            vision_range=vision_range,
        )

        # Auto-detect dimensions from environment (avoids hardcoded config values)
        obs_dim = self.env.observation_dim
        action_dim = self.env.action_dim

        # Create curriculum
        self.curriculum = AdversarialCurriculum(
            max_steps_per_episode=curriculum_cfg.get('max_steps_per_episode', 500),
            survival_advance_threshold=curriculum_cfg.get('survival_advance_threshold', 0.7),
            survival_retreat_threshold=curriculum_cfg.get('survival_retreat_threshold', 0.3),
            entropy_gate=curriculum_cfg.get('entropy_gate', 0.5),
            min_steps_at_stage=curriculum_cfg.get('min_steps_at_stage', 1000),
            device=device,
        )

        # Create exploration (use auto-detected obs_dim)
        self.exploration = AdaptiveIntrinsicExploration(
            obs_dim=obs_dim,
            embed_dim=exploration_cfg.get('embed_dim', 128),
            initial_intrinsic_weight=exploration_cfg.get('initial_intrinsic_weight', 1.0),
            variance_threshold=exploration_cfg.get('variance_threshold', 100.0),  # Increased from 10.0
            survival_window=exploration_cfg.get('survival_window', 100),
            device=device,
        )

        # Get population parameters from config
        learning_rate = population_cfg.get('learning_rate', 0.00025)
        gamma = population_cfg.get('gamma', 0.99)
        replay_buffer_capacity = population_cfg.get('replay_buffer_capacity', 10000)
        network_type = population_cfg.get('network_type', 'simple')  # 'simple' or 'recurrent'
        vision_window_size = 2 * vision_range + 1  # 5 for vision_range=2

        # Create agent IDs
        agent_ids = [f"agent_{i}" for i in range(num_agents)]

        # Create population with correct API
        self.population = VectorizedPopulation(
            env=self.env,
            curriculum=self.curriculum,
            exploration=self.exploration,
            agent_ids=agent_ids,
            device=device,
            obs_dim=obs_dim,
            action_dim=action_dim,
            learning_rate=learning_rate,
            gamma=gamma,
            replay_buffer_capacity=replay_buffer_capacity,
            network_type=network_type,
            vision_window_size=vision_window_size,
        )

        self.curriculum.initialize_population(num_agents)

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

                # Reset environment and population
                self.env.reset()
                self.population.reset()  # No argument

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
    max_episodes = int(sys.argv[4]) if len(sys.argv) > 4 else 10000

    runner = DemoRunner(
        config_path=config_path,
        db_path=db_path,
        checkpoint_dir=checkpoint_dir,
        max_episodes=max_episodes,
    )
    runner.run()
