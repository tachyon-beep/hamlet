"""Demo runner for multi-day training."""

from __future__ import annotations

import logging
import signal
import time
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import yaml

from townlet.config import HamletConfig
from townlet.curriculum.adversarial import AdversarialCurriculum
from townlet.demo.database import DemoDatabase
from townlet.environment.vectorized_env import VectorizedHamletEnv
from townlet.exploration.adaptive_intrinsic import AdaptiveIntrinsicExploration
from townlet.population.vectorized import VectorizedPopulation
from townlet.training.checkpoint_utils import (
    assert_checkpoint_dimensions,
    attach_universe_metadata,
    config_hash_warning,
)
from townlet.training.state import BatchedAgentState
from townlet.training.tensorboard_logger import TensorBoardLogger

if TYPE_CHECKING:
    from townlet.universe.compiled import CompiledUniverse

logger = logging.getLogger(__name__)


class DemoRunner:
    """Orchestrates multi-day demo training with checkpointing."""

    def __init__(
        self,
        config_dir: Path | str,
        db_path: Path | str,
        checkpoint_dir: Path | str,
        max_episodes: int | None = None,
        training_config_path: Path | str | None = None,
    ):
        """Initialize demo runner.

        Args:
            config_dir: Directory containing configuration pack
            db_path: Path to SQLite database
            checkpoint_dir: Directory for checkpoint files
            max_episodes: Maximum number of episodes to run (if None, reads from config YAML)
            training_config_path: Optional explicit path to training YAML file
        """
        self.config_dir = Path(config_dir)
        if training_config_path is None:
            self.training_config_path = self.config_dir / "training.yaml"
        else:
            self.training_config_path = Path(training_config_path)
        if not self.training_config_path.exists():
            raise FileNotFoundError(f"Training config not found: {self.training_config_path}")
        self.db_path = Path(db_path)
        self.checkpoint_dir = Path(checkpoint_dir)

        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Pre-flight check: detect old checkpoints (Phase 4 breaking change)
        self._validate_checkpoint_compatibility()

        # Initialize database
        self.db = DemoDatabase(self.db_path)

        # Initialize TensorBoard logger
        # If checkpoint_dir is in a structured run (e.g., runs/L1_full_observability/2025-11-02_143022/checkpoints)
        # then tensorboard should be at sibling level, not inside checkpoints
        parent_dir = self.checkpoint_dir.parent
        if parent_dir.parent.parent.name == "runs" and self.checkpoint_dir.name == "checkpoints":
            # New structure: runs/LX_name/timestamp/checkpoints -> use runs/LX_name/timestamp/tensorboard
            tb_log_dir = parent_dir / "tensorboard"
        else:
            # Old structure or direct invocation: put tensorboard inside checkpoint_dir
            tb_log_dir = self.checkpoint_dir / "tensorboard"

        self.tb_logger = TensorBoardLogger(
            log_dir=tb_log_dir,
            flush_every=10,
        )

        # Load config using HamletConfig DTO (enforces no-defaults validation)
        self.hamlet_config = HamletConfig.load(self.config_dir, training_config_path=self.training_config_path)

        # Also load raw YAML for backward compatibility and optional sections (e.g., recording)
        with open(self.training_config_path) as f:
            self.config = yaml.safe_load(f)

        # Set max_episodes: use provided value, otherwise read from config
        if max_episodes is not None:
            self.max_episodes = max_episodes
        else:
            self.max_episodes = self.hamlet_config.training.max_episodes

        self.current_episode = 0

        # Initialize components (will be created in run())
        self.env = None
        self.population = None
        self.curriculum = None
        self.exploration = None
        self.recorder = None  # Episode recorder (initialized if recording enabled)

        # Shutdown flag
        self.should_shutdown = False
        # Signal handlers only work in main thread
        try:
            signal.signal(signal.SIGTERM, self._handle_shutdown)
            signal.signal(signal.SIGINT, self._handle_shutdown)
        except ValueError:
            # Running in a worker thread (e.g., unified server)
            # Signal handling will be done by the orchestrator
            pass

    def _validate_checkpoint_compatibility(self) -> None:
        """Validate checkpoint directory doesn't contain old checkpoints.

        BREAKING CHANGE: Phase 4 changed checkpoint format.
        Old checkpoints (Version 2) will not load.

        Raises:
            ValueError: If old checkpoints detected
        """
        if not self.checkpoint_dir.exists():
            return  # No checkpoints yet (fresh start)

        checkpoint_files = list(self.checkpoint_dir.glob("*.pt"))
        if not checkpoint_files:
            return  # Empty directory (fresh start)

        # Check first checkpoint for substrate_metadata
        first_checkpoint_path = checkpoint_files[0]

        try:
            checkpoint = torch.load(first_checkpoint_path, weights_only=False)

            # Phase 4+ checkpoints have substrate_metadata
            if "substrate_metadata" not in checkpoint:
                raise ValueError(
                    f"Old checkpoints detected in {self.checkpoint_dir}.\n"
                    "\n"
                    "BREAKING CHANGE: Phase 4 changed checkpoint format.\n"
                    "Legacy checkpoints (Version 2) are no longer compatible.\n"
                    "\n"
                    "Action required:\n"
                    f"  1. Delete checkpoint directory: {self.checkpoint_dir}\n"
                    "  2. Retrain model from scratch with Phase 4+ code\n"
                    "\n"
                    "If you need to preserve old models, checkout pre-Phase 4 git commit.\n"
                    "\n"
                    f"Detected old checkpoint: {first_checkpoint_path.name}"
                )
        except Exception as e:
            # If we can't load checkpoint, let the normal loading code handle it
            # (might be corrupted, wrong format, etc.)
            if "Old checkpoints detected" in str(e):
                raise  # Re-raise our validation error
            # Otherwise ignore (will fail later during actual load)

    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signal gracefully."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.should_shutdown = True

    def _cleanup(self):
        """Internal cleanup method for resources.

        Safe to call multiple times (idempotent).
        Handles partial initialization gracefully.
        """
        # Shutdown recorder if enabled
        if hasattr(self, "recorder") and self.recorder is not None:
            logger.info("Shutting down episode recorder...")
            try:
                self.recorder.shutdown()
            except Exception as e:
                logger.warning(f"Error shutting down recorder: {e}")

        # Close database connection
        if hasattr(self, "db"):
            try:
                self.db.close()
            except Exception as e:
                logger.warning(f"Error closing database: {e}")

        # Close TensorBoard logger
        if hasattr(self, "tb_logger"):
            try:
                self.tb_logger.close()
            except Exception as e:
                logger.warning(f"Error closing TensorBoard logger: {e}")

    def __enter__(self):
        """Enter context manager - return self for 'with' statement."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager - cleanup resources.

        Args:
            exc_type: Exception type if exception occurred
            exc_val: Exception value if exception occurred
            exc_tb: Exception traceback if exception occurred

        Returns:
            False to propagate exceptions (don't suppress)
        """
        self._cleanup()
        return False  # Don't suppress exceptions

    def flush_all_agents(self):
        """Flush all agents' episodes to replay buffer before checkpoint."""
        if not self.population:
            return

        for agent_idx in range(self.population.num_agents):
            self.population.flush_episode(agent_idx)

    def save_checkpoint(self, universe: CompiledUniverse | None = None):
        """Save checkpoint at current episode."""
        # P1.1 Phase 5: Flush all agents before checkpoint
        self.flush_all_agents()

        checkpoint_path = self.checkpoint_dir / f"checkpoint_ep{self.current_episode:05d}.pt"

        # P1.1 Phase 2: Use population's complete checkpoint state
        checkpoint = {
            "version": 3,  # Phase 5: Version 3 includes substrate metadata
            "episode": self.current_episode,
            "timestamp": time.time(),
        }

        # Phase 5: Add substrate metadata for validation
        if self.env:
            checkpoint["substrate_metadata"] = {
                "position_dim": self.env.substrate.position_dim,
                "substrate_type": type(self.env.substrate).__name__,
            }

        # Add full population state (includes q_network, optimizer, replay_buffer, etc.)
        if self.population:
            checkpoint["population_state"] = self.population.get_checkpoint_state()

        # P1.1 Phase 3: Add curriculum state (agent stages, performance trackers)
        if self.curriculum:
            checkpoint["curriculum_state"] = self.curriculum.state_dict()

        # P1.1 Phase 4: Add affordance layout (grid positions)
        if self.env:
            checkpoint["affordance_layout"] = self.env.get_affordance_positions()

        # P1.1 Phase 6: Add agent_ids for multi-agent coordination
        if self.population:
            checkpoint["agent_ids"] = self.population.agent_ids

        # Add epsilon for inference server display (use population's epsilon getter for all exploration types)
        if self.population:
            checkpoint["epsilon"] = self.population._get_current_epsilon_value()

        # Persist the training configuration for provenance
        checkpoint["training_config"] = self.config
        checkpoint["config_dir"] = str(self.config_dir)

        if universe is not None:
            attach_universe_metadata(checkpoint, universe)

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

        # Update system state
        self.db.set_system_state("last_checkpoint", str(checkpoint_path))

    def load_checkpoint(self, universe: CompiledUniverse | None = None) -> int | None:
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

        if universe is not None:
            warning = config_hash_warning(checkpoint, universe)
            if warning:
                logger.warning(warning)
            assert_checkpoint_dimensions(checkpoint, universe)

        # P1.1: Check checkpoint version
        checkpoint_version = checkpoint.get("version", 1)  # Default to v1 for legacy checkpoints
        if checkpoint_version != 2:
            logger.warning(f"Loading v{checkpoint_version} checkpoint (current version is 2)")
            # Future: Add migration logic here if needed

        self.current_episode = checkpoint["episode"]

        # P1.1 Phase 2: Load full population state
        if "population_state" in checkpoint and self.population:
            self.population.load_checkpoint_state(checkpoint["population_state"])

        # P1.1 Phase 3: Load curriculum state (agent stages, performance trackers)
        if "curriculum_state" in checkpoint and self.curriculum:
            self.curriculum.load_state_dict(checkpoint["curriculum_state"])

        # P1.1 Phase 4: Load affordance layout (grid positions)
        if "affordance_layout" in checkpoint and self.env:
            self.env.set_affordance_positions(checkpoint["affordance_layout"])

        # P1.1 Phase 6: Restore agent_ids for multi-agent coordination
        if "agent_ids" in checkpoint and self.population:
            self.population.agent_ids = checkpoint["agent_ids"]

        logger.info(f"Resumed from episode {self.current_episode}")
        return self.current_episode

    def run(self):
        """Run demo training loop."""
        logger.info(f"Starting demo runner: {self.max_episodes} episodes")
        logger.info(f"Database: {self.db_path}")
        logger.info(f"Checkpoints: {self.checkpoint_dir}")

        # PDR-002: No-defaults validation done automatically by HamletConfig DTO
        # All required parameters validated at load time (lines 109-110)

        # Initialize training components
        device_str = self.hamlet_config.training.device
        device = torch.device(device_str if torch.cuda.is_available() else "cpu")
        if device_str == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")

        # Extract config parameters from DTOs (all required, validated at load time)
        num_agents = self.hamlet_config.population.num_agents
        grid_size = self.hamlet_config.environment.grid_size
        partial_observability = self.hamlet_config.environment.partial_observability
        vision_range = self.hamlet_config.environment.vision_range
        # enabled_affordances: None = all affordances (semantic meaning)
        enabled_affordances = self.hamlet_config.environment.enabled_affordances
        enable_temporal_mechanics = self.hamlet_config.environment.enable_temporal_mechanics
        move_energy_cost = self.hamlet_config.environment.energy_move_depletion
        wait_energy_cost = self.hamlet_config.environment.energy_wait_depletion
        interact_energy_cost = self.hamlet_config.environment.energy_interact_depletion

        # TODO(UAC): agent_lifespan should be in config (TASK-006: BRAIN_AS_CODE)
        # For now, use constant 1000 (standard test value)
        agent_lifespan = 1000

        # Create environment FIRST (need it to auto-detect dimensions)
        self.env = VectorizedHamletEnv(
            num_agents=num_agents,
            grid_size=grid_size,
            partial_observability=partial_observability,
            vision_range=vision_range,
            enable_temporal_mechanics=enable_temporal_mechanics,
            move_energy_cost=move_energy_cost,
            wait_energy_cost=wait_energy_cost,
            interact_energy_cost=interact_energy_cost,
            agent_lifespan=agent_lifespan,
            device=device,
            enabled_affordances=enabled_affordances,
            config_pack_path=self.config_dir,
        )

        # Auto-detect dimensions from environment (avoids hardcoded config values)
        obs_dim = self.env.observation_dim
        action_dim = self.env.action_dim

        # Create curriculum (all params required per PDR-002)
        self.curriculum = AdversarialCurriculum(
            max_steps_per_episode=self.hamlet_config.curriculum.max_steps_per_episode,
            survival_advance_threshold=self.hamlet_config.curriculum.survival_advance_threshold,
            survival_retreat_threshold=self.hamlet_config.curriculum.survival_retreat_threshold,
            entropy_gate=self.hamlet_config.curriculum.entropy_gate,
            min_steps_at_stage=self.hamlet_config.curriculum.min_steps_at_stage,
            device=device,
        )

        # Create exploration (all params required per PDR-002)
        self.exploration = AdaptiveIntrinsicExploration(
            obs_dim=obs_dim,
            embed_dim=self.hamlet_config.exploration.embed_dim,
            initial_intrinsic_weight=self.hamlet_config.exploration.initial_intrinsic_weight,
            variance_threshold=self.hamlet_config.exploration.variance_threshold,
            survival_window=self.hamlet_config.exploration.survival_window,
            epsilon_start=self.hamlet_config.training.epsilon_start,
            epsilon_decay=self.hamlet_config.training.epsilon_decay,
            epsilon_min=self.hamlet_config.training.epsilon_min,
            device=device,
        )

        # Get population parameters from config (all required per PDR-002)
        learning_rate = self.hamlet_config.population.learning_rate
        gamma = self.hamlet_config.population.gamma
        replay_buffer_capacity = self.hamlet_config.population.replay_buffer_capacity
        network_type = self.hamlet_config.population.network_type  # 'simple' or 'recurrent'
        vision_window_size = 2 * vision_range + 1  # 5 for vision_range=2

        # Get training hyperparameters from config (all required per PDR-002)
        train_frequency = self.hamlet_config.training.train_frequency
        target_update_frequency = self.hamlet_config.training.target_update_frequency
        batch_size = self.hamlet_config.training.batch_size
        sequence_length = self.hamlet_config.training.sequence_length
        max_grad_norm = self.hamlet_config.training.max_grad_norm

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
            tb_logger=self.tb_logger,
            train_frequency=train_frequency,
            target_update_frequency=target_update_frequency,
            batch_size=batch_size,
            sequence_length=sequence_length,
            max_grad_norm=max_grad_norm,
        )

        self.curriculum.initialize_population(num_agents)

        # Initialize episode recorder if enabled
        recording_cfg = self.config.get("recording", {})
        if recording_cfg.get("enabled", False):
            from townlet.recording.recorder import EpisodeRecorder

            # Create recording output directory
            recording_output_dir = self.checkpoint_dir / recording_cfg.get("output_dir", "recordings")
            recording_output_dir.mkdir(parents=True, exist_ok=True)

            self.recorder = EpisodeRecorder(
                config=recording_cfg,
                output_dir=recording_output_dir,
                database=self.db,
                curriculum=self.curriculum,
            )
            logger.info(f"Episode recording enabled: {recording_output_dir}")
        else:
            logger.info("Episode recording disabled")

        # Try to resume from checkpoint
        loaded_episode = self.load_checkpoint()
        if loaded_episode is not None:
            self.current_episode = loaded_episode + 1

        # Phase 4 - Log hyperparameters to TensorBoard
        self.hparams = {
            "learning_rate": learning_rate,
            "gamma": gamma,
            "network_type": network_type,
            "replay_buffer_capacity": replay_buffer_capacity,
            "grid_size": grid_size,
            "partial_observability": partial_observability,
            "vision_range": vision_range,
            "enable_temporal": enable_temporal_mechanics,
            "initial_intrinsic_weight": self.hamlet_config.exploration.initial_intrinsic_weight,
            "variance_threshold": self.hamlet_config.exploration.variance_threshold,
            "max_steps_per_episode": self.hamlet_config.curriculum.max_steps_per_episode,
        }
        # Note: final metrics will be logged at end of training
        self.tb_logger.log_hyperparameters(hparams=self.hparams, metrics={})

        # Mark training started
        self.db.set_system_state("training_status", "running")
        self.db.set_system_state("start_time", str(time.time()))

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

                    self.db.set_system_state("affordance_randomization_episode", "5000")
                    self.db.set_system_state("old_affordance_positions", json.dumps(old_positions))
                    self.db.set_system_state("new_affordance_positions", json.dumps(new_positions))

                # Reset environment and population
                # NOTE: Only call population.reset(), which internally calls env.reset()
                # Calling both causes double reset which breaks affordance randomization
                self.population.reset()

                # Run episode
                num_agents = self.population.num_agents
                max_steps = self.hamlet_config.curriculum.max_steps_per_episode
                episode_reward = torch.zeros(num_agents, device=self.env.device)
                episode_extrinsic_reward = torch.zeros(num_agents, device=self.env.device)
                episode_intrinsic_reward = torch.zeros(num_agents, device=self.env.device)
                final_meters = [None for _ in range(num_agents)]
                affordance_visits = [defaultdict(int) for _ in range(num_agents)]
                # NEW: Transition tracking
                affordance_transitions = [defaultdict(lambda: defaultdict(int)) for _ in range(num_agents)]
                last_affordance = [None for _ in range(num_agents)]
                last_agent_state: BatchedAgentState | None = None

                for step in range(max_steps):
                    agent_state = self.population.step_population(self.env)
                    last_agent_state = agent_state

                    if self.curriculum.transition_events:
                        self.tb_logger.log_curriculum_transitions(
                            episode=self.current_episode,
                            events=list(self.curriculum.transition_events),
                        )
                        # Log curriculum transitions to console
                        for event in self.curriculum.transition_events:
                            logger.info(
                                f"🎓 Curriculum {event['type'].upper()}: Stage {event['from_stage']} → {event['to_stage']} "
                                f"(Survival: {event['survival_rate']:.1%}, Entropy: {event['entropy']:.3f})"
                            )
                        self.curriculum.transition_events.clear()

                    # Note: agent_state.rewards contains COMBINED rewards (extrinsic + weighted intrinsic)
                    # We need to extract pure extrinsic for separate logging
                    intrinsic_weight = self.exploration.get_intrinsic_weight() if hasattr(self.exploration, "get_intrinsic_weight") else 1.0
                    weighted_intrinsic = agent_state.intrinsic_rewards * intrinsic_weight
                    extrinsic_only = agent_state.rewards - weighted_intrinsic

                    # Accumulate rewards for episode totals
                    episode_reward += agent_state.rewards  # Combined (what agent actually receives)
                    episode_extrinsic_reward += extrinsic_only  # Pure extrinsic (for analysis)
                    episode_intrinsic_reward += agent_state.intrinsic_rewards  # Unweighted intrinsic (for analysis)

                    if "successful_interactions" in agent_state.info:
                        for agent_idx, affordance_name in agent_state.info["successful_interactions"].items():
                            if 0 <= agent_idx < num_agents:
                                # Existing: count tracking
                                affordance_visits[agent_idx][affordance_name] += 1

                                # NEW: transition tracking
                                prev = last_affordance[agent_idx]
                                if prev is not None:
                                    # Record transition: prev → current
                                    affordance_transitions[agent_idx][prev][affordance_name] += 1

                                # Update last affordance for next transition
                                last_affordance[agent_idx] = affordance_name

                    for idx in range(num_agents):
                        if agent_state.dones[idx] and final_meters[idx] is None:
                            final_meters[idx] = self.env.meters[idx].detach().cpu()

                    # Record step if recording enabled (only agent 0 for now)
                    if self.recorder is not None:
                        # Get temporal mechanics fields if available
                        time_of_day = agent_state.info.get("time_of_day", [None] * num_agents)[0]
                        interaction_progress = agent_state.info.get("interaction_progress", [None] * num_agents)[0]

                        # Get action masks for debugging (which actions were valid)
                        action_masks = self.env.get_action_masks()[0]  # Agent 0 action masks

                        self.recorder.record_step(
                            step=step,
                            positions=self.env.positions[0],  # Agent 0 position
                            meters=self.env.meters[0],  # Agent 0 meters
                            action=(
                                agent_state.actions[0].item() if hasattr(agent_state.actions[0], "item") else int(agent_state.actions[0])
                            ),
                            reward=float(extrinsic_only[0].item()),
                            intrinsic_reward=float(agent_state.intrinsic_rewards[0].item()),
                            done=bool(agent_state.dones[0].item()),
                            q_values=agent_state.info.get("q_values", [None] * num_agents)[0],
                            epsilon=float(agent_state.epsilons[0].item()),
                            action_masks=action_masks,
                            time_of_day=time_of_day,
                            interaction_progress=interaction_progress,
                        )

                    if torch.all(agent_state.dones).item():
                        break

                if last_agent_state is None:
                    continue

                for idx in range(num_agents):
                    if final_meters[idx] is None:
                        final_meters[idx] = self.env.meters[idx].detach().cpu()

                step_counts = last_agent_state.info.get("step_counts", self.population.episode_step_counts.clone())
                step_counts = step_counts.to(self.env.device)
                curriculum_survival_tensor = step_counts.float()
                curriculum_done_tensor = torch.ones_like(last_agent_state.dones, dtype=torch.bool, device=self.env.device)
                self.population.update_curriculum_tracker(curriculum_survival_tensor, curriculum_done_tensor)

                # P1.2: Flush episode if agent survived to max_steps (recurrent networks only)
                # Without this, successful episodes never reach replay buffer → memory leak + data loss
                # CRITICAL: Loop over all agents to support multi-agent configs (not just agent 0)
                for agent_idx in range(self.population.num_agents):
                    if not last_agent_state.dones[agent_idx]:  # Agent survived to max_steps without dying
                        self.population.flush_episode(agent_idx=agent_idx)

                epsilon_value = (
                    self.exploration.rnd.epsilon if hasattr(self.exploration, "rnd") else getattr(self.exploration, "epsilon", 0.0)
                )
                intrinsic_weight_value = (
                    self.exploration.get_intrinsic_weight() if hasattr(self.exploration, "get_intrinsic_weight") else 0.0
                )

                episode_reward_cpu = episode_reward.detach().cpu()
                episode_extrinsic_cpu = episode_extrinsic_reward.detach().cpu()
                episode_intrinsic_cpu = episode_intrinsic_reward.detach().cpu()
                step_counts_cpu = step_counts.detach().cpu().long()
                stages_cpu = self.curriculum.tracker.agent_stages.detach().cpu()

                # Log metrics to database
                episode_timestamp = time.time()
                self.db.insert_episode(
                    episode_id=self.current_episode,
                    timestamp=episode_timestamp,
                    survival_time=int(step_counts_cpu[0].item()),
                    total_reward=float(episode_reward_cpu[0].item()),
                    extrinsic_reward=float(episode_extrinsic_cpu[0].item()),
                    intrinsic_reward=float(episode_intrinsic_cpu[0].item()),
                    intrinsic_weight=intrinsic_weight_value,
                    curriculum_stage=int(stages_cpu[0].item()),
                    epsilon=epsilon_value,
                )

                # NEW: Insert affordance transitions for agent 0
                if affordance_transitions[0]:
                    # Convert nested defaultdict to regular dict for database insertion
                    transitions_dict = {from_aff: dict(to_affs) for from_aff, to_affs in affordance_transitions[0].items()}
                    self.db.insert_affordance_visits(episode_id=self.current_episode, transitions=transitions_dict)

                # Finish episode recording if enabled
                if self.recorder is not None:
                    from townlet.recording.data_structures import EpisodeMetadata

                    metadata = EpisodeMetadata(
                        episode_id=self.current_episode,
                        survival_steps=int(step_counts_cpu[0].item()),
                        total_reward=float(episode_reward_cpu[0].item()),
                        extrinsic_reward=float(episode_extrinsic_cpu[0].item()),
                        intrinsic_reward=float(episode_intrinsic_cpu[0].item()),
                        curriculum_stage=int(stages_cpu[0].item()),
                        epsilon=epsilon_value,
                        intrinsic_weight=intrinsic_weight_value,
                        timestamp=episode_timestamp,
                        affordance_layout=self.env.get_affordance_positions(),
                        affordance_visits={k: v for k, v in affordance_visits[0].items()},  # Agent 0 visits
                    )
                    self.recorder.finish_episode(metadata)

                agent_metrics = []
                for idx, agent_id in enumerate(self.population.agent_ids):
                    agent_metrics.append(
                        {
                            "agent_id": agent_id,
                            "survival_time": int(step_counts_cpu[idx].item()),
                            "total_reward": float(episode_reward_cpu[idx].item()),
                            "extrinsic_reward": float(episode_extrinsic_cpu[idx].item()),
                            "intrinsic_reward": float(episode_intrinsic_cpu[idx].item()),
                            "curriculum_stage": int(stages_cpu[idx].item()),
                            "epsilon": epsilon_value,
                            "intrinsic_weight": intrinsic_weight_value,
                        }
                    )

                # Log to TensorBoard (Phase 1 - Episode metrics)
                self.tb_logger.log_multi_agent_episode(
                    episode=self.current_episode,
                    agents=agent_metrics,
                )

                # Phase 2 - Training metrics (if training occurred this episode)
                training_metrics = self.population.get_training_metrics()
                if training_metrics["training_step"] > 0:
                    self.tb_logger.log_training_step(
                        step=training_metrics["training_step"],
                        td_error=training_metrics["td_error"],
                        loss=training_metrics["loss"],
                    )

                # Phase 3 - Meter dynamics and affordance usage
                meter_names = [
                    "energy",
                    "hygiene",
                    "satiation",
                    "money",
                    "mood",
                    "social",
                    "health",
                    "fitness",
                ]
                for idx, agent_id in enumerate(self.population.agent_ids):
                    meters_tensor = final_meters[idx]
                    if meters_tensor is not None:
                        meter_dict = {name: meters_tensor[i].item() for i, name in enumerate(meter_names)}
                        self.tb_logger.log_meters(
                            episode=self.current_episode,
                            step=int(step_counts_cpu[idx].item()),
                            meters=meter_dict,
                            agent_id=agent_id,
                        )

                for idx, visits in enumerate(affordance_visits):
                    if visits:
                        self.tb_logger.log_affordance_usage(
                            episode=self.current_episode,
                            affordance_counts=dict(visits),
                            agent_id=self.population.agent_ids[idx],
                        )

                # Heartbeat log every 10 episodes
                if self.current_episode % 10 == 0:
                    elapsed = time.time() - episode_start
                    stage_overview = "/".join(str(int(s.item())) for s in stages_cpu)

                    # Calculate reward breakdown
                    total_reward = episode_reward_cpu[0].item()
                    extrinsic_reward = episode_extrinsic_cpu[0].item()
                    intrinsic_reward = episode_intrinsic_cpu[0].item()
                    weighted_intrinsic = intrinsic_reward * intrinsic_weight_value

                    logger.info(
                        f"Episode {self.current_episode}/{self.max_episodes} | "
                        f"Survival: {int(step_counts_cpu[0].item())} steps | "
                        f"Reward: {total_reward:.2f} (Extrinsic: {extrinsic_reward:.2f}, "
                        f"Intrinsic: {intrinsic_reward:.2f}×{intrinsic_weight_value:.3f}={weighted_intrinsic:.2f}) | "
                        f"ε: {epsilon_value:.3f} | "
                        f"Stage: {stage_overview} | "
                        f"Time: {elapsed:.2f}s"
                    )

                # Detailed summary every 50 episodes
                if self.current_episode % 50 == 0:
                    # Get training metrics
                    training_metrics = self.population.get_training_metrics()

                    # Get final meters for agent 0
                    final_meter_values = {}
                    if final_meters[0] is not None:
                        meter_names = ["energy", "hygiene", "satiation", "money", "mood", "social", "health", "fitness"]
                        final_meter_values = {name: final_meters[0][i].item() for i, name in enumerate(meter_names)}

                    # Get affordance usage for agent 0
                    affordance_summary = (
                        ", ".join(f"{name}: {count}" for name, count in affordance_visits[0].items()) if affordance_visits[0] else "None"
                    )

                    logger.info("=" * 80)
                    logger.info(f"📊 SUMMARY - Episode {self.current_episode}/{self.max_episodes}")
                    logger.info("-" * 80)
                    logger.info(f"Performance:    Survival: {int(step_counts_cpu[0].item())} steps | Stage: {stage_overview}")
                    logger.info(
                        f"Rewards:        Total: {total_reward:.2f} | "
                        f"Extrinsic: {extrinsic_reward:.2f} | Intrinsic: {weighted_intrinsic:.2f}"
                    )
                    logger.info(f"Exploration:    ε: {epsilon_value:.3f} | Intrinsic Weight: {intrinsic_weight_value:.3f}")
                    if training_metrics["training_step"] > 0:
                        logger.info(
                            f"Training:       Steps: {training_metrics['training_step']} | "
                            f"Loss: {training_metrics['loss']:.4f} | TD Error: {training_metrics['td_error']:.4f}"
                        )
                    if final_meter_values:
                        logger.info(
                            f"Final Meters:   Energy: {final_meter_values.get('energy', 0):.2f} | "
                            f"Health: {final_meter_values.get('health', 0):.2f} | "
                            f"Money: ${final_meter_values.get('money', 0) * 100:.1f}"
                        )
                    logger.info(f"Affordances:    {affordance_summary}")
                    logger.info("=" * 80)

                # Checkpoint every 100 episodes
                if self.current_episode % 100 == 0:
                    self.save_checkpoint()

                # Decay epsilon for next episode and sync telemetry
                self.exploration.decay_epsilon()
                self.population.sync_exploration_metrics()

                self.current_episode += 1

        finally:
            # Save final checkpoint
            logger.info("Training complete, saving final checkpoint...")
            self.save_checkpoint()

            # Phase 4 - Log final metrics with hyperparameters
            if self.population is not None:
                final_metrics = {
                    "final_episode": self.current_episode,
                    "total_training_steps": self.population.total_steps,
                }
                # Re-log hyperparameters with final metrics for comparison
                if hasattr(self, "hparams"):
                    self.tb_logger.log_hyperparameters(hparams=self.hparams, metrics=final_metrics)

            self.db.set_system_state("training_status", "completed")

            # Use extracted cleanup method
            self._cleanup()


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
    )

    # Get paths from environment or args
    config_arg = sys.argv[1] if len(sys.argv) > 1 else "configs/test"
    db_path = sys.argv[2] if len(sys.argv) > 2 else "demo_state.db"
    checkpoint_dir = sys.argv[3] if len(sys.argv) > 3 else "checkpoints"
    # If max_episodes provided via CLI, use it; otherwise pass None to read from config
    max_episodes = int(sys.argv[4]) if len(sys.argv) > 4 else None

    config_path = Path(config_arg)
    if config_path.is_dir():
        config_dir = config_path
        training_config = config_dir / "training.yaml"
    else:
        config_dir = config_path.parent
        training_config = config_path

    runner = DemoRunner(
        config_dir=config_dir,
        db_path=db_path,
        checkpoint_dir=checkpoint_dir,
        max_episodes=max_episodes,  # None = read from config
        training_config_path=training_config,
    )
    runner.run()
