"""Live inference server for multi-day demo.

Runs inference on the latest checkpoint while training happens in background.
Provides step-by-step visualization at human-watchable speed.
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional, Set
import time

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from townlet.environment.vectorized_env import VectorizedHamletEnv
from townlet.population.vectorized import VectorizedPopulation
from townlet.curriculum.adversarial import AdversarialCurriculum
from townlet.exploration.adaptive_intrinsic import AdaptiveIntrinsicExploration

logger = logging.getLogger(__name__)


class LiveInferenceServer:
    """Runs inference on latest checkpoint with step-by-step WebSocket streaming."""

    def __init__(
        self,
        checkpoint_dir: Path | str,
        port: int = 8766,
        step_delay: float = 0.2,
        total_episodes: int = 5000,  # Expected total episodes in training run
        config_path: Optional[Path | str] = None,  # Optional training config
    ):
        """Initialize live inference server.

        Args:
            checkpoint_dir: Directory containing training checkpoints
            port: WebSocket port
            step_delay: Delay between steps in seconds (0.2 = 5 steps/sec)
            total_episodes: Expected total episodes for training run (for progress gauge)
            config_path: Optional path to training config YAML (for matching environment settings)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.port = port
        self.step_delay = step_delay
        self.total_episodes = total_episodes
        self.config_path = Path(config_path) if config_path else None
        self.config = None

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.clients: Set[WebSocket] = set()

        # Current checkpoint tracking
        self.current_checkpoint_path: Optional[Path] = None
        self.current_checkpoint_episode: int = 0
        self.current_epsilon: float = 0.0

        # Environment and agent
        self.env: Optional[VectorizedHamletEnv] = None
        self.population: Optional[VectorizedPopulation] = None
        self.curriculum: Optional[AdversarialCurriculum] = None
        self.exploration: Optional[AdaptiveIntrinsicExploration] = None

        # Episode state
        self.is_running = False
        self.current_episode = 0
        self.current_step = 0

        # Affordance interaction tracking (for UI display)
        self.affordance_interactions = {}  # {affordance_name: count}

        # Checkpoint auto-update mode
        self.auto_checkpoint_mode = False  # If true, automatically check for new checkpoints after each episode

        # FastAPI app
        self.app = FastAPI(title="Hamlet Live Inference Server")

        # Enable CORS for frontend
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Register routes
        self.app.websocket("/ws")(self.websocket_endpoint)
        self.app.websocket("/ws/training")(self.websocket_endpoint)  # Same endpoint, different name
        self.app.on_event("startup")(self.startup)
        self.app.on_event("shutdown")(self.shutdown)

    async def startup(self):
        """Initialize environment and start checkpoint monitoring."""
        logger.info("Starting live inference server")

        # Load config if provided
        if self.config_path and self.config_path.exists():
            import yaml
            with open(self.config_path) as f:
                self.config = yaml.safe_load(f)
            logger.info(f"Loaded training config: {self.config_path}")

        # Initialize environment and components
        self._initialize_components()

        # Load initial checkpoint
        await self._check_and_load_checkpoint()

        logger.info(f"Loaded checkpoint: {self.current_checkpoint_path.name if self.current_checkpoint_path else 'None'}")

    async def shutdown(self):
        """Cleanup on shutdown."""
        self.is_running = False
        logger.info("Live inference server shut down")

    def _initialize_components(self):
        """Initialize environment and agent components."""
        # Get environment config (use config if available, otherwise defaults)
        num_agents = 1
        grid_size = 8
        partial_observability = False
        vision_range = 2
        enable_temporal_mechanics = False
        network_type = "simple"
        vision_window_size = 5

        if self.config:
            env_cfg = self.config.get('environment', {})
            pop_cfg = self.config.get('population', {})

            grid_size = env_cfg.get('grid_size', 8)
            partial_observability = env_cfg.get('partial_observability', False)
            vision_range = env_cfg.get('vision_range', 2)
            enable_temporal_mechanics = env_cfg.get('enable_temporal_mechanics', False)
            network_type = pop_cfg.get('network_type', 'simple')
            vision_window_size = 2 * vision_range + 1

            logger.info(f"Environment config: grid={grid_size}, POMDP={partial_observability}, vision={vision_range}, temporal={enable_temporal_mechanics}")
            logger.info(f"Network type: {network_type}")

        # Create environment with config settings
        self.env = VectorizedHamletEnv(
            num_agents=num_agents,
            grid_size=grid_size,
            device=self.device,
            partial_observability=partial_observability,
            vision_range=vision_range,
            enable_temporal_mechanics=enable_temporal_mechanics,
        )

        # Auto-detect observation dimension from environment
        obs_dim = self.env.observation_dim

        # Create curriculum
        self.curriculum = AdversarialCurriculum(
            max_steps_per_episode=500,
            survival_advance_threshold=0.7,
            survival_retreat_threshold=0.3,
            entropy_gate=0.5,
            min_steps_at_stage=1000,
            device=self.device,
        )

        # Create exploration (for inference, we want greedy)
        self.exploration = AdaptiveIntrinsicExploration(
            obs_dim=obs_dim,  # Auto-detected from environment
            embed_dim=128,
            initial_intrinsic_weight=0.0,  # Pure exploitation for inference
            variance_threshold=10.0,
            survival_window=100,
            device=self.device,
        )

        # Create population (use auto-detected dimensions and network type from config)
        agent_ids = [f"agent_{i}" for i in range(num_agents)]
        self.population = VectorizedPopulation(
            env=self.env,
            curriculum=self.curriculum,
            exploration=self.exploration,
            agent_ids=agent_ids,
            device=self.device,
            obs_dim=obs_dim,
            action_dim=self.env.action_dim,
            replay_buffer_capacity=10000,
            network_type=network_type,
            vision_window_size=vision_window_size,
        )

        self.curriculum.initialize_population(num_agents)

    async def _check_and_load_checkpoint(self) -> bool:
        """Check for new checkpoints and load if available.

        Returns:
            True if a new checkpoint was loaded
        """
        # Find all checkpoints
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_ep*.pt"))
        if not checkpoints:
            logger.warning(f"No checkpoints found in {self.checkpoint_dir}")
            return False

        # Get latest checkpoint
        latest_checkpoint = checkpoints[-1]

        # Check if it's newer than current
        if latest_checkpoint == self.current_checkpoint_path:
            return False

        # Extract episode number from filename
        try:
            episode_str = latest_checkpoint.stem.split("_ep")[1]
            episode_num = int(episode_str)
        except:
            logger.error(f"Could not parse episode number from {latest_checkpoint.name}")
            return False

        # Load checkpoint
        logger.info(f"Loading checkpoint: {latest_checkpoint.name} (episode {episode_num})")

        checkpoint = torch.load(latest_checkpoint, weights_only=False)

        # Load Q-network weights
        if 'population_state' in checkpoint:
            self.population.q_network.load_state_dict(checkpoint['population_state']['q_network'])
            logger.info("Loaded Q-network weights")

        # Calculate epsilon based on training progress
        # For inference, we estimate epsilon from episode number (linear decay from 1.0 to 0.05)
        epsilon = checkpoint.get('epsilon', None)
        if epsilon is None:
            # Estimate epsilon based on training progress (assuming linear decay over total_episodes)
            progress = episode_num / self.total_episodes if self.total_episodes > 0 else 0
            epsilon = max(0.05, 1.0 - (progress * 0.95))  # Decay from 1.0 to 0.05
            logger.info(f"Estimated epsilon from training progress: {epsilon:.3f}")

        # Update tracking
        self.current_checkpoint_path = latest_checkpoint
        self.current_checkpoint_episode = episode_num
        self.current_epsilon = epsilon

        # Broadcast model update to all clients
        await self._broadcast_to_clients({
            'type': 'model_loaded',
            'model': f"checkpoint_ep{episode_num:05d}",
            'episode': episode_num,
            'total_episodes': self.total_episodes,
            'epsilon': epsilon,
            'message': f"Loaded model from episode {episode_num}"
        })

        return True

    async def websocket_endpoint(self, websocket: WebSocket):
        """WebSocket endpoint for client connections."""
        await websocket.accept()
        self.clients.add(websocket)
        logger.info(f"Client connected. Total clients: {len(self.clients)}")

        # Send connection message
        await websocket.send_json({
            'type': 'connected',
            'message': 'Connected to live inference server',
            'available_models': [],
            'mode': 'inference',
            'checkpoint': f"checkpoint_ep{self.current_checkpoint_episode:05d}" if self.current_checkpoint_path else "None",
            'checkpoint_episode': self.current_checkpoint_episode,
            'total_episodes': self.total_episodes,
            'epsilon': self.current_epsilon,
            'auto_checkpoint_mode': self.auto_checkpoint_mode
        })

        try:
            # Handle incoming commands
            while True:
                data = await websocket.receive_json()
                await self._handle_command(websocket, data)
        except WebSocketDisconnect:
            logger.info("Client disconnected")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            self.clients.discard(websocket)
            logger.info(f"Client removed. Total clients: {len(self.clients)}")

    async def _handle_command(self, websocket: WebSocket, data: dict):
        """Handle incoming WebSocket commands."""
        command = data.get('command') or data.get('type')

        if command == 'play':
            if not self.is_running:
                self.is_running = True
                asyncio.create_task(self._run_inference_loop())
                logger.info("Started inference loop")

        elif command == 'pause':
            self.is_running = False
            logger.info("Paused inference loop")

        elif command == 'step':
            # Run a single step
            if not self.is_running:
                asyncio.create_task(self._run_single_episode())

        elif command == 'reset':
            # Reset episode
            self.current_episode = 0
            self.current_step = 0
            logger.info("Reset episode counter")

        elif command == 'refresh_checkpoint':
            # Manually check for and load new checkpoint
            logger.info("Manual checkpoint refresh requested")
            checkpoint_loaded = await self._check_and_load_checkpoint()
            if checkpoint_loaded:
                logger.info("New checkpoint loaded")
            else:
                logger.info("No new checkpoint available")

        elif command == 'toggle_auto_checkpoint':
            # Toggle auto checkpoint mode
            self.auto_checkpoint_mode = not self.auto_checkpoint_mode
            status = "enabled" if self.auto_checkpoint_mode else "disabled"
            logger.info(f"Auto checkpoint mode {status}")
            # Broadcast mode change to all clients
            await self._broadcast_to_clients({
                'type': 'auto_checkpoint_mode',
                'enabled': self.auto_checkpoint_mode
            })

    async def _run_inference_loop(self):
        """Main inference loop - runs episodes continuously."""
        logger.info("Inference loop started")

        while self.is_running:
            # Check for new checkpoint before each episode (if auto mode enabled)
            if self.auto_checkpoint_mode:
                await self._check_and_load_checkpoint()

            # Run one inference episode
            await self._run_single_episode()

            # Small delay between episodes
            await asyncio.sleep(0.5)

    async def _run_single_episode(self):
        """Run a single inference episode with step-by-step updates."""
        self.current_episode += 1
        self.current_step = 0

        # Reset affordance interaction tracking
        self.affordance_interactions = {name: 0 for name in self.env.affordances.keys()}

        # Randomize affordance positions for each episode (like training)
        self.env.randomize_affordance_positions()

        # Log affordance positions for verification
        affordance_positions = {name: pos.tolist() for name, pos in self.env.affordances.items()}
        logger.info(f"Episode {self.current_episode} affordance positions: {affordance_positions}")

        # Reset environment
        self.env.reset()
        self.population.reset()

        # Send episode start
        await self._broadcast_to_clients({
            'type': 'episode_start',
            'episode': self.current_episode,
            'checkpoint': f"checkpoint_ep{self.current_checkpoint_episode:05d}",
            'checkpoint_episode': self.current_checkpoint_episode,
            'total_episodes': self.total_episodes,
            'epsilon': self.current_epsilon
        })

        # Run episode
        done = False
        cumulative_reward = 0.0
        max_steps = 500

        while not done and self.current_step < max_steps:
            # Get action using epsilon-greedy (respects current epsilon from checkpoint)
            actions = self.population.select_epsilon_greedy_actions(self.env, self.current_epsilon)

            # Get Q-values for display
            with torch.no_grad():
                q_output = self.population.q_network(self.population.current_obs)
                q_values = q_output[0] if isinstance(q_output, tuple) else q_output

            # Step environment
            next_obs, rewards, dones, info = self.env.step(actions)

            # Track successful interactions (only count if interaction actually succeeded)
            successful_interactions = info.get('successful_interactions', {})
            if 0 in successful_interactions:
                affordance_name = successful_interactions[0]
                self.affordance_interactions[affordance_name] += 1

            # Update state
            self.population.current_obs = next_obs
            done = dones[0].item()
            cumulative_reward += rewards[0].item()
            self.current_step += 1

            # Send state update with Q-values
            await self._broadcast_state_update(cumulative_reward, actions[0].item(), q_values[0])

            # Delay for human viewing
            if self.is_running:
                await asyncio.sleep(self.step_delay)

        # Episode complete
        await self._broadcast_to_clients({
            'type': 'episode_end',
            'episode': self.current_episode,
            'steps': self.current_step,
            'total_reward': cumulative_reward,
            'reason': 'done' if done else 'max_steps',
            'checkpoint': f"checkpoint_ep{self.current_checkpoint_episode:05d}",
            'checkpoint_episode': self.current_checkpoint_episode,
            'total_episodes': self.total_episodes,
            'epsilon': self.current_epsilon
        })

        logger.info(f"Episode {self.current_episode} complete: {self.current_step} steps, reward: {cumulative_reward:.2f}")

    async def _broadcast_state_update(self, cumulative_reward: float, last_action: int, q_values: torch.Tensor):
        """Broadcast current state to all clients."""
        # Get agent position (unpack for frontend compatibility)
        agent_pos = self.env.positions[0].cpu().tolist()

        # Get meters (all 8: energy, hygiene, satiation, money, health, mood, social, fitness)
        # Note: Backend order is [energy, hygiene, satiation, money, mood, social, health, fitness]
        # We reorder for UI display to group the primary/secondary meters together
        meter_indices = {
            'energy': 0,
            'hygiene': 1,
            'satiation': 2,
            'money': 3,
            'health': 6,    # Primary (direct top-up)
            'mood': 4,      # Primary (direct top-up)
            'social': 5,    # Secondary (modulates mood)
            'fitness': 7,   # Secondary (modulates health)
        }
        meters = {}
        for meter_name, idx in meter_indices.items():
            meters[meter_name] = self.env.meters[0, idx].item()

        # Get affordances (unpack position for frontend compatibility)
        affordances = []
        for name, pos in self.env.affordances.items():
            pos_list = pos.cpu().tolist()
            affordances.append({
                'type': name,  # Frontend expects 'type' not 'name'
                'x': pos_list[0],
                'y': pos_list[1],
            })

        # Convert Q-values to list for JSON serialization
        q_values_list = q_values.cpu().tolist()

        # Prepare affordance interaction counts (sorted by count descending)
        affordance_stats = [
            {'name': name, 'count': count}
            for name, count in sorted(
                self.affordance_interactions.items(),
                key=lambda x: x[1],
                reverse=True
            )
        ]

        # Build state update message
        update = {
            'type': 'state_update',
            'step': self.current_step,
            'cumulative_reward': cumulative_reward,
            'grid': {
                'width': self.env.grid_size,
                'height': self.env.grid_size,
                'agents': [{
                    'id': 'agent_0',
                    'x': agent_pos[0],  # Frontend expects x, y not position
                    'y': agent_pos[1],
                    'color': '#4CAF50',  # Green color for agent
                    'last_action': last_action,
                }],
                'affordances': affordances,
            },
            'agent_meters': {
                'agent_0': {
                    'meters': meters  # MeterPanel expects agent_0.meters nested structure
                }
            },
            'q_values': q_values_list,  # Q-values for all 5 actions
            'affordance_stats': affordance_stats,  # Interaction counts sorted by frequency
        }

        # Add temporal mechanics data if enabled
        if hasattr(self.env, 'time_of_day'):
            update['temporal'] = {
                'time_of_day': self.env.time_of_day,
                'interaction_progress': self.env.interaction_progress[0].item(),
            }

        await self._broadcast_to_clients(update)

    async def _broadcast_to_clients(self, message: dict):
        """Broadcast message to all connected clients."""
        dead_clients = set()
        for client in self.clients:
            try:
                await client.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send to client: {e}")
                dead_clients.add(client)

        # Remove dead clients
        self.clients -= dead_clients


def run_server(
    checkpoint_dir: str = "checkpoints",
    port: int = 8766,
    step_delay: float = 0.2,
    total_episodes: int = 5000,
    config_path: str = None,
):
    """Run live inference server."""
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s'
    )

    server = LiveInferenceServer(checkpoint_dir, port, step_delay, total_episodes, config_path)

    logger.info(f"Starting live inference server on port {port}")
    logger.info(f"Checkpoint directory: {checkpoint_dir}")
    logger.info(f"Step delay: {step_delay}s ({1/step_delay:.1f} steps/sec)")
    logger.info(f"Expected total training episodes: {total_episodes}")
    if config_path:
        logger.info(f"Training config: {config_path}")
    logger.info(f"Connect Vue frontend to: ws://localhost:{port}/ws")

    uvicorn.run(server.app, host="0.0.0.0", port=port)


if __name__ == '__main__':
    import sys

    checkpoint_dir = sys.argv[1] if len(sys.argv) > 1 else "checkpoints"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8766
    step_delay = float(sys.argv[3]) if len(sys.argv) > 3 else 0.2
    total_episodes = int(sys.argv[4]) if len(sys.argv) > 4 else 5000
    config_path = sys.argv[5] if len(sys.argv) > 5 else None

    run_server(checkpoint_dir, port, step_delay, total_episodes, config_path)
