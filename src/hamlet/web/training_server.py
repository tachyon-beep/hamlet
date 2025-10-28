"""
Training server with live WebSocket broadcasting.

Runs training episodes and broadcasts state updates to connected clients
so you can watch the agent learn in real-time.
"""

import asyncio
import json
from typing import List, Set
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import torch
import numpy as np

from hamlet.environment.hamlet_env import HamletEnv
from hamlet.environment.renderer import Renderer
from hamlet.agent.drl_agent import DRLAgent
from hamlet.agent.replay_buffer import ReplayBuffer
from hamlet.agent.observation_utils import preprocess_observation


app = FastAPI(title="Hamlet Training Server")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TrainingBroadcaster:
    """Manages training and broadcasting to WebSocket clients."""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.training_task = None
        self.is_training = False
        self.pause_event = asyncio.Event()
        self.pause_event.set()  # Start unpaused

        # Training state
        self.current_episode = 0
        self.total_episodes = 0
        self.episode_metrics = []

        # Position heat map tracking (grid_size x grid_size)
        self.position_visits = {}  # Dict[(x, y), count]
        self.grid_size = 8

    async def connect(self, websocket: WebSocket):
        """Accept new WebSocket connection."""
        await websocket.accept()
        self.active_connections.add(websocket)
        print(f"Client connected. Total: {len(self.active_connections)}")

        # Send current training state
        await websocket.send_text(json.dumps({
            "type": "training_status",
            "is_training": self.is_training,
            "current_episode": self.current_episode,
            "total_episodes": self.total_episodes,
        }))

    async def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection."""
        self.active_connections.discard(websocket)
        print(f"Client disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients."""
        if not self.active_connections:
            return

        json_message = json.dumps(message)
        disconnected = []

        for connection in self.active_connections:
            try:
                await connection.send_text(json_message)
            except Exception as e:
                print(f"Error broadcasting: {e}")
                disconnected.add(connection)

        # Clean up disconnected clients
        for connection in disconnected:
            await self.disconnect(connection)

    async def start_training(
        self,
        num_episodes: int = 100,
        batch_size: int = 32,
        buffer_capacity: int = 10000,
        show_every: int = 5,  # Show agent behavior every N episodes
        step_delay: float = 0.2,  # Delay between steps when visualizing (seconds)
    ):
        """Start training with live broadcasting."""
        if self.is_training:
            return

        self.is_training = True
        self.current_episode = 0
        self.total_episodes = num_episodes
        self.episode_metrics = []

        # Reset position heat map
        self.position_visits = {}

        # Broadcast training started
        await self.broadcast({
            "type": "training_started",
            "num_episodes": num_episodes,
        })

        # Initialize environment and agent
        env = HamletEnv()
        agent = DRLAgent(
            agent_id="learner",
            state_dim=72,  # 2 pos + 6 meters + 64 grid
            action_dim=5,
            learning_rate=1e-3,
            gamma=0.99,
            epsilon=1.0,
            epsilon_min=0.05,
            epsilon_decay=0.995,
            device="cpu",
            network_type="relational",  # Use attention network!
            grid_size=8
        )
        buffer = ReplayBuffer(capacity=buffer_capacity)

        # Training metrics
        episode_rewards = []
        episode_lengths = []
        losses = []

        print(f"Starting training: {num_episodes} episodes")
        print(f"Network: RelationalQNetwork (attention-based)")
        print(f"Broadcasting to {len(self.active_connections)} clients")

        for episode in range(num_episodes):
            self.current_episode = episode + 1

            obs = env.reset()
            episode_reward = 0.0
            episode_loss = []
            step = 0
            done = False

            # Decide if we should broadcast this episode
            broadcast_episode = (episode % show_every == 0)

            if broadcast_episode:
                # Send episode start
                await self.broadcast({
                    "type": "episode_start",
                    "episode": episode + 1,
                    "epsilon": agent.epsilon,
                })

            while not done:
                # Wait if paused
                await self.pause_event.wait()

                # Select action
                action = agent.select_action(obs, explore=True)

                # Execute action
                next_obs, reward, done, info = env.step(action)
                episode_reward += reward
                step += 1

                # Track agent position for heat map (position is numpy array [x, y])
                agent_pos = (int(next_obs["position"][0]), int(next_obs["position"][1]))
                self.position_visits[agent_pos] = self.position_visits.get(agent_pos, 0) + 1

                # Broadcast state if this is a broadcast episode
                if broadcast_episode:
                    # Use Renderer for proper formatting (like inference mode)
                    renderer = Renderer(
                        grid=env.grid,
                        agents=list(env.agents.values()),
                        affordances=env.affordances
                    )
                    rendered = renderer.render_to_dict()

                    # Convert action index to name
                    action_names = ["up", "down", "left", "right", "interact"]
                    action_name = action_names[action] if action < len(action_names) else "unknown"

                    # Add action and reward to agent info (rendered["agents"] is the meters dict)
                    if "agent_0" in rendered["agents"]:
                        rendered["agents"]["agent_0"]["last_action"] = action_name
                        rendered["agents"]["agent_0"]["reward"] = reward

                    # Extract agents list from grid (for Grid component)
                    agents_list = rendered["grid"]["agents"]

                    # Also add action/reward to agents list
                    for agent_data in agents_list:
                        if agent_data["id"] == "agent_0":
                            agent_data["last_action"] = action_name
                            agent_data["reward"] = reward

                    # Normalize heat map data (0-1 scale)
                    max_visits = max(self.position_visits.values()) if self.position_visits else 1
                    heat_map = {}
                    for (x, y), count in self.position_visits.items():
                        heat_map[f"{x},{y}"] = count / max_visits

                    await self.broadcast({
                        "type": "state_update",
                        "episode": episode + 1,
                        "step": step,
                        "action": action,
                        "reward": reward,
                        "cumulative_reward": episode_reward,
                        "done": done,
                        "grid": {
                            "width": rendered["grid"]["width"],
                            "height": rendered["grid"]["height"],
                        },
                        "agents": agents_list,  # Array for Grid component
                        "agent_meters": rendered["agents"],  # Dict for MeterPanel
                        "affordances": rendered["grid"]["affordances"],
                        "heat_map": heat_map,  # Position visit frequencies (normalized 0-1)
                    })

                    # Delay so humans can watch (configurable)
                    await asyncio.sleep(step_delay)

                # Store experience
                state_vec = preprocess_observation(obs)
                next_state_vec = preprocess_observation(next_obs)
                buffer.push(state_vec, action, reward, next_state_vec, done)

                # Learn from experience
                if buffer.is_ready(batch_size):
                    batch = buffer.sample(batch_size)
                    loss = agent.learn(batch)
                    episode_loss.append(loss)

                obs = next_obs

                # Limit episode length
                if step >= 500:
                    break

            # Decay exploration
            agent.decay_epsilon()

            # Update target network periodically
            if (episode + 1) % 10 == 0:
                agent.update_target_network()

            # Record metrics
            episode_rewards.append(episode_reward)
            episode_lengths.append(step)
            if episode_loss:
                losses.append(np.mean(episode_loss))

            # Broadcast episode summary
            recent_rewards = episode_rewards[-5:] if len(episode_rewards) >= 5 else episode_rewards
            recent_lengths = episode_lengths[-5:] if len(episode_lengths) >= 5 else episode_lengths
            recent_losses = losses[-5:] if len(losses) >= 5 else losses

            summary = {
                "type": "episode_complete",
                "episode": episode + 1,
                "reward": episode_reward,
                "length": step,
                "loss": np.mean(episode_loss) if episode_loss else 0,
                "epsilon": agent.epsilon,
                "avg_reward_5": float(np.mean(recent_rewards)),
                "avg_length_5": float(np.mean(recent_lengths)),
                "avg_loss_5": float(np.mean(recent_losses)) if recent_losses else 0,
                "buffer_size": len(buffer),
            }

            await self.broadcast(summary)

            # Print progress
            if (episode + 1) % 5 == 0:
                print(f"Episode {episode + 1}/{num_episodes}")
                print(f"  Avg Reward (last 5):  {np.mean(recent_rewards):7.2f}")
                print(f"  Avg Length (last 5):  {np.mean(recent_lengths):7.1f}")
                if recent_losses:
                    print(f"  Avg Loss (last 5):    {np.mean(recent_losses):7.4f}")
                print(f"  Epsilon:              {agent.epsilon:7.3f}")
                print()

        # Save trained agent
        import os
        os.makedirs("models", exist_ok=True)
        save_path = "models/trained_agent.pt"
        agent.save(save_path)

        # Training complete
        await self.broadcast({
            "type": "training_complete",
            "model_saved": save_path,
            "final_avg_reward": float(np.mean(episode_rewards[-10:])),
            "final_avg_length": float(np.mean(episode_lengths[-10:])),
        })

        print(f"Training complete! Model saved to {save_path}")
        self.is_training = False


# Global broadcaster instance
broadcaster = TrainingBroadcaster()


@app.websocket("/ws/training")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for training visualization."""
    await broadcaster.connect(websocket)

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            command = message.get("command")

            if command == "start_training":
                # Start training in background
                num_episodes = message.get("num_episodes", 100)
                batch_size = message.get("batch_size", 32)
                buffer_capacity = message.get("buffer_capacity", 10000)
                show_every = message.get("show_every", 5)
                step_delay = message.get("step_delay", 0.2)  # Default 200ms per step

                asyncio.create_task(broadcaster.start_training(
                    num_episodes=num_episodes,
                    batch_size=batch_size,
                    buffer_capacity=buffer_capacity,
                    show_every=show_every,
                    step_delay=step_delay,
                ))

            elif command == "pause":
                broadcaster.pause_event.clear()
                await websocket.send_text(json.dumps({"type": "paused"}))

            elif command == "resume":
                broadcaster.pause_event.set()
                await websocket.send_text(json.dumps({"type": "resumed"}))

            elif command == "status":
                await websocket.send_text(json.dumps({
                    "type": "training_status",
                    "is_training": broadcaster.is_training,
                    "current_episode": broadcaster.current_episode,
                    "total_episodes": broadcaster.total_episodes,
                }))

    except WebSocketDisconnect:
        await broadcaster.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        await broadcaster.disconnect(websocket)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "running",
        "service": "Hamlet Training Server",
        "clients": len(broadcaster.active_connections),
        "training": broadcaster.is_training,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("HAMLET TRAINING SERVER")
    print("=" * 60)
    print("WebSocket endpoint: ws://localhost:8765/ws/training")
    print("Frontend should connect to this endpoint")
    print("=" * 60)
    print()

    uvicorn.run(app, host="0.0.0.0", port=8765)
