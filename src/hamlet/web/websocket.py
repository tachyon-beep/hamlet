"""
WebSocket manager for Hamlet visualization.

Manages WebSocket connections, coordinates simulation runner,
and broadcasts state updates to all connected clients.
"""

from typing import List, Optional, Dict, Any
from fastapi import WebSocket
import json
import asyncio

from hamlet.web.simulation_runner import SimulationRunner
from hamlet.environment.renderer import Renderer


class WebSocketManager:
    """
    Manages WebSocket connections and simulation broadcasting.

    Coordinates between simulation runner and connected clients,
    handling state updates and control commands.
    """

    def __init__(self):
        """Initialize WebSocket manager."""
        self.active_connections: List[WebSocket] = []
        self.simulation_runner: Optional[SimulationRunner] = None
        self.broadcast_task: Optional[asyncio.Task] = None
        self._running = False

    async def connect(self, websocket: WebSocket):
        """
        Accept and register a new WebSocket connection.

        Args:
            websocket: WebSocket connection to accept
        """
        await websocket.accept()
        self.active_connections.append(websocket)

        print(f"Client connected. Total connections: {len(self.active_connections)}")

        # Start simulation if this is the first client
        if len(self.active_connections) == 1:
            await self.start_simulation()

        # Send connection acknowledgment with available models
        from pathlib import Path
        models_dir = Path("models")
        available_models = []
        if models_dir.exists():
            available_models = [f.name for f in models_dir.glob("*.pt")]

        await websocket.send_text(json.dumps({
            "type": "connected",
            "available_models": available_models,
        }))

    async def disconnect(self, websocket: WebSocket):
        """
        Remove a WebSocket connection.

        Args:
            websocket: WebSocket connection to remove
        """
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            print(f"Client disconnected. Total connections: {len(self.active_connections)}")

        # Stop simulation if no clients remain
        if len(self.active_connections) == 0:
            await self.stop_simulation()

    async def broadcast(self, message: dict):
        """
        Broadcast message to all connected clients.

        Args:
            message: Dictionary to send (will be JSON serialized)
        """
        if not self.active_connections:
            return

        json_message = json.dumps(message)
        disconnected = []

        for connection in self.active_connections:
            try:
                await connection.send_text(json_message)
            except Exception as e:
                print(f"Error broadcasting to client: {e}")
                disconnected.append(connection)

        # Clean up disconnected clients
        for connection in disconnected:
            await self.disconnect(connection)

    async def start_simulation(self):
        """Start the simulation runner and broadcasting task."""
        if self._running:
            return

        print("Starting simulation...")
        self.simulation_runner = SimulationRunner()
        self._running = True

        # Start broadcasting task
        self.broadcast_task = asyncio.create_task(self._broadcast_loop())

    async def stop_simulation(self):
        """Stop the simulation runner and broadcasting task."""
        if not self._running:
            return

        print("Stopping simulation...")
        self._running = False

        if self.broadcast_task:
            self.broadcast_task.cancel()
            try:
                await self.broadcast_task
            except asyncio.CancelledError:
                pass
            self.broadcast_task = None

        self.simulation_runner = None

    async def _broadcast_loop(self):
        """
        Main broadcast loop that streams simulation updates.

        Consumes state updates from simulation runner and broadcasts to clients.
        """
        try:
            async for update in self.simulation_runner.run():
                if not self._running:
                    break

                # Enrich state_update messages with rendered state
                if update["type"] == "state_update":
                    # Add rendered state to the update
                    obs = update.get("observation")
                    if obs:
                        # Get renderer from environment
                        renderer = Renderer(
                            grid=self.simulation_runner.env.grid,
                            agents=list(self.simulation_runner.env.agents.values()),
                            affordances=self.simulation_runner.env.affordances
                        )
                        rendered = renderer.render_to_dict()

                        # Merge rendered state with update
                        update["grid"] = rendered["grid"]
                        update["agents"] = rendered["agents"]

                        # Add last action info
                        action_names = ["up", "down", "left", "right", "interact"]
                        action_idx = update.get("action", 0)
                        update["agents"]["agent_0"]["last_action"] = action_names[action_idx]
                        update["agents"]["agent_0"]["reward"] = update.get("reward", 0.0)

                        # Remove observation from message (too large)
                        del update["observation"]

                    # Add RND metrics if available (Phase 3)
                    # This will be populated when using adaptive intrinsic exploration
                    if hasattr(self.simulation_runner, 'rnd_metrics') and self.simulation_runner.rnd_metrics:
                        update["rnd_metrics"] = self.simulation_runner.rnd_metrics

                # Broadcast to all clients
                await self.broadcast(update)

        except asyncio.CancelledError:
            print("Broadcast loop cancelled")
        except Exception as e:
            print(f"Error in broadcast loop: {e}")
            import traceback
            traceback.print_exc()

    async def shutdown(self):
        """Clean shutdown of all connections and simulation."""
        await self.stop_simulation()

        # Close all connections
        for connection in list(self.active_connections):
            try:
                await connection.close()
            except:
                pass
        self.active_connections.clear()

    # Control commands

    def play(self):
        """Start simulation playback."""
        if self.simulation_runner:
            self.simulation_runner.play()

    def pause(self):
        """Pause simulation playback."""
        if self.simulation_runner:
            self.simulation_runner.pause()

    def step(self):
        """Advance simulation by one step."""
        if self.simulation_runner:
            self.simulation_runner.step()

    def reset(self):
        """Reset current episode."""
        if self.simulation_runner:
            self.simulation_runner.reset()

    def set_speed(self, speed: float):
        """Set simulation speed multiplier."""
        if self.simulation_runner:
            self.simulation_runner.set_speed(speed)

    async def load_model(self, model_name: str):
        """
        Load a different trained agent model.

        Args:
            model_name: Name of the model file in models/ directory
        """
        if self.simulation_runner:
            from pathlib import Path
            model_path = Path("models") / model_name
            if model_path.exists():
                self.simulation_runner.load_agent(str(model_path))
                await self.broadcast({
                    "type": "model_loaded",
                    "model": model_name,
                })
            else:
                await self.broadcast({
                    "type": "error",
                    "message": f"Model not found: {model_name}",
                })

    # Status queries

    def get_connection_count(self) -> int:
        """Get number of active connections."""
        return len(self.active_connections)

    def is_running(self) -> bool:
        """Check if simulation is running."""
        return self._running
