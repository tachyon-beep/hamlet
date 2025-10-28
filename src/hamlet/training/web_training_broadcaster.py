"""
Training broadcaster for web visualization.

Streams training state to web UI via WebSocket.
"""

import json
import asyncio
from typing import Dict, Any, List, Optional
from hamlet.training.metrics_manager import MetricsManager


class WebTrainingBroadcaster:
    """
    Broadcasts training state to connected WebSocket clients.

    Integrates with MetricsManager to stream real-time training updates.
    """

    def __init__(self):
        """Initialize broadcaster."""
        self.websockets: List[Any] = []
        self.is_training = False
        self.current_episode = 0
        self.current_step = 0
        self.episode_reward = 0.0
        self.agent_position = (0, 0)
        self.agent_meters = {}

    def add_websocket(self, websocket: Any):
        """Add a WebSocket connection to broadcast to."""
        self.websockets.append(websocket)

    def remove_websocket(self, websocket: Any):
        """Remove a WebSocket connection."""
        if websocket in self.websockets:
            self.websockets.remove(websocket)

    async def broadcast_training_state(
        self,
        episode: int,
        step: int,
        reward: float,
        env_state: Dict[str, Any],
        metrics: Optional[Dict[str, float]] = None
    ):
        """
        Broadcast current training state to all connected clients.

        Args:
            episode: Current episode number
            step: Current step in episode
            reward: Episode reward so far
            env_state: Environment render() output
            metrics: Additional training metrics
        """
        # Build message
        message = {
            "type": "training_update",
            "episode": episode,
            "step": step,
            "reward": reward,
            "state": env_state,
            "metrics": metrics or {},
            "is_training": True,
        }

        # Broadcast to all connected clients
        await self._broadcast(message)

    async def broadcast_episode_end(
        self,
        episode: int,
        total_reward: float,
        episode_length: int,
        agent_survived: bool
    ):
        """
        Broadcast episode completion.

        Args:
            episode: Episode number
            total_reward: Total episode reward
            episode_length: Number of steps
            agent_survived: Whether agent survived
        """
        message = {
            "type": "episode_complete",
            "episode": episode,
            "total_reward": total_reward,
            "episode_length": episode_length,
            "survived": agent_survived,
        }

        await self._broadcast(message)

    async def broadcast_training_start(self, config: Dict[str, Any]):
        """
        Broadcast training start event.

        Args:
            config: Training configuration
        """
        message = {
            "type": "training_start",
            "config": config,
        }

        await self._broadcast(message)

    async def broadcast_training_end(self, final_metrics: Dict[str, float]):
        """
        Broadcast training completion.

        Args:
            final_metrics: Final training metrics
        """
        message = {
            "type": "training_complete",
            "metrics": final_metrics,
        }

        await self._broadcast(message)

    async def _broadcast(self, message: Dict[str, Any]):
        """
        Send message to all connected WebSocket clients.

        Args:
            message: Message dictionary to broadcast
        """
        if not self.websockets:
            return

        message_json = json.dumps(message)

        # Send to all clients
        for websocket in self.websockets[:]:  # Copy list to avoid modification during iteration
            try:
                await websocket.send_text(message_json)
            except Exception as e:
                # Remove failed websocket
                self.websockets.remove(websocket)
                print(f"Removed failed websocket: {e}")


# Global broadcaster instance
_broadcaster: Optional[WebTrainingBroadcaster] = None


def get_broadcaster() -> WebTrainingBroadcaster:
    """Get or create global broadcaster instance."""
    global _broadcaster
    if _broadcaster is None:
        _broadcaster = WebTrainingBroadcaster()
    return _broadcaster
