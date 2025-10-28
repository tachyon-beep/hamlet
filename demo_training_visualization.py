"""
Launch Hamlet live training visualization server.

Starts the training server with WebSocket broadcasting on port 8765.
Frontend should be run separately with: cd frontend && npm run dev

This allows you to watch the agent learn in real-time!
"""

from hamlet.web.training_server import app
import uvicorn

if __name__ == "__main__":
    print("=" * 60)
    print("HAMLET LIVE TRAINING VISUALIZATION SERVER")
    print("=" * 60)
    print()
    print("Training server starting on http://localhost:8765")
    print("WebSocket endpoint: ws://localhost:8765/ws/training")
    print()
    print("To start the frontend (in another terminal):")
    print("  cd frontend")
    print("  npm run dev")
    print()
    print("Then open: http://localhost:5173")
    print("Select 'Training' mode and configure training parameters")
    print("=" * 60)
    print()

    uvicorn.run(app, host="0.0.0.0", port=8765)
