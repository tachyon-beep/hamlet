"""
Launch Hamlet visualization server.

Starts the FastAPI backend with WebSocket streaming on port 8765.
Frontend should be run separately with: cd frontend && npm run dev
"""

from hamlet.web.server import main

if __name__ == "__main__":
    print("=" * 60)
    print("HAMLET VISUALIZATION SERVER")
    print("=" * 60)
    print()
    print("Backend server starting on http://localhost:8765")
    print("WebSocket endpoint: ws://localhost:8765/ws")
    print()
    print("To start the frontend (in another terminal):")
    print("  cd frontend")
    print("  npm run dev")
    print()
    print("Then open: http://localhost:5173")
    print("=" * 60)
    print()

    main()
