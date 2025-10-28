"""
Launch Hamlet unified metaserver.

Runs BOTH inference and training modes on a single port (8765).
Frontend connects once and can switch between modes dynamically.

This is the recommended way to run Hamlet visualization!
"""

from hamlet.web.metaserver import main

if __name__ == "__main__":
    print("=" * 60)
    print("HAMLET UNIFIED METASERVER")
    print("=" * 60)
    print()
    print("Starting metaserver on http://localhost:8765")
    print()
    print("Available endpoints:")
    print("  • Inference:  ws://localhost:8765/ws")
    print("  • Training:   ws://localhost:8765/ws/training")
    print()
    print("To start the frontend (in another terminal):")
    print("  cd frontend")
    print("  npm run dev")
    print()
    print("Then open: http://localhost:5173")
    print()
    print("✨ BOTH modes work simultaneously!")
    print("  - Watch trained agents (Inference mode)")
    print("  - Watch agents learn (Training mode)")
    print("  - Switch between modes without restarting!")
    print("=" * 60)
    print()

    main()
