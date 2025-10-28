"""
Verification script for Hamlet visualization system.

Checks that all components are properly set up.
"""

from pathlib import Path
import sys

def check_backend():
    """Check backend files exist."""
    print("Checking backend...")

    files = [
        "src/hamlet/web/server.py",
        "src/hamlet/web/websocket.py",
        "src/hamlet/web/simulation_runner.py",
        "src/hamlet/environment/renderer.py",
    ]

    for file in files:
        path = Path(file)
        if path.exists():
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} MISSING")
            return False

    return True

def check_frontend():
    """Check frontend files exist."""
    print("\nChecking frontend...")

    files = [
        "frontend/package.json",
        "frontend/vite.config.js",
        "frontend/index.html",
        "frontend/src/main.js",
        "frontend/src/App.vue",
        "frontend/src/stores/simulation.js",
        "frontend/src/components/Grid.vue",
        "frontend/src/components/MeterPanel.vue",
        "frontend/src/components/Controls.vue",
        "frontend/src/components/StatsPanel.vue",
    ]

    for file in files:
        path = Path(file)
        if path.exists():
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} MISSING")
            return False

    # Check node_modules
    node_modules = Path("frontend/node_modules")
    if node_modules.exists():
        print(f"  ✓ frontend/node_modules (dependencies installed)")
    else:
        print(f"  ✗ frontend/node_modules (run: cd frontend && npm install)")
        return False

    return True

def check_models():
    """Check if trained models exist."""
    print("\nChecking trained models...")

    models_dir = Path("models")
    if not models_dir.exists():
        print("  ⚠ models/ directory not found (agent will run untrained)")
        return True

    models = list(models_dir.glob("*.pt"))
    if models:
        for model in models:
            print(f"  ✓ {model}")
        return True
    else:
        print("  ⚠ No .pt files found in models/ (agent will run untrained)")
        return True

def main():
    print("=" * 60)
    print("HAMLET VISUALIZATION VERIFICATION")
    print("=" * 60)
    print()

    backend_ok = check_backend()
    frontend_ok = check_frontend()
    models_ok = check_models()

    print()
    print("=" * 60)

    if backend_ok and frontend_ok:
        print("✓ VERIFICATION PASSED")
        print()
        print("To start the visualization:")
        print("  1. Terminal 1: uv run python demo_visualization.py")
        print("  2. Terminal 2: cd frontend && npm run dev")
        print("  3. Browser: http://localhost:5173")
        print()
    else:
        print("✗ VERIFICATION FAILED")
        print()
        print("Please fix the missing files above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
