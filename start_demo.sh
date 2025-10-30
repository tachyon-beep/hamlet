#!/bin/bash
# Helper script to start the multi-day demo

set -e

echo "🎯 Hamlet Multi-Day Demo Launcher"
echo "=================================="
echo ""

# Check if checkpoints directory exists
if [ ! -d "checkpoints" ]; then
    echo "📁 Creating checkpoints directory..."
    mkdir checkpoints
fi

# Check if config exists
if [ ! -f "configs/townlet/sparse_adaptive.yaml" ]; then
    echo "❌ Config file not found: configs/townlet/sparse_adaptive.yaml"
    exit 1
fi

echo "Starting demo components in separate terminals..."
echo ""
echo "This will open 3 terminal windows:"
echo "  1. Training (burn mode) - Fast background training"
echo "  2. Live Inference - Human-speed visualization server"
echo "  3. Vue Frontend - Web UI"
echo ""
echo "Press Ctrl+C in each terminal to stop"
echo ""

# Function to open new terminal with command
open_terminal() {
    local title=$1
    local command=$2

    # Detect terminal emulator
    if command -v gnome-terminal &> /dev/null; then
        gnome-terminal --title="$title" -- bash -c "$command; exec bash"
    elif command -v xterm &> /dev/null; then
        xterm -T "$title" -e "$command; bash" &
    elif command -v konsole &> /dev/null; then
        konsole --new-tab -e bash -c "$command; exec bash" &
    else
        echo "⚠️  No supported terminal emulator found"
        echo "Please run these commands manually in separate terminals:"
        echo ""
        echo "Terminal 1: $command"
        return 1
    fi
}

# Start training
echo "🔥 Terminal 1: Starting background training..."
open_terminal "Hamlet Training" \
    "cd $(pwd) && source .venv/bin/activate && python -m hamlet.demo.runner configs/townlet/sparse_adaptive.yaml demo_state.db checkpoints"

sleep 2

# Start inference server
echo "🎬 Terminal 2: Starting live inference server..."
open_terminal "Hamlet Inference" \
    "cd $(pwd) && source .venv/bin/activate && python -m hamlet.demo.live_inference checkpoints 8766 0.2"

sleep 2

# Start frontend
echo "🌐 Terminal 3: Starting Vue frontend..."
open_terminal "Hamlet Frontend" \
    "cd $(pwd)/frontend && npm run dev"

echo ""
echo "✅ Demo started!"
echo ""
echo "📊 Open browser to: http://localhost:5173"
echo ""
echo "Monitor progress:"
echo "  - Training terminal: Episode progress, metrics"
echo "  - Inference terminal: Checkpoint loading, episode inference"
echo "  - Browser: Real-time agent visualization"
echo ""
echo "To stop: Ctrl+C in each terminal window"
