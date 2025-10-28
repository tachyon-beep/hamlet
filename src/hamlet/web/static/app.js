// Hamlet Web Interface JavaScript

class HamletVisualization {
    constructor() {
        this.ws = null;
        this.gridSize = 8;
        this.initializeGrid();
        this.initializeControls();
        this.connectWebSocket();
    }

    initializeGrid() {
        const gridElement = document.getElementById('grid');
        gridElement.innerHTML = '';

        for (let i = 0; i < this.gridSize * this.gridSize; i++) {
            const cell = document.createElement('div');
            cell.className = 'cell empty';
            cell.id = `cell-${i}`;
            gridElement.appendChild(cell);
        }
    }

    initializeControls() {
        document.getElementById('start-btn').addEventListener('click', () => {
            this.sendCommand('start');
        });

        document.getElementById('pause-btn').addEventListener('click', () => {
            this.sendCommand('pause');
        });

        document.getElementById('reset-btn').addEventListener('click', () => {
            this.sendCommand('reset');
            this.resetDisplay();
        });
    }

    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;

        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
            console.log('WebSocket connected');
        };

        this.ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this.handleUpdate(data);
            } catch (e) {
                console.log('Received:', event.data);
            }
        };

        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };

        this.ws.onclose = () => {
            console.log('WebSocket disconnected, reconnecting in 3s...');
            setTimeout(() => this.connectWebSocket(), 3000);
        };
    }

    handleUpdate(data) {
        if (data.type === 'state_update') {
            this.updateGrid(data.data.grid);
            this.updateMeters(data.data.meters);
            this.updateInfo(data.data.episode_info);
        }
    }

    updateGrid(gridData) {
        // Clear all cells
        for (let i = 0; i < this.gridSize * this.gridSize; i++) {
            const cell = document.getElementById(`cell-${i}`);
            cell.className = 'cell empty';
            cell.textContent = '';
        }

        // Update cells based on grid data
        // This will be implemented when backend sends grid data
    }

    updateMeters(meters) {
        if (!meters) return;

        this.updateMeter('energy', meters.energy);
        this.updateMeter('hygiene', meters.hygiene);
        this.updateMeter('satiation', meters.satiation);
        this.updateMeter('money', meters.money);
    }

    updateMeter(name, value) {
        const bar = document.getElementById(`${name}-bar`);
        const valueSpan = document.getElementById(`${name}-value`);

        if (bar && valueSpan) {
            const percentage = Math.max(0, Math.min(100, value));
            bar.style.width = `${percentage}%`;
            valueSpan.textContent = Math.round(value);

            // Color coding based on value
            if (percentage < 25) {
                bar.style.background = 'linear-gradient(90deg, #e74c3c, #c0392b)';
            } else if (percentage < 50) {
                bar.style.background = 'linear-gradient(90deg, #f39c12, #e67e22)';
            } else {
                bar.style.background = 'linear-gradient(90deg, #4a9eff, #6ac3ff)';
            }
        }
    }

    updateInfo(info) {
        if (!info) return;

        const episodeNum = document.getElementById('episode-num');
        const stepNum = document.getElementById('step-num');
        const totalReward = document.getElementById('total-reward');

        if (episodeNum) episodeNum.textContent = info.episode || 0;
        if (stepNum) stepNum.textContent = info.step || 0;
        if (totalReward) totalReward.textContent = (info.total_reward || 0).toFixed(2);
    }

    sendCommand(command) {
        fetch(`/api/${command}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => response.json())
        .then(data => {
            console.log(`${command} command sent:`, data);
        })
        .catch(error => {
            console.error(`Error sending ${command} command:`, error);
        });
    }

    resetDisplay() {
        this.updateMeters({
            energy: 100,
            hygiene: 100,
            satiation: 100,
            money: 50
        });
        this.updateInfo({
            episode: 0,
            step: 0,
            total_reward: 0
        });
        this.initializeGrid();
    }
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', () => {
    const viz = new HamletVisualization();
    console.log('Hamlet visualization initialized');
});
