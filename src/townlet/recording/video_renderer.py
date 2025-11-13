"""
Video renderer for episode replay.

Renders episode frames to numpy arrays for video encoding.
Uses matplotlib for high-quality visualization suitable for YouTube.
"""

import logging

import matplotlib
import numpy as np

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

# Affordance colors (matching frontend)
AFFORDANCE_COLORS = {
    "Bed": "#9b59b6",
    "Shower": "#3498db",
    "Fridge": "#2ecc71",
    "Job": "#f39c12",
    "Gym": "#e74c3c",
    "Bar": "#e67e22",
    "CoffeeShop": "#95a5a6",
    "Hospital": "#1abc9c",
}

# Meter colors
METER_COLORS = {
    "energy": "#f1c40f",
    "hygiene": "#3498db",
    "satiation": "#2ecc71",
    "money": "#27ae60",
    "health": "#e74c3c",
    "fitness": "#e67e22",
    "mood": "#9b59b6",
    "social": "#1abc9c",
}

# Action names
# Note: Using simple characters to avoid font glyph warnings with DejaVu Sans
ACTION_NAMES = ["↑ Up", "↓ Down", "← Left", "→ Right", "⚡ Interact", "• Wait"]


class EpisodeVideoRenderer:
    """Renders episode frames for video export.

    Produces high-quality matplotlib visualizations optimized for YouTube.
    """

    def __init__(self, grid_size: int = 8, dpi: int = 100, style: str = "dark"):
        """Initialize video renderer.

        Args:
            grid_size: Grid dimensions (8 for 8×8 grid)
            dpi: Dots per inch (100 = 1600×900, 150 = 2400×1350)
            style: Visual style ("dark" or "light")
        """
        self.grid_size = grid_size
        self.dpi = dpi
        self.style = style

        # Set style
        if style == "dark":
            plt.style.use("dark_background")
            self.bg_color = "#1a1a2e"
            self.grid_color = "#16213e"
            self.text_color = "#eee"
            self.agent_color = "#00d9ff"
        else:
            self.bg_color = "#ffffff"
            self.grid_color = "#f0f0f0"
            self.text_color = "#333"
            self.agent_color = "#0066cc"

    def render_frame(
        self,
        step_data: dict,
        metadata: dict,
        affordances: dict,
    ) -> np.ndarray:
        """Render single frame to numpy array.

        Args:
            step_data: Step data dict from recording
            metadata: Episode metadata dict
            affordances: Affordance layout dict {name: [x, y]}

        Returns:
            RGB numpy array (H, W, 3) uint8
        """
        # Create figure (16:9 aspect ratio for YouTube)
        fig = Figure(figsize=(16, 9), dpi=self.dpi)
        fig.patch.set_facecolor(self.bg_color)
        canvas = FigureCanvasAgg(fig)

        # Create grid layout
        gs = fig.add_gridspec(2, 3, width_ratios=[2, 1, 1], height_ratios=[3, 1])

        # Main grid view (left, spans both rows)
        ax_grid = fig.add_subplot(gs[:, 0])
        self._render_grid(ax_grid, step_data, affordances)

        # Meters (top right)
        ax_meters = fig.add_subplot(gs[0, 1])
        self._render_meters(ax_meters, step_data)

        # Info panel (top right, second column)
        ax_info = fig.add_subplot(gs[0, 2])
        self._render_info(ax_info, step_data, metadata)

        # Q-values (bottom right, spans two columns)
        ax_qvalues = fig.add_subplot(gs[1, 1:])
        self._render_qvalues(ax_qvalues, step_data)

        # Tight layout
        fig.tight_layout()

        # Render to numpy array
        canvas.draw()
        buf = canvas.buffer_rgba()
        frame = np.asarray(buf)

        # Convert RGBA to RGB
        frame_rgb = frame[:, :, :3].copy()

        plt.close(fig)

        return frame_rgb

    def _render_grid(self, ax, step_data: dict, affordances: dict):
        """Render grid with agent and affordances."""
        ax.set_xlim(-0.5, self.grid_size - 0.5)
        ax.set_ylim(-0.5, self.grid_size - 0.5)
        ax.set_aspect("equal")
        ax.invert_yaxis()  # Y increases downward

        # Grid background
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                rect = patches.Rectangle(
                    (x - 0.5, y - 0.5),
                    1,
                    1,
                    linewidth=1,
                    edgecolor=self.grid_color,
                    facecolor=self.grid_color,
                    alpha=0.3,
                )
                ax.add_patch(rect)

        # Extract affordance positions (handle both formats)
        # Format 1: {name: [x, y], ...}
        # Format 2: {positions: {name: [x, y], ...}, ordering: [...]}
        if "positions" in affordances and isinstance(affordances["positions"], dict):
            affordance_positions = affordances["positions"]
        else:
            affordance_positions = affordances

        # Draw affordances
        for name, pos in affordance_positions.items():
            x, y = pos[0], pos[1]
            color = AFFORDANCE_COLORS.get(name, "#95a5a6")

            rect = patches.Rectangle(
                (x - 0.4, y - 0.4),
                0.8,
                0.8,
                linewidth=2,
                edgecolor=color,
                facecolor=color,
                alpha=0.7,
            )
            ax.add_patch(rect)

            # Affordance label
            ax.text(x, y, name[:3], ha="center", va="center", fontsize=8, color="white", weight="bold")

        # Draw agent
        agent_pos = step_data["position"]
        agent_x, agent_y = agent_pos[0], agent_pos[1]

        circle = patches.Circle(
            (agent_x, agent_y),
            0.35,
            facecolor=self.agent_color,
            linewidth=2,
            edgecolor="white",
            zorder=10,
        )
        ax.add_patch(circle)

        # Agent label
        ax.text(agent_x, agent_y, "A", ha="center", va="center", fontsize=12, color="white", weight="bold", zorder=11)

        # Action arrow
        action = step_data["action"]
        if action < 4:  # Movement actions
            dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][action]
            ax.arrow(agent_x, agent_y, dx * 0.3, dy * 0.3, head_width=0.2, head_length=0.15, fc="yellow", ec="yellow", alpha=0.8, zorder=9)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Grid View", color=self.text_color, fontsize=14, weight="bold")

    def _render_meters(self, ax, step_data: dict):
        """Render meter bars."""
        meters = step_data["meters"]
        # TODO: Get meter names from episode metadata to support variable meter counts
        # For now, assumes 8 standard meters (sufficient for all production configs)
        meter_names = ["energy", "hygiene", "satiation", "money", "health", "fitness", "mood", "social"]

        ax.set_xlim(0, 1)
        ax.set_ylim(0, len(meters))
        ax.set_yticks(range(len(meters)))
        ax.set_yticklabels([name.capitalize() for name in meter_names])
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_xticklabels(["0", "50", "100"])

        # Draw bars
        for i, (name, value) in enumerate(zip(meter_names, meters)):
            color = METER_COLORS.get(name, "#95a5a6")

            # Background bar
            ax.barh(i, 1.0, height=0.8, left=0, color=self.grid_color, alpha=0.5)

            # Value bar
            ax.barh(i, value, height=0.8, left=0, color=color, alpha=0.9)

            # Value text
            ax.text(value + 0.02, i, f"{value * 100:.0f}%", va="center", fontsize=9, color=self.text_color)

        ax.invert_yaxis()
        ax.set_title("Agent Meters", color=self.text_color, fontsize=12, weight="bold")
        ax.tick_params(colors=self.text_color)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color(self.text_color)
        ax.spines["left"].set_color(self.text_color)

    def _render_info(self, ax, step_data: dict, metadata: dict):
        """Render episode info panel."""
        ax.axis("off")

        info_lines = [
            f"Episode: {metadata['episode_id']}",
            f"Stage: {metadata['curriculum_stage']}",
            "",
            f"Step: {step_data['step']}",
            f"Survival: {metadata['survival_steps']}",
            "",
            f"Action: {ACTION_NAMES[step_data['action']]}",
            f"Reward: {step_data['reward']:.2f}",
            f"Total: {metadata['total_reward']:.1f}",
        ]

        # Add temporal mechanics if present
        if step_data.get("time_of_day") is not None:
            hour = step_data["time_of_day"]
            time_str = f"{hour:02d}:00"
            info_lines.append("")
            info_lines.append(f"Time: {time_str}")

        if step_data.get("interaction_progress") is not None:
            progress = step_data["interaction_progress"]
            info_lines.append(f"Progress: {progress * 100:.0f}%")

        # Render text
        y_pos = 0.95
        for line in info_lines:
            ax.text(
                0.1,
                y_pos,
                line,
                transform=ax.transAxes,
                fontsize=10,
                color=self.text_color,
                verticalalignment="top",
                weight="bold" if line.startswith(("Episode", "Stage", "Step")) else "normal",
            )
            y_pos -= 0.08

        ax.set_title("Episode Info", color=self.text_color, fontsize=12, weight="bold", loc="left")

    def _render_qvalues(self, ax, step_data: dict):
        """Render Q-values bar chart."""
        q_values = step_data.get("q_values")

        if q_values is None or len(q_values) == 0:
            ax.axis("off")
            ax.text(
                0.5,
                0.5,
                "Q-values not recorded",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=10,
                color=self.text_color,
                alpha=0.5,
            )
            return

        # Action names (handle both 5 and 6 action cases)
        all_action_names = ["Up", "Down", "Left", "Right", "Interact", "Wait"]
        action_names = all_action_names[: len(q_values)]
        action = step_data["action"]

        # Color bars
        colors = ["#95a5a6"] * len(q_values)
        colors[action] = "#2ecc71"  # Highlight chosen action

        # Draw bars
        bars = ax.bar(action_names, q_values, color=colors, alpha=0.8, edgecolor="white", linewidth=1.5)

        # Value labels on bars
        for bar, val in zip(bars, q_values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color=self.text_color,
                weight="bold",
            )

        ax.set_ylabel("Q-Value", color=self.text_color, fontsize=10)
        ax.set_title("Action Q-Values (Green = Chosen)", color=self.text_color, fontsize=12, weight="bold")
        ax.tick_params(colors=self.text_color)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color(self.text_color)
        ax.spines["left"].set_color(self.text_color)
        ax.grid(axis="y", alpha=0.3, color=self.text_color)
