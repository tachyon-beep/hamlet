"""
Minimal ASCII renderer for terminal visualization during training.

Keep it SIMPLE - just enough to see what's happening.
"""

from typing import Dict, Any


def render_simple_state(env_state: Dict[str, Any], episode: int, step: int, reward: float) -> str:
    """
    Render a simple ASCII view of the current state.

    Args:
        env_state: Environment render() output
        episode: Current episode number
        step: Current step in episode
        reward: Current episode reward

    Returns:
        ASCII string to print
    """
    # Get first agent (single agent for now)
    agent = env_state['agents'][0] if env_state.get('agents') else None
    if not agent:
        return "No agent found"

    # Simple grid (8x8)
    grid_size = env_state['grid']['width']
    grid = [['.' for _ in range(grid_size)] for _ in range(grid_size)]

    # Place agent
    agent_x, agent_y = agent['x'], agent['y']
    if 0 <= agent_x < grid_size and 0 <= agent_y < grid_size:
        grid[agent_y][agent_x] = 'A'

    # Place affordances
    if 'affordances' in env_state:
        for aff in env_state['affordances']:
            x, y = aff['x'], aff['y']
            if 0 <= x < grid_size and 0 <= y < grid_size:
                # Use first letter of affordance name
                grid[y][x] = aff['name'][0]

    # Build output
    lines = []
    lines.append(f"Episode {episode} | Step {step} | Reward: {reward:.1f}")
    lines.append("=" * 30)

    # Grid
    for row in grid:
        lines.append(' '.join(row))

    lines.append("=" * 30)

    # Meters (simple bars)
    meters = agent['meters']
    energy = meters.get('energy', 0)
    hygiene = meters.get('hygiene', 0)
    satiation = meters.get('satiation', 0)
    money = meters.get('money', 0)

    lines.append(f"Energy:    {_meter_bar(energy, 100)}")
    lines.append(f"Hygiene:   {_meter_bar(hygiene, 100)}")
    lines.append(f"Satiation: {_meter_bar(satiation, 100)}")
    lines.append(f"Money:     ${money:.0f}")

    return '\n'.join(lines)


def _meter_bar(value: float, max_value: float, width: int = 20) -> str:
    """Create a simple ASCII progress bar."""
    filled = int((value / max_value) * width)
    bar = '█' * filled + '░' * (width - filled)
    return f"[{bar}] {value:.0f}/{max_value:.0f}"


def clear_terminal():
    """Clear terminal screen (ANSI escape code)."""
    print('\033[2J\033[H', end='')
