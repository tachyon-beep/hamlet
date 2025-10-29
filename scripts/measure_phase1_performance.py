"""Measure Phase 1 performance baseline."""

import time
import torch

from townlet.environment.vectorized_env import VectorizedHamletEnv
from townlet.curriculum.static import StaticCurriculum
from townlet.exploration.epsilon_greedy import EpsilonGreedyExploration
from townlet.population.vectorized import VectorizedPopulation


def measure_performance(num_agents: int, num_steps: int = 1000, device_type: str = 'cpu'):
    """Measure training performance."""
    device = torch.device(device_type)

    # Setup
    env = VectorizedHamletEnv(num_agents=num_agents, grid_size=8, device=device)
    curriculum = StaticCurriculum(difficulty_level=0.5, reward_mode='shaped')
    exploration = EpsilonGreedyExploration(epsilon=0.1, epsilon_decay=1.0)

    population = VectorizedPopulation(
        env=env,
        curriculum=curriculum,
        exploration=exploration,
        agent_ids=[f'agent_{i}' for i in range(num_agents)],
        device=device,
    )

    # Warmup
    population.reset()
    for _ in range(10):
        population.step_population(env)

    # Measure performance
    population.reset()
    start_time = time.time()

    for _ in range(num_steps):
        population.step_population(env)

    end_time = time.time()

    # Calculate metrics
    elapsed = end_time - start_time
    fps = num_steps / elapsed

    print(f"\n{'='*50}")
    print(f"Performance Baseline: n={num_agents}, device={device_type}")
    print(f"{'='*50}")
    print(f"Total steps: {num_steps}")
    print(f"Elapsed time: {elapsed:.2f}s")
    print(f"Steps/sec: {fps:.1f} FPS")
    print(f"{'='*50}\n")

    return {
        'num_agents': num_agents,
        'device': device_type,
        'num_steps': num_steps,
        'elapsed': elapsed,
        'fps': fps,
    }


if __name__ == '__main__':
    print("Measuring Phase 1 Performance Baseline...")

    results = []

    # CPU baseline
    results.append(measure_performance(num_agents=1, num_steps=1000, device_type='cpu'))
    results.append(measure_performance(num_agents=5, num_steps=1000, device_type='cpu'))

    # GPU baseline (if available)
    if torch.cuda.is_available():
        print("\nGPU available - measuring GPU performance...")
        results.append(measure_performance(num_agents=1, num_steps=1000, device_type='cuda'))
        results.append(measure_performance(num_agents=5, num_steps=1000, device_type='cuda'))
    else:
        print("\nGPU not available - skipping GPU measurements")

    # Summary
    print("\n" + "="*50)
    print("PERFORMANCE SUMMARY")
    print("="*50)
    for result in results:
        print(f"n={result['num_agents']}, device={result['device']}: "
              f"{result['fps']:.1f} FPS")
    print("="*50)
