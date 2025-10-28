"""
Simple demonstration of the Hamlet environment.

Run this to see the environment in action.
"""

from hamlet.environment.hamlet_env import HamletEnv


def main():
    """Run a simple demonstration episode."""
    print("=" * 60)
    print("Hamlet Environment Demonstration")
    print("=" * 60)

    # Create environment
    env = HamletEnv()
    print("\n✓ Environment created")

    # Reset
    obs = env.reset()
    print("✓ Environment reset")

    # Show initial state
    agent = env.agents["agent_0"]
    print(f"\nInitial agent position: ({agent.x}, {agent.y})")
    print(f"Initial meters:")
    for name, meter in agent.meters.meters.items():
        print(f"  {name:10s}: {meter.value:6.1f}")

    print("\n" + "-" * 60)
    print("Running episode with random actions...")
    print("-" * 60)

    import random

    total_reward = 0
    steps = 0
    max_steps = 100

    while steps < max_steps:
        # Random action
        action = random.randint(0, 4)
        action_names = ["UP", "DOWN", "LEFT", "RIGHT", "INTERACT"]

        obs, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1

        # Print every 10 steps
        if steps % 10 == 0:
            print(f"\nStep {steps}:")
            print(f"  Action: {action_names[action]}")
            print(f"  Position: ({agent.x}, {agent.y})")
            print(f"  Meters:")
            for name, meter in agent.meters.meters.items():
                status = "CRITICAL!" if meter.is_critical() else "OK"
                print(f"    {name:10s}: {meter.value:6.1f} [{status}]")
            print(f"  Step Reward: {reward:+.2f}")
            print(f"  Total Reward: {total_reward:+.2f}")

        if done:
            print(f"\n{'=' * 60}")
            print(f"Episode terminated at step {steps}")
            print(f"Final total reward: {total_reward:+.2f}")
            print(f"{'=' * 60}")
            break

    if not done:
        print(f"\n{'=' * 60}")
        print(f"Episode completed {steps} steps (max reached)")
        print(f"Final total reward: {total_reward:+.2f}")
        print(f"Agent survived!")
        print(f"{'=' * 60}")

    # Show final state
    print(f"\nFinal meters:")
    for name, meter in agent.meters.meters.items():
        normalized = meter.normalize()
        bar_length = int(normalized * 20)
        bar = "█" * bar_length + "░" * (20 - bar_length)
        print(f"  {name:10s}: [{bar}] {meter.value:6.1f}")

    print("\n✓ Demonstration complete!")


if __name__ == "__main__":
    main()
