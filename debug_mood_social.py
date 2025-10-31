"""Debug mood and social depletion."""

import torch
from src.townlet.environment.vectorized_env import VectorizedHamletEnv

# Test mood
print("=" * 60)
print("MOOD DEPLETION TEST")
print("=" * 60)

env = VectorizedHamletEnv(num_agents=1, grid_size=8, device=torch.device("cpu"))
env.reset()
env.positions[0] = torch.tensor([4, 4])

print(
    f"Initial: Mood={env.meters[0, 4].item() * 100:.1f}%, "
    f"Social={env.meters[0, 5].item() * 100:.1f}%, "
    f"Hygiene={env.meters[0, 1].item() * 100:.1f}%"
)

for step in range(10):
    before_mood = env.meters[0, 4].item()
    before_social = env.meters[0, 5].item()

    env.step(torch.tensor([4]))  # INTERACT

    after_mood = env.meters[0, 4].item()
    after_social = env.meters[0, 5].item()

    mood_loss = before_mood - after_mood
    social_loss = before_social - after_social

    print(
        f"Step {step + 1:2d}: Mood {before_mood * 100:5.1f}% → {after_mood * 100:5.1f}% "
        f"(loss: {mood_loss * 100:.3f}%) | "
        f"Social {before_social * 100:5.1f}% → {after_social * 100:5.1f}% "
        f"(loss: {social_loss * 100:.3f}%)"
    )

print(f"\nAfter 10 steps:")
print(f"  Mood: {env.meters[0, 4].item() * 100:.1f}%")
print(f"  Social: {env.meters[0, 5].item() * 100:.1f}%")
print(f"  Hygiene: {env.meters[0, 1].item() * 100:.1f}%")

# Check for cascading
print("\n" + "=" * 60)
print("HYPOTHESIS: Low social → cascading to mood?")
print("=" * 60)

env2 = VectorizedHamletEnv(num_agents=1, grid_size=8, device=torch.device("cpu"))
env2.reset()
env2.positions[0] = torch.tensor([4, 4])

# Run until social drops below 30% threshold
for step in range(100):
    social = env2.meters[0, 5].item()
    mood = env2.meters[0, 4].item()

    if step % 10 == 0:
        print(f"Step {step:3d}: Social={social * 100:5.1f}%, Mood={mood * 100:5.1f}%")

    env2.step(torch.tensor([4]))

    # Check if we crossed the 30% threshold
    social_after = env2.meters[0, 5].item()
    if social > 0.3 and social_after <= 0.3:
        print(f"\n🚨 CROSSED THRESHOLD at step {step}")
        print(f"   Social: {social * 100:.1f}% → {social_after * 100:.1f}%")

print(
    f"\nFinal: Social={env2.meters[0, 5].item() * 100:.1f}%, "
    f"Mood={env2.meters[0, 4].item() * 100:.1f}%"
)
