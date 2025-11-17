"""
Quickstart Example - F1 Racing RL

Demonstrates basic usage of the system.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.envs.f1_racing_env import F1RacingEnv
from src.agents.sac_adaptive import create_sac_agent
import numpy as np


def main():
    print("="*60)
    print("F1 Racing RL - Quickstart Example")
    print("="*60 + "\n")

    # 1. Create environment
    print("Creating F1 Racing environment (Silverstone)...")
    env = F1RacingEnv(
        circuit_name='silverstone',
        enable_dynamic_conditions=True
    )
    print("✓ Environment created\n")

    # 2. Test random agent
    print("Testing with random policy (5 episodes)...")
    for episode in range(5):
        obs, info = env.reset()
        done = False
        episode_reward = 0.0
        steps = 0

        while not done and steps < 1000:
            # Random action
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            steps += 1

        print(f"  Episode {episode + 1}: Reward = {episode_reward:.2f}, "
              f"Steps = {steps}, Distance = {info.get('distance', 0):.1f}m")

    print("\n✓ Random policy test complete\n")

    # 3. Create RL agent
    print("Creating SAC agent...")
    agent = create_sac_agent(env)
    print("✓ SAC agent created\n")

    # 4. Train agent (short training for demo)
    print("Training agent (10,000 timesteps - DEMO ONLY)...")
    print("For real training, use 1,000,000+ timesteps\n")

    agent.train(
        total_timesteps=10000,
        log_interval=1
    )

    print("\n✓ Training complete\n")

    # 5. Test trained agent
    print("Testing trained agent (3 episodes)...")
    for episode in range(3):
        obs, info = env.reset()
        done = False
        episode_reward = 0.0
        steps = 0

        while not done and steps < 1000:
            # Get action from trained agent
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            steps += 1

        print(f"  Episode {episode + 1}: Reward = {episode_reward:.2f}, "
              f"Steps = {steps}, Distance = {info.get('distance', 0):.1f}m")

    print("\n✓ Evaluation complete\n")

    # 6. Save agent
    save_path = "models/quickstart_agent"
    print(f"Saving agent to {save_path}...")
    agent.save(save_path)
    print("✓ Agent saved\n")

    print("="*60)
    print("Quickstart Complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Train for longer: python scripts/train.py --total-timesteps 1000000")
    print("2. Evaluate: python scripts/evaluate.py --checkpoint models/quickstart_agent.zip")
    print("3. Try different circuits: --circuit monaco")
    print("4. Read USAGE_GUIDE.md for detailed instructions")

    env.close()


if __name__ == "__main__":
    main()
