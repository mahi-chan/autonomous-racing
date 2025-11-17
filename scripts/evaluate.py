#!/usr/bin/env python3
"""
Evaluation script for trained F1 Racing RL agents.

Usage:
    python scripts/evaluate.py --checkpoint models/sac_silverstone_best.zip --circuit silverstone --num-episodes 10
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from stable_baselines3 import SAC
from sb3_contrib import RecurrentPPO

from src.envs.f1_racing_env import F1RacingEnv


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate F1 Racing RL Agent")

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )

    parser.add_argument(
        "--circuit",
        type=str,
        default="silverstone",
        choices=["silverstone", "monaco", "spa"],
        help="Circuit to evaluate on"
    )

    parser.add_argument(
        "--num-episodes",
        type=int,
        default=10,
        help="Number of episodes to evaluate"
    )

    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic policy"
    )

    parser.add_argument(
        "--render",
        action="store_true",
        help="Render environment"
    )

    parser.add_argument(
        "--save-telemetry",
        type=str,
        default=None,
        help="Path to save telemetry data"
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to environment config"
    )

    return parser.parse_args()


def evaluate_agent(model, env, num_episodes, deterministic=True, render=False):
    """
    Evaluate agent performance.

    Args:
        model: Trained model
        env: Environment
        num_episodes: Number of episodes
        deterministic: Use deterministic policy
        render: Render environment

    Returns:
        Dictionary with evaluation metrics
    """
    episode_rewards = []
    episode_lengths = []
    lap_times = []
    best_lap_times = []
    final_distances = []

    all_telemetry = []

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0.0
        episode_length = 0

        # Reset state for recurrent policies
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)

        while not done:
            # Predict action
            if hasattr(model, 'predict'):
                if isinstance(model, RecurrentPPO):
                    action, lstm_states = model.predict(
                        obs,
                        state=lstm_states,
                        episode_start=episode_starts,
                        deterministic=deterministic
                    )
                    episode_starts = np.zeros((1,), dtype=bool)
                else:
                    action, _ = model.predict(obs, deterministic=deterministic)
            else:
                action = model.predict(obs, deterministic=deterministic)[0]

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1

            if render:
                env.render()

        # Collect metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        if 'lap_time' in info:
            lap_times.append(info['lap_time'])
        if 'best_lap_time' in info and info['best_lap_time'] is not None:
            best_lap_times.append(info['best_lap_time'])
        if 'distance' in info:
            final_distances.append(info['distance'])

        # Save telemetry
        episode_telemetry = env.get_episode_telemetry()
        all_telemetry.append(episode_telemetry)

        print(f"Episode {episode + 1}/{num_episodes}: "
              f"Reward = {episode_reward:.2f}, "
              f"Length = {episode_length}, "
              f"Distance = {info.get('distance', 0):.1f}m")

        if best_lap_times and best_lap_times[-1] > 0:
            print(f"  Best lap time: {best_lap_times[-1]:.3f}s")

    # Calculate statistics
    results = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_episode_length': np.mean(episode_lengths),
        'mean_distance': np.mean(final_distances) if final_distances else 0.0,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'telemetry': all_telemetry
    }

    if best_lap_times:
        results['mean_lap_time'] = np.mean(best_lap_times)
        results['best_lap_time'] = np.min(best_lap_times)
        results['lap_times'] = best_lap_times

    return results


def print_results(results):
    """Print evaluation results."""
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    print(f"Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"Mean Episode Length: {results['mean_episode_length']:.1f} steps")
    print(f"Mean Distance: {results['mean_distance']:.1f} meters")

    if 'mean_lap_time' in results:
        print(f"Mean Lap Time: {results['mean_lap_time']:.3f} seconds")
        print(f"Best Lap Time: {results['best_lap_time']:.3f} seconds")

    print("="*60 + "\n")


def save_telemetry(telemetry_list, filepath):
    """Save telemetry data to CSV."""
    # Combine all episodes
    all_data = []

    for episode_idx, episode_telemetry in enumerate(telemetry_list):
        for step_data in episode_telemetry:
            step_data['episode'] = episode_idx
            all_data.append(step_data)

    # Create DataFrame
    df = pd.DataFrame(all_data)

    # Save to CSV
    df.to_csv(filepath, index=False)
    print(f"Telemetry saved to: {filepath}")


def main():
    """Main evaluation function."""
    args = parse_args()

    print("\n" + "="*60)
    print(f"F1 Racing RL Evaluation")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Circuit: {args.circuit}")
    print(f"Num episodes: {args.num_episodes}")
    print(f"Deterministic: {args.deterministic}")
    print("="*60 + "\n")

    # Create environment
    if args.config:
        env = F1RacingEnv.from_config(args.config)
    else:
        env = F1RacingEnv(
            circuit_name=args.circuit,
            render_mode="human" if args.render else None
        )

    # Load model
    print("Loading model...")
    if "ppo" in args.checkpoint.lower() and "lstm" in args.checkpoint.lower():
        model = RecurrentPPO.load(args.checkpoint, env=env)
    else:
        # Assume SAC or standard model
        try:
            model = SAC.load(args.checkpoint, env=env)
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Attempting to load as generic model...")
            import torch
            model = torch.load(args.checkpoint)

    print("Model loaded successfully!\n")

    # Evaluate
    results = evaluate_agent(
        model=model,
        env=env,
        num_episodes=args.num_episodes,
        deterministic=args.deterministic,
        render=args.render
    )

    # Print results
    print_results(results)

    # Save telemetry if requested
    if args.save_telemetry:
        save_telemetry(results['telemetry'], args.save_telemetry)

    env.close()


if __name__ == "__main__":
    main()
