#!/usr/bin/env python3
"""
Training script for F1 Racing RL agents.

Usage:
    python scripts/train.py --circuit silverstone --algorithm sac --config configs/f1_2024.yaml
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import yaml

from src.envs.f1_racing_env import F1RacingEnv
from src.agents.sac_adaptive import create_sac_agent
from src.agents.ppo_lstm import create_ppo_lstm_agent
from src.agents.model_based import ModelBasedAgent
from src.agents.meta_rl import MetaRLAgent


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train F1 Racing RL Agent")

    parser.add_argument(
        "--circuit",
        type=str,
        default="silverstone",
        choices=["silverstone", "monaco", "spa"],
        help="Circuit to train on"
    )

    parser.add_argument(
        "--algorithm",
        type=str,
        default="sac",
        choices=["sac", "ppo-lstm", "model-based", "meta-rl"],
        help="RL algorithm to use"
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML file"
    )

    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=1000000,
        help="Total timesteps to train"
    )

    parser.add_argument(
        "--save-dir",
        type=str,
        default="models",
        help="Directory to save models"
    )

    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory for tensorboard logs"
    )

    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Number of parallel environments"
    )

    parser.add_argument(
        "--eval-freq",
        type=int,
        default=10000,
        help="Evaluation frequency"
    )

    parser.add_argument(
        "--save-freq",
        type=int,
        default=50000,
        help="Checkpoint save frequency"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use for training"
    )

    return parser.parse_args()


def make_env(circuit_name: str, config_path: str = None, rank: int = 0, seed: int = 0):
    """
    Create environment function for vectorization.

    Args:
        circuit_name: Circuit name
        config_path: Path to config file
        rank: Process rank
        seed: Base random seed

    Returns:
        Callable that creates environment
    """
    def _init():
        if config_path:
            env = F1RacingEnv.from_config(config_path)
        else:
            env = F1RacingEnv(
                circuit_name=circuit_name,
                enable_dynamic_conditions=True
            )

        env.reset(seed=seed + rank)
        return env

    return _init


def train_sac(env, args, config):
    """Train SAC agent."""
    print("\n" + "="*60)
    print("Training SAC with Adaptive Temperature")
    print("="*60 + "\n")

    # Create agent
    agent = create_sac_agent(env, config.get('sac', {}))

    # Create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=os.path.join(args.save_dir, f"sac_{args.circuit}"),
        name_prefix="sac_model"
    )

    eval_env = make_env(args.circuit, args.config)()
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(args.save_dir, f"sac_{args.circuit}_best"),
        log_path=os.path.join(args.log_dir, f"sac_{args.circuit}_eval"),
        eval_freq=args.eval_freq,
        deterministic=True,
        render=False
    )

    # Train
    agent.train(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        tb_log_name=f"SAC_{args.circuit}"
    )

    # Save final model
    final_path = os.path.join(args.save_dir, f"sac_{args.circuit}_final")
    agent.save(final_path)
    print(f"\nFinal model saved to: {final_path}")

    return agent


def train_ppo_lstm(env, args, config):
    """Train PPO-LSTM agent."""
    print("\n" + "="*60)
    print("Training PPO with LSTM for Temporal Dynamics")
    print("="*60 + "\n")

    # Create agent
    agent = create_ppo_lstm_agent(env, config.get('ppo_lstm', {}))

    # Create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=os.path.join(args.save_dir, f"ppo_lstm_{args.circuit}"),
        name_prefix="ppo_lstm_model"
    )

    # Train
    agent.train(
        total_timesteps=args.total_timesteps,
        callback=checkpoint_callback,
        tb_log_name=f"PPO_LSTM_{args.circuit}"
    )

    # Save final model
    final_path = os.path.join(args.save_dir, f"ppo_lstm_{args.circuit}_final")
    agent.save(final_path)
    print(f"\nFinal model saved to: {final_path}")

    return agent


def train_model_based(env, args, config):
    """Train model-based agent."""
    print("\n" + "="*60)
    print("Training Model-Based RL (Dreamer)")
    print("="*60 + "\n")

    agent = ModelBasedAgent(
        env=env,
        algorithm="dreamer",
        **config.get('model_based', {})
    )

    agent.train(total_timesteps=args.total_timesteps)

    final_path = os.path.join(args.save_dir, f"dreamer_{args.circuit}_final.pth")
    agent.save(final_path)
    print(f"\nFinal model saved to: {final_path}")

    return agent


def train_meta_rl(env, args, config):
    """Train meta-RL agent."""
    print("\n" + "="*60)
    print("Training Meta-RL for Track Adaptation")
    print("="*60 + "\n")

    agent = MetaRLAgent(
        env=env,
        **config.get('meta_rl', {})
    )

    print("Meta-RL requires multiple tracks for meta-training.")
    print("This is a placeholder - implement full meta-training loop as needed.")

    return agent


def main():
    """Main training function."""
    args = parse_args()

    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Load config
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

    print("\n" + "="*60)
    print(f"F1 Racing RL Training")
    print("="*60)
    print(f"Circuit: {args.circuit}")
    print(f"Algorithm: {args.algorithm}")
    print(f"Total timesteps: {args.total_timesteps:,}")
    print(f"Num environments: {args.num_envs}")
    print(f"Seed: {args.seed}")
    print(f"Device: {args.device}")
    print("="*60 + "\n")

    # Create environment(s)
    if args.num_envs > 1:
        print(f"Creating {args.num_envs} parallel environments...")
        env = SubprocVecEnv([
            make_env(args.circuit, args.config, i, args.seed)
            for i in range(args.num_envs)
        ])
    else:
        env = make_env(args.circuit, args.config, 0, args.seed)()

    # Train based on algorithm
    if args.algorithm == "sac":
        agent = train_sac(env, args, config)
    elif args.algorithm == "ppo-lstm":
        agent = train_ppo_lstm(env, args, config)
    elif args.algorithm == "model-based":
        agent = train_model_based(env, args, config)
    elif args.algorithm == "meta-rl":
        agent = train_meta_rl(env, args, config)
    else:
        raise ValueError(f"Unknown algorithm: {args.algorithm}")

    print("\n" + "="*60)
    print("Training completed successfully!")
    print("="*60 + "\n")

    env.close()


if __name__ == "__main__":
    main()
