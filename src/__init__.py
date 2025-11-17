"""F1 Racing RL - State-of-the-art reinforcement learning for F1 racing optimization."""

__version__ = "1.0.0"
__author__ = "F1 Racing RL Team"

from src.envs.f1_racing_env import F1RacingEnv
from src.physics.f1_car import F1Car
from src.tracks.circuit import Circuit

__all__ = ["F1RacingEnv", "F1Car", "Circuit"]
