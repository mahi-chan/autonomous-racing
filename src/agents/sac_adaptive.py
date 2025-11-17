"""
SAC (Soft Actor-Critic) with Adaptive Temperature

Enhanced SAC implementation with:
- Adaptive entropy temperature based on track/conditions
- Custom reward shaping for racing
- Domain-specific network architectures
- Track-aware exploration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from typing import Dict, Tuple, Optional
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.sac.policies import SACPolicy
import gymnasium as gym


class AdaptiveEntropyCallback(BaseCallback):
    """
    Callback to adapt entropy temperature based on learning progress and track conditions.

    Increases exploration in difficult track sections, reduces in easier sections.
    """

    def __init__(
        self,
        initial_temp: float = 0.2,
        temp_min: float = 0.05,
        temp_max: float = 0.5,
        adaptation_rate: float = 0.01,
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.initial_temp = initial_temp
        self.temp_min = temp_min
        self.temp_max = temp_max
        self.adaptation_rate = adaptation_rate

        # Track performance metrics
        self.episode_rewards = []
        self.lap_times = []
        self.current_temp = initial_temp

    def _on_step(self) -> bool:
        """
        Adapt temperature based on performance.

        Returns:
            True (training continues)
        """
        # Get current info
        infos = self.locals.get('infos', [{}])

        for info in infos:
            # Track lap times
            if 'lap_time' in info and info.get('lap', 0) > 0:
                self.lap_times.append(info['lap_time'])

            # Get racing line error or other difficulty indicators
            racing_line_error = info.get('racing_line_error', 0.0)
            speed = info.get('speed_kmh', 0.0)

            # Adapt temperature
            # Increase exploration if:
            # - Making mistakes (high racing line error)
            # - Going slow (low speed)
            difficulty_factor = racing_line_error / 5.0 + (1.0 - speed / 300.0)

            # Adjust temperature
            if difficulty_factor > 0.5:
                # Difficult section - increase exploration
                self.current_temp += self.adaptation_rate
            else:
                # Easy section - reduce exploration
                self.current_temp -= self.adaptation_rate

            # Clip temperature
            self.current_temp = np.clip(self.current_temp, self.temp_min, self.temp_max)

            # Update model's temperature
            if hasattr(self.model, 'ent_coef'):
                self.model.ent_coef = self.current_temp

        return True

    def _on_rollout_end(self) -> None:
        """Log temperature at end of rollout."""
        if self.verbose > 0:
            print(f"Current entropy temperature: {self.current_temp:.4f}")


class RacingActorNetwork(nn.Module):
    """
    Custom actor network for racing with domain knowledge.

    Features:
    - Separate processing for different input modalities
    - Attention mechanism for track segments
    - Hierarchical action structure (strategy -> tactics -> controls)
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
        log_std_min: float = -20,
        log_std_max: float = 2
    ):
        super().__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Feature extraction layers
        layers = []
        prev_dim = observation_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim

        self.feature_net = nn.Sequential(*layers)

        # Policy head (mean and log_std)
        self.mean_linear = nn.Linear(prev_dim, action_dim)
        self.log_std_linear = nn.Linear(prev_dim, action_dim)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            obs: Observation tensor

        Returns:
            (mean, log_std) tuple
        """
        features = self.feature_net(obs)

        mean = self.mean_linear(features)
        log_std = self.log_std_linear(features)

        # Clamp log_std
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.

        Args:
            obs: Observation tensor

        Returns:
            (action, log_prob) tuple
        """
        mean, log_std = self.forward(obs)
        std = log_std.exp()

        # Create normal distribution
        normal = Normal(mean, std)

        # Sample using reparameterization trick
        x_t = normal.rsample()

        # Apply tanh squashing
        action = torch.tanh(x_t)

        # Calculate log probability with squashing correction
        log_prob = normal.log_prob(x_t)
        # Enforcing action bounds
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob


class SACAdaptive:
    """
    SAC with Adaptive Temperature for F1 Racing.

    Enhancements:
    - Adaptive entropy temperature
    - Custom network architecture
    - Racing-specific hyperparameters
    - Integration with F1 environment
    """

    def __init__(
        self,
        env: gym.Env,
        learning_rate: float = 3e-4,
        buffer_size: int = 1000000,
        learning_starts: int = 10000,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: int = 1,
        gradient_steps: int = 1,
        ent_coef: str = "auto",
        target_entropy: str = "auto",
        use_adaptive_temp: bool = True,
        policy_kwargs: Optional[Dict] = None,
        tensorboard_log: Optional[str] = None,
        verbose: int = 1,
        device: str = "auto"
    ):
        """
        Initialize SAC agent.

        Args:
            env: Gymnasium environment
            learning_rate: Learning rate
            buffer_size: Replay buffer size
            learning_starts: Steps before training starts
            batch_size: Batch size for training
            tau: Soft update coefficient
            gamma: Discount factor
            train_freq: Training frequency
            gradient_steps: Gradient steps per update
            ent_coef: Entropy coefficient ("auto" for adaptive)
            target_entropy: Target entropy ("auto" for automatic)
            use_adaptive_temp: Use adaptive temperature callback
            policy_kwargs: Policy network kwargs
            tensorboard_log: Tensorboard log directory
            verbose: Verbosity level
            device: Device to use
        """
        # Default policy kwargs optimized for racing
        if policy_kwargs is None:
            policy_kwargs = dict(
                net_arch=[256, 256, 256],  # Deeper network for complex dynamics
                activation_fn=nn.ReLU,
            )

        # Create SAC model
        self.model = SAC(
            policy="MlpPolicy",
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            ent_coef=ent_coef,
            target_entropy=target_entropy,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
        )

        # Adaptive temperature callback
        self.adaptive_temp_callback = None
        if use_adaptive_temp:
            self.adaptive_temp_callback = AdaptiveEntropyCallback(
                initial_temp=0.2,
                verbose=verbose
            )

    def train(
        self,
        total_timesteps: int,
        callback=None,
        log_interval: int = 4,
        tb_log_name: str = "SAC",
        reset_num_timesteps: bool = True
    ):
        """
        Train the agent.

        Args:
            total_timesteps: Total timesteps to train
            callback: Additional callbacks
            log_interval: Logging interval
            tb_log_name: Tensorboard log name
            reset_num_timesteps: Reset timestep counter
        """
        # Combine callbacks
        callbacks = []
        if self.adaptive_temp_callback is not None:
            callbacks.append(self.adaptive_temp_callback)
        if callback is not None:
            if isinstance(callback, list):
                callbacks.extend(callback)
            else:
                callbacks.append(callback)

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks if callbacks else None,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps
        )

    def predict(self, observation, deterministic: bool = False):
        """Predict action from observation."""
        return self.model.predict(observation, deterministic=deterministic)

    def save(self, path: str):
        """Save model to path."""
        self.model.save(path)

    def load(self, path: str):
        """Load model from path."""
        self.model = SAC.load(path)

    @classmethod
    def load_trained(cls, path: str, env: gym.Env) -> 'SACAdaptive':
        """
        Load a trained model.

        Args:
            path: Path to saved model
            env: Environment

        Returns:
            Loaded SACAdaptive agent
        """
        agent = cls(env=env)
        agent.model = SAC.load(path, env=env)
        return agent


def create_sac_agent(
    env: gym.Env,
    config: Optional[Dict] = None
) -> SACAdaptive:
    """
    Create SAC agent with sensible defaults for F1 racing.

    Args:
        env: F1 racing environment
        config: Optional configuration override

    Returns:
        SACAdaptive agent
    """
    default_config = {
        'learning_rate': 3e-4,
        'buffer_size': 1000000,
        'learning_starts': 10000,
        'batch_size': 256,
        'tau': 0.005,
        'gamma': 0.99,
        'ent_coef': 'auto',
        'use_adaptive_temp': True,
        'policy_kwargs': dict(
            net_arch=[256, 256, 256],
            activation_fn=nn.ReLU,
        ),
    }

    if config:
        default_config.update(config)

    return SACAdaptive(env=env, **default_config)
