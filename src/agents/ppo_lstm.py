"""
PPO with LSTM/GRU for Temporal Dynamics

Implements PPO with recurrent policies to handle:
- Tire degradation over time
- Fuel load changes
- Racing line consistency
- Strategy planning
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple, Type
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
import gymnasium as gym
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy


class LSTMFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom LSTM-based feature extractor for racing.

    Processes temporal sequences to learn:
    - Tire degradation patterns
    - Fuel consumption trends
    - Racing line consistency
    - Track learning progression
    """

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 256,
        lstm_hidden_size: int = 256,
        num_lstm_layers: int = 2,
        use_gru: bool = False
    ):
        super().__init__(observation_space, features_dim)

        # Input dimension
        self.input_dim = int(np.prod(observation_space.shape))

        # Choose RNN type
        rnn_class = nn.GRU if use_gru else nn.LSTM

        # RNN layers
        self.rnn = rnn_class(
            input_size=self.input_dim,
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=0.1 if num_lstm_layers > 1 else 0.0
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(lstm_hidden_size, features_dim),
            nn.ReLU(),
        )

        self.use_gru = use_gru
        self.num_layers = num_lstm_layers
        self.hidden_size = lstm_hidden_size

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LSTM.

        Args:
            observations: Observation tensor [batch, seq_len, obs_dim] or [batch, obs_dim]

        Returns:
            Features [batch, features_dim]
        """
        # Handle both sequential and single-step inputs
        if len(observations.shape) == 2:
            # Add sequence dimension
            observations = observations.unsqueeze(1)  # [batch, 1, obs_dim]

        # Pass through RNN
        rnn_output, _ = self.rnn(observations)

        # Take last timestep output
        last_output = rnn_output[:, -1, :]  # [batch, hidden_size]

        # Project to features
        features = self.output_proj(last_output)

        return features


class RacingLSTMPolicy(RecurrentActorCriticPolicy):
    """
    Custom recurrent policy for racing with domain-specific architecture.
    """

    def __init__(
        self,
        *args,
        lstm_hidden_size: int = 256,
        num_lstm_layers: int = 2,
        use_gru: bool = False,
        **kwargs
    ):
        self.lstm_hidden_size = lstm_hidden_size
        self.num_lstm_layers = num_lstm_layers
        self.use_gru = use_gru

        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        """Build the LSTM-based feature extractor."""
        # Override to use LSTM features
        pass


class PPO_LSTM:
    """
    PPO with LSTM for F1 Racing.

    Uses recurrent policies to handle:
    - Long-term dependencies (tire deg, fuel)
    - Temporal patterns
    - Strategy planning
    """

    def __init__(
        self,
        env: gym.Env,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        lstm_hidden_size: int = 256,
        num_lstm_layers: int = 2,
        use_gru: bool = False,
        policy_kwargs: Optional[Dict] = None,
        tensorboard_log: Optional[str] = None,
        verbose: int = 1,
        device: str = "auto"
    ):
        """
        Initialize PPO-LSTM agent.

        Args:
            env: Gymnasium environment
            learning_rate: Learning rate
            n_steps: Steps per rollout
            batch_size: Minibatch size
            n_epochs: Epochs per update
            gamma: Discount factor
            gae_lambda: GAE lambda
            clip_range: PPO clip range
            ent_coef: Entropy coefficient
            vf_coef: Value function coefficient
            max_grad_norm: Max gradient norm
            lstm_hidden_size: LSTM hidden size
            num_lstm_layers: Number of LSTM layers
            use_gru: Use GRU instead of LSTM
            policy_kwargs: Additional policy kwargs
            tensorboard_log: Tensorboard log directory
            verbose: Verbosity level
            device: Device to use
        """
        # Build policy kwargs with LSTM
        if policy_kwargs is None:
            policy_kwargs = {}

        policy_kwargs.update({
            'lstm_hidden_size': lstm_hidden_size,
            'n_lstm_layers': num_lstm_layers,
            'enable_critic_lstm': True,
            'shared_lstm': False,  # Separate LSTM for actor and critic
        })

        # Create RecurrentPPO model (from sb3-contrib)
        self.model = RecurrentPPO(
            policy="MlpLstmPolicy",
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
        )

        self.lstm_hidden_size = lstm_hidden_size
        self.num_lstm_layers = num_lstm_layers

    def train(
        self,
        total_timesteps: int,
        callback=None,
        log_interval: int = 1,
        tb_log_name: str = "PPO_LSTM",
        reset_num_timesteps: bool = True
    ):
        """
        Train the agent.

        Args:
            total_timesteps: Total timesteps to train
            callback: Callback(s)
            log_interval: Logging interval
            tb_log_name: Tensorboard log name
            reset_num_timesteps: Reset timestep counter
        """
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps
        )

    def predict(
        self,
        observation,
        state=None,
        episode_start=None,
        deterministic: bool = False
    ):
        """
        Predict action from observation.

        Args:
            observation: Current observation
            state: LSTM hidden state
            episode_start: Episode start flag
            deterministic: Use deterministic policy

        Returns:
            (action, state) tuple
        """
        return self.model.predict(
            observation,
            state=state,
            episode_start=episode_start,
            deterministic=deterministic
        )

    def save(self, path: str):
        """Save model to path."""
        self.model.save(path)

    def load(self, path: str):
        """Load model from path."""
        self.model = RecurrentPPO.load(path)

    @classmethod
    def load_trained(cls, path: str, env: gym.Env) -> 'PPO_LSTM':
        """
        Load a trained model.

        Args:
            path: Path to saved model
            env: Environment

        Returns:
            Loaded PPO_LSTM agent
        """
        agent = cls(env=env)
        agent.model = RecurrentPPO.load(path, env=env)
        return agent


class TemporalAwareCallback:
    """
    Callback to monitor temporal learning.

    Tracks how well the agent learns long-term dependencies.
    """

    def __init__(self, check_freq: int = 1000):
        self.check_freq = check_freq
        self.tire_wear_predictions = []
        self.fuel_predictions = []

    def on_step(self, locals_dict, globals_dict):
        """Called at each step."""
        if locals_dict['self'].num_timesteps % self.check_freq == 0:
            # Could analyze how well agent predicts future states
            pass

        return True


def create_ppo_lstm_agent(
    env: gym.Env,
    config: Optional[Dict] = None
) -> PPO_LSTM:
    """
    Create PPO-LSTM agent with sensible defaults for F1 racing.

    Args:
        env: F1 racing environment
        config: Optional configuration override

    Returns:
        PPO_LSTM agent
    """
    default_config = {
        'learning_rate': 3e-4,
        'n_steps': 2048,
        'batch_size': 64,
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.01,
        'lstm_hidden_size': 256,
        'num_lstm_layers': 2,
        'use_gru': False,
    }

    if config:
        default_config.update(config)

    return PPO_LSTM(env=env, **default_config)


# Additional: Attention-based variant
class AttentionRacingPolicy(nn.Module):
    """
    Experimental: Attention-based policy for racing.

    Uses self-attention to focus on relevant parts of track/history.
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2
    ):
        super().__init__()

        self.embedding = nn.Linear(observation_dim, hidden_dim)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs_sequence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            obs_sequence: Sequence of observations [batch, seq_len, obs_dim]

        Returns:
            (action_logits, values) tuple
        """
        # Embed observations
        embedded = self.embedding(obs_sequence)  # [batch, seq_len, hidden]

        # Apply transformer
        attended = self.transformer(embedded)  # [batch, seq_len, hidden]

        # Take last timestep for action
        last_hidden = attended[:, -1, :]  # [batch, hidden]

        # Compute policy and value
        action_logits = self.policy_head(last_hidden)
        values = self.value_head(last_hidden)

        return action_logits, values
